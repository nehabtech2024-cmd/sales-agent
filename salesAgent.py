# ============================================================
# SCRIPT 2 — SALES VOICE AGENT  (Canary-Qwen STT  ->  Groq LLM  ->  Parler TTS  ->  Twilio)
#   GPU0: Canary-Qwen-2.5B (STT only now)   |   GPU1: Indic Parler-TTS (separate venv)
#   LLM replies now come from Groq's API instead of the local Canary-Qwen LLM head.
# Keep this cell running for the whole call.
# ============================================================
import os, sys, json, re, time, uuid, logging, threading, subprocess, traceback

CFG = json.load(open("/kaggle/working/agent_config.json"))
for k, v in CFG.items():
    os.environ[k] = str(v)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["HF_HOME"]        = "/kaggle/working/hf_cache"
os.environ["NEMO_CACHE_DIR"] = "/kaggle/working/nemo_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests, numpy as np, soundfile as sf, librosa, torch
from flask import Flask, request, Response, send_file
from twilio.rest import Client
from pyngrok import ngrok

WORK, AUDIO_DIR = "/kaggle/working", "/kaggle/working/agent_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
VPY, PORT, TTS_PORT = f"{WORK}/tts_env/bin/python", 5050, 5051
TTS_URL = f"http://127.0.0.1:{TTS_PORT}"
assert os.path.exists(VPY), "TTS venv missing — run SCRIPT 1 first"

SID, TOKEN = CFG["TWILIO_ACCOUNT_SID"], CFG["TWILIO_AUTH_TOKEN"]
FROM_N, TO_N = CFG["TWILIO_FROM_NUMBER"], CFG["MY_VERIFIED_NUMBER"]
AGENT_NAME, COMPANY, PRODUCT = CFG["AGENT_NAME"], CFG["COMPANY"], CFG["PRODUCT"]
VOICE_DESC, MAX_TURNS = CFG["VOICE_DESCRIPTION"], int(CFG["MAX_TURNS"])
PUBLIC_URL = ""

# --- Groq LLM config (add GROQ_API_KEY to agent_config.json, GROQ_MODEL optional) ---

GROQ_MODEL   = CFG.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
assert GROQ_API_KEY, "GROQ_API_KEY missing — add it to agent_config.json (or set env var)"

# ============================================================
# 1) START THE TTS SERVICE (separate env, GPU 1)
# ============================================================
def tts_alive():
    try: return requests.get(f"{TTS_URL}/health", timeout=3).status_code == 200
    except Exception: return False

def start_tts():
    if tts_alive():
        print("[tts] already running ✅"); return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1" if torch.cuda.device_count() > 1 else "0"
    env["TTS_PORT"] = str(TTS_PORT)
    log = open(f"{WORK}/tts_server.log", "w")
    p = subprocess.Popen([VPY, f"{WORK}/tts_server.py"], stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[tts] starting on GPU {env['CUDA_VISIBLE_DEVICES']} (first run downloads ~3.6 GB) ...")
    t0 = time.time()
    while time.time() - t0 < 2400:
        if tts_alive():
            print(f"[tts] ready in {time.time()-t0:.0f}s ✅"); return p
        if p.poll() is not None:
            print(open(f"{WORK}/tts_server.log").read()[-3000:])
            raise RuntimeError("TTS server crashed — see log above")
        time.sleep(5)
    raise RuntimeError("TTS server did not become ready in time")

TTS_PROC = start_tts()

# ============================================================
# 2) LOAD CANARY-QWEN (STT ONLY NOW — LLM head no longer used)
# ============================================================
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)
from nemo.collections.speechlm2.models import SALM

DEVICE, IN_SR = ("cuda:0" if torch.cuda.is_available() else "cpu"), 16000
if "salm" not in globals():
    print("[stt] loading Canary-Qwen-2.5B (for transcription only) ...")
    t0 = time.time()
    salm = SALM.from_pretrained("nvidia/canary-qwen-2.5b").to(DEVICE).eval()
    print(f"[stt] loaded in {time.time()-t0:.1f}s ✅")
GPU_LOCK = threading.Lock()
file=open("prompt.txt","r")
content=file.read()
file.close()
SYSTEM =content

def to_16k(src, out=None):
    out = out or f"{AUDIO_DIR}/in_{uuid.uuid4().hex[:8]}_16k.wav"
    wav, _ = librosa.load(src, sr=IN_SR, mono=True)
    sf.write(out, wav, IN_SR)
    return out, len(wav) / IN_SR

@torch.inference_mode()
def transcribe(wav_path):
    ids = salm.generate(prompts=[[{"role": "user",
        "content": f"Transcribe the following: {salm.audio_locator_tag}",
        "audio": [wav_path]}]], max_new_tokens=256)
    return salm.tokenizer.ids_to_text(ids[0].cpu()).strip()

def clean(t):
    t = (t or "").strip().strip('"').strip()
    t = re.sub(rf'^\s*({AGENT_NAME}|agent|assistant|reply)\s*[:\-]\s*', '', t, flags=re.I)
    t = re.sub(r'[*#`_>\[\]]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    parts = re.split(r'(?<=[.!?])\s+', t)
    return (" ".join(parts[:2]))[:260] or "Sorry, could you repeat that?"

# ------------------------------------------------------------
# LLM reply generation
# ------------------------------------------------------------
# --- Original local-LLM version (Canary-Qwen) — commented out ---
# @torch.inference_mode()
# def agent_reply(user_text, history):
#     convo = "\n".join(f"{'Customer' if r=='customer' else AGENT_NAME}: {x}" for r, x in history[-6:])
#     prompt = (f"{SYSTEM}\n\nConversation so far:\n{convo}\n\nCustomer: {user_text}\n\n"
#               f"Write only {AGENT_NAME}'s next reply.")
#     with salm.llm.disable_adapter():
#         ids = salm.generate(prompts=[[{"role": "user", "content": prompt}]], max_new_tokens=70)
#     return clean(salm.tokenizer.ids_to_text(ids[0].cpu()))

# --- New Groq API version ---
def agent_reply(user_text, history):
    convo = "\n".join(f"{'Customer' if r=='customer' else AGENT_NAME}: {x}" for r, x in history[-6:])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Conversation so far:\n{convo}\n\nCustomer: {user_text}\n\n"
            f"Write only {AGENT_NAME}'s next reply."
        )},
    ]
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": 70, "temperature": 0.7}
    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[groq] error: {e}", flush=True)
        text = "Sorry, could you repeat that?"
    return clean(text)

def tts(text, filename):
    """Parler -> 8 kHz mono PCM16 WAV (the format Twilio <Play> likes best)."""
    r = requests.post(f"{TTS_URL}/tts", json={"text": text, "description": VOICE_DESC}, timeout=600)
    r.raise_for_status()
    raw = f"{AUDIO_DIR}/_raw_{filename}"
    open(raw, "wb").write(r.content)
    wav, sr = sf.read(raw, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=8000)
    peak = float(np.max(np.abs(wav))) or 1.0
    sf.write(f"{AUDIO_DIR}/{filename}", (wav / peak * 0.95), 8000, subtype="PCM_16")
    return filename

# ============================================================
# 3) PRE-RENDER THE FIXED CLIPS
# ============================================================
print("[tts] rendering fixed clips ...")
CLIPS = {}
for key, txt in [("greeting", CFG["GREETING"]),
                 ("filler",   "Sure, one moment please."),
                 ("repeat",   "Sorry, I did not catch that. Could you say it again?"),
                 ("bye",      CFG["CLOSING"]),
                 ("oops",     "Sorry, I am having a technical issue. I will call you back later.")]:
    CLIPS[key] = tts(txt, f"{key}.wav"); print("   ", key, "✅")

# ============================================================
# 4) TWILIO WEBHOOK APP
# ============================================================
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
STATE, JOBS, TRANSCRIPT, CALL_DONE = {}, {}, [], threading.Event()
BYE_WORDS = ("not interested", "no thanks", "no thank you", "goodbye", "good bye",
             "bye", "stop calling", "remove me", "don't call", "do not call", "busy right now")

def xml(body): return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>',
                               mimetype="text/xml")
def play(f):   return f"<Play>{PUBLIC_URL}/audio/{f}</Play>"
def record():  return (f'<Record action="{PUBLIC_URL}/handle" method="POST" maxLength="20" '
                       f'timeout="3" trim="trim-silence" playBeep="false"/>'
                       f'<Redirect method="POST">{PUBLIC_URL}/noinput</Redirect>')

@app.route("/audio/<path:name>", methods=["GET", "POST"])
def audio(name):
    p = os.path.join(AUDIO_DIR, os.path.basename(name))
    if not os.path.exists(p): return "not found", 404
    return send_file(p, mimetype="audio/wav", conditional=False)

@app.route("/voice", methods=["POST", "GET"])
def voice():
    sid = request.values.get("CallSid", "local")
    STATE[sid] = {"history": [], "turn": 0, "nudge": 0}
    print("\n[call] answered — starting conversation\n", flush=True)
    print(f"  {AGENT_NAME}: {CFG['GREETING']}", flush=True)
    return xml('<Pause length="1"/>' + play(CLIPS["greeting"]) + record())

def fetch_recording(url, tries=15):
    u = url if url.endswith(".wav") else url + ".wav"
    for _ in range(tries):
        try:
            r = requests.get(u, auth=(SID, TOKEN), timeout=30)
            if r.status_code == 200 and len(r.content) > 2000:
                p = f"{AUDIO_DIR}/rec_{uuid.uuid4().hex[:8]}.wav"
                open(p, "wb").write(r.content); return p
        except Exception: pass
        time.sleep(1.0)
    return None

def process_turn(job, sid, rec_url):
    try:
        st = STATE.setdefault(sid, {"history": [], "turn": 0, "nudge": 0})
        st["turn"] += 1
        text, dur = "", 0.0
        path = fetch_recording(rec_url) if rec_url else None
        if path:
            wav16, dur = to_16k(path)
            if dur > 0.4:
                with GPU_LOCK:
                    t0 = time.time(); text = transcribe(wav16)
                print(f"  [stt {time.time()-t0:.1f}s] Customer: {text}", flush=True)
        if not text.strip():
            JOBS[job] = {"status": "done", "file": CLIPS["repeat"], "hangup": False}; return
        TRANSCRIPT.append(("Customer", text))
        hangup = st["turn"] >= MAX_TURNS or any(w in text.lower() for w in BYE_WORDS)
        # Groq call doesn't touch the GPU, so no need to hold GPU_LOCK here.
        t0 = time.time(); reply = agent_reply(text, st["history"])
        print(f"  [llm {time.time()-t0:.1f}s] {AGENT_NAME}: {reply}", flush=True)
        if hangup: reply = f"{reply} {CFG['CLOSING']}"
        st["history"] += [("customer", text), ("agent", reply)]
        TRANSCRIPT.append((AGENT_NAME, reply))
        t0 = time.time(); f = tts(reply, f"reply_{job}.wav")
        print(f"  [tts {time.time()-t0:.1f}s] audio ready", flush=True)
        JOBS[job] = {"status": "done", "file": f, "hangup": hangup}
    except Exception as e:
        traceback.print_exc()
        JOBS[job] = {"status": "error", "error": str(e)}

@app.route("/handle", methods=["POST"])
def handle():
    sid, rec = request.values.get("CallSid", "local"), request.values.get("RecordingUrl")
    job = uuid.uuid4().hex[:8]
    JOBS[job] = {"status": "pending"}
    threading.Thread(target=process_turn, args=(job, sid, rec), daemon=True).start()
    return xml(f'<Redirect method="POST">{PUBLIC_URL}/wait?job={job}&amp;n=0</Redirect>')

@app.route("/wait", methods=["POST", "GET"])
def wait():
    job, n = request.values.get("job"), int(request.values.get("n", 0))
    j = JOBS.get(job, {})
    if j.get("status") == "done":
        return xml(play(j["file"]) + ("<Hangup/>" if j.get("hangup") else record()))
    if j.get("status") == "error" or n > 28:
        return xml(play(CLIPS["oops"]) + "<Hangup/>")
    head = play(CLIPS["filler"]) if n == 0 else '<Pause length="2"/>'
    return xml(head + f'<Redirect method="POST">{PUBLIC_URL}/wait?job={job}&amp;n={n+1}</Redirect>')

@app.route("/noinput", methods=["POST", "GET"])
def noinput():
    st = STATE.setdefault(request.values.get("CallSid", "local"), {"history": [], "turn": 0, "nudge": 0})
    st["nudge"] += 1
    if st["nudge"] > 2:
        return xml(play(CLIPS["bye"]) + "<Hangup/>")
    return xml(play(CLIPS["repeat"]) + record())

@app.route("/status", methods=["POST"])
def status():
    s = request.values.get("CallStatus", "")
    print(f"[call] status = {s}", flush=True)
    if s in ("completed", "failed", "busy", "no-answer", "canceled"):
        CALL_DONE.set()
    return ("", 204)

# ============================================================
# 5) TUNNEL + PLACE THE CALL
# ============================================================
threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, threaded=True,
                                        debug=False, use_reloader=False), daemon=True).start()
time.sleep(2)
try: ngrok.kill()
except Exception: pass
ngrok.set_auth_token(CFG["NGROK_AUTHTOKEN"])
PUBLIC_URL = ngrok.connect(PORT, "http").public_url.replace("http://", "https://")
print("[net] public URL:", PUBLIC_URL)

client = Client(SID, TOKEN)

def place_call():
    CALL_DONE.clear(); TRANSCRIPT.clear()
    c = client.calls.create(to=TO_N, from_=FROM_N, url=f"{PUBLIC_URL}/voice", method="POST",
                            status_callback=f"{PUBLIC_URL}/status", status_callback_method="POST",
                            status_callback_event=["initiated", "ringing", "answered", "completed"],
                            timeout=30)
    print(f"\n📞 calling {TO_N} ... (sid {c.sid}) — pick up!\n")
    t0 = time.time()
    while not CALL_DONE.wait(timeout=10):
        if time.time() - t0 > 900: break
        if client.calls(c.sid).fetch().status in ("completed", "failed", "busy", "no-answer", "canceled"):
            break
    print("\n===== CALL ENDED — TRANSCRIPT =====")
    for who, what in TRANSCRIPT: print(f"{who}: {what}")
    print("===================================\nrun place_call() again for another call (models stay loaded)")
    return c.sid

place_call()