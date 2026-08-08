# ============================================================
# SCRIPT 2 — VOICE AGENT PIPELINE  (run top to bottom)
#   Section 1: STT  — Canary-Qwen-2.5B  (ASR mode)
#   Section 2: LLM  — Canary-Qwen-2.5B  (LLM mode, SAME model)
# ============================================================

# ---- 0. Env flags BEFORE importing torch/nemo (must match Script 1) ----
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")
os.environ.setdefault("NEMO_CACHE_DIR", "/kaggle/working/nemo_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time, urllib.request, logging
import torch, soundfile as sf, librosa

from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)                      # quieter logs

from nemo.collections.speechlm2.models import SALM        # Canary-Qwen

# ---- Device ----
# Only one model is loaded now (Canary-Qwen), so a single T4 is plenty.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IN_SR = 16000                                             # Canary input sample rate

# ============================================================
# LOAD MODEL ONCE (heavy) — kept resident for low latency
# ============================================================
print("Loading Canary-Qwen-2.5B ...")
t = time.time()
# Kept in native (bf16) precision. T4 has no bf16 tensor cores so it runs via
# math kernels (correct, just not the fastest path). For a speed experiment you
# can try `.half()` instead of native, but test output for fp16 NaNs first.
salm = SALM.from_pretrained("nvidia/canary-qwen-2.5b").to(DEVICE).eval()
print(f"  loaded in {time.time()-t:.1f}s")

# ============================================================
# SECTION 1 — SPEECH-TO-TEXT  (Canary-Qwen, ASR mode)
# ============================================================
def prepare_audio_16k(src, out="/kaggle/working/input_16k.wav"):
    """Download (if URL) + convert ANY audio to 16 kHz mono WAV for Canary."""
    local = src
    if str(src).startswith(("http://", "https://")):
        local = "/kaggle/working/_dl_audio"
        urllib.request.urlretrieve(src, local)
    wav, _ = librosa.load(local, sr=IN_SR, mono=True)     # resample to 16k mono
    sf.write(out, wav, IN_SR)
    return out

@torch.inference_mode()
def transcribe(wav_path):
    ids = salm.generate(
        prompts=[[{"role": "user",
                   "content": f"Transcribe the following: {salm.audio_locator_tag}",
                   "audio": [wav_path]}]],
        max_new_tokens=256,
    )
    return salm.tokenizer.ids_to_text(ids[0].cpu()).strip()

# ---- Choose input ----
# (A) From the web (default): any direct link to .wav/.mp3/.flac
AUDIO_SOURCE = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"

# (B) From a local path instead (e.g. an uploaded Kaggle dataset file):
#     comment out (A) and uncomment the next line.
# AUDIO_SOURCE = "/kaggle/input/<your-dataset>/your_audio.wav"

wav16 = prepare_audio_16k(AUDIO_SOURCE)
t = time.time()
transcript = transcribe(wav16)
print(f"\n[STT {time.time()-t:.2f}s]  Transcript:\n  {transcript}\n")

# ============================================================
# SECTION 2 — LLM RESPONSE  (Canary-Qwen, LLM mode — SAME model)
# ============================================================
AGENT_SYSTEM = (
    "You are a helpful, friendly voice assistant. "
    "Reply to the user in 1-3 short, natural spoken sentences. "
    "Do not use markdown, lists, or emojis."
    "do not generate more than 7 words at a time."
)

@torch.inference_mode()
def agent_reply(user_text):
    # LLM mode: disable the ASR LoRA adapter so the base Qwen LLM answers.
    with salm.llm.disable_adapter():
        ids = salm.generate(
            prompts=[[{"role": "user",
                       "content": f'{AGENT_SYSTEM}\n\nUser said: "{user_text}"'}]],
            max_new_tokens=200,
        )
    return salm.tokenizer.ids_to_text(ids[0].cpu()).strip()

t = time.time()
response_text = agent_reply(transcript)
print(f"[LLM {time.time()-t:.2f}s]  Agent reply:\n  {response_text}\n")