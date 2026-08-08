# ============================================================
# Indic Parler-TTS on Kaggle — single cell, run top to bottom
# Requires: Internet = ON, Accelerator = GPU (T4), HF_TOKEN in Secrets,
# and terms accepted at huggingface.co/ai4bharat/indic-parler-tts
# ============================================================
HF_TOKEN=
import importlib
import importlib.util
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

# --- disable TensorFlow BEFORE transformers is imported anywhere ---
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)


# --- install only what's missing ---
if importlib.util.find_spec("parler_tts") is None:
    print("[setup] installing parler-tts ...")
    _pip("git+https://github.com/huggingface/parler-tts.git")

if importlib.util.find_spec("soundfile") is None:
    print("[setup] installing soundfile ...")
    _pip("soundfile")

# protobuf must stay <5 for descript-audiotools (parler-tts dependency)
try:
    from google.protobuf import __version__ as _pb
    if int(_pb.split(".")[0]) >= 5:
        print(f"[setup] protobuf {_pb} too new -> pinning 4.25.9")
        _pip("protobuf==4.25.9")
except Exception:
    _pip("protobuf==4.25.9")

importlib.invalidate_caches()

# --- imports ---
import torch
import soundfile as sf
import transformers
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from IPython.display import Audio, display

print(f"transformers {transformers.__version__} | torch {torch.__version__}")

# --- auth ---
token =HF_TOKEN
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"[warn] could not read HF_TOKEN secret: {e}")

if token:
    from huggingface_hub import login
    login(token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = token
    print("[auth] logged in to Hugging Face")
else:
    print("[warn] no token — the gated download will fail with 401")

# --- config ---
MODEL_ID = "ai4bharat/indic-parler-tts"
OUT_DIR = "/kaggle/working"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device.startswith("cuda") else torch.float32
if device == "cpu":
    print("[warn] no GPU detected — generation will be very slow")
print(f"device={device} dtype={dtype}")

# --- load (first run downloads ~3.6 GB) ---
# NOTE: do NOT pass attn_implementation here — it propagates to the T5
# text encoder, which has no SDPA support in transformers 4.46.1.
print("[load] fetching model, this takes a few minutes on first run ...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device).eval()

# apply SDPA to the decoder only (that's where the speed win is)
try:
    model.decoder.config._attn_implementation = "sdpa"
    print("[load] SDPA enabled on decoder")
except Exception as e:
    print(f"[load] SDPA not applied ({e}) — using eager attention")

tok = AutoTokenizer.from_pretrained(MODEL_ID)                                      # transcript
desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)  # caption

SR = model.config.sampling_rate
print(f"[load] ready | sampling_rate={SR}")


# --- synthesis ---
def synth(prompt: str, description: str):
    """prompt = what to say; description = how it should sound."""
    d = desc_tok(description, return_tensors="pt").to(device)
    p = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(
            input_ids=d.input_ids,
            attention_mask=d.attention_mask,
            prompt_input_ids=p.input_ids,
            prompt_attention_mask=p.attention_mask,
        )
    return gen.to(torch.float32).cpu().numpy().squeeze()


def speak(prompt, description, filename="out.wav"):
    audio = synth(prompt, description)
    path = os.path.join(OUT_DIR, filename)
    sf.write(path, audio, SR)
    print(f"\n{filename}  ({len(audio)/SR:.2f}s)  <- {prompt[:60]}")
    display(Audio(path))
    return path


# --- demo ---
HINDI_DESC = ("Divya's voice is clear and slightly expressive, at a moderate pace. "
              "The recording is very high quality with no background noise.")
ENGLISH_DESC = ("Mary speaks clearly at a moderate pace. The recording is very "
                "high quality with no background noise.")

speak("अरे, तुम आज कैसे हो? मैं ठीक हूँ, धन्यवाद।", HINDI_DESC, "hindi.wav")
speak("Sure, I've cancelled that booking for you.", ENGLISH_DESC, "english.wav")