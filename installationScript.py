# ============================================================
# SCRIPT 1 — ONE-TIME SETUP   (Kaggle · GPU T4 x2 · Internet ON)
# Run this ONCE per session. If any import in Script 2 fails right
# after this, restart the kernel once and run Script 2 (cache survives).
# ============================================================
import os, sys, subprocess

# Flags that MUST be set before torch/nemo are ever imported.
# NeMo .nemo checkpoints need full (non weights-only) torch.load on PyTorch >= 2.6
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"        # cache on fast local disk
os.environ["NEMO_CACHE_DIR"] = "/kaggle/working/nemo_cache"

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# 1) System audio libs (ffmpeg -> mp3/misc decoding, libsndfile -> soundfile)
subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg", "libsndfile1"], check=False)

# 2) Light audio I/O + HF helpers
pip("-U", "huggingface_hub", "soundfile", "librosa", "kaldialign")

# 3) NVIDIA NeMo (ASR + Speech-LLM + TTS collections).
#    Plain PyPI install: pip picks the newest release compatible with Kaggle's
#    Python/torch and installs ON TOP of the existing CUDA PyTorch (not reinstalled).
pip("-U", "nemo_toolkit[asr,tts]")

# 4) FIX: Canary-Qwen uses plain LoRA (no torchao). Kaggle ships an OLD torchao
#    (0.10) that the newer `peft` rejects with an ImportError during LoRA install.
#    Removing it makes peft skip that path cleanly. (We do NOT upgrade torchao,
#    because newer versions tend to drag in a different torch and break CUDA.)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

# 5) Sanity check
import torch
print("\n--- versions ---")
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "| CUDA:", torch.version.cuda,
      "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
import nemo
print("nemo  :", nemo.__version__)

try:
    import torchao  # noqa
    print("⚠️  torchao still present — run this cell again or restart the kernel.")
except ImportError:
    print("torchao removed ✅")

print("\nSetup OK ✅  — continue to Script 2")