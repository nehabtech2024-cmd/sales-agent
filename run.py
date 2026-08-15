#!/usr/bin/env python3
"""
run.py - start the voice agent.

Finds the binary and weights that the notebook's setup cells put down, locates the app
files wherever they happen to live in this repo, then hands off to the bridge.

    python run.py                  # serve the UI on :8998
    python run.py --selftest 25    # no browser: prove the audio pipes work
    python run.py --port 9000
"""

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("PPLEX_HOME", "/kaggle/temp/pplex-runtime"
                              if Path("/kaggle").exists()
                              else Path.home() / ".cache" / "pplex"))


def locate(filename):
    """Look in the usual spots, then anywhere in the repo. Layout shouldn't matter."""
    for candidate in (REPO / "app" / filename, REPO / filename):
        if candidate.is_file():
            return candidate
    hits = sorted(p for p in REPO.rglob(filename)
                  if p.is_file() and ".git" not in p.parts)
    return hits[0] if hits else None


def inventory():
    """Everything tracked in the repo, so a missing file is obvious at a glance."""
    out = []
    for p in sorted(REPO.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            out.append(str(p.relative_to(REPO)))
    return out


def find_binary():
    hits = [p for p in glob.glob(str(RUNTIME / "**" / "personaplex"), recursive=True)
            if os.path.isfile(p)]
    if not hits:
        sys.exit(f"No personaplex binary under {RUNTIME}.\n"
                 "Run the setup cells in kaggle-vscode.ipynb first, or point PPLEX_HOME\n"
                 "at a directory that already has the moshi.cpp release in it.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--selftest", type=float, default=0,
                    help="seconds; run headless and report whether audio flows")
    ap.add_argument("--voice", default="NATF1")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--rate", type=float, default=0.75,
                    help="speaking pace; 0.75 = three quarter speed, pitch unchanged")
    args = ap.parse_args()

    bridge = locate("bridge.py")
    ui = locate("index.html")

    if bridge is None or ui is None:
        missing = [n for n, v in (("bridge.py", bridge), ("index.html", ui)) if v is None]
        print(f"Can't find {' and '.join(missing)} anywhere under {REPO}\n")
        files = inventory()
        if files:
            print("This repo contains:")
            for f in files:
                print("   ", f)
        else:
            print("This repo appears to be empty.")
        sys.exit("\nPush the app files (bridge.py, index.html) and pull again.")

    ppx = find_binary()
    bindir = os.path.dirname(ppx)
    print("bridge :", bridge)
    print("ui     :", ui)
    print("binary :", ppx)
    print("weights:", bindir)

    cmd = [sys.executable, "-u", str(bridge),
           "--binary", ppx,
           "--workdir", bindir,
           "--ui", str(ui),
           "--host", args.host,
           "--port", str(args.port),
           "--rate", str(args.rate)]

    if args.selftest:
        prompt = ""
        pf = Path(args.prompt_file) if args.prompt_file else None
        if pf is None:
            for name in ("riya.txt", "prompt.txt"):
                pf = locate(name)
                if pf:
                    break
        if pf and Path(pf).is_file():
            prompt = Path(pf).read_text().strip()
            print("prompt :", pf)
        cmd += ["--selftest", str(args.selftest), "--voice", args.voice,
                "--prompt", prompt]
    else:
        print(f"\nUI on http://localhost:{args.port}")
        print(f"Through code-server in Kaggle: <ngrok-url>/proxy/{args.port}/\n")

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
