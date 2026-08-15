#!/usr/bin/env python3
"""
run.py - start the voice agent.

Finds the binary and weights that the notebook's setup cells put down, then hands off
to the bridge.

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
    ap.add_argument("--prompt-file", default=str(REPO / "prompts" / "riya.txt"))
    args = ap.parse_args()

    ppx = find_binary()
    bindir = os.path.dirname(ppx)
    print("binary :", ppx)
    print("weights:", bindir)

    cmd = [sys.executable, "-u", str(REPO / "app" / "bridge.py"),
           "--binary", ppx,
           "--workdir", bindir,
           "--ui", str(REPO / "app" / "index.html"),
           "--host", args.host,
           "--port", str(args.port)]

    if args.selftest:
        prompt = ""
        pf = Path(args.prompt_file)
        if pf.exists():
            prompt = pf.read_text().strip()
        cmd += ["--selftest", str(args.selftest), "--voice", args.voice,
                "--prompt", prompt]
    else:
        print(f"\nUI on http://localhost:{args.port}")
        print(f"Through code-server in Kaggle: <ngrok-url>/proxy/{args.port}/\n")

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
