#!/usr/bin/env python3
"""
bridge.py - puts a browser in front of moshi.cpp's `personaplex` binary.

The binary is a desktop app: it talks to a sound card through SDL. A Kaggle box
has no sound card, so we point SDL at its `disk` driver and hand it two named
pipes where it expects a device:

    browser mic  --ws-->  [pipe]  -->  SDL capture   -->  personaplex
    browser spkr <--ws--  [pipe]  <--  SDL playback  <--  personaplex

Both pipes carry float32 mono @ 24 kHz, which is exactly what personaplex asks
SDL for (want.freq=24000, want.format=AUDIO_F32, want.channels=1). On the
websocket we use int16 to halve the bytes going through the tunnel.

Pacing: Python is the clock. We read the playback pipe on a wall-clock budget of
24000*4 bytes/sec and no faster, so when the model outruns us the pipe fills, SDL's
write blocks, and the model throttles itself. Mic audio arrives from the browser in
real time already; if the model falls behind we drop the oldest buffered audio
rather than let latency grow without bound.
"""

import argparse
import asyncio
import errno
import fcntl
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
from aiohttp import web, WSMsgType

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920            # 80 ms, one mimi frame at 12.5 fps
FRAME_BYTES_F32 = FRAME_SAMPLES * 4
BYTES_PER_SEC = SAMPLE_RATE * 4  # float32 mono
TICK = 0.02                      # bridge loop period, seconds
MAX_INPUT_BACKLOG = FRAME_BYTES_F32 * 4   # ~320 ms, then we start dropping
F_SETPIPE_SZ = 1031

VOICES = ["NATF0", "NATF1", "NATF2", "NATF3",
          "NATM0", "NATM1", "NATM2", "NATM3",
          "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
          "VARM0", "VARM1", "VARM2", "VARM3", "VARM4"]


def log(*a):
    print(f"[bridge {time.strftime('%H:%M:%S')}]", *a, flush=True)


def f32_to_i16(buf: bytes) -> bytes:
    x = np.frombuffer(buf, dtype="<f4")
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def i16_to_f32(buf: bytes) -> bytes:
    x = np.frombuffer(buf, dtype="<i2").astype("<f4") / 32768.0
    return x.tobytes()


def set_pipe_size(fd: int, size: int):
    try:
        fcntl.fcntl(fd, F_SETPIPE_SZ, size)
    except OSError:
        pass  # not permitted / not supported; defaults are fine


class Engine:
    """One personaplex process plus its two pipes."""

    def __init__(self, binary: str, workdir: str, on_event):
        self.binary = binary
        self.workdir = workdir
        self.on_event = on_event          # called with dict, from the event loop
        self.proc = None
        self.in_fd = None                 # we write mic audio here
        self.out_fd = None                # we read model audio here
        self.in_buf = bytearray()
        self.state = "idle"
        self.started_at = 0.0
        self.ready = False
        self.out_bytes = 0
        self._fps_window = []
        self._tasks = []
        self._pipe_dir = Path(workdir) / "_pipes"

    # ---------- lifecycle ----------

    async def start(self, prompt, voice, temperature, context, seed=None):
        await self.stop()

        self._pipe_dir.mkdir(parents=True, exist_ok=True)
        mic = self._pipe_dir / "mic.f32"
        spk = self._pipe_dir / "spk.f32"
        for p in (mic, spk):
            if p.exists():
                p.unlink()
            os.mkfifo(p, 0o600)

        env = dict(os.environ)
        env["SDL_AUDIODRIVER"] = "disk"
        env["SDL_DISKAUDIOFILEIN"] = str(mic)   # SDL reads our mic audio from here
        env["SDL_DISKAUDIOFILE"] = str(spk)     # SDL writes model audio to here
        env["SDL_DISKAUDIODELAY"] = "0"         # we do the pacing, not SDL
        env["SDL_VIDEODRIVER"] = "dummy"

        cmd = [self.binary, "-v", voice, "-c", str(int(context)),
               "-t", str(float(temperature))]
        if seed is not None:
            cmd += ["-s", str(int(seed))]
        if prompt and prompt.strip():
            cmd += ["-p", prompt.strip()]

        log("launching:", " ".join(repr(c) if " " in c else c for c in cmd))
        self.state = "starting"
        self.ready = False
        self.out_bytes = 0
        self._fps_window = []
        self.in_buf = bytearray()
        self.started_at = time.monotonic()

        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        await self._emit({"t": "state", "state": "starting",
                          "msg": "Loading the model. This takes a few seconds."})

        # Read end of a fifo opens immediately with O_NONBLOCK even with no writer.
        self.out_fd = os.open(spk, os.O_RDONLY | os.O_NONBLOCK)
        set_pipe_size(self.out_fd, 65536)

        self._tasks = [
            asyncio.create_task(self._pump_stdout()),
            asyncio.create_task(self._open_mic(mic)),
            asyncio.create_task(self._pump_audio()),
            asyncio.create_task(self._watch_proc()),
        ]

    async def _open_mic(self, path):
        """Write end of a fifo raises ENXIO until the reader (SDL) shows up."""
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.returncode is not None:
                return
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                if e.errno != errno.ENXIO:
                    raise
                await asyncio.sleep(0.2)
                continue
            set_pipe_size(fd, MAX_INPUT_BACKLOG)
            self.in_fd = fd
            log("mic pipe connected")
            return
        log("mic pipe never opened - SDL may not have a disk audio driver")

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None
        for attr in ("in_fd", "out_fd"):
            fd = getattr(self, attr)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)
        self.state = "idle"
        self.ready = False

    async def _watch_proc(self):
        rc = await self.proc.wait()
        self.state = "stopped"
        self.ready = False
        await self._emit({"t": "state", "state": "stopped",
                          "msg": f"The engine exited (code {rc}). See the log."})

    # ---------- audio ----------

    def feed(self, i16_bytes: bytes):
        """Mic audio from the browser."""
        self.in_buf.extend(i16_to_f32(i16_bytes))
        if len(self.in_buf) > MAX_INPUT_BACKLOG:
            # Model is behind. Drop the oldest audio so latency stays bounded.
            drop = len(self.in_buf) - MAX_INPUT_BACKLOG
            del self.in_buf[:drop]
        self._flush_in()

    def _flush_in(self):
        if self.in_fd is None or not self.in_buf:
            return
        try:
            n = os.write(self.in_fd, self.in_buf)
        except BlockingIOError:
            return
        except OSError:
            self.in_fd = None
            return
        if n:
            del self.in_buf[:n]

    async def _pump_audio(self):
        """Drain the playback pipe at real-time rate. That rate is the backpressure."""
        last = time.monotonic()
        credit = 0.0
        while True:
            await asyncio.sleep(TICK)
            now = time.monotonic()
            credit = min(credit + (now - last) * BYTES_PER_SEC, BYTES_PER_SEC * 0.15)
            last = now
            self._flush_in()

            n = int(credit) & ~3          # whole float32 samples only
            if n <= 0 or self.out_fd is None:
                continue
            try:
                data = os.read(self.out_fd, n)
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""
            if not data:
                continue
            credit -= len(data)
            self.out_bytes += len(data)
            self._fps_window.append((now, len(data)))
            cutoff = now - 3.0
            while self._fps_window and self._fps_window[0][0] < cutoff:
                self._fps_window.pop(0)
            await self._emit({"t": "audio", "pcm": f32_to_i16(data)})

    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1][0] - self._fps_window[0][0]
        if span <= 0.2:
            return 0.0
        total = sum(b for _, b in self._fps_window)
        return (total / FRAME_BYTES_F32) / span

    # ---------- text ----------

    async def _pump_stdout(self):
        """personaplex prints its transcript token by token, with no newlines."""
        pending = b""
        while True:
            chunk = await self.proc.stdout.read(512)
            if not chunk:
                return
            pending += chunk
            try:
                text = pending.decode("utf-8")
                pending = b""
            except UnicodeDecodeError:
                continue  # split multi-byte char, wait for the rest
            sys.stdout.write(text)
            sys.stdout.flush()
            if not self.ready:
                await self._emit({"t": "log", "v": text})
                if "ready" in text:
                    self.ready = True
                    self.state = "ready"
                    took = time.monotonic() - self.started_at
                    await self._emit({"t": "state", "state": "ready",
                                      "msg": f"Engine up in {took:.1f}s. Say hello."})
            else:
                await self._emit({"t": "text", "v": text})

    async def _emit(self, msg):
        await self.on_event(msg)


class Bridge:
    def __init__(self, binary, workdir, ui_file):
        self.ui_file = ui_file
        self.ws = None
        self.engine = Engine(binary, workdir, self._on_event)

    async def _on_event(self, msg):
        ws = self.ws
        if ws is None or ws.closed:
            return
        try:
            if msg["t"] == "audio":
                await ws.send_bytes(msg["pcm"])
            else:
                await ws.send_json(msg)
        except (ConnectionResetError, RuntimeError):
            pass

    async def index(self, request):
        return web.FileResponse(self.ui_file)

    async def config(self, request):
        return web.json_response({"voices": VOICES, "sampleRate": SAMPLE_RATE})

    async def websocket(self, request):
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        if self.ws is not None and not self.ws.closed:
            await ws.send_json({"t": "state", "state": "busy",
                                "msg": "Another tab already has the line. Close it and reload."})
            await ws.close()
            return ws

        self.ws = ws
        log("client connected from", request.remote)
        await ws.send_json({"t": "state", "state": "idle", "msg": "Ready when you are."})
        stats = asyncio.create_task(self._stats_loop())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    self.engine.feed(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._command(json.loads(msg.data))
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            stats.cancel()
            self.ws = None
            log("client gone; stopping engine")
            await self.engine.stop()
        return ws

    async def _command(self, cmd):
        kind = cmd.get("t")
        if kind == "start":
            await self.engine.start(
                prompt=cmd.get("prompt", ""),
                voice=cmd.get("voice", "NATF1"),
                temperature=cmd.get("temp", 0.8),
                context=cmd.get("ctx", 1500),
                seed=cmd.get("seed"),
            )
        elif kind == "stop":
            await self.engine.stop()
            await self._on_event({"t": "state", "state": "idle", "msg": "Call ended."})

    async def _stats_loop(self):
        while True:
            await asyncio.sleep(0.5)
            await self._on_event({
                "t": "stats",
                "fps": round(self.engine.fps(), 2),
                "state": self.engine.state,
                "uptime": round(time.monotonic() - self.engine.started_at, 1)
                if self.engine.started_at else 0,
            })


async def selftest(binary, workdir, voice, prompt, seconds):
    """Run the whole pipe rig without a browser: feed silence, see if audio comes back."""
    got = {"bytes": 0, "text": ""}

    async def sink(msg):
        if msg["t"] == "audio":
            got["bytes"] += len(msg["pcm"])
        elif msg["t"] == "text":
            got["text"] += msg["v"]

    eng = Engine(binary, workdir, sink)
    await eng.start(prompt, voice, 0.8, 1500)

    deadline = time.monotonic() + seconds
    silence = (np.zeros(FRAME_SAMPLES, dtype="<i2")).tobytes()
    next_frame = time.monotonic()
    while time.monotonic() < deadline:
        if eng.proc is None or eng.proc.returncode is not None:
            print("\nengine exited early")
            break
        if eng.in_fd is not None:
            eng.feed(silence)
        next_frame += 0.08
        await asyncio.sleep(max(0.0, next_frame - time.monotonic()))

    secs = got["bytes"] / (SAMPLE_RATE * 2)
    print("\n" + "=" * 62)
    print(f"audio returned : {got['bytes']} bytes  ({secs:.1f}s of speech)")
    print(f"frame rate     : {eng.fps():.2f} fps   (12.5 = real time)")
    print(f"transcript     : {got['text'][:400] or '(nothing)'}")
    print("=" * 62)
    if got["bytes"] == 0:
        print("FAIL - nothing came back through the playback pipe.")
        print("       SDL probably has no disk audio driver. See the notebook's")
        print("       troubleshooting section.")
    else:
        print("PASS - the pipe rig works. Start the server.")
    await eng.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--ui", default=str(Path(__file__).with_name("index.html")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--selftest", type=float, default=0,
                    help="seconds; run without a browser and report")
    ap.add_argument("--voice", default="NATF1")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()

    if args.selftest:
        asyncio.run(selftest(args.binary, args.workdir, args.voice,
                             args.prompt, args.selftest))
        return

    bridge = Bridge(args.binary, args.workdir, args.ui)
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.add_routes([
        web.get("/", bridge.index),
        web.get("/api/config", bridge.config),
        web.get("/ws", bridge.websocket),
    ])
    log(f"serving on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
