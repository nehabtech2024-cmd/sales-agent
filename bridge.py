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


class Stretcher:
    """WSOLA: changes duration without changing pitch.

    Speech gets stretched by `rate`; silence passes through untouched, and gets
    compressed when the browser is running a backlog. That last part is what keeps
    the call from drifting - the model still emits audio at exactly real time, so
    the extra duration has to come out of the gaps between her sentences.

    The analysis grid advances by exactly hop*rate regardless of where the
    similarity search lands, so search offsets stay bounded perturbations instead
    of accumulating into drift. The search is also biased toward the grid position,
    which stops it locking onto a neighbouring pitch period.
    """

    def __init__(self, rate=1.0, win=1024, search=128, sr=SAMPLE_RATE):
        self.rate = float(rate)
        self.win = win
        self.hop_s = win // 2
        self.search = search
        self.sr = sr
        self.window = np.hanning(win).astype(np.float32)
        # prefer offsets near the grid; sigma ~ half the search range
        d = np.arange(-search, search + 1, dtype=np.float32)
        self.bias = np.exp(-0.5 * (d / (search * 0.6)) ** 2)
        self.buf = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(self.hop_s, dtype=np.float32)
        self.template = None
        self.ideal = 0.0
        self.in_n = 0
        self.out_n = 0
        self.silence_thresh = 0.005
        self.backlog_cap = 1.5

    def backlog(self):
        return (self.out_n - self.in_n) / self.sr

    def set_rate(self, rate):
        self.rate = float(rate)

    def _local_rate(self, frame):
        speech = float(np.sqrt(np.mean(frame * frame))) > self.silence_thresh
        b = self.backlog()
        if speech:
            r = self.rate
            if b > self.backlog_cap:
                r = min(1.0, self.rate + 0.15)
            return r
        return 3.0 if b > 0.25 else 1.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.rate == 1.0 and self.backlog() <= 0.25:
            self.in_n += len(x)
            self.out_n += len(x)
            return x

        self.buf = np.concatenate([self.buf, x])
        self.in_n += len(x)
        L, Hs, S = self.win, self.hop_s, self.search
        out = []

        while True:
            nominal = int(round(self.ideal))
            if nominal + L + Hs + S >= len(self.buf):
                break

            # find the frame that continues most smoothly from what we just emitted
            if self.template is None:
                pos = nominal
            else:
                lo = max(0, nominal - S)
                hi = min(len(self.buf) - L - Hs, nominal + S)
                if hi <= lo:
                    pos = nominal
                else:
                    region = self.buf[lo:hi + Hs]
                    corr = np.correlate(region, self.template, "valid")
                    energy = np.sqrt(np.convolve(region * region,
                                                 np.ones(Hs, np.float32),
                                                 "valid")) + 1e-6
                    score = corr / energy
                    b = self.bias[(lo - nominal + S):(hi - nominal + S + 1)]
                    if len(b) == len(score):
                        score = score * b
                    pos = lo + int(np.argmax(score))

            frame = self.buf[pos:pos + L] * self.window
            frame[:Hs] += self.tail
            out.append(frame[:Hs].copy())
            self.tail = frame[Hs:].copy()
            self.template = self.buf[pos + Hs:pos + 2 * Hs].copy()

            # grid advances by the ideal amount, not by where the search landed
            r = self._local_rate(self.buf[pos:pos + L])
            self.ideal += Hs * r
            self.out_n += Hs

            if self.ideal > 4 * L:
                drop = int(self.ideal) - 2 * L
                self.buf = self.buf[drop:]
                self.ideal -= drop

        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class Engine:
    """One personaplex process plus its two pipes."""

    def __init__(self, binary: str, workdir: str, on_event, rate: float = 1.0):
        self.binary = binary
        self.workdir = workdir
        self.rate = float(rate)
        self.stretch = Stretcher(self.rate)
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
        self.stretch = Stretcher(self.rate)
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
            # fps is measured on what the model produced, before any time-stretch
            self._fps_window.append((now, len(data)))
            cutoff = now - 3.0
            while self._fps_window and self._fps_window[0][0] < cutoff:
                self._fps_window.pop(0)

            pcm = self.stretch.process(np.frombuffer(data, dtype="<f4"))
            if len(pcm):
                await self._emit({"t": "audio", "pcm": f32_to_i16(pcm.tobytes())})

    def set_rate(self, rate: float):
        rate = max(0.5, min(1.5, float(rate)))
        self.rate = rate
        self.stretch.set_rate(rate)

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
    def __init__(self, binary, workdir, ui_file, rate=1.0):
        self.ui_file = ui_file
        self.ws = None
        self.engine = Engine(binary, workdir, self._on_event, rate=rate)

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
            if "rate" in cmd:
                self.engine.set_rate(cmd["rate"])
            await self.engine.start(
                prompt=cmd.get("prompt", ""),
                voice=cmd.get("voice", "NATF1"),
                temperature=cmd.get("temp", 0.8),
                context=cmd.get("ctx", 1500),
                seed=cmd.get("seed"),
            )
        elif kind == "rate":
            self.engine.set_rate(cmd.get("v", 1.0))
            await self._on_event({"t": "state", "state": self.engine.state,
                                  "msg": f"Pace set to {self.engine.rate:.2f}x."})
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
                "rate": round(self.engine.rate, 2),
                "uptime": round(time.monotonic() - self.engine.started_at, 1)
                if self.engine.started_at else 0,
            })


async def selftest(binary, workdir, voice, prompt, seconds, rate=1.0):
    """Run the whole pipe rig without a browser: feed silence, see if audio comes back."""
    got = {"bytes": 0, "text": ""}

    async def sink(msg):
        if msg["t"] == "audio":
            got["bytes"] += len(msg["pcm"])
        elif msg["t"] == "text":
            got["text"] += msg["v"]

    eng = Engine(binary, workdir, sink, rate=rate)
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
    print(f"pace           : {rate:.2f}x")
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
    ap.add_argument("--rate", type=float, default=1.0,
                    help="speaking pace; 0.75 = three quarter speed, pitch unchanged")
    args = ap.parse_args()

    if args.selftest:
        asyncio.run(selftest(args.binary, args.workdir, args.voice,
                             args.prompt, args.selftest, args.rate))
        return

    bridge = Bridge(args.binary, args.workdir, args.ui, rate=args.rate)
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
