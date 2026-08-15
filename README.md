# sales-agent

A full-duplex voice sales agent: NVIDIA PersonaPlex 7B (q4_k) running under
[moshi.cpp](https://github.com/Codes4Fun/moshi.cpp), with a browser front end so you can
call it from a phone or laptop. Runs on a single T4.

The model binary is a native SDL desktop app with no server. This repo adds the part that
was missing: a websocket bridge that hands SDL two named pipes instead of a sound card, and
a web UI on the other end.

```
browser mic   --ws-->  [pipe]  -->  SDL capture   -->  personaplex
browser spkr  <--ws--  [pipe]  <--  SDL playback  <--  personaplex
```

Audio on the pipes is float32 mono @ 24 kHz, which is what personaplex asks SDL for. The
bridge drains the playback pipe on a wall clock at exactly real-time rate and no faster, so
when the model runs ahead the pipe fills and SDL blocks, and when it runs behind you see it
on the meter.

## Layout

```
run.py              start the agent
app/bridge.py       websocket server, pipe handling, pacing
app/index.html      the call UI
prompts/riya.txt    the persona prompt
```

This repo is the application only. Environment setup — apt packages, the moshi.cpp binary,
the 5 GB of weights — lives in `kaggle-vscode.ipynb`, because it's Kaggle-specific and has
to run once per session regardless.

The binary and weights land in `$PPLEX_HOME` (default `/kaggle/temp/pplex-runtime`), outside
the repo. Nothing large is ever committed.

## Running

Once the notebook's setup cells have run:

```bash
python run.py --selftest 25   # no browser needed: does audio actually flow?
python run.py                 # serve the UI on :8998
```

`--selftest` starts the engine, feeds it silence, and reports whether audio came back and at
what frame rate. Run it before opening a browser — a failure there is about SDL and pipes,
and a failure after it is about your microphone or your tunnel.

## On Kaggle

Requires **GPU T4** (one is enough, q4_k needs ~6 GB) and **Internet on**.

The notebook opens two ngrok tunnels — one for the editor, one for the agent — using a
token from a different account for each, since a free ngrok account allows one agent session
at a time.

If you only have one token, code-server can proxy the agent instead:

```
https://<your-code-server-url>/proxy/8998/
```

The UI addresses its websocket relative to the page, so it works at the server root and
behind that prefix without changes.

## Writing the prompt

PersonaPlex was fine-tuned on customer service roles written as one flat sentence:

> You work for **{company}** which is a **{industry}** and your name is **{name}**.
> Information: **{facts}**

Matching that shape matters more than the wording inside it.

**Facts after `Information:` are the only thing the agent can say.** There is no database and
no retrieval. Any fee, duration or offer that isn't in the string gets invented on the spot,
confidently. If a number matters, it goes in the string.

**Length works against you.** Every fact dilutes role adherence. Six to ten facts is the
useful range. If she drifts off-script, cut the prompt before changing anything else, then
drop temperature to 0.6.

Write "four thousand rupees" rather than a currency symbol — the tokenizer is English and
reads symbols poorly aloud. Avoid `$` in prompts passed on a command line, where the shell
eats it.

Don't prompt for conversational behaviour. Turn-taking, backchannelling, handling
interruptions and keeping replies short are architectural; the model is full duplex and does
them whether you ask or not.

**The model is English only.** It's built on Moshi, which was trained on English data, and
NVIDIA have confirmed it can't translate. Hindi or Hinglish needs a different stack.

## Reading the meter

Mimi runs at 12.5 frames a second. The bridge never forwards faster than real time, so 12.5
is the ceiling — falling short of the red mark means the model can't keep up: replies lag,
and some of your speech gets dropped on the way in. Lowering context (`-c 800` in the UI) is
the lever that helps most on a T4.

## Troubleshooting

**The notebook says there's no disk driver.** SDL's disk audio driver is a compile-time
option. Without it this approach can't work and you need a real audio device — PulseAudio
with a null sink and a pipe source is the usual substitute.

**Self-test returns zero bytes.** Check whether the engine printed `ready`. If it never did,
it's a model or VRAM problem, not a pipe problem, and `-c 800` is the first thing to try.

**Self-test passes but the browser is silent.** Check the engine log panel in the UI. If the
state reaches *On call* and the frame rate is moving, audio is being produced and the problem
is browser-side — check the tab isn't muted and the page is on `https://`.

**She talks over you.** Wear headphones. There is no second fix; the model is full duplex and
will respond to its own voice coming out of your speakers.

**She ignores the script.** Shorten the prompt, then drop temperature to 0.6.

## Licence notes

PersonaPlex weights are under the NVIDIA Open Model License. moshi.cpp and the Moshi base
model carry their own terms. Check both before shipping this anywhere commercial.
