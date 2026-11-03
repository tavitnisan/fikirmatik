# Run your own private AI with tools — in ~20 minutes

This is **fikirmatik.ai**: your own open-source LLM running on a GPU you rent by the hour, with a
clean chat app and real tools — **web search, a browser it can drive, a Python sandbox, file access,
image generation, and memory**. It runs on your machine and talks to your own model. Nobody else
sees your chats.

You don't set it up by hand. **You open this folder in [Claude Code](https://claude.com/claude-code)
and ask Claude to do it for you.**

---

## ⚡ The fast path: ask Claude

1. Download this folder (clone the repo, or grab the ZIP and unzip it).
2. Open it in **Claude Code** (the `claude` terminal app, the desktop app's Code tab, or the VS Code
   extension — all at [claude.com/claude-code](https://claude.com/claude-code)).
3. Paste this to Claude:

> **Read `RUN_WITH_CLAUDE.md` and set up fikirmatik.ai for me. Generate my SSH key, walk me through
> renting the RunPod GPU step by step (tell me exactly what to click), fill in my `.env`, run the
> install, and launch the chat app. Run all the local commands for me.**

Claude will do the terminal work and tell you the few things only you can do (make a RunPod account,
paste a key, click Deploy). That's it.

> **Prefer to do it yourself?** Every step is below. It's all `./gpu.sh` commands plus a few clicks
> on RunPod.

---

## What you need

- A **Mac** (the one-click app is Mac-only; on Linux everything works except the `.app` — use
  `./gpu.sh ui`).
- **Python 3** (already on macOS; the app uses only the standard library).
- A **[RunPod](https://runpod.io) account** with a little credit (~$10 gets you going).
- About **20 minutes**, most of it a one-time download.

No GPU of your own, no CUDA install, no Docker. The GPU is rented; your Mac just drives it.

---

## Setup in 5 steps

```bash
# 1. Create an SSH key and copy it (you'll paste it into RunPod)
./gpu.sh keygen

# 2. ...rent a RunPod GPU (see "The RunPod part" below), then copy its SSH command into .env:
cp .env.example .env
#    edit POD_HOST and POD_PORT (from RunPod's "SSH over exposed TCP")

# 3. Install everything on the box and download the model (~15–30 min, one time)
./gpu.sh setup

# 4. Launch the chat app  (or just double-click fikirmatik.ai.app)
./gpu.sh ui              # opens http://127.0.0.1:7777

# 5. In the app: click "Start a model", wait for it to load, and chat.
```

The model, environments, and everything heavy live on a **RunPod network volume**, so they survive
when you stop or delete the pod. You download them once.

---

## The RunPod part, explained

RunPod rents you a GPU server by the second. Here's the whole flow for a newcomer.

### 1. Account + credit
Sign up at [runpod.io](https://runpod.io) and add a few dollars of credit (Billing). Turn on
auto-pay if you want it to never run dry.

### 2. Add your SSH key
`./gpu.sh keygen` prints a public key (and copies it to your clipboard). In RunPod:
**Settings → SSH Public Keys → add it.** Do this *before* you deploy — the key is installed when the
box starts.

### 3. Create a network volume (the important part)
**Storage → Network Volume → Create.** Name it `fikirmatik`, size **~250 GB**, and pick a **data
center** (note which one — your pod must be in the same one). This volume is where the model and
environments live, so **it survives stopping or deleting pods** — you never re-download. It costs
about **$0.07/GB/month** (~$17.50/mo for 250 GB) whether or not a pod is running.

### 4. Deploy a pod on the volume
From your volume → **Deploy**, or **Pods → Deploy** and attach the volume. Choose:
- **GPU** — see the table below.
- **Template/image** — any `runpod/pytorch` image with **CUDA 12.8+** (13+ is best).
- **Expose SSH** — make sure port **22/tcp** is on.

### 5. Connect it
On the running pod: **Connect → SSH over exposed TCP** → copy the `ssh root@HOST -p PORT` line into
your `.env` (or paste it into the app's **Pod** box in Settings). Done.

### GPU recommendations

The default model ([**tavitnisanyan/fikirmatik**](https://huggingface.co/tavitnisanyan/fikirmatik),
~56 GB) needs **one 80–96 GB GPU**:

| What you're running | GPU | Rough price |
|---|---|---|
| **Default (fikirmatik 27B)** | **RTX PRO 6000 (96 GB)** — usually best stock | ~$2.49/hr |
| Default (fikirmatik 27B) | A100 80 GB, or H100 | ~$1.79–3.99/hr |
| A small model (≤9B) to try it cheaply | RTX 4090 / 5090 (24–32 GB) | ~$0.70–1.20/hr |
| Bigger models (120B+) | 4× 80 GB, or quantize | more |

**Recommendation:** start with the **RTX PRO 6000** — it has the most reliable stock and 96 GB gives
the model and image generation room to share the card. Pick a data center where it's actually in
stock; EU and US regions both work.

### Stopping, and the one gotcha
- **Stop the pod when you're done** (in the app: Settings → *Stop pod and quit*, or RunPod's console).
  You pay for the GPU only while it runs; the volume keeps billing as storage regardless.
- **If a stopped pod won't start again** ("not enough free GPUs"), its host was reassigned. Don't
  fight it: **terminate that pod and deploy a fresh one on the same volume.** Nothing is lost —
  everything's on the volume. (Ask Claude to do this; it takes a minute.)

> **Even easier:** add the **RunPod connector** to Claude (claude.ai → Settings → Connectors). Then
> Claude can rent, start, stop, and replace the pod for you — just ask. Without it, you do the clicks
> above and Claude does everything local.

---

## The tools

Turn them on with the **Tools** button under the message box. The model decides when to use one; the
app runs it and feeds the result back, showing each step as a card.

| Tool | What it does |
|---|---|
| Web search + read page | DuckDuckGo and page reading, no API key |
| Run a command | Runs shell commands on your Mac |
| Run Python | A Python session that remembers its variables between calls |
| Files | Read, write, edit, list files |
| Browser | Drives a real automated Chromium (for JS-heavy or interactive sites) |
| Generate an image | Makes images with the self-hosted MOSS‑Image model |
| Memory | Remembers notes about you across every chat |

**A one-time install for two of them:**
```bash
pip3 install playwright && python3 -m playwright install chromium   # the browser tool
./gpu.sh image-setup                                                # image generation (~40 GB on the volume)
```

**Safety — read this.** Commands, Python, file writes, and browser actions **run on your real Mac**,
and every one asks for your **approval** (a Run / Skip button) before it happens. That gate is there
on purpose: the model will run whatever it decides to, and a web page it reads could try to steer it.
There's a "Run commands without asking" switch for hands-off sessions — leave it **off** unless you're
watching. Guests you share with over Tailscale get plain chat only, never the tools.

---

## Costs, honestly

- **GPU:** ~$1.79–2.49/hr **while the pod runs.** Stop it when idle.
- **Storage:** ~$17.50/mo for a 250 GB volume, always (it's what lets you skip re-downloading).
- A casual evening of use is a couple of dollars. Leaving a pod running overnight by accident is not —
  so stop it.

---

## The model

`./gpu.sh setup` pulls [**tavitnisanyan/fikirmatik**](https://huggingface.co/tavitnisanyan/fikirmatik):
an abliterated Qwen3.8‑27B — the refusal direction is removed from the weights, so it answers
directly rather than declining. Nothing else about the base model changed (only the residual-stream
writers differ from stock; the model card has the tensor diff). Apache‑2.0, multimodal (it reads
images you attach), and good at tool calling. Since it won't refuse, you're the only filter on what
it produces — worth remembering before you share the chat with guests over Tailscale.

Prefer the stock model? Set `MODEL=Qwen/Qwen3.8-27B` in `.env` before running setup.

## Make it yours

- **Different model:** set `MODEL=` in `.env` to any Hugging Face chat model, then `./gpu.sh setup`.
  Smaller models fit cheaper GPUs.
- **Its personality:** `CHAT_SYSTEM=` in `.env` is the system prompt, or change it live in Settings.
- **Your own variant:** `box/lab.py` and `./gpu.sh tune` can make a custom version of the model (the
  README covers this). Optional.

---

## Before you share your copy

If you fork or publish this, the included **`.gitignore` already keeps your `.env`, your `chats/`,
`memory.txt`, and generated `images/` out** — those are private. Double-check they're not in anything
you upload, and never share your `.env` or the SSH key in `~/.ssh/`.

---

## If something breaks

The full reference — every command, bigger models, image LoRA training, troubleshooting — is in
**[README.md](README.md)**. Or just tell Claude what you're seeing and it'll sort it out.
