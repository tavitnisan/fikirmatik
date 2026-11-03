# gpu-lab

Rent a GPU, pull an open-weight model, tune it if you like, and serve it privately,
with self-hosted image generation and LoRA training on the side. You need any provider that gives you a
`ssh -p PORT root@HOST` box with a persistent `/workspace` volume (RunPod and Vast.ai both do).

> **New here? Start with [RUN_WITH_CLAUDE.md](RUN_WITH_CLAUDE.md)** — the ~20-minute, ask-Claude-to-do-it
> setup for the private chat app and its tools. This README is the full technical reference.

```
gpu.sh           runs on your Mac, drives everything over SSH
chat_client.py   streaming chat client (stdlib only) for the served model
fikirmatik.ai.app  double-click: the same chat in your browser, no terminal (ui.py + ui.html, stdlib only)
imagine.py       text-to-image from the terminal via the box's ComfyUI (stdlib only)
box/             copied to the GPU box
  bootstrap.sh     installs the lab + vLLM venvs, downloads $MODEL
  lab.py           hands-on model lab: collect / test / chat / bake
  tune.sh          automatic tuning run (tune.toml)
  serve.sh         vLLM OpenAI-compatible server on the box's localhost:8000
  image_setup.sh   ComfyUI v0.42.4 + MOSS-Image models
  comfy.sh         ComfyUI on the box's localhost:8188
  train_setup.sh   ai-toolkit for LoRA training
  train.sh         train a MOSS-Image LoRA (lora_moss.yaml)
  caption.py       caption a training set with the served vision model
```

## Where things live

| What | Where | Notes |
|---|---|---|
| Scripts, `.env`, generated `images/`, fikirmatik.ai's `chats/` (guests' in `chats/guest/`) | this folder | Run every `./gpu.sh` command from here. `.env` holds the pod address, the models and the chat system prompt. |
| SSH key | `~/.ssh/gpu_rental_ed25519` (+ `.pub`) | Kept out of this folder on purpose: it's a private key. `.env` points to it. |
| Pod identity | `~/.ssh/known_hosts_gpu` | Pinned host key of the current pod; `./gpu.sh forget` after re-renting. |
| Models, environments, ComfyUI, logs | RunPod network volume **fikirmatik** (250 GB, data center EU-RO-1), mounted at `/workspace` | Not tied to one machine: it survives stopping and terminating pods, and any new pod in EU-RO-1 can attach it. Logs are in `/workspace/runs/`. |
| RunPod access for AI tools | Claude app → Settings → Connectors | Remove the RunPod connector there to revoke it. |

Working on this with another AI: point it at this README first, run commands from this
folder, and keep `.env` and the SSH key out of anything you share or upload.

## 1. One-time setup

```bash
./gpu.sh keygen
```
Creates `~/.ssh/gpu_rental_ed25519` and copies the public key. Paste it into
RunPod (Settings → SSH Public Keys) or Vast (Account → Keys) **before** renting,
because the key is installed when the instance starts.

## 2. Rent a box

Default model is `Qwen/Qwen3.8-27B` (dense, Apache-2.0, ~56 GB in BF16). It needs one 80-96 GB GPU.

| | RunPod | Vast.ai |
|---|---|---|
| GPU | A100 80GB, H100, or RTX PRO 6000 | same, filter "Verified" + reliability > 98% |
| Image | any `runpod/pytorch` template (CUDA 12.8+) | "PyTorch" template, launch mode **SSH** |
| Disk | network volume at `/workspace` (Secure Cloud; template **fikirmatik** has the rest) | disk 250 GB |

Then copy the SSH host and port into `.env`:
```bash
cp .env.example .env
```

Disk budget: Qwen3.8-27B 56 GB + each saved model copy 56 GB + image models 40 GB
+ venvs ~35 GB.

## 3. Install

```bash
./gpu.sh setup
```
Runs in tmux on the box (safe to close your laptop). `Ctrl-b d` detaches,
`./gpu.sh attach setup` re-attaches. Ends with a check line showing CUDA is visible.

## 4. Model lab (optional)

Two ways to make your own variant of `$MODEL`, saved under `/workspace/models/` and served like any other.

```bash
./gpu.sh tune                                # automatic: ~100 trials, a few hours for a 27B
```
When it finishes it lists the best trials: pick one (KL ideally < 0.1–0.2), choose **Save**, and
save to `/workspace/models/qwen3.8-27b-tuned`. You can also chat with it right there. If the box
restarts, run the same command again; it resumes from its checkpoint. Settings: `box/tune.toml`.

The same idea by hand, step by step (`./gpu.sh lab <command> --help` for every option):

```bash
./gpu.sh lab collect                         # per-layer directions
./gpu.sh lab test --layers 30%:75%:2         # sweep candidates (hooks, no weight edits)
./gpu.sh lab chat --layer 28                 # talk to it; /ablate off|on, /layer N, /scale X
./gpu.sh lab bake --layer 28 --out /workspace/models/qwen3.8-27b-L28
./gpu.sh lab test --model /workspace/models/qwen3.8-27b-L28   # verify the baked copy
```

- `test` prints a count and KL for each layer you list. Pick a low count with the lowest KL.
  KL below ~0.1 is gentle; above ~0.5 usually shows.
- Knobs: `--scale` (1.0 = full strength; 1.2 pushes past it), `--apply-layers 20%:100%`
  (only edit later layers), fractional `--layer 27.5` (interpolates two directions),
  custom prompt files, `--no-embed`.
- `bake` writes exactly the transformation `test` measured, as a standalone model.

## 5. Serve and chat

```bash
./gpu.sh serve <HF id or /workspace/models/...> [name]   # no arguments: $MODEL
./gpu.sh logs serve                                      # wait for "Application startup complete"
./gpu.sh tunnel                                          # leave running in one terminal
./gpu.sh chat                                            # in another
```
`[name]` is what chat shows for the model (default: its folder name).
The first `serve` or `comfy` after a pod boots unpacks its venv from `/workspace/venvs/*.pack` onto
the pod's local disk (a network volume is slow to open many small files); `setup` and `image-setup`
re-pack after changing a venv. Delete a `.pack` folder to run that venv straight from the volume.
In the chat, `/think off` skips the model's reasoning for quick answers (`/think on` brings it
back; `./gpu.sh chat --no-think` starts that way). `/reset` clears the conversation.

### Without a terminal: fikirmatik.ai

Double-click **fikirmatik.ai.app** in this folder (drag it to the Dock to keep it handy), or run
`./gpu.sh ui`. It opens http://127.0.0.1:7777 and does the steps above for you: opens the
tunnel, offers to start a model when none is loaded (`SERVE_MODEL` and `SERVE_NAME` from `.env`,
then whatever you last served; the vLLM log shows while it loads), then
chats. Replies stream with the reasoning in a collapsible "Thinking" section; the Think switch
is `/think`, and sampling follows the same per-mode Qwen3.8 settings as `chat_client.py`.
Paste a new pod's SSH command into it after re-renting and it updates `.env`.
Chats are saved as JSON in `chats/`. **Quit** (in Settings) also closes the tunnel it opened;
the pod keeps billing. **Stop pod and quit**, shown only while the pod is on, also stops the pod.
Its log is `~/Library/Logs/fikirmatik.log`.

With a RunPod API key (paste it under Settings → Pod; it's saved in `.env` as `RUNPOD_API_KEY`),
fikirmatik.ai also starts the pod: **Start pod** wakes the last pod it used on the network volume
`RUNPOD_VOLUME_ID`, tries your other stopped pods on that volume if its GPU is taken, and offers to
create a new one (same settings, a GPU type you already use) when all of them are busy. It then saves the
new SSH address and connects. Stopping goes through the API too. Without a key, stop works over SSH
(`./gpu.sh stop-pod`) and you start the pod from RunPod's console (see "Starting the pod again" below).

### Tools: let the model search the web and act on your Mac

The **Tools** button under the message box lets the model do things for itself instead of only
answering. The model decides when to use a tool; fikirmatik.ai runs it and hands the result back,
looping until the model is done (up to 16 rounds per turn). What it did shows as a card in the
reply, with the output under "Output"; screenshots and generated images show in the card too.

| Tool | What it does | Approval |
|---|---|---|
| Web search | DuckDuckGo; titles, URLs and snippets. No API key. | runs on its own |
| Read a web page | Fetches a URL and returns its text. | runs on its own |
| Run a command | Runs a shell command (your login shell, from your home folder). | **asks you first** |
| Run Python | Runs Python in a session that keeps its variables between calls (calculation, data work, scripts). | **asks you first** |
| Read / list files | Reads a text file, or lists a folder. | runs on its own |
| Write / edit files | Creates, overwrites, or edits a text file. | **asks you first** |
| Browser | Opens a page in a real automated browser (Playwright/Chromium) and can read it, click, type, and screenshot — for sites that need JavaScript or interaction. | navigate/click/type **ask first**; read/screenshot run on their own |
| Generate an image | Makes an image with the self-hosted MOSS-Image model (needs ComfyUI running; see §6). | **asks you first** |
| Remember / forget | Saves or removes a lasting note about you, included in every future chat. | runs on its own |

The browser tool needs Playwright on your Mac (one-time): `pip3 install playwright && python3 -m playwright install chromium`. Image generation uses the pod's ComfyUI, which shares the one GPU with the chat model — start it with `./gpu.sh comfy` and expect the model to pause while it renders.

Every command, Python cell, file write, or browser action shows with its exact content and a
**Run** / **Skip** choice; nothing runs on your Mac until you click Run. This matters because the
model will run whatever it decides to — and a web page it reads can try to steer it ("prompt
injection"). The approval step is your check on both. Settings → Tools turns individual tools on or
off. The same screen has **Run commands without asking**, which skips approval for a hands-off
session — off by default; only turn it on while you're watching. Tool output is capped (10K
characters) so one call can't fill the context.

**Memory** is kept in `memory.txt` next to the scripts, one note per line, and prepended to the
system prompt so the model carries it into every chat. Delete lines there (or ask it to forget) to
clear it.

**Attaching images:** the paperclip by the message box (or pasting an image) attaches pictures for
the model to look at — it's multimodal, so it can read screenshots, photos, and text in images.

**Interrupting a reply:** while a reply is streaming, type a new message and press Enter — it stops
the current reply (keeping what it wrote) and sends your new input, so you can redirect it mid-thought.
With the box empty, the button just stops.

This is host-only: the Tailscale guest page can't see or run tools, only plain chat. Tools need the
model to support tool calling — the Qwen3.8 default does, and `serve.sh` already enables vLLM's
Qwen tool parser (`TOOL_ARGS`). For other model families, set the matching `--tool-call-parser`
(or `TOOL_ARGS=""` to turn tool calling off). The web tools go out over HTTPS from your Mac; if your
Python has no root certificates, fikirmatik.ai falls back to the system bundle at `/etc/ssl/cert.pem`.

### Sharing fikirmatik.ai with a friend (Tailscale)

While [Tailscale](https://tailscale.com/download) is running on this Mac, fikirmatik.ai also opens a
**chat-only** copy of the page on the Mac's Tailscale address, same port. Settings → Sharing shows it
(e.g. `http://100.101.102.103:7777`). Share this Mac from the Tailscale admin console (Machines → ⋯ →
Share); your friend installs Tailscale (Mac, Windows, Linux or phone), accepts the invite and opens
that address in any browser. Nothing else is needed on their side.

Guests can chat with whatever is being served. They can't start or stop models, see or change the
pod, quit the app, or see your chats; theirs are saved in `chats/guest/<their device's Tailscale IP>/`.
Who counts as a guest depends only on which address they connect to: the full page stays on
127.0.0.1. Keep in mind:
- It works only while this Mac is awake and fikirmatik.ai is open, and only while a model is served.
- The pod bills while it runs, whoever is chatting; you share one GPU, so replies slow down a little
  when you both generate at once.
- Limit your friend to port 7777 in Tailscale's access controls, so the share doesn't also reach
  other services on this Mac (file or screen sharing, say).

vLLM listens only on the box's localhost; the tunnel is the only way in.
Any OpenAI-compatible client or agent harness works at `http://127.0.0.1:8000/v1`;
tool calling is on by default (Qwen parsers; `TOOL_ARGS=""` for other model families).

`serve.sh` asks for 128K of context (`MAX_LEN=131072`), which agent and tool work fill quickly.
Qwen3.8 only keeps a KV cache for one layer in four, so a 27B in BF16 on one 80 GB GPU still has
room for ~230K cached tokens at that length. Override per run with `MAX_LEN=...`; on a smaller GPU,
or a dense model that won't fit, quantize on the fly to free memory for the cache:
`VLLM_ARGS="--quantization fp8" ./gpu.sh serve <model>` (halves the weights to ~28 GB).
`MAX_SEQS` (default 64) caps how many requests vLLM runs at once; its own default on Blackwell GPUs
(1024) doesn't fit next to 128K of context on this model and stops it at startup.
`MAX_LEN`, `MAX_SEQS`, `VLLM_ARGS`, `GPU_UTIL` and `TOOL_ARGS` set before `./gpu.sh serve` are carried to the box.
Stop serving (to free the GPU for `tune` or the lab) with `./gpu.sh attach serve`, then `Ctrl-C`.

## 6. Image generation (MOSS-Image)

Self-hosted ComfyUI. The model is
[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (OpenMOSS, Oct 2026,
Apache-2.0): a ~5B single-stream DiT with Qwen3.5-9B as its text encoder, strong at
photoreal, anime and rendered text, with long prompts (~4K tokens).
Two checkpoints are installed: **SFT** (better layouts and text) and **Base**
(no post-training; rawer, and the one LoRAs are trained on).

```bash
./gpu.sh image-setup          # once: ComfyUI + ~40 GB of weights
./gpu.sh stop serve           # if vLLM is running: one big job per GPU at a time
./gpu.sh comfy                # start ComfyUI
./gpu.sh tunnel               # (other terminal) then open http://127.0.0.1:8188
```

In the browser: Templates → search "MOSS" for the official graphs. They point at
int8/small files you don't have; switch the loaders to `moss_image_sft_bf16`,
`qwen3.5_9b_bf16` and `flux2-vae`.

Or skip the UI and generate from the terminal (images and their settings land in `./images`):
```bash
./gpu.sh imagine "rain-soaked neon alley, 35mm film, low angle" --size 1344x768 --batch 4
./gpu.sh imagine "..." --model base --steps 50 --cfg 3 --seed 1234
./gpu.sh imagine "..." --lora my_style:0.8          # after training one
./gpu.sh imagine --list-loras
```
Defaults follow the ComfyUI template (25 steps, CFG 2.5, er_sde, skip-layer guidance on
layer 22). `--steps 50 --cfg 3` matches OpenMOSS's own settings; `--no-slg` looks more
photographic but can lose detail. 1–2 megapixels per image is the sweet spot.

### Teaching it new things: LoRA training

The base model only draws what it saw in training. To add a style, character, product or
concept, train a LoRA on 20–200 examples (~30–60 min for 2000 steps on an A100/H100, rough estimate):

```bash
./gpu.sh train-setup                         # once
./gpu.sh dataset ~/Pictures/my_style my_style # images, optionally with same-named .txt captions
./gpu.sh serve                               # caption with the vision LLM...
./gpu.sh caption my_style --trigger my_style  # ...writes a .txt for every image without one
./gpu.sh stop serve
./gpu.sh train my_style 2000 my_style        # name, steps, trigger word
```
Every 250 steps a checkpoint and sample images are written to `/workspace/loras/my_style/`
(`./gpu.sh pull loras/my_style/samples` to look at them here). Pick the step where samples look
right before they start to overcook, then `--lora my_style_000001500` in `imagine`
(the final save is just `--lora my_style`). Captions matter more than step count:
describe everything *except* the thing you want the trigger word to carry.

The captioner uses whatever vLLM is serving.

## 7. Costs and stopping

- You pay per second while the instance runs. **Stop it from the provider's console when idle.**
- RunPod: the network volume bills $0.07/GB/month (250 GB = $17.50) whether or not a pod runs; a stopped
  pod adds nothing. Pods can be terminated freely. Delete the **volume** only when you're done for real,
  after `./gpu.sh pull` of anything you want to keep. Vast bills disk while stopped.
- Keep the RunPod balance above zero (auto-pay helps): at $0 pods stop, and an unpaid volume can be deleted.
- `./gpu.sh pull runs` copies directions and logs home (small). Models are large: push them
  to a private Hugging Face repo from `tune`'s save menu instead of downloading.

### Starting the pod again (RunPod)

With a RunPod API key saved, fikirmatik.ai's **Start pod** does all of this. By hand:
Pods → ⋮ → **Start Pod**. If RunPod says the pod's GPUs are no longer available, nothing is lost,
since everything lives on the network volume: terminate that pod, then Storage → **fikirmatik** → ⋮ →
**Deploy**, pick any 80–96 GB GPU in stock (A100 SXM or PCIe, RTX PRO 6000) and the **fikirmatik**
template. Either way the SSH port changes: copy the pod's SSH command (Connect → SSH over exposed TCP)
into fikirmatik.ai's pod box.

## Bigger models

| Model class | GPUs for `tune` and the lab (BF16) |
|---|---|
| ≤ 9B | 1× 24–32 GB (RTX 4090/5090) |
| 27–35B (Qwen3.8-27B, Gemma 4 31B, Qwen3.6-35B-A3B) | 1× 80–96 GB |
| ~120B (gpt-oss-120b, Ornith-1.5-122B) | 4× 80 GB, or 1× 96 GB with `quantization = "bnb_4bit"` |
| 180B (Qwen3.8-Flash-Next) | 4× H200, or 1× H200 with bnb_4bit |
| 500B–2.8T (GLM-5.3, DeepSeek V4.1, MiMo-V2.6-Pro, Kimi K3) | 8× H200/B200 just to hold FP8 weights; not practical for these tools |

Multi-GPU works automatically (`device_map="auto"` for the lab/`tune`, tensor parallel for vLLM).

## Trying it on a Mac first

`box/lab.py` runs on Apple Silicon with a small model:
```bash
pip install torch "transformers>=5.6,<6" accelerate datasets
python box/lab.py collect --model Qwen/Qwen2.5-0.5B-Instruct --n 64 --bs 8
python box/lab.py test --model Qwen/Qwen2.5-0.5B-Instruct --layers 30%:80%:2 --bs 8
```

## Troubleshooting

- `REMOTE HOST IDENTIFICATION HAS CHANGED`: you re-rented the same IP:port. `./gpu.sh forget`.
- `./gpu.sh status` shows petabytes free: that's RunPod's shared storage behind the network volume.
  Your limit is the volume's 250 GB (RunPod → Storage); the `du` lines below it are what you use.
- CUDA out of memory in the lab: lower `--bs`. In vLLM: `MAX_LEN=16384 ./gpu.sh serve ...`.
- `tune` says the architecture is unsupported: the lab script handles more layouts
  (fused MoE experts, hybrid attention); try it there.
