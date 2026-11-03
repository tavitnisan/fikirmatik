#!/usr/bin/env python3
"""
Generate images with MOSS-Image on the box's ComfyUI (via ./gpu.sh tunnel). Stdlib only.

  ./gpu.sh imagine "a lighthouse in a storm, oil painting"
  ./gpu.sh imagine "..." --model base --lora mystyle:0.8 --size 1344x768 --batch 4
  ./gpu.sh imagine --list-loras

The graph mirrors ComfyUI's built-in MOSS-Image template; settings and seed are
saved next to each image so any result can be reproduced.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

MODELS = {"sft": "moss_image_sft_bf16.safetensors", "base": "moss_image_base_bf16.safetensors"}
TEXT_ENCODER = "qwen3.5_9b_bf16.safetensors"
VAE = "flux2-vae.safetensors"
# Negative prompt, adapted from the official template.
NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"


def api(url, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI rejected the request ({e.code}):\n{e.read().decode()[:2000]}")


def available_loras(url):
    info = api(url, "/object_info/LoraLoaderModelOnly")
    return info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]


def resolve_lora(name, loras):
    if name in loras:
        return name
    hits = [l for l in loras if name.lower() in l.lower()]
    finals = [l for l in hits if Path(l).stem.lower() == name.lower()]  # trained/x/x.safetensors = final save
    if len(finals) == 1:
        return finals[0]
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"LoRA '{name}' matches {len(hits)} files; use the full name:\n  " + "\n  ".join(hits or loras))


def build_graph(a, seed, loras):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS[a.model], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": a.width, "height": a.height, "batch_size": a.batch}},
    }
    model = ["1", 0]
    for i, (name, strength) in enumerate(loras):
        nid = str(20 + i)
        g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [nid, 0]
    if not a.no_slg:  # skip-layer guidance on the negative branch, as in the template
        g["15"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
            "model": model, "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 0.5}}
        model = ["15", 0]
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
        "seed": seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": a.sampler, "scheduler": a.scheduler, "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "imagine"}}
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?")
    p.add_argument("--negative", default=NEGATIVE)
    p.add_argument("--model", choices=MODELS, default="sft", help="sft: better layouts/text; base: rawer, what LoRAs train on")
    p.add_argument("--size", default="1024x1024", help="WxH, multiples of 16; ~1-2 megapixels is the sweet spot")
    p.add_argument("--steps", type=int, default=25, help="template default; 50 matches OpenMOSS's own setting")
    p.add_argument("--cfg", type=float, default=2.5, help="template default; OpenMOSS uses 3")
    p.add_argument("--sampler", default="er_sde")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lora", action="append", default=[], help="name[:strength], repeatable; substring of the file name is enough")
    p.add_argument("--no-slg", action="store_true", help="disable skip-layer guidance (more photographic, can lose detail)")
    p.add_argument("--out", default=str(Path(__file__).parent / "images"))
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--list-loras", action="store_true")
    p.add_argument("--dump", action="store_true", help="print the API graph instead of running it")
    a = p.parse_args()

    try:
        api(a.url, "/system_stats")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"ComfyUI not reachable at {a.url} ({e}). Is './gpu.sh comfy' running and './gpu.sh tunnel' open?")

    if a.list_loras:
        print("\n".join(available_loras(a.url)) or "(no LoRAs yet)")
        return
    if not a.prompt:
        p.error("a prompt is required")
    a.width, a.height = (int(x) for x in a.size.lower().split("x"))
    loras = []
    if a.lora:
        names = available_loras(a.url)
        for spec in a.lora:
            name, _, strength = spec.partition(":")
            loras.append((resolve_lora(name, names), float(strength or 1.0)))
    seed = a.seed if a.seed is not None else random.randrange(2**48)
    graph = build_graph(a, seed, loras)
    if a.dump:
        print(json.dumps(graph, indent=1))
        return

    resp = api(a.url, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    pid = resp["prompt_id"]
    t0 = time.time()
    while True:
        hist = api(a.url, f"/history/{pid}").get(pid) or {}
        if hist.get("status", {}).get("completed") or hist.get("status", {}).get("status_str") == "error":
            break
        print(f"\rgenerating... {time.time() - t0:.0f}s", end="", flush=True)
        time.sleep(1)
    print()
    status = hist.get("status", {})
    if status.get("status_str") == "error":
        errs = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
        sys.exit("generation failed: " + "; ".join(
            f"{e.get('node_type')} ({e.get('exception_type', '').split('.')[-1]}): {e.get('exception_message', '').strip()}" for e in errs
        ) + "\nfull trace: ./gpu.sh logs comfy")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"prompt": a.prompt, "negative": a.negative, "model": a.model, "size": a.size, "steps": a.steps, "cfg": a.cfg,
            "sampler": a.sampler, "scheduler": a.scheduler, "seed": seed, "loras": loras, "slg": not a.no_slg}
    for img in hist["outputs"]["9"]["images"]:
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
        dest = out / img["filename"]
        with urllib.request.urlopen(f"{a.url}/view?{q}", timeout=120) as r:
            dest.write_bytes(r.read())
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"saved {dest}  (seed {seed}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
