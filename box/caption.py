#!/usr/bin/env python3
"""
Caption a LoRA dataset with the vision model vLLM is serving (./gpu.sh serve).
Writes <image>.txt next to each image that doesn't have one yet. Stdlib only.

  python caption.py NAME|FOLDER [--trigger WORD] [--overwrite]
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.request
from pathlib import Path

INSTRUCTION = (
    "Write a caption for this image to be used as training data for an image generation model. "
    "One paragraph, plain description, no preamble: subject, appearance, clothing, pose and expression, "
    "setting, lighting, camera angle and framing, and visual style or medium."
)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def chat(url, model, image, max_tokens):
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    data = base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": INSTRUCTION},
        ]}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    return " ".join(re.sub(r"<think>.*?</think>", "", text, flags=re.S).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="folder, or a dataset name under $WORK/datasets")
    p.add_argument("--trigger", default=None, help="prefix every caption with this word")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--max-tokens", type=int, default=300)
    a = p.parse_args()

    try:
        model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
    except OSError as e:
        sys.exit(f"vLLM not reachable at {a.url} ({e}); start it with ./gpu.sh serve")
    folder = Path(a.dataset)
    if not folder.is_dir():
        folder = Path(os.environ.get("WORK", ".")) / "datasets" / a.dataset
    if not folder.is_dir():
        sys.exit(f"no dataset folder {folder}; upload one with ./gpu.sh dataset <folder> {a.dataset}")
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTS)
    todo = [f for f in images if a.overwrite or not f.with_suffix(".txt").exists()]
    print(f"{len(images)} images, captioning {len(todo)} with {model}")
    for i, img in enumerate(todo, 1):
        caption = chat(a.url, model, img, a.max_tokens)
        if a.trigger:
            caption = f"{a.trigger}, {caption}"
        img.with_suffix(".txt").write_text(caption + "\n")
        print(f"[{i}/{len(todo)}] {img.name}: {caption[:110]}...", flush=True)


if __name__ == "__main__":
    main()
