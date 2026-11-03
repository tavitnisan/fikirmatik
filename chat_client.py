#!/usr/bin/env python3
"""Minimal streaming chat against an OpenAI-compatible server (vLLM via ./gpu.sh tunnel). Stdlib only."""

import argparse
import json
import os
import sys
import urllib.request

# Qwen3.8's recommended sampling per mode (model card).
SAMPLING = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
}
HELP = "/think on|off toggles reasoning, /reset clears history, /quit exits"


def post_stream(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--system", default=os.environ.get("CHAT_SYSTEM") or None, help="default: CHAT_SYSTEM from .env")
    p.add_argument("--no-think", action="store_true", help="start with reasoning off (faster, shorter answers)")
    p.add_argument("--max-tokens", type=int, default=None, help="cap per reply; default is whatever context is left")
    a = p.parse_args()

    try:
        models = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"]
    except OSError as e:
        sys.exit(f"can't reach {a.url} ({e}). Is './gpu.sh serve' up and './gpu.sh tunnel' running?")
    model = models[0]["id"]
    history = [{"role": "system", "content": a.system}] if a.system else []
    think = not a.no_think
    print(f"chatting with {model}, reasoning {'on' if think else 'off'}  ({HELP})")

    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "/quit":
            break
        if msg == "/reset":
            history = history[:1] if a.system else []
            continue
        if msg.startswith("/think"):
            think = msg.split()[-1] != "off"
            print(f"reasoning {'on' if think else 'off'}")
            continue
        if msg.startswith("/"):
            print(HELP)
            continue
        if not msg:
            continue
        history.append({"role": "user", "content": msg})
        payload = {"model": model, "messages": history, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": think}, **SAMPLING[think]}
        if a.max_tokens:
            payload["max_tokens"] = a.max_tokens
        reply, thinking, finish = [], False, None
        print("model> ", end="", flush=True)
        try:
            for chunk in post_stream(f"{a.url}/chat/completions", payload):
                choice = chunk["choices"][0] if chunk.get("choices") else {}
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    if not thinking:
                        print("\033[2m", end="")
                        thinking = True
                    print(reasoning, end="", flush=True)
                if delta.get("content"):
                    if thinking:
                        print("\033[0m\n", end="")
                        thinking = False
                    reply.append(delta["content"])
                    print(delta["content"], end="", flush=True)
        except KeyboardInterrupt:
            finish = "interrupted"
        except OSError as e:
            finish = f"connection lost ({e}); is the tunnel still up?"
        print("\033[0m")
        if finish == "length":
            print("[cut off: hit the length limit. Try /think off, /reset, or a larger MAX_LEN on the server]")
        elif finish not in (None, "stop", "tool_calls"):
            print(f"[{finish}]")
        if reply:
            history.append({"role": "assistant", "content": "".join(reply)})
        else:
            history.pop()  # don't keep a question that got no answer


if __name__ == "__main__":
    main()
