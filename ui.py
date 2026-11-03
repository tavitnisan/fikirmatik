#!/usr/bin/env python3
"""
fikirmatik.ai: chat with the served model in your browser instead of a terminal. Stdlib only.

  ./gpu.sh ui                       (or double-click "fikirmatik.ai.app")
  python3 ui.py --port 7777 --no-open

From the page you can connect the tunnel, start or stop vLLM on the box, start or stop the pod
(with a RunPod API key in .env), and chat with whatever is served. It drives ./gpu.sh for everything that touches the box, so .env
is still where the pod address lives. Chats are saved as JSON in ./chats. Quitting from the
page also closes the tunnel it opened (a tunnel you started in a terminal is left alone).

While Tailscale is running, a chat-only copy of the page also listens on this Mac's Tailscale
address (same port) for people you share the Mac with. Guests can chat with whatever is served;
their chats are saved in ./chats/guest/<their device's Tailscale IP>. The pod and model controls,
Quit, and your own chats exist only on 127.0.0.1.
"""

import argparse
import base64
import html as htmllib
import http.client
import ipaddress
import json
import os
import queue
import random
import re
import select
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KIT = Path(__file__).resolve().parent
GPU = ["bash", str(KIT / "gpu.sh")]
PAGE = KIT / "ui.html"
LOGO = KIT / "ui-logo.png"
CHATS = KIT / "chats"
GUEST_CHATS = CHATS / "guest"
TAILSCALE = ["tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale"]  # CLI on PATH, or the Mac app's
TAILNET = ipaddress.ip_network("100.64.0.0/10")  # where Tailscale addresses come from
VLLM_PORT = 8000
VLLM = f"http://127.0.0.1:{VLLM_PORT}"
APP_ID = "fikirmatik"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]|[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


class BadRequest(Exception):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


# --- .env (the same file gpu.sh sources) ---------------------------------------------

def read_env():
    env = {}
    path = KIT / ".env"
    if path.exists():
        lines = iter(path.read_text().splitlines())
        for line in lines:
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)$", line)
            if m:
                v = m.group(2).strip()
                if v[:1] in "'\"" and (len(v) < 2 or v[-1] != v[0]):  # a quoted value spanning lines, as bash allows
                    for more in lines:
                        v += "\n" + more
                        if more.rstrip().endswith(v[0]):
                            v = v.rstrip()
                            break
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                env[m.group(1)] = v
    return env


def write_env(updates):
    """Set KEY=value lines in .env, creating it from .env.example the first time."""
    path = KIT / ".env"
    lines = (path if path.exists() else KIT / ".env.example").read_text().splitlines()
    for key, value in updates.items():
        for i, line in enumerate(lines):
            if re.match(rf"\s*(?:export\s+)?{key}=", line):
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


# --- the box, through gpu.sh --------------------------------------------------------

def clean(text):
    """Terminal output (tmux logs, progress bars, colors) as plain lines."""
    lines = []
    for line in text.split("\n"):
        parts = [p for p in line.split("\r") if p.strip()]  # a progress bar keeps its last redraw
        lines.append(ANSI.sub("", parts[-1] if parts else "").rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


def gpu(*args, timeout=60, **env):
    try:
        p = subprocess.run(GPU + list(args), cwd=KIT, stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=timeout, env={**os.environ, **env})
        # bytes, not text=True: universal newlines would turn each progress-bar \r into a line
        return p.returncode, clean((p.stdout + p.stderr).decode(errors="replace"))
    except subprocess.TimeoutExpired:
        return 124, f"./gpu.sh {args[0]} gave no answer within {timeout}s"


class Tunnel:
    """`./gpu.sh tunnel` as a child process, so the page can open and close it."""

    def __init__(self):
        self.proc = None
        self.output = ""
        self.lock = threading.Lock()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        with self.lock:
            if self.alive():
                return
            self.output = ""
            self.proc = subprocess.Popen(GPU + ["tunnel"], cwd=KIT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
            threading.Thread(target=self._collect, args=(self.proc,), daemon=True).start()

    def _collect(self, proc):
        for line in proc.stdout:
            if proc is self.proc:
                self.output = (self.output + line)[-4000:]

    def error(self):
        """Why the last tunnel we opened closed, minus gpu.sh's banner."""
        if self.proc is None or self.alive():
            return None
        text = "\n".join(l for l in clean(self.output).splitlines() if not re.match(r"(vLLM|ComfyUI) +http", l))
        return text.strip() or f"the tunnel closed (exit code {self.proc.returncode})"

    def stop(self):
        with self.lock:
            if self.alive():
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc = None


tunnel = Tunnel()


def port_open(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


def probe_model():
    try:
        with urllib.request.urlopen(VLLM + "/v1/models", timeout=4) as r:
            m = json.load(r)["data"][0]
        return {"model": m["id"], "root": m.get("root") or m["id"], "max_len": m.get("max_model_len")}
    except (OSError, ValueError, LookupError, http.client.HTTPException):
        return None


def status():
    env = read_env()
    pod = {"host": env.get("POD_HOST", ""), "port": env.get("POD_PORT", ""), "user": env.get("POD_USER", "")}
    out = {"pod": pod, "tunnel": {"ours": tunnel.alive(), "error": tunnel.error()}, "guest": guests.url,
           "runpod": podwatch.snapshot()}
    model = probe_model()
    if model:
        out.update(state="ready", **model)
    elif port_open(VLLM_PORT):  # the tunnel answers but vLLM behind it doesn't
        out["state"] = "no_model"
    elif tunnel.alive():
        out["state"] = "connecting"
    else:
        out["state"] = "offline" if pod["host"] else "no_pod"
    return out


def box_info():
    """vLLM on the box: is its tmux session there, how does its log end, what is the GPU doing."""
    results = {}

    def run(key, *args, **env):
        results[key] = gpu(*args, timeout=40, **env)

    jobs = [threading.Thread(target=run, args=("status", "status")),
            threading.Thread(target=run, args=("log", "logs", "serve"), kwargs={"NO_FOLLOW": "1"})]
    for t in jobs:
        t.start()
    for t in jobs:
        t.join()
    rc, out = results["status"]
    if rc:
        return {"error": out[-800:] or f"./gpu.sh status failed (exit {rc})"}
    session = re.search(r"^serve:", out, re.M) is not None
    rc, log = results["log"]
    lines = [l for l in (log if rc == 0 else "").splitlines() if l.strip()]
    # session.sh keeps the tmux session open after vLLM exits and prints this marker.
    exited = re.search(r"\[exited with (\d+)\]", "\n".join(lines[-4:]))
    return {"session": session, "loading": session and not exited, "exited": bool(exited),
            "exit_code": int(exited.group(1)) if exited else None,
            "log": "\n".join(lines[-60:]), "gpu": out.splitlines()[0] if out else "",
            "sessions": re.findall(r"^([\w.-]+): \d+ windows?", out, re.M)}


def stop_pod():
    if read_env().get("RUNPOD_API_KEY"):
        stopped = podwatch.stop()
        tunnel.stop()
        return {"output": f"stopped {', '.join(stopped) or 'nothing (no pod was running)'}"}
    rc, out = gpu("stop-pod", timeout=90)
    # The pod can go down before ssh relays runpodctl's answer, and then ssh itself exits 255.
    # A 255 without a connection error means it got in and the stop was sent.
    if rc and not (rc == 255 and not re.search(r"connect to host|Permission denied|resolve hostname", out)):
        raise BadRequest(out[-800:] or f"./gpu.sh stop-pod failed (exit {rc})", 502)
    tunnel.stop()
    return {"output": out}


def start_model(body):
    model, name = str(body.get("model") or "").strip(), str(body.get("name") or "").strip()
    if not re.fullmatch(r"[\w./:@+-]+", model) or model.startswith("-"):
        raise BadRequest("Enter a Hugging Face model id (org/name) or a folder on the pod.")
    if name and not re.fullmatch(r"[\w.:-]+", name):
        raise BadRequest("The name can only use letters, digits, dots, dashes and underscores.")
    rc, out = gpu("serve", model, *([name] if name else []), timeout=300)
    if rc:
        raise BadRequest(out[-800:] or f"./gpu.sh serve failed (exit {rc})", 502)
    return {"output": out}


def set_pod(body):
    host, port, user = (str(body.get(k) or d).strip() for k, d in (("host", ""), ("port", "22"), ("user", "root")))
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host):
        raise BadRequest("That doesn't look like a host name or IP address.")
    if not port.isdigit() or not 0 < int(port) < 65536:
        raise BadRequest("The port should be a number.")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise BadRequest("That doesn't look like an SSH user name.")
    write_env({"POD_HOST": host, "POD_PORT": port, "POD_USER": user})
    tunnel.stop()
    if not port_open(VLLM_PORT):
        tunnel.start()
    return {"ok": True}


# --- RunPod: start and stop the pod itself, which SSH can't do while it's off ------------------
# Every pod on the network volume RUNPOD_VOLUME_ID can serve (the volume holds everything), so
# Start tries them in turn and only offers a new pod when all of their machines' GPUs are taken.

RUNPOD_API = os.environ.get("RUNPOD_API", "https://api.runpod.io/v2")
GPU_TAKEN = re.compile(r"not enough free GPUs|no longer any instances|no instances available", re.I)
LIVE = ("RUNNING", "PROVISIONING", "STARTING")


def runpod(method, path, body=None):
    key = read_env().get("RUNPOD_API_KEY", "")
    if not key:
        raise BadRequest("Add a RunPod API key under Pod in Settings first.")
    req = urllib.request.Request(RUNPOD_API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read() or b"{}")
        except ValueError:
            d = {}
        if e.code == 401:
            raise BadRequest("RunPod didn't accept the API key. Make a new one and paste it under Pod in Settings.", 401)
        raise BadRequest(d.get("detail") or d.get("title") or f"RunPod answered {e.code}", e.code)
    except OSError as e:
        raise BadRequest(f"Can't reach RunPod ({e}).", 502)
    return json.loads(data) if data else {}


def gpu_name(p):
    g = (p.get("gpu") or {}).get("id") or "CPU"
    return re.sub(r"^NVIDIA |\s*Blackwell Server Edition$", "", g)


def volume_pods():
    """The pods on our network volume, newest first."""
    vol = read_env().get("RUNPOD_VOLUME_ID", "")
    if not vol:
        raise BadRequest("fikirmatik.ai doesn't know which network volume is yours: add RUNPOD_VOLUME_ID to .env.")
    pods = runpod("GET", "/pods").get("pods") or []
    mine = [p for p in pods if any(m.get("volumeId") == vol for m in (p.get("mounts") or {}).get("network") or [])]
    return sorted(mine, key=lambda p: p.get("createdAt") or "", reverse=True)


class PodWatch:
    """What RunPod says about our pods (refreshed in the background as the page polls), and the
    Start job, which runs in a thread because booting a pod takes minutes."""

    def __init__(self):
        self.pods, self.error, self.at = None, "", 0
        self.lock = threading.Lock()
        self.job = {"busy": False, "step": "", "error": "", "offer": None}

    def snapshot(self):
        if not read_env().get("RUNPOD_API_KEY"):
            return {"configured": False}
        if time.time() - self.at > 15 and self.lock.acquire(blocking=False):
            threading.Thread(target=self._refresh, daemon=True).start()
        pods = self.pods or []
        return {"configured": True, "known": self.pods is not None, "error": self.error, "job": self.job,
                "running": any(p["status"] in LIVE for p in pods), "pods": pods}

    def _refresh(self):
        try:
            self.reload()
        finally:
            self.lock.release()

    def reload(self):
        try:
            self.pods = [{"id": p["id"], "status": p.get("status"), "gpu": gpu_name(p), "cost": p.get("cost")}
                         for p in volume_pods()]
            self.error = ""
        except BadRequest as e:
            self.error = str(e)
        self.at = time.time()

    def start(self, create=False):
        if self.job["busy"]:
            return
        self.job = job = {"busy": True, "step": "Asking RunPod…", "error": "", "offer": None}
        threading.Thread(target=self._start, args=(job, create), daemon=True).start()

    def _start(self, job, create):
        try:
            pod = self._create(job) if create else self._resume(job)
            if pod:
                self._connect(job, pod)
        except BadRequest as e:
            job["error"] = str(e)
        finally:
            self.reload()  # before the page stops waiting, so it never sees the list from before the start
            job.update(busy=False, step="")

    def _resume(self, job):
        pods = volume_pods()
        if not pods:
            raise BadRequest("RunPod lists no pod on your network volume. Deploy one onto it from RunPod's Storage page.")
        live = [p for p in pods if p.get("status") in LIVE]
        if live:
            return live[0]
        last = read_env().get("RUNPOD_POD_ID")
        for p in sorted(pods, key=lambda p: (p["id"] != last, p.get("cost") or 99)):  # last used, then cheapest
            if "start" not in (p.get("actions") or []):
                continue
            job["step"] = f"Starting the pod on its {gpu_name(p)}…"
            try:
                return runpod("POST", f"/pods/{p['id']}/action", {"action": "start"})
            except BadRequest as e:
                if not GPU_TAKEN.search(str(e)):
                    raise
        offer = {}
        for p in pods:
            if p.get("gpu"):
                offer.setdefault(p["gpu"]["id"], {"gpu": gpu_name(p), "cost": p.get("cost")})
        job["offer"] = sorted(offer.values(), key=lambda o: o["cost"] or 99)
        return None

    def _create(self, job):
        """A new pod on the same volume, set up like the newest one, on whichever of our GPU types is free."""
        pods = volume_pods()
        if not pods:
            raise BadRequest("RunPod lists no pod on your network volume to copy the settings from.")
        base = pods[0]
        mount = (base.get("mounts") or {}).get("network")[0]
        types = sorted({p["gpu"]["id"]: p.get("cost") or 99 for p in pods if p.get("gpu")}.items(), key=lambda t: t[1])
        for gpu, _ in types:
            job["step"] = f"Creating a pod on {gpu_name({'gpu': {'id': gpu}})}…"
            body = {"name": base["name"], "image": base["image"], "disk": base["disk"], "ports": base["ports"],
                    "env": base.get("env") or {}, "cloud": base.get("cloud") or "SECURE", "dataCenterIds": [base["dataCenterId"]],
                    # the vLLM venv is built for CUDA 13, so older host drivers can't run it
                    "gpu": {"id": gpu, "count": (base.get("gpu") or {}).get("count", 1), "minCudaVersion": "13.0"},
                    "mounts": {"network": [{"volumeId": mount["volumeId"], "path": mount["path"]}]}}
            try:
                return runpod("POST", "/pods", body)
            except BadRequest as e:
                if e.code != 400:  # 402 is an empty balance: no other GPU will do better
                    raise
        raise BadRequest(f"No GPU is free in {base['dataCenterId']} right now. Try again in a few minutes.", 503)

    def _connect(self, job, pod):
        job["step"] = f"Waiting for the pod on its {gpu_name(pod)} to boot…"
        deadline = time.time() + 600
        while not (pod.get("status") == "RUNNING" and (pod.get("ssh") or {}).get("direct")):
            if pod.get("status") in ("EXITED", "ERROR", "TERMINATED"):
                raise BadRequest(f"The pod went to {pod['status']} while starting. Check it in RunPod's console.", 502)
            if time.time() > deadline:
                raise BadRequest("The pod is up but never offered SSH. Check it in RunPod's console.", 504)
            time.sleep(5)
            pod = runpod("GET", f"/pods/{pod['id']}")
        d = pod["ssh"]["direct"]
        write_env({"POD_HOST": d["host"], "POD_PORT": str(d["port"]), "POD_USER": d.get("username") or "root",
                   "RUNPOD_POD_ID": pod["id"]})
        job["step"] = "Connecting…"
        for _ in range(24):  # sshd comes up a little after RunPod publishes the port
            try:
                socket.create_connection((d["host"], int(d["port"])), timeout=3).close()
                break
            except OSError:
                time.sleep(5)
        tunnel.stop()
        if not port_open(VLLM_PORT):
            tunnel.start()

    def stop(self):
        """Stop every running pod on the volume (normally just one)."""
        stopped = [p["id"] for p in volume_pods() if "stop" in (p.get("actions") or [])]
        for pid in stopped:
            runpod("POST", f"/pods/{pid}/action", {"action": "stop"})
        self.reload()
        return stopped


podwatch = PodWatch()


def save_runpod_key(body):
    key = str(body.get("key") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{20,200}", key):
        raise BadRequest("That doesn't look like a RunPod API key.")
    old = read_env().get("RUNPOD_API_KEY", "")
    write_env({"RUNPOD_API_KEY": key})
    try:
        if not read_env().get("RUNPOD_VOLUME_ID"):  # one volume: that's the one
            vols = runpod("GET", "/network-volumes").get("networkVolumes") or []
            if len(vols) != 1:
                raise BadRequest(f"Your account has {len(vols)} network volumes; add the right one as RUNPOD_VOLUME_ID in .env.")
            write_env({"RUNPOD_VOLUME_ID": vols[0]["id"]})
        volume_pods()  # proves the key can read pods
    except BadRequest:
        write_env({"RUNPOD_API_KEY": old})
        raise
    podwatch.reload()
    return {"ok": True}


def config():
    env = read_env()
    return {"system": env.get("CHAT_SYSTEM", ""), "model": env.get("MODEL", "Qwen/Qwen3.8-27B"),
            "serve_model": env.get("SERVE_MODEL", ""), "serve_name": env.get("SERVE_NAME", ""),
            "memory": memory_text(), "env_file": (KIT / ".env").exists()}


# --- tools the model can call (the agent loop lives in the browser; this runs each tool) --------

def ca_context():
    """An SSL context with a working trust store. The python.org build on macOS ships an empty
    one, so https fails until we point it at certifi or the system bundle."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    for path in ("/etc/ssl/cert.pem", "/private/etc/ssl/cert.pem", "/usr/local/etc/openssl/cert.pem"):
        if os.path.exists(path):
            try:
                return ssl.create_default_context(cafile=path)
            except OSError:
                continue
    return ssl.create_default_context()


SSL_CTX = ca_context()
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/18.0 Safari/605.1.15")
MAX_TOOL_OUTPUT = 10000  # characters fed back to the model, so one tool call can't flood the context
SHELL = os.environ.get("SHELL") or "/bin/zsh"

DDG_RESULT = re.compile(r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>', re.S)
DDG_SNIPPET = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S)


def strip_tags(s):
    return htmllib.unescape(re.sub(r"<.*?>", "", s)).strip()


def unwrap_ddg(url):
    url = htmllib.unescape(url)
    m = re.search(r"[?&]uddg=([^&]+)", url)  # DDG sometimes wraps the target in a redirect
    if m:
        url = urllib.parse.unquote(m.group(1))
    return "https:" + url if url.startswith("//") else url


def html_to_text(raw):
    raw = re.sub(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<head[^>]*>.*?</head>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr|/section|/article)\s*/?>", "\n", raw)
    text = htmllib.unescape(re.sub(r"(?s)<.*?>", " ", raw))
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text).strip()


def tool_web_search(args):
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: web_search needs a 'query'."
    n = max(1, min(int(args.get("max_results") or 5), 10))
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data, headers={
        "User-Agent": BROWSER_UA, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as r:
        body = r.read().decode("utf-8", "replace")
    hits = DDG_RESULT.findall(body)
    snips = DDG_SNIPPET.findall(body)
    if not hits:
        return f"No results for {query!r}."
    out = []
    for i, (href, title) in enumerate(hits[:n]):
        snip = strip_tags(snips[i]) if i < len(snips) else ""
        out.append(f"{i + 1}. {strip_tags(title)}\n   {unwrap_ddg(href)}\n   {snip}")
    return "\n\n".join(out)


def tool_fetch_url(args):
    url = str(args.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        return "Error: fetch_url needs a full http(s):// URL."
    # Some servers (python.org, for one) compress whether or not we ask. Ask for the two kinds the
    # standard library can undo, so nobody picks brotli, and unpack them with a size cap.
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html,*/*",
                                               "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=25, context=SSL_CTX) as r:
        charset = r.headers.get_content_charset() or "utf-8"
        ctype = r.headers.get("Content-Type", "")
        encoding = (r.headers.get("Content-Encoding") or "").lower()
        data = r.read(3_000_000)
    if encoding in ("gzip", "x-gzip", "deflate"):
        for wbits in (47, -15):  # 47: gzip or zlib header; -15: bare deflate
            try:
                data = zlib.decompressobj(wbits).decompress(data, 10_000_000)
                break
            except zlib.error:
                continue
    raw = data.decode(charset, "replace")
    text = html_to_text(raw) if ("html" in ctype or "<html" in raw[:2000].lower()) else raw
    text = text.strip() or "(the page had no readable text)"
    return text[:MAX_TOOL_OUTPUT] + ("\n\n[...truncated]" if len(text) > MAX_TOOL_OUTPUT else "")


def tool_run_command(args):
    command = str(args.get("command") or "").strip()
    if not command:
        return "Error: run_command needs a 'command'."
    timeout = max(1, min(int(args.get("timeout") or 120), 600))
    cwd = str(args.get("cwd") or "").strip()
    cwd = os.path.expanduser(cwd) if cwd else os.path.expanduser("~")
    if not os.path.isdir(cwd):
        return f"Error: no such directory {cwd!r}."
    try:
        p = subprocess.run([SHELL, "-lc", command], cwd=cwd, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)
        out, code = (p.stdout or ""), p.returncode
        if p.stderr and p.stderr.strip():
            out += ("\n" if out and not out.endswith("\n") else "") + "[stderr]\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        # On a timeout the captured output comes back as raw bytes even with text=True.
        partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out, code = partial + f"\n[timed out after {timeout}s]", None
    out = out.strip() or "(no output)"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + "\n[...truncated]"
    return f"exit code {code}\n{out}" if code is not None else out


# --- memory: notes the model keeps about the user, injected into the system prompt ---------

MEMORY_FILE = KIT / "memory.txt"


def memory_lines():
    try:
        return [l.strip() for l in MEMORY_FILE.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def memory_text():
    return "\n".join(f"- {l}" for l in memory_lines())


def tool_remember(args):
    note = str(args.get("note") or "").strip()
    if not note:
        return "Error: remember needs a 'note'."
    lines = memory_lines()
    if note in lines:
        return "Already remembered that."
    lines.append(note)
    MEMORY_FILE.write_text("\n".join(lines) + "\n")
    return f"Remembered: {note}  ({len(lines)} notes saved.)"


def tool_forget(args):
    q = str(args.get("note") or "").strip().lower()
    if not q:
        return "Error: forget needs a 'note' (text to match)."
    lines = memory_lines()
    kept = [l for l in lines if q not in l.lower()]
    MEMORY_FILE.write_text(("\n".join(kept) + "\n") if kept else "")
    return f"Forgot {len(lines) - len(kept)} note(s). {len(kept)} left." if len(kept) != len(lines) else "No matching note."


# --- files -----------------------------------------------------------------------------

def _abspath(p):
    return os.path.abspath(os.path.expanduser(str(p or "").strip()))


def tool_read_file(args):
    path = _abspath(args.get("path"))
    if not os.path.isfile(path):
        return f"Error: no such file {path!r}."
    try:
        with open(path, "r", errors="replace") as f:
            data = f.read(2 * MAX_TOOL_OUTPUT + 1)
    except OSError as e:
        return f"Error reading {path}: {e}"
    if len(data) > 2 * MAX_TOOL_OUTPUT:
        data = data[:2 * MAX_TOOL_OUTPUT] + "\n[...truncated]"
    return data or "(empty file)"


def tool_write_file(args):
    path = _abspath(args.get("path"))
    if args.get("content") is None:
        return "Error: write_file needs 'content'."
    content = str(args.get("content"))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
    except OSError as e:
        return f"Error writing {path}: {e}"
    return f"Wrote {len(content)} characters to {path}."


def tool_edit_file(args):
    path = _abspath(args.get("path"))
    old, new = args.get("old_str"), args.get("new_str")
    if old is None or new is None:
        return "Error: edit_file needs 'old_str' and 'new_str'."
    if not os.path.isfile(path):
        return f"Error: no such file {path!r}."
    try:
        with open(path, "r", errors="replace") as f:
            data = f.read()
    except OSError as e:
        return f"Error: {e}"
    n = data.count(str(old))
    if n == 0:
        return "Error: old_str not found in the file."
    if n > 1:
        return f"Error: old_str appears {n} times; include more context so it matches once."
    with open(path, "w") as f:
        f.write(data.replace(str(old), str(new), 1))
    return f"Edited {path}."


def tool_list_dir(args):
    path = _abspath(args.get("path") or "~")
    if not os.path.isdir(path):
        return f"Error: no such directory {path!r}."
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        return f"Error: {e}"
    rows = [f"{'d' if os.path.isdir(os.path.join(path, n)) else '-'} {n}" for n in names[:300]]
    return f"{path}:\n" + "\n".join(rows) if rows else f"{path} is empty."


# --- a persistent Python session, so variables and imports survive across calls ---------

class PyKernel:
    # Runs in a child Python. Echoes the last bare expression's repr, like a notebook cell, and
    # streams one framed JSON result per cell back on the real stdout (cell output is captured).
    DRIVER = r"""
import sys, json, io, ast, contextlib, traceback
ns = {'__name__': '__main__'}
def _send(o): sys.__stdout__.write('\x00' + json.dumps(o) + '\n'); sys.__stdout__.flush()
_send({'ready': True})
for _line in sys.stdin:
    try: _req = json.loads(_line)
    except Exception: continue
    _buf = io.StringIO(); _ok = True
    try:
        _mod = ast.parse(_req.get('code', ''), '<cell>', 'exec')
        _tail = _mod.body.pop().value if _mod.body and isinstance(_mod.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            exec(compile(_mod, '<cell>', 'exec'), ns)
            if _tail is not None:
                _v = eval(compile(ast.Expression(_tail), '<cell>', 'eval'), ns)
                if _v is not None: print(repr(_v))
    except Exception:
        _ok = False; traceback.print_exc(file=_buf)
    _send({'ok': _ok, 'output': _buf.getvalue()})
"""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()

    def _start(self):
        self.proc = subprocess.Popen([sys.executable, "-I", "-u", "-c", self.DRIVER],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, cwd=os.path.expanduser("~"))
        self._readline(15)  # the {'ready': True} line

    def _readline(self, timeout):
        r, _, _ = select.select([self.proc.stdout], [], [], max(0.0, timeout))
        return self.proc.stdout.readline() if r else None

    def run(self, code, timeout):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                line = self._readline(deadline - time.time())
                if line is None:  # still running past the limit: reset the session to regain control
                    self.proc.kill()
                    self.proc = None
                    return {"ok": False, "output": f"[timed out after {timeout}s; the Python session was reset]"}
                if line.startswith("\x00"):
                    return json.loads(line[1:])


PY_KERNEL = PyKernel()


def tool_run_python(args):
    code = str(args.get("code") or "")
    if not code.strip():
        return "Error: run_python needs 'code'."
    timeout = max(1, min(int(args.get("timeout") or 120), 600))
    r = PY_KERNEL.run(code, timeout)
    out = (r.get("output") or "").strip() or "(ran, no output)"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + "\n[...truncated]"
    return out


# --- a real browser the model drives (Playwright), on one worker thread it owns ----------

class BrowserSession:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = None
        self.lock = threading.Lock()

    def _worker(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            err = f"Playwright isn't installed on this Mac ({e}). Run: pip3 install playwright && python3 -m playwright install chromium"
            while True:
                item = self.q.get()
                if item is None:
                    return
                item[1]["error"] = err
                item[1]["event"].set()
        with sync_playwright() as p:
            # A real window you can watch by default. BROWSER_HEADED=0 (in .env or the environment)
            # runs it headless instead, for setups with no GUI; you then follow the per-step screenshots.
            headless = (os.environ.get("BROWSER_HEADED") or read_env().get("BROWSER_HEADED") or "1") == "0"
            state = {"browser": None, "ctx": None}

            def live_page():
                st = state
                if st["browser"] is None or not st["browser"].is_connected():
                    st["browser"] = p.chromium.launch(headless=headless)
                    # Chromium's own user agent: a Safari one on a Chromium engine gets served the wrong
                    # code, and sites like Google flag the mismatch.
                    st["ctx"] = st["browser"].new_context(viewport={"width": 1280, "height": 860})
                # Drive the newest open tab, so a link that opened a new tab is the one we act on.
                tabs = [t for t in st["ctx"].pages if not t.is_closed()]
                return tabs[-1] if tabs else st["ctx"].new_page()

            while True:
                item = self.q.get()
                if item is None:
                    try:
                        state["browser"] and state["browser"].close()
                    except Exception:
                        pass
                    return
                fn, res = item
                try:
                    res["value"] = fn(live_page())  # rebuilds the page/browser if a crash closed it
                except Exception as e:
                    res["error"] = f"{type(e).__name__}: {e}"
                res["event"].set()

    def call(self, fn, timeout=60):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()
        res = {"event": threading.Event()}
        self.q.put((fn, res))
        if not res["event"].wait(timeout):
            return {"error": f"the browser didn't respond within {timeout}s"}
        return res


BROWSER = BrowserSession()


# Numbers every visible thing in one frame you can act on, so the model picks elements by number
# instead of guessing CSS selectors or matching a hidden label's text.
_COLLECT_JS = r"""(start) => {
  const SEL = 'a[href],button,input:not([type=hidden]),textarea,select,summary,[contenteditable=""],[contenteditable="true"],' +
    '[role=button],[role=link],[role=checkbox],[role=radio],[role=switch],[role=tab],[role=menuitem],[role=option],' +
    '[role=combobox],[role=textbox],[role=searchbox],[onclick]';
  const CLICKY = 'a[href],button,[role=button],[role=link]';
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const byIds = ids => (ids || '').split(/\s+/).map(id => (document.getElementById(id) || {}).innerText || '').join(' ');
  document.querySelectorAll('[data-fk]').forEach(e => e.removeAttribute('data-fk'));
  const picked = new Set(), out = [];
  let n = start;
  for (const e of document.querySelectorAll(SEL)) {
    const r = e.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || getComputedStyle(e).visibility === 'hidden' || e.closest('[aria-hidden="true"],[inert]')) continue;
    const tag = e.tagName.toLowerCase(), type = (e.getAttribute('type') || '').toLowerCase(), role = e.getAttribute('role') || '';
    const btnInput = tag === 'input' && ['button', 'submit', 'reset', 'image'].includes(type);
    const isField = !btnInput && (['input', 'textarea', 'select'].includes(tag) || e.isContentEditable || ['textbox', 'searchbox', 'combobox'].includes(role));
    const outer = e.parentElement && e.parentElement.closest(CLICKY);
    if (!isField && outer && picked.has(outer)) continue;  // a button inside a link: the outer one is enough
    if (!e.matches(CLICKY) && !isField && !role && r.width * r.height > innerWidth * innerHeight / 4) continue;  // a page-sized onclick wrapper
    const kind = btnInput ? 'button' : ['checkbox', 'radio'].includes(type) ? type : tag === 'select' ? 'select' : isField ? 'field'
      : ['checkbox', 'radio', 'switch', 'tab', 'option', 'menuitem'].includes(role) ? role : tag === 'a' || role === 'link' ? 'link' : 'button';
    let name = e.getAttribute('aria-label') || byIds(e.getAttribute('aria-labelledby'));
    if (!clean(name) && e.labels && e.labels.length) name = [...e.labels].map(l => l.innerText).join(' ');
    if (!clean(name) && btnInput) name = e.value;
    if (!clean(name) && isField) name = e.getAttribute('placeholder') || e.getAttribute('title') || e.getAttribute('name');
    if (!clean(name)) name = e.innerText || e.getAttribute('title') || e.getAttribute('alt') || (e.querySelector('img[alt]') || {}).alt;
    name = clean(name).slice(0, 70);
    if (!name && kind === 'link') continue;
    const info = [];
    if (kind === 'field') {
      if (type && type !== 'text') info.push(type);
      const v = clean(e.isContentEditable ? e.innerText : e.value);
      if (v) info.push(type === 'password' ? 'filled in' : `value "${v.slice(0, 50)}"`);
    }
    if (kind === 'select') info.push(`"${clean((e.selectedOptions[0] || {}).text)}" selected; options: ` + [...e.options].slice(0, 15).map(o => clean(o.text)).join(' | '));
    if (e.checked || e.getAttribute('aria-checked') === 'true') info.push('checked');
    if (e.disabled || e.getAttribute('aria-disabled') === 'true') info.push('disabled');
    const inView = r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
    if (!inView) info.push('off-screen');
    e.setAttribute('data-fk', n);
    picked.add(e);
    out.push({n: n++, kind, name, info: info.join(', '), inView});
  }
  return out;
}"""

# Draws each numbered element's number on the page for the screenshot (true), then takes it away (false).
_MARKS_JS = r"""(on) => {
  const old = document.getElementById('__fk_marks');
  if (old) old.remove();
  if (!on) return;
  const box = document.createElement('div');
  box.id = '__fk_marks';
  box.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647';
  for (const e of document.querySelectorAll('[data-fk]')) {
    const r = e.getBoundingClientRect();
    if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
    const m = document.createElement('div');
    m.style.cssText = `position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;outline:1px solid rgba(230,160,0,.85)`;
    const tag = document.createElement('span');
    tag.textContent = e.getAttribute('data-fk');
    tag.style.cssText = 'position:absolute;left:-1px;top:-1px;font:bold 11px/13px Menlo,monospace;color:#000;background:#ffd400;padding:0 3px;border-radius:2px';
    m.append(tag);
    box.append(m);
  }
  document.documentElement.append(box);
}"""

_REFS = {}  # element number -> the frame it's in, from the latest page read (only the browser thread touches it)
MAX_ELEMENTS = 110
CAPTCHA_FRAME = re.compile(r"recaptcha/(api2|enterprise)/(anchor|bframe)|hcaptcha\.com|challenges\.cloudflare\.com|arkoselabs\.com|funcaptcha", re.I)
HUMAN_CHECK = ("A CAPTCHA or 'verify you are human' check is on this page. If the screenshot shows it still needs doing, "
               "don't try to solve or get around it, and don't reload or switch sites: tell the user to complete it in "
               "the browser window on their screen, end your turn, and call browser_read once they say it's done.")
BLANK_PAGE = ("This page shows no text or controls. It may still be loading (browser_act 'wait'), the site may be "
              "blocking automated browsers, or a person may need to sign in. Look at the screenshot; if the user has to "
              "do something in the browser window, ask them and wait for their reply instead of retrying.")


def _visible_frame(frame):
    try:
        return frame.frame_element().evaluate(
            "e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);"
            " return r.width > 30 && r.height > 30 && r.bottom > 0 && r.top < innerHeight && s.visibility !== 'hidden' && +s.opacity > 0.1 }")
    except Exception:
        return False


def _number_elements(page):
    """Number what can be acted on, across the page and its visible iframes. Returns (elements, captcha_showing)."""
    _REFS.clear()
    elements, captcha = [], False
    for frame in page.frames:
        if frame.is_detached():
            continue
        if frame.parent_frame is not None:
            if not _visible_frame(frame):
                continue
            if CAPTCHA_FRAME.search(frame.url):
                captcha = captcha or "size=invisible" not in frame.url
                continue  # a CAPTCHA is for the person watching, not the model
        try:
            found = frame.evaluate(_COLLECT_JS, len(_REFS) + 1)
        except Exception:
            continue  # a frame that navigated or closed mid-read
        for e in found:
            _REFS[e["n"]] = frame
        elements += found
    if len(elements) > MAX_ELEMENTS:  # keep what's on screen and the controls; far-away links go first
        keep = {e["n"] for e in sorted(elements, key=lambda e: (not e["inView"], e["kind"] == "link"))[:MAX_ELEMENTS]}
        elements = [e for e in elements if e["n"] in keep]
    return elements, captcha


def _marks(on):
    for frame in {id(f): f for f in _REFS.values()}.values():
        try:
            frame.evaluate(_MARKS_JS, on)
        except Exception:
            pass


def _shot(page):
    # A JPEG of what's on screen with each element's number drawn on, so the model can match what it
    # sees to the numbers it acts with (and the user sees the same picture in the chat).
    _marks(True)
    try:
        return "data:image/jpeg;base64," + base64.b64encode(page.screenshot(type="jpeg", quality=60)).decode()
    except Exception:
        return None
    finally:
        _marks(False)


def _state_and_shot(page, note=""):
    content = _page_state(page, note)[:MAX_TOOL_OUTPUT]  # numbers the elements, which the screenshot then shows
    return {"content": content, "image": _shot(page)}


def _page_state(page, note=""):
    elements, captcha = _number_elements(page)
    title = page.title()
    text = " ".join((page.evaluate("() => document.body ? document.body.innerText : ''") or "").split())
    if re.match(r"just a moment|attention required", title, re.I) or re.search(r"verify (that )?you are (a )?human", text[:3000], re.I):
        captcha = True
    head = [f"URL: {page.url}", f"Title: {title}"]
    tabs = sum(not t.is_closed() for t in page.context.pages)
    if tabs > 1:
        head.append(f"{tabs} tabs are open; you're on the newest.")
    if note:
        head.append(note)
    if captcha:
        head.append(HUMAN_CHECK)
    elif len(text) < 15 and not elements:
        head.append(BLANK_PAGE)
    els = "\n".join(f"[{e['n']}] {e['kind']} \"{e['name']}\"" + (f" ({e['info']})" if e["info"] else "") for e in elements)
    room = max(800, min(5000, MAX_TOOL_OUTPUT - len(els) - 800))  # the element list matters more than the prose
    return ("\n".join(head) + f"\n\nPage text:\n{text[:room]}\n\n"
            "Elements you can act on (pass the number as `ref` to browser_act; the screenshot shows the same numbers):\n"
            + (els or "(none)"))


def _settle(page, load_ms=6000, pause_ms=500):
    # Let the page finish what it started (load, then a quiet network) without waiting forever on
    # sites that never stop polling.
    for state, ms in (("load", load_ms), ("networkidle", 2500)):
        try:
            page.wait_for_load_state(state, timeout=ms)
        except Exception:
            pass
    page.wait_for_timeout(pause_ms)


def _act_error(e):
    # Playwright's errors end in a long call log; keep the part that says what went wrong.
    s = str(e)
    if "intercepts pointer events" in s:
        return "something is covering that element (a dialog, banner, ad, or loading overlay); deal with that first, or wait"
    return s.split("Call log:")[0].strip().removeprefix("Error: ").rstrip(".")[:300]


def tool_browser_navigate(args):
    url = str(args.get("url") or "").strip()
    if not url:
        return "Error: browser_navigate needs a 'url'."
    if not re.match(r"^https?://", url):
        url = "https://" + url

    def go(page):
        note = ""
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            note = f"Opening {url} didn't finish: {_act_error(e)}."
        _settle(page)
        return _state_and_shot(page, note)
    res = BROWSER.call(go, timeout=60)
    return f"Error: {res['error']}" if res.get("error") else res["value"]


def tool_browser_read(args):
    res = BROWSER.call(_state_and_shot)
    if res.get("error"):
        return f"Error: {res['error']} (open a page first with browser_navigate)"
    return res["value"]


def tool_browser_act(args):
    action = str(args.get("action") or "").strip().lower()
    ref, target = args.get("ref"), str(args.get("target") or "").strip()
    text = "" if args.get("text") is None else str(args.get("text"))

    def element(page):
        if ref not in (None, ""):
            m = re.search(r"\d+", str(ref))
            frame = _REFS.get(int(m.group())) if m else None
            if frame is None or frame.is_detached():
                raise ValueError(f"there's no element {ref} on the page now; use a number from the list below")
            return frame.locator(f'[data-fk="{m.group()}"]').first
        if not target:
            raise ValueError(f"{action} needs a 'ref': the number of an element from the list")
        if re.match(r"^[.#\[]|^(a|button|input|textarea|select|div|span|h[1-6])[\s.#\[>:]", target):
            return page.locator(target).first
        return page.get_by_text(target, exact=False).first

    def act(page):
        before, note = list(page.context.pages), ""
        try:
            if action == "click":
                element(page).click(timeout=8000)
            elif action == "type":
                el = element(page)
                try:
                    el.fill(text, timeout=8000)
                except Exception as e:
                    if "not an <input>" not in str(e):
                        raise
                    el.click(timeout=8000)  # a custom editor rather than a plain field: type like a person
                    page.keyboard.type(text, delay=15)
                if args.get("submit"):
                    page.keyboard.press("Enter")
            elif action == "select":
                el = element(page)
                try:
                    el.select_option(label=text, timeout=8000)
                except Exception:
                    el.select_option(text, timeout=8000)
            elif action == "press":
                page.keyboard.press(text or target or "Enter")
            elif action == "scroll":
                if ref not in (None, ""):
                    element(page).scroll_into_view_if_needed(timeout=8000)
                else:
                    page.mouse.wheel(0, -700 if text.lower() == "up" else 700)
            elif action == "back":
                page.go_back(timeout=15000)
            elif action == "wait":
                m = re.search(r"\d+", text)
                page.wait_for_timeout(1000 * min(int(m.group()) if m else 3, 20))
            else:
                raise ValueError("action must be click, type, select, press, scroll, back, or wait")
        except Exception as e:
            note = (f"That {action} didn't work: {_act_error(e)}. The page as it is now is below; "
                    "pick an element by its number rather than reloading.")
        _settle(page, 5000, 600)
        opened = [t for t in page.context.pages if t not in before and not t.is_closed()]  # it shows up a moment after the click
        if opened:
            page = opened[-1]
            page.bring_to_front()
            _settle(page, 5000, 300)
            note = (note + "\n" if note else "") + "That opened a new tab; you're on it now."
        return _state_and_shot(page, note)
    res = BROWSER.call(act, timeout=75)
    return f"Error: {res['error']}" if res.get("error") else res["value"]


def tool_browser_screenshot(args):
    # Every browser step already comes with a screenshot; this one is for a fresh look on demand.
    return tool_browser_read(args)


# --- image generation on the pod's ComfyUI (MOSS-Image), reusing imagine.py's graph ------

COMFY_URL = "http://127.0.0.1:8188"


def tool_generate_image(args):
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return "Error: generate_image needs a 'prompt'."
    try:
        sys.path.insert(0, str(KIT))
        import imagine as im
    except Exception as e:
        return f"Error: couldn't load the image generator ({e})."
    try:
        im.api(COMFY_URL, "/system_stats")
    except Exception:
        return ("ComfyUI isn't running. Image generation uses the pod's ComfyUI, which shares the one GPU "
                "with the chat model — start it with `./gpu.sh comfy` (the model pauses while it renders), "
                "then try again.")

    class A:
        pass
    a = A()
    a.prompt = prompt
    a.negative = str(args.get("negative") or im.NEGATIVE)
    a.model = args.get("model") if args.get("model") in im.MODELS else "sft"
    size = str(args.get("size") or "1024x1024").lower()
    try:
        a.width, a.height = (max(256, min(2048, int(x))) for x in size.split("x"))
    except Exception:
        a.width, a.height = 1024, 1024
    a.steps = max(1, min(int(args.get("steps") or 25), 60))
    a.cfg = float(args.get("cfg") or 2.5)
    a.sampler, a.scheduler, a.batch, a.no_slg = "er_sde", "simple", 1, False
    seed = int(args["seed"]) if str(args.get("seed") or "").lstrip("-").isdigit() else random.randrange(2 ** 48)

    graph = im.build_graph(a, seed, [])
    pid = im.api(COMFY_URL, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 300:
        hist = im.api(COMFY_URL, f"/history/{pid}").get(pid) or {}
        st = hist.get("status", {})
        if st.get("completed") or st.get("status_str") == "error":
            break
        time.sleep(1.5)
    else:
        return "Error: the image took too long (over 5 minutes)."
    if hist.get("status", {}).get("status_str") == "error":
        return "Error: generation failed. See `./gpu.sh logs comfy` for the trace."
    img = hist["outputs"]["9"]["images"][0]
    q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
    with urllib.request.urlopen(f"{COMFY_URL}/view?{q}", timeout=120) as r:
        raw = r.read()
    try:
        (KIT / "images").mkdir(exist_ok=True)
        (KIT / "images" / img["filename"]).write_bytes(raw)
    except OSError:
        pass
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    return {"content": f"Generated a {a.width}x{a.height} image (seed {seed}) and showed it to the user.", "image": data_url}


# Each tool: metadata for the UI + an OpenAI function spec the browser passes to vLLM.
# risk "safe" runs on its own; risk "confirm" waits for the user's approval in the page.
TOOLS = [
    {"name": "web_search", "label": "Web search", "icon": "search", "risk": "safe", "fn": tool_web_search,
     "spec": {"type": "function", "function": {
         "name": "web_search",
         "description": "Search the web and get a list of results with titles, URLs and snippets. Use this to find current information or pages to read.",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string", "description": "What to search for."},
             "max_results": {"type": "integer", "description": "How many results (1-10, default 5)."},
         }, "required": ["query"]}}}},
    {"name": "fetch_url", "label": "Read a web page", "icon": "globe", "risk": "safe", "fn": tool_fetch_url,
     "spec": {"type": "function", "function": {
         "name": "fetch_url",
         "description": "Fetch a web page and return its readable text. Use after web_search to read a result, or on any URL the user gives.",
         "parameters": {"type": "object", "properties": {
             "url": {"type": "string", "description": "The full http(s):// URL to read."},
         }, "required": ["url"]}}}},
    {"name": "run_command", "label": "Run a command on this Mac", "icon": "terminal", "risk": "confirm", "fn": tool_run_command,
     "spec": {"type": "function", "function": {
         "name": "run_command",
         "description": "Run a shell command on the user's Mac and return its output. Use this to inspect files, run scripts, or take actions on the computer. The user approves each command before it runs.",
         "parameters": {"type": "object", "properties": {
             "command": {"type": "string", "description": "The shell command to run."},
             "cwd": {"type": "string", "description": "Working directory (default: the home folder)."},
             "timeout": {"type": "integer", "description": "Seconds to wait before giving up (default 120, max 600)."},
         }, "required": ["command"]}}}},
    {"name": "run_python", "label": "Run Python", "icon": "code", "risk": "confirm", "fn": tool_run_python,
     "spec": {"type": "function", "function": {
         "name": "run_python",
         "description": "Run Python code in a session that persists between calls (variables, imports and data stay). Use for calculation, data work, plotting (save figures to files), and quick scripts. Prints and the last expression's output come back.",
         "parameters": {"type": "object", "properties": {
             "code": {"type": "string", "description": "Python source to run."},
             "timeout": {"type": "integer", "description": "Seconds before giving up (default 120, max 600)."},
         }, "required": ["code"]}}}},
    {"name": "read_file", "label": "Read a file", "icon": "file", "risk": "safe", "fn": tool_read_file,
     "spec": {"type": "function", "function": {
         "name": "read_file", "description": "Read a text file from the Mac and return its contents.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Path to the file (~ is allowed)."}}, "required": ["path"]}}}},
    {"name": "write_file", "label": "Write a file", "icon": "file", "risk": "confirm", "fn": tool_write_file,
     "spec": {"type": "function", "function": {
         "name": "write_file", "description": "Create or overwrite a text file on the Mac.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Path to write (~ is allowed; folders are created)."},
             "content": {"type": "string", "description": "The full file contents."}}, "required": ["path", "content"]}}}},
    {"name": "edit_file", "label": "Edit a file", "icon": "file", "risk": "confirm", "fn": tool_edit_file,
     "spec": {"type": "function", "function": {
         "name": "edit_file", "description": "Replace one exact, unique passage in a text file with new text.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Path to the file."},
             "old_str": {"type": "string", "description": "Exact text to find (must occur once)."},
             "new_str": {"type": "string", "description": "What to replace it with."}}, "required": ["path", "old_str", "new_str"]}}}},
    {"name": "list_dir", "label": "List a folder", "icon": "file", "risk": "safe", "fn": tool_list_dir,
     "spec": {"type": "function", "function": {
         "name": "list_dir", "description": "List the entries of a folder on the Mac (d = directory).",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Folder path (default: home)."}}, "required": []}}}},
    {"name": "browser_navigate", "label": "Open a page in the browser", "icon": "globe", "risk": "confirm", "fn": tool_browser_navigate,
     "spec": {"type": "function", "function": {
         "name": "browser_navigate",
         "description": ("Open a URL in the browser: a real Chromium window on the user's screen that they can watch and take over. "
                         "Returns the page's text, a numbered list of the elements you can act on, and a screenshot with the same numbers drawn on it. "
                         "Use this (not fetch_url) when a page needs JavaScript or you need to click or type. If a page needs a CAPTCHA, a sign-in, "
                         "a code, or a 'verify you are human' check, ask the user to do that part in the browser window and wait for their reply."),
         "parameters": {"type": "object", "properties": {
             "url": {"type": "string", "description": "The address to open."}}, "required": ["url"]}}}},
    {"name": "browser_read", "label": "Read the browser page", "icon": "globe", "risk": "safe", "fn": tool_browser_read,
     "spec": {"type": "function", "function": {
         "name": "browser_read", "description": "Look at the current browser page again: its text, numbered elements, and a fresh screenshot. Use it after the user says they did something in the browser window, or when a page was still loading.",
         "parameters": {"type": "object", "properties": {}, "required": []}}}},
    {"name": "browser_act", "label": "Act in the browser", "icon": "globe", "risk": "confirm", "fn": tool_browser_act,
     "spec": {"type": "function", "function": {
         "name": "browser_act",
         "description": ("Act on the current browser page and get it back afterwards (text, numbered elements, screenshot). "
                         "Pick the element by its number from the latest list or screenshot; the numbers change every time the page is read, so use the newest ones. "
                         "If an action fails, the result still shows the page as it is now: choose another element or deal with what's in the way, rather than reloading or opening the page again."),
         "parameters": {"type": "object", "properties": {
             "action": {"type": "string", "enum": ["click", "type", "select", "press", "scroll", "back", "wait"],
                        "description": "click an element, type into a field, select a dropdown option, press a key, scroll, go back a page, or wait for the page."},
             "ref": {"type": "integer", "description": "The element's number from the list, e.g. 12. Needed for click, type and select."},
             "text": {"type": "string", "description": "For type: the text. For select: the option's label. For press: the key (Enter, Tab, Escape, ArrowDown...). For scroll: up or down. For wait: seconds (default 3)."},
             "submit": {"type": "boolean", "description": "For type: press Enter afterwards."},
             "target": {"type": "string", "description": "Only when the element has no number: its visible text or a CSS selector."}},
             "required": ["action"]}}}},
    {"name": "browser_screenshot", "label": "Screenshot the browser", "icon": "globe", "risk": "safe", "fn": tool_browser_screenshot,
     "spec": {"type": "function", "function": {
         "name": "browser_screenshot", "description": "Take a fresh screenshot of the current browser page (you see it too), along with its text and numbered elements.",
         "parameters": {"type": "object", "properties": {}, "required": []}}}},
    {"name": "generate_image", "label": "Generate an image", "icon": "image", "risk": "confirm", "fn": tool_generate_image,
     "spec": {"type": "function", "function": {
         "name": "generate_image",
         "description": "Generate an image from a text prompt with the self-hosted MOSS-Image model and show it to the user. Needs ComfyUI running on the pod.",
         "parameters": {"type": "object", "properties": {
             "prompt": {"type": "string", "description": "A detailed description of the image."},
             "size": {"type": "string", "description": "WxH, e.g. 1024x1024 or 1344x768 (default 1024x1024)."},
             "steps": {"type": "integer", "description": "Sampling steps (default 25)."},
             "seed": {"type": "integer", "description": "Seed for a reproducible image (optional)."}},
             "required": ["prompt"]}}}},
    {"name": "remember", "label": "Remember something", "icon": "memory", "risk": "safe", "fn": tool_remember,
     "spec": {"type": "function", "function": {
         "name": "remember",
         "description": "Save a lasting note about the user or their preferences, available in every future chat. Use when they share something worth keeping.",
         "parameters": {"type": "object", "properties": {
             "note": {"type": "string", "description": "The fact to remember, in one line."}}, "required": ["note"]}}}},
    {"name": "forget", "label": "Forget something", "icon": "memory", "risk": "safe", "fn": tool_forget,
     "spec": {"type": "function", "function": {
         "name": "forget", "description": "Remove saved notes that match some text.",
         "parameters": {"type": "object", "properties": {
             "note": {"type": "string", "description": "Text to match against saved notes."}}, "required": ["note"]}}}},
]
TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def tools_public():
    """What the page needs: UI metadata plus the spec it forwards to the model."""
    return [{k: t[k] for k in ("name", "label", "icon", "risk", "spec")} for t in TOOLS]


def run_tool(body):
    name = str(body.get("name") or "")
    args = body.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    tool = TOOLS_BY_NAME.get(name)
    if not tool:
        return {"content": f"Error: unknown tool {name!r}."}
    try:
        out = tool["fn"](args)
    except Exception as e:  # feed failures back to the model as data, so it can adapt
        return {"content": f"Error running {name}: {e}"}
    if isinstance(out, dict):  # a tool may return an image to show the user alongside its text
        res = {"content": str(out.get("content", ""))}
        if out.get("image"):
            res["image"] = out["image"]
        return res
    return {"content": str(out)}


# --- chats on disk ------------------------------------------------------------------

def list_chats(folder):
    items = []
    for f in folder.glob("*.json"):
        try:
            c = json.loads(f.read_text())
            items.append({"id": c["id"], "title": c.get("title") or "New chat", "updated": c.get("updated", 0)})
        except (OSError, ValueError, LookupError, TypeError):
            continue
    return sorted(items, key=lambda c: c["updated"], reverse=True)


def save_chat(folder, cid, chat):
    if not isinstance(chat, dict) or not isinstance(chat.get("messages"), list):
        raise BadRequest("a chat needs a messages list")
    chat["id"] = cid
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{cid}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(chat, ensure_ascii=False))
    tmp.replace(folder / f"{cid}.json")
    return {"ok": True}


# --- HTTP ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "fikirmatik"
    chat_dir = CHATS

    def log_request(self, code="-", size="-"):
        if self.command != "GET" or str(code)[:1] in "45":
            super().log_request(code, size)

    def do_GET(self):
        self.route()

    def do_POST(self):
        self.route()

    def do_PUT(self):
        self.route()

    def do_DELETE(self):
        self.route()

    def trusted(self):
        # Only this page may drive the app: the Host check stops DNS rebinding, the Origin
        # check stops other websites from posting to it.
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in {"http://" + h for h in hosts})

    def route(self):
        if not self.trusted():
            return self.send_json(403, {"error": "forbidden"})
        try:
            self.dispatch(self.command, self.path.split("?", 1)[0])
        except BadRequest as e:
            self.send_json(e.code, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def dispatch(self, method, path):
        if path.startswith("/v1/"):
            return self.proxy()
        if method == "GET" and path == "/":
            return self.send_bytes(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if method == "GET" and path == "/logo.png":
            return self.send_bytes(200, LOGO.read_bytes(), "image/png")
        if method == "GET" and path == "/api/ping":
            return self.send_json(200, {"app": APP_ID})
        if method == "GET" and path == "/api/status":
            return self.send_json(200, status())
        if method == "GET" and path == "/api/config":
            return self.send_json(200, config())
        if method == "GET" and path == "/api/box":
            return self.send_json(200, box_info())
        if method == "GET" and path == "/api/tools":
            return self.send_json(200, {"tools": tools_public()})
        if method == "POST" and path == "/api/tools/run":
            return self.send_json(200, run_tool(self.body()))
        if method == "POST" and path == "/api/pod":
            return self.send_json(200, set_pod(self.body()))
        if method == "POST" and path == "/api/connect":
            tunnel.start()
            return self.send_json(200, {"ok": True})
        if method == "POST" and path == "/api/disconnect":
            tunnel.stop()
            return self.send_json(200, {"ok": True})
        if method == "POST" and path == "/api/forget":
            rc, out = gpu("forget")
            if rc:
                raise BadRequest(out or "couldn't forget the old key", 502)
            tunnel.stop()
            tunnel.start()
            return self.send_json(200, {"ok": True})
        if method == "POST" and path == "/api/serve":
            return self.send_json(200, start_model(self.body()))
        if method == "POST" and path == "/api/stop-serve":
            rc, out = gpu("stop", "serve")
            if rc and not re.search(r"can't find session|no server running", out):
                raise BadRequest(out or "couldn't stop the model", 502)
            return self.send_json(200, {"ok": True})
        if method == "POST" and path == "/api/stop-pod":
            return self.send_json(200, stop_pod())
        if method == "POST" and path == "/api/pod/start":
            podwatch.start(create=bool(self.body().get("create")))
            return self.send_json(200, {"ok": True})
        if method == "POST" and path == "/api/runpod-key":
            return self.send_json(200, save_runpod_key(self.body()))
        if method == "POST" and path == "/api/quit":
            self.send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path.startswith("/api/chats"):
            return self.chats(method, path)
        self.send_json(404, {"error": "not found"})

    def chats(self, method, path):
        """The saved chats in self.chat_dir: list, read, save, delete."""
        if method == "GET" and path == "/api/chats":
            return self.send_json(200, list_chats(self.chat_dir))
        m = re.fullmatch(r"/api/chats/([a-z0-9-]{1,64})", path)
        if m:
            f = self.chat_dir / f"{m.group(1)}.json"
            if method == "GET":
                if not f.exists():
                    raise BadRequest("no such chat", 404)
                return self.send_bytes(200, f.read_bytes(), "application/json")
            if method == "PUT":
                return self.send_json(200, save_chat(self.chat_dir, m.group(1), self.body()))
            if method == "DELETE":
                f.unlink(missing_ok=True)
                return self.send_json(200, {"ok": True})
        self.send_json(404, {"error": "not found"})

    def proxy(self):
        """Pass /v1/* to vLLM through the tunnel, streaming the answer back as it arrives."""
        n = int(self.headers.get("Content-Length") or 0)
        req = urllib.request.Request(VLLM + self.path, data=self.rfile.read(n) if n else None, method=self.command,
                                     headers={"Content-Type": "application/json"})
        try:
            upstream = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            return self.send_bytes(e.code, e.read(), e.headers.get("Content-Type", "application/json"))
        except (OSError, http.client.HTTPException) as e:
            return self.send_json(502, {"error": {"message": f"The model isn't reachable ({e})."}})
        with upstream:
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Stop was pressed; closing upstream makes vLLM abort the request

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise BadRequest("the request body must be JSON")

    def send_json(self, code, obj):
        self.send_bytes(code, json.dumps(obj).encode(), "application/json")

    def send_bytes(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


# --- the guest page, for people you share this Mac with on Tailscale ------------------

class GuestHandler(Handler):
    """Chat only: see whether a model is ready, talk to it, keep your own chats. Who counts as a
    guest is decided by which listener they reached, never by anything their browser sends."""

    @property
    def chat_dir(self):
        return GUEST_CHATS / self.client_address[0]  # one folder per guest device

    def trusted(self):
        # The same Host/Origin checks as the main page, for this Mac's names on Tailscale.
        hosts, origin = self.server.hosts, self.headers.get("Origin")
        return (self.headers.get("Host") or "").lower() in hosts and (
            origin is None or origin.lower() in {"http://" + h for h in hosts})

    def dispatch(self, method, path):
        if method == "POST" and path == "/v1/chat/completions":
            return self.proxy()
        if method == "GET" and path == "/":  # the same page, told to hide everything but the chat
            return self.send_bytes(200, PAGE.read_bytes().replace(b"<html", b"<html data-guest", 1),
                                   "text/html; charset=utf-8")
        if method == "GET" and path == "/logo.png":
            return self.send_bytes(200, LOGO.read_bytes(), "image/png")
        if method == "GET" and path == "/api/status":
            model = probe_model()
            if model:
                model.pop("root")  # where the weights came from is the host's business
            return self.send_json(200, {"state": "ready", **model} if model else {"state": "host_offline"})
        if method == "GET" and path == "/api/config":
            return self.send_json(200, {"system": read_env().get("CHAT_SYSTEM", "")})
        if path.startswith("/api/chats"):
            return self.chats(method, path)
        self.send_json(404, {"error": "not found"})


def tailscale_self():
    """This Mac's Tailscale IPv4 address and MagicDNS name, or None while Tailscale isn't running."""
    for exe in TAILSCALE:
        try:
            p = subprocess.run([exe, "status", "--json"], stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
            s = json.loads(p.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            continue
        if not isinstance(s, dict) or s.get("BackendState") != "Running":
            return None
        me = s.get("Self") or {}
        for ip in me.get("TailscaleIPs") or []:
            try:
                if ipaddress.ip_address(ip) in TAILNET:
                    return ip, (me.get("DNSName") or "").rstrip(".").lower()
            except ValueError:
                continue
        return None
    return None


class GuestPage:
    """The guest page's own server, on this Mac's Tailscale address. It waits for Tailscale, so
    fikirmatik.ai can start before Tailscale has connected."""

    def __init__(self):
        self.url = None

    def run(self, port):
        said = None
        while True:
            me = tailscale_self()
            if me:
                ip, name = me
                try:
                    server = ThreadingHTTPServer((ip, port), GuestHandler)
                    break
                except OSError as e:
                    if str(e) != said:
                        said = str(e)
                        print(f"guest page: can't listen on {ip}:{port} ({e}); trying again every 30s", flush=True)
            time.sleep(30)
        server.hosts = {f"{h}:{port}" for h in (ip, name, name.split(".")[0]) if h}
        self.url = f"http://{ip}:{port}"
        print(f"guest page (chat only) on {self.url}" + (f" and http://{name}:{port}" if name else ""), flush=True)
        server.serve_forever()


guests = GuestPage()


def ours(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2) as r:
            return json.load(r).get("app") == APP_ID
    except (OSError, ValueError, http.client.HTTPException):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", 7777)))
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = p.parse_args()

    server = None
    for port in range(a.port, a.port + 10):
        if ours(port):  # already running: just show it
            print(f"fikirmatik.ai is already running at http://127.0.0.1:{port}")
            if not a.no_open:
                webbrowser.open(f"http://127.0.0.1:{port}")
            return
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit(f"ports {a.port}-{a.port + 9} are all taken; pass --port")

    if not port_open(VLLM_PORT) and read_env().get("POD_HOST"):  # nobody has a tunnel open yet
        tunnel.start()
    podwatch.snapshot()  # ask RunPod now, so the page knows whether the pod is on when it first loads
    threading.Thread(target=guests.run, args=(server.server_port,), daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"fikirmatik.ai on {url}  (Quit in the page, or Ctrl-C)", flush=True)
    if not a.no_open:
        webbrowser.open(url)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.stop()
        server.server_close()


if __name__ == "__main__":
    main()
