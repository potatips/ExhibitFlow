"""Hardened auto-tunnel launcher for the standalone ExhibitFlow GPU Worker.

Supports:
  - Cloudflare Quick Tunnel (`--tunnel quick`, fallback `localhost.run`)
  - Cloudflare Named Tunnel (`--tunnel named --tunnel-token ... --public-url ...`)
  - Local / Vast.ai direct mode (`--tunnel none` / `--no-tunnel`)
  - Strict readiness verification (`evaluate_readiness`, `probe_external_health`)
  - Token-safe logging (`--log-file`, never writes GPU_TOKEN / HF_TOKEN / tunnel token to disk)

Usage:
    python launch.py --port 8001 [--demo] [--data-dir ./gpu_data] [--tunnel quick|named|none]
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
URL_RE = re.compile(r"https://[a-z0-9-]+\.(?:trycloudflare\.com|lhr\.life|localhost\.run)")


def _download(urls: list[str], dst: Path, tries: int = 3) -> bool:
    last: Optional[Exception] = None
    for url in urls:
        for k in range(tries):
            try:
                print(f"Downloading cloudflared ({k + 1}/{tries}): {url}", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "ExhibitFlow-GPU-Worker"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as fh:
                    fh.write(resp.read())
                return True
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"  download error: {exc}", flush=True)
                time.sleep(2 * (k + 1))
    print(f"Failed to download cloudflared: {last}", flush=True)
    return False


def get_cloudflared(dst_dir: Path) -> Optional[str]:
    """Locate cloudflared in PATH or download binary into dst_dir (Linux, macOS .tgz, Windows)."""
    found = shutil.which("cloudflared")
    if found:
        return found
    exe = dst_dir / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
    if exe.exists() and exe.stat().st_size > 1_000_000:
        return str(exe)

    sys_name = platform.system().lower()
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("aarch64", "arm64") else "amd64"
    base = "https://github.com/cloudflare/cloudflared/releases"
    dst_dir.mkdir(parents=True, exist_ok=True)

    if sys_name == "darwin":
        archive = dst_dir / f"cloudflared-darwin-{arch}.tgz"
        urls = [
            f"{base}/latest/download/cloudflared-darwin-{arch}.tgz",
            f"{base}/download/2024.12.2/cloudflared-darwin-{arch}.tgz",
        ]
        if not _download(urls, archive):
            return None
        try:
            with tarfile.open(archive, "r:gz") as tf:
                member = next((m for m in tf.getmembers() if Path(m.name).name == "cloudflared"), None)
                if member is None:
                    return None
                src = tf.extractfile(member)
                if src is None:
                    return None
                with open(exe, "wb") as out:
                    shutil.copyfileobj(src, out)
        finally:
            archive.unlink(missing_ok=True)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        return str(exe)

    name = "windows-amd64.exe" if os.name == "nt" else f"linux-{arch}"
    urls = [
        f"{base}/latest/download/cloudflared-{name}",
        f"{base}/download/2024.12.2/cloudflared-{name}",
    ]
    if not _download(urls, exe):
        return None
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def build_quick_tunnel_cmd(exe: str, port: int) -> list[str]:
    return [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]


def build_named_tunnel_cmd(exe: str, tunnel_token: str) -> list[str]:
    if not tunnel_token:
        raise ValueError("Named tunnel requires a non-empty tunnel token")
    return [exe, "tunnel", "--no-autoupdate", "run", "--token", tunnel_token]


def start_cloudflared(port: int, bin_dir: Path) -> Optional[subprocess.Popen[str]]:
    exe = get_cloudflared(bin_dir)
    if not exe:
        return None
    return subprocess.Popen(
        build_quick_tunnel_cmd(exe, port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_named_tunnel(tunnel_token: str, bin_dir: Path) -> Optional[subprocess.Popen[str]]:
    exe = get_cloudflared(bin_dir)
    if not exe:
        return None
    return subprocess.Popen(
        build_named_tunnel_cmd(exe, tunnel_token),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_localhost_run(port: int) -> Optional[subprocess.Popen[str]]:
    ssh = shutil.which("ssh")
    if not ssh:
        return None
    return subprocess.Popen(
        [
            ssh,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ServerAliveInterval=30",
            "-R",
            f"80:127.0.0.1:{port}",
            "nokey@localhost.run",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_url(proc: subprocess.Popen[str], timeout: float = 60.0) -> Optional[str]:
    found = threading.Event()
    box: dict[str, str] = {}

    def _pump() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m and "url" not in box and "admin." not in m.group(0):
                box["url"] = m.group(0)
                found.set()

    threading.Thread(target=_pump, daemon=True).start()
    found.wait(timeout)
    return box.get("url")


def _drain_output(proc: subprocess.Popen[str]) -> None:
    def _pump() -> None:
        if proc.stdout is None:
            return
        for _ in proc.stdout:
            pass

    threading.Thread(target=_pump, daemon=True).start()


def fetch_local_health(port: int, token: str, timeout: float = 2.0) -> dict[str, Any]:
    headers = {"X-GPU-Token": token} if token else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/internal/health", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_models(port: int, token: str, timeout: int = 3600) -> dict[str, Any]:
    t0 = time.time()
    last = 0.0
    last_msg = ""
    print("Waiting for GPU models & warmup to finish...", flush=True)
    while time.time() - t0 < timeout:
        try:
            h = fetch_local_health(port, token, timeout=3.0)
            m = h.get("models") or {"state": "loading", "msg": ""}
            v = h.get("vlm") or {}
        except Exception:  # noqa: BLE001
            m = {"state": "loading", "msg": ""}
            v = {}
        m_state = m.get("state")
        v_state = v.get("state")
        if m_state == "error":
            return m
        if m_state == "ready" and v_state != "starting":
            print(f"  [warmup] All GPU models warmed up and ready ({int(time.time() - t0)}s).", flush=True)
            return m
        msg = m.get("msg") or ""
        now = time.time()
        if (msg and msg != last_msg) or (now - last > 10):
            last = now
            last_msg = msg
            print(f"  ... loading/warming up ({int(now - t0)}s): {msg or m_state or 'initializing'}", flush=True)
        time.sleep(2.0)
    return {"state": "timeout", "msg": "Timed out waiting for models"}


def wait_tunnel_dns(
    public_url: str,
    timeout_s: float = 20.0,
    interval_s: float = 1.0,
) -> bool:
    """Wait for Cloudflare DoH (1.1.1.1) to resolve a new trycloudflare.com host before probing via OS DNS.

    Prevents poisoning the local OS resolver negative DNS cache with NXDOMAIN right when cloudflared starts.
    """
    host = (urllib.parse.urlparse(public_url).hostname or "").strip().lower()
    if not host.endswith(".trycloudflare.com"):
        return True
    doh_url = f"https://1.1.1.1/dns-query?{urllib.parse.urlencode({'name': host, 'type': 'A'})}"
    deadline = time.time() + max(1.0, timeout_s)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                doh_url,
                headers={"Accept": "application/dns-json", "User-Agent": "ExhibitFlow-GPU-Worker"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("Status") == 0 and data.get("Answer"):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(3.5)
            return False
        time.sleep(interval_s)
    return False


def probe_external_health(
    public_url: str,
    token: str,
    timeout_s: float = 45.0,
    interval_s: float = 1.5,
) -> tuple[bool, Optional[bool], str]:
    """Probe public_url/internal/health with retries for tunnel/DNS propagation.

    Returns (auth_probe_ok, unauth_rejected_401_or_None, message).
    """
    base = public_url.rstrip("/")
    deadline = time.time() + max(1.0, timeout_s)
    last_err = "no response"
    auth_ok = False

    while time.time() < deadline:
        try:
            headers = {"X-GPU-Token": token} if token else {}
            req = urllib.request.Request(f"{base}/internal/health", headers=headers)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("ok") is True:
                        auth_ok = True
                        break
                    last_err = f"unexpected payload ok={payload.get('ok')}"
                else:
                    last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(interval_s)

    if not auth_ok:
        return False, None, f"Authenticated probe failed ({last_err})"

    if not token:
        return True, None, "ok (no token configured)"

    unauth_401 = False
    try:
        req_no_auth = urllib.request.Request(f"{base}/internal/health")
        with urllib.request.urlopen(req_no_auth, timeout=5.0) as resp:
            unauth_401 = resp.status == 401
    except urllib.error.HTTPError as exc:
        unauth_401 = exc.code == 401
    except Exception:  # noqa: BLE001
        unauth_401 = False

    msg = "ok (200 authed, 401 unauthed)" if unauth_401 else "unauthenticated request was not rejected with 401"
    return True, unauth_401, msg


def evaluate_readiness(
    health: dict[str, Any],
    *,
    demo: bool,
    wait_models_flag: bool = True,
    tunnel_requested: bool = False,
    public_url: Optional[str] = None,
    external_ok: Optional[bool] = None,
    external_auth_ok: Optional[bool] = None,
) -> tuple[bool, list[str]]:
    """Evaluate strict readiness across worker health, models, VLM, and external tunnel."""
    reasons: list[str] = []
    if not health or health.get("ok") is not True:
        reasons.append("Local /internal/health did not return ok=true")
        return False, reasons

    if not demo and wait_models_flag:
        models = health.get("models") or {}
        vlm = health.get("vlm") or {}
        m_state = models.get("state")
        if m_state != "ready":
            reasons.append(f"models.state is {m_state!r} ({models.get('msg') or 'not ready'})")
        if vlm.get("state") in ("error", "starting"):
            reasons.append(f"vlm.state is {vlm.get('state')!r}")

    if tunnel_requested:
        if not public_url:
            reasons.append("Public HTTPS tunnel URL was not established")
        else:
            if external_ok is not True:
                reasons.append(f"External health probe failed for {public_url}")
            elif external_auth_ok is False:
                reasons.append(f"External unauthenticated probe to {public_url} did not return 401")

    return len(reasons) == 0, reasons


def detect_vast_direct_url(port: int) -> Optional[str]:
    ip = (os.environ.get("PUBLIC_IPADDR") or "").strip()
    ext_port = (os.environ.get(f"VAST_TCP_PORT_{port}") or "").strip()
    if ip and ext_port:
        return f"http://{ip}:{ext_port}"
    return None


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def print_banner(
    *,
    local_url: str,
    public_url: Optional[str],
    vast_url: Optional[str],
    token: str,
    demo: bool,
    tunnel_mode: str,
    ready: bool,
    reasons: list[str],
    log_file: Path,
) -> None:
    print(flush=True)
    print("=" * 72, flush=True)
    if ready:
        print(f"  STATUS: READY (SẴN SÀNG) — ExhibitFlow GPU Worker ({'DEMO CPU' if demo else 'GPU'})", flush=True)
    else:
        print(f"  STATUS: NOT READY (CHƯA SẴN SÀNG) — ExhibitFlow GPU Worker ({'DEMO CPU' if demo else 'GPU'})", flush=True)
    print(f"  Local URL   : {local_url}", flush=True)
    if public_url:
        print(f"  Public HTTPS: {public_url}  (mode={tunnel_mode})", flush=True)
    if vast_url:
        print(f"  Vast Direct : {vast_url}  (HTTP direct; Cloudflare HTTPS preferred)", flush=True)
    if token:
        print(f"  X-GPU-Token : {token}", flush=True)
    print(f"  Worker Log  : {log_file}  (secrets are NEVER written to this log)", flush=True)
    if ready:
        print("  Paste Public HTTPS URL + X-GPU-Token into Backend / Web Settings.", flush=True)
    else:
        for reason in reasons:
            print(f"  [!] {reason}", flush=True)
        print("  Action: Check worker log or weights/tunnel settings above.", flush=True)
    print("=" * 72, flush=True)


def _monitor_tunnel(tun: subprocess.Popen[str], srv: subprocess.Popen[str], stop_ev: threading.Event) -> None:
    while not stop_ev.is_set() and srv.poll() is None:
        rc = tun.poll()
        if rc is not None:
            print(
                f"\n[!] CẢNH BÁO / WARNING: Tunnel process exited unexpectedly (code {rc})! "
                "Restart launch.py to establish a new public URL.",
                flush=True,
            )
            return
        time.sleep(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch ExhibitFlow GPU Worker with strict readiness & tunnel")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--tunnel", choices=("quick", "named", "none"), default="quick")
    ap.add_argument("--no-tunnel", action="store_true", help="Alias for --tunnel none")
    ap.add_argument("--tunnel-token", default=None, help="Cloudflare Named Tunnel token (or CLOUDFLARE_TUNNEL_TOKEN)")
    ap.add_argument("--public-url", default=None, help="Public HTTPS URL for Named Tunnel (or PUBLIC_GPU_URL)")
    ap.add_argument("--log-file", default=None, help="Path for worker stdout/stderr log (default: <data-dir>/worker.log)")
    ap.add_argument("--no-wait-models", action="store_true")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--mode", choices=("tracking", "tracking_seg", "full"), default=None)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()

    tunnel_mode = "none" if args.no_tunnel else args.tunnel
    tunnel_token = (
        args.tunnel_token if args.tunnel_token is not None else os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
    ).strip()
    named_public_url = (
        args.public_url if args.public_url is not None else os.environ.get("PUBLIC_GPU_URL", "")
    ).strip()

    if tunnel_mode == "named" and (not tunnel_token or not named_public_url):
        print(
            "ERROR: --tunnel named requires both --tunnel-token (or CLOUDFLARE_TUNNEL_TOKEN) "
            "and --public-url (or PUBLIC_GPU_URL).",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    token = args.token if args.token is not None else os.environ.get("GPU_TOKEN", "")
    if not token:
        token = secrets.token_urlsafe(24)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (HERE / "gpu_data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file).resolve() if args.log_file else (data_dir / "worker.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    stop_ev = threading.Event()
    atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])

    if port_in_use(args.port):
        new_port = free_port()
        print(f"Port {args.port} is already in use -> switching to port {new_port}", flush=True)
        args.port = new_port

    cmd = [sys.executable, "-u", "-m", "worker.app", "--host", args.host, "--port", str(args.port)]
    if args.demo:
        cmd.append("--demo")
    if args.no_preload:
        cmd.append("--no-preload")
    if args.local:
        cmd.append("--local")
    if args.mode:
        cmd.extend(["--mode", args.mode])
    cmd.extend(["--data-dir", str(data_dir)])

    env = dict(os.environ)
    env["GPU_TOKEN"] = token
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "1")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    log_fh = open(log_file, "a", encoding="utf-8")
    srv = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    procs.append(srv)

    health: dict[str, Any] = {}
    for _ in range(120):
        if srv.poll() is not None:
            print(f"GPU worker exited early (code {srv.returncode}). Check log: {log_file}", flush=True)
            sys.exit(1)
        try:
            health = fetch_local_health(args.port, token, timeout=1.0)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    else:
        print(f"Timed out waiting for GPU worker /internal/health. Check log: {log_file}", flush=True)
        sys.exit(1)

    local_url = f"http://127.0.0.1:{args.port}"
    print(f"Internal GPU worker listening at {local_url}", flush=True)

    wait_models_flag = (not args.no_wait_models) and (not args.no_preload)
    if wait_models_flag and not args.demo:
        m = wait_models(args.port, token)
        try:
            health = fetch_local_health(args.port, token, timeout=2.0)
        except Exception:  # noqa: BLE001
            health["models"] = m

    public_url: Optional[str] = None
    tun: Optional[subprocess.Popen[str]] = None
    tunnel_requested = tunnel_mode != "none"
    bin_dir = HERE / "work" / "bin"

    if tunnel_mode == "named":
        print(f"Starting Cloudflare Named Tunnel ({named_public_url})...", flush=True)
        try:
            tun = start_named_tunnel(tunnel_token, bin_dir)
            if tun is not None:
                procs.append(tun)
                _drain_output(tun)
                public_url = named_public_url.rstrip("/")
        except Exception as exc:  # noqa: BLE001
            print(f"  named tunnel error: {exc}", flush=True)
            tun = None
    elif tunnel_mode == "quick":
        for name, starter in (
            ("cloudflared", lambda: start_cloudflared(args.port, bin_dir)),
            ("localhost.run (ssh)", lambda: start_localhost_run(args.port)),
        ):
            print(f"Starting tunnel via {name} ...", flush=True)
            try:
                tun = starter()
                if tun is not None:
                    procs.append(tun)
            except Exception as exc:  # noqa: BLE001
                print(f"  tunnel start error: {exc}", flush=True)
                tun = None
            if tun is None:
                continue
            public_url = wait_url(tun)
            if public_url:
                break
            print(f"  {name} did not return a public URL, trying fallback...", flush=True)
            tun.terminate()
            tun = None

    external_ok: Optional[bool] = None
    external_auth_ok: Optional[bool] = None
    if tunnel_requested and public_url:
        if ".trycloudflare.com" in public_url:
            print(f"Waiting for tunnel DNS propagation ({public_url}) via 1.1.1.1 DoH...", flush=True)
            wait_tunnel_dns(public_url, timeout_s=20.0)
        print(f"Probing external health at {public_url}/internal/health ...", flush=True)
        external_ok, external_auth_ok, probe_msg = probe_external_health(public_url, token, timeout_s=45.0)
        print(f"  external probe: {probe_msg}", flush=True)

    ready, reasons = evaluate_readiness(
        health,
        demo=args.demo,
        wait_models_flag=wait_models_flag,
        tunnel_requested=tunnel_requested,
        public_url=public_url,
        external_ok=external_ok,
        external_auth_ok=external_auth_ok,
    )

    if tun is not None and tun.poll() is None:
        threading.Thread(target=_monitor_tunnel, args=(tun, srv, stop_ev), daemon=True).start()

    print_banner(
        local_url=local_url,
        public_url=public_url,
        vast_url=detect_vast_direct_url(args.port),
        token=token,
        demo=args.demo,
        tunnel_mode=tunnel_mode,
        ready=ready,
        reasons=reasons,
        log_file=log_file,
    )

    try:
        srv.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        for p in (tun, srv):
            if p is not None and p.poll() is None:
                p.terminate()
        try:
            log_fh.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
