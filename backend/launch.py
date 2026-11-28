"""Production-ready local launcher for ExhibitFlow CPU Backend + Cloudflare Tunnel.

Usage:
    python backend/launch.py --port 8000 --tunnel quick
    python backend/launch.py --tunnel named --tunnel-token <TOKEN> --public-url https://api.example.com
    python backend/launch.py --no-tunnel
"""

from __future__ import annotations

import argparse
import io
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
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def resolve_tunnel_mode(tunnel: str, no_tunnel: bool) -> str:
    if no_tunnel:
        return "none"
    mode = (tunnel or "quick").strip().lower()
    if mode not in ("quick", "named", "none"):
        raise ValueError(f"Unsupported tunnel mode: {mode!r}")
    return mode


def resolve_api_key(
    cli_key: str | None,
    no_auth: bool,
    env: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Return (api_key, was_auto_generated). Empty string when --no-auth is set."""
    if no_auth:
        return "", False
    if cli_key is not None and cli_key.strip():
        return cli_key.strip(), False
    source_env = env if env is not None else os.environ
    env_key = (source_env.get("BACKEND_API_KEY") or "").strip()
    if env_key:
        return env_key, False
    return secrets.token_urlsafe(24), True


def build_tunnel_command(
    mode: str,
    port: int,
    exe: str,
    tunnel_token: str = "",
    public_url: str = "",
) -> list[str]:
    """Construct the cloudflared CLI command for quick or named tunnels."""
    if mode == "quick":
        return [
            exe,
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{port}",
        ]
    if mode == "named":
        tok = (tunnel_token or "").strip()
        pub = (public_url or "").strip().rstrip("/")
        if not tok:
            raise ValueError(
                "Named tunnel requires --tunnel-token or CLOUDFLARE_TUNNEL_TOKEN."
            )
        if not pub.startswith(("https://", "http://")):
            raise ValueError(
                "Named tunnel requires --public-url or PUBLIC_BACKEND_URL (https://...)."
            )
        return [
            exe,
            "tunnel",
            "--no-autoupdate",
            "run",
            "--token",
            tok,
        ]
    raise ValueError(f"Cannot build tunnel command for mode={mode!r}")


def _download_bytes(urls: list[str], tries: int = 3) -> bytes | None:
    last_err: Exception | None = None
    for url in urls:
        for attempt in range(tries):
            try:
                print(
                    f"Downloading cloudflared ({attempt + 1}/{tries}): {url}",
                    flush=True,
                )
                req = urllib.request.Request(
                    url, headers={"User-Agent": "ExhibitFlow-Backend-Launcher"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"  download error: {exc}", flush=True)
                time.sleep(min(4, 1 + attempt))
    print(f"Failed to download cloudflared: {last_err}", flush=True)
    return None


def _extract_tgz_binary(tgz_bytes: bytes, dst_exe: Path) -> bool:
    with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if Path(member.name).name == "cloudflared":
                src = tf.extractfile(member)
                if src is None:
                    continue
                dst_exe.parent.mkdir(parents=True, exist_ok=True)
                with dst_exe.open("wb") as out:
                    shutil.copyfileobj(src, out)
                dst_exe.chmod(dst_exe.stat().st_mode | stat.S_IEXEC)
                return True
    return False


def get_cloudflared(
    dst_dir: Path,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> str | None:
    """Locate cloudflared on PATH or download it for Linux, macOS (.tgz), or Windows."""
    found = shutil.which("cloudflared")
    if found:
        return found

    sys_name = (system_name or platform.system()).lower()
    mach = (machine_name or platform.machine()).lower()
    arch = "arm64" if mach in ("aarch64", "arm64") else "amd64"
    is_win = sys_name == "windows" or os.name == "nt"
    exe = dst_dir / ("cloudflared.exe" if is_win else "cloudflared")
    if exe.is_file() and exe.stat().st_size > 1_000_000:
        return str(exe)

    base = "https://github.com/cloudflare/cloudflared/releases"
    if is_win:
        artifact = f"cloudflared-windows-{arch}.exe"
        is_tgz = False
    elif sys_name == "darwin":
        artifact = f"cloudflared-darwin-{arch}.tgz"
        is_tgz = True
    else:
        artifact = f"cloudflared-linux-{arch}"
        is_tgz = False

    urls = [
        f"{base}/latest/download/{artifact}",
        f"{base}/download/2024.12.2/{artifact}",
    ]
    payload = _download_bytes(urls)
    if not payload:
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    if is_tgz:
        if not _extract_tgz_binary(payload, exe):
            print("Failed to locate cloudflared binary inside .tgz archive.", flush=True)
            return None
        return str(exe)

    exe.write_bytes(payload)
    if not is_win:
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def wait_quick_tunnel_url(
    proc: subprocess.Popen[str], timeout: float = 45.0
) -> str | None:
    found = threading.Event()
    box: dict[str, str] = {}

    def _pump() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m and "url" not in box:
                box["url"] = m.group(0)
                found.set()

    threading.Thread(target=_pump, daemon=True).start()
    found.wait(timeout)
    return box.get("url")


def _drain_stream(proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for _ in proc.stdout:
        pass


def wait_dns_propagation(
    hostname: str,
    timeout_s: float = 20.0,
    interval_s: float = 1.0,
    fallback_sleep_s: float = 3.5,
) -> bool:
    """Wait for Cloudflare DoH (1.1.1.1) to resolve hostname before querying OS DNS.

    Prevents poisoning the local OS negative DNS cache with NXDOMAIN while a
    freshly created *.trycloudflare.com subdomain is still propagating.
    """
    if not hostname:
        return False
    doh_url = f"https://1.1.1.1/dns-query?name={urllib.parse.quote(hostname)}&type=A"
    deadline = time.monotonic() + timeout_s
    doh_succeeded_once = False
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                doh_url,
                headers={"Accept": "application/dns-json", "User-Agent": "ExhibitFlow"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            doh_succeeded_once = True
            if data.get("Status") == 0 and data.get("Answer"):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval_s)
    if not doh_succeeded_once and fallback_sleep_s > 0:
        time.sleep(fallback_sleep_s)
    return False


def probe_health(
    url: str, api_key: str = "", timeout: float = 4.0
) -> tuple[int, dict[str, Any] | None, str | None]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{url.rstrip('/')}/api/health", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data, None
    except urllib.error.HTTPError as exc:
        return exc.code, None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return 0, None, str(exc)


def verify_health_contract(
    base_url: str, api_key: str, timeout: float = 4.0
) -> tuple[bool, str]:
    """Verify /api/health returns ok:true and enforces auth when api_key is set."""
    code, data, err = probe_health(base_url, api_key="", timeout=timeout)
    if code != 200 or not isinstance(data, dict) or data.get("ok") is not True:
        return False, f"HTTP {code} ({err or 'invalid /api/health response'})"

    if api_key:
        if not data.get("auth_required") or data.get("authenticated") is not False:
            return False, "Auth enabled but unauthenticated /api/health did not report auth_required=true, authenticated=false"
        if "gpu" in data:
            return False, "Unauthenticated /api/health leaked gpu metadata"

        acode, adata, aerr = probe_health(base_url, api_key=api_key, timeout=timeout)
        if acode != 200 or not isinstance(adata, dict) or adata.get("authenticated") is not True:
            return False, f"Authenticated /api/health failed (HTTP {acode}: {aerr})"
    return True, "ok"


def wait_for_readiness(
    base_url: str,
    api_key: str,
    timeout_s: float = 30.0,
    interval_s: float = 1.0,
    proc: subprocess.Popen[Any] | None = None,
) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_s
    last_reason = "Timed out"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, f"Process exited prematurely with code {proc.returncode}"
        ok, reason = verify_health_contract(base_url, api_key, timeout=3.0)
        if ok:
            return True, "ok"
        last_reason = reason
        time.sleep(interval_s)
    return False, last_reason


def print_banner(
    *,
    local_url: str,
    public_url: str | None,
    api_key: str,
    auto_key: bool,
    data_dir: Path,
    tunnel_mode: str,
    log_file: Path,
    ready: bool,
    fail_reason: str = "",
) -> None:
    print(flush=True)
    print("=" * 72, flush=True)
    print("  ExhibitFlow CPU Backend", flush=True)
    print(f"  Local URL          : {local_url}", flush=True)
    if public_url:
        print(f"  Public HTTPS URL   : {public_url}", flush=True)
    print(f"  Tunnel Mode        : {tunnel_mode}", flush=True)
    if api_key:
        tag = " (auto-generated)" if auto_key else " (configured)"
        print(f"  Backend Access Key : {api_key}{tag}", flush=True)
        print(
            "  Note               : Nhập URL + Backend Access Key vào Settings trên Web (tách biệt với X-GPU-Token).",
            flush=True,
        )
    else:
        print("  Backend Access Key : DISABLED (--no-auth)", flush=True)
    print(f"  DATA_DIR           : {data_dir}", flush=True)
    print(f"  Log File           : {log_file}", flush=True)
    if ready:
        print("  STATUS: READY (SẴN SÀNG)", flush=True)
    else:
        print(f"  STATUS: NOT READY ({fail_reason})", flush=True)
    print("=" * 72, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Launch ExhibitFlow CPU Backend with optional Cloudflare Tunnel"
    )
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or "8000"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR") or "./backend_data")
    ap.add_argument("--api-key", default=None, help="Backend Access Key (BACKEND_API_KEY)")
    ap.add_argument("--no-auth", action="store_true", help="Disable Backend Access Key auth")
    ap.add_argument(
        "--tunnel",
        choices=("quick", "named", "none"),
        default="quick",
        help="Cloudflare Tunnel mode (default: quick)",
    )
    ap.add_argument("--no-tunnel", action="store_true", help="Alias for --tunnel none")
    ap.add_argument(
        "--tunnel-token",
        default=os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", ""),
        help="Cloudflare Zero Trust tunnel token (for --tunnel named)",
    )
    ap.add_argument(
        "--public-url",
        default=os.environ.get("PUBLIC_BACKEND_URL", ""),
        help="Public HTTPS URL of named tunnel (for --tunnel named)",
    )
    ap.add_argument(
        "--cors-origins",
        default=os.environ.get("CORS_ORIGINS", "*"),
        help="Comma-separated CORS origins (default: *)",
    )
    ap.add_argument("--log-file", default=None, help="Uvicorn log file path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tunnel_mode = resolve_tunnel_mode(args.tunnel, args.no_tunnel)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr, flush=True)
        return 2

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else (data_dir / "backend.log")
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    api_key, auto_key = resolve_api_key(args.api_key, args.no_auth)

    port = args.port
    if port_in_use(port, "127.0.0.1"):
        new_port = free_port("127.0.0.1")
        print(
            f"Port {port} is already in use -> switching to port {new_port}",
            flush=True,
        )
        port = new_port

    if tunnel_mode == "named":
        try:
            # Validate named tunnel parameters upfront before starting processes
            build_tunnel_command(
                "named",
                port,
                "cloudflared",
                tunnel_token=args.tunnel_token,
                public_url=args.public_url,
            )
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr, flush=True)
            return 2

    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env["BACKEND_API_KEY"] = api_key
    env["CORS_ORIGINS"] = args.cors_origins
    env["PORT"] = str(port)
    if not env.get("WEB_DIR") and (REPO_ROOT / "web").is_dir():
        env["WEB_DIR"] = str(REPO_ROOT / "web")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{BACKEND_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(BACKEND_DIR)
    )

    log_fh = log_file.open("a", encoding="utf-8")
    srv = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            args.host,
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    tun: subprocess.Popen[str] | None = None
    shutting_down = threading.Event()

    def _cleanup() -> None:
        shutting_down.set()
        for proc in (tun, srv):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        log_fh.close()

    local_url = f"http://127.0.0.1:{port}"
    try:
        local_ok, local_reason = wait_for_readiness(
            local_url, api_key, timeout_s=45.0, interval_s=0.3, proc=srv
        )
        if not local_ok:
            print_banner(
                local_url=local_url,
                public_url=None,
                api_key=api_key,
                auto_key=auto_key,
                data_dir=data_dir,
                tunnel_mode=tunnel_mode,
                log_file=log_file,
                ready=False,
                fail_reason=f"Local backend check failed: {local_reason}",
            )
            return 1

        public_url: str | None = None
        if tunnel_mode != "none":
            exe = get_cloudflared(data_dir / "bin")
            if not exe:
                print_banner(
                    local_url=local_url,
                    public_url=None,
                    api_key=api_key,
                    auto_key=auto_key,
                    data_dir=data_dir,
                    tunnel_mode=tunnel_mode,
                    log_file=log_file,
                    ready=False,
                    fail_reason="Could not find or download cloudflared",
                )
                return 1

            tcmd = build_tunnel_command(
                tunnel_mode,
                port,
                exe,
                tunnel_token=args.tunnel_token,
                public_url=args.public_url,
            )
            tun = subprocess.Popen(
                tcmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if tunnel_mode == "quick":
                public_url = wait_quick_tunnel_url(tun, timeout=45.0)
                if not public_url:
                    print_banner(
                        local_url=local_url,
                        public_url=None,
                        api_key=api_key,
                        auto_key=auto_key,
                        data_dir=data_dir,
                        tunnel_mode=tunnel_mode,
                        log_file=log_file,
                        ready=False,
                        fail_reason="Cloudflare Quick Tunnel did not return a public HTTPS URL",
                    )
                    return 1
                host = urllib.parse.urlparse(public_url).hostname or ""
                if host:
                    wait_dns_propagation(host, timeout_s=20.0)
            else:
                public_url = args.public_url.strip().rstrip("/")
                threading.Thread(target=_drain_stream, args=(tun,), daemon=True).start()

            ext_ok, ext_reason = wait_for_readiness(
                public_url, api_key, timeout_s=45.0, interval_s=1.5, proc=tun
            )
            if not ext_ok:
                print_banner(
                    local_url=local_url,
                    public_url=public_url,
                    api_key=api_key,
                    auto_key=auto_key,
                    data_dir=data_dir,
                    tunnel_mode=tunnel_mode,
                    log_file=log_file,
                    ready=False,
                    fail_reason=f"External HTTPS probe failed: {ext_reason}",
                )
                return 1

            def _watch_tunnel() -> None:
                assert tun is not None
                rc = tun.wait()
                if not shutting_down.is_set() and srv.poll() is None:
                    print(
                        f"\nCẢNH BÁO: Cloudflare Tunnel đã ngắt kết nối! (exit code {rc})",
                        flush=True,
                    )

            threading.Thread(target=_watch_tunnel, daemon=True).start()

        print_banner(
            local_url=local_url,
            public_url=public_url,
            api_key=api_key,
            auto_key=auto_key,
            data_dir=data_dir,
            tunnel_mode=tunnel_mode,
            log_file=log_file,
            ready=True,
        )

        try:
            return srv.wait()
        except KeyboardInterrupt:
            return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
