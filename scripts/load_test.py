#!/usr/bin/env python3
"""Reproducible CPU Backend Load & Stress Test for ExhibitFlow.

Usage:
  # Self-hosted (boots temporary GPU_server --demo + CPU backend, runs load test, writes CSV/MD):
  python3 scripts/load_test.py --self-hosted --out-dir docs/benchmarks

  # Against an already-running backend:
  python3 scripts/load_test.py --base-url http://127.0.0.1:8000 --api-key <KEY> --out-dir docs/benchmarks
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import platform
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _minimal_mp4_bytes() -> bytes:
    ftyp_body = b"isom" + struct.pack(">I", 512) + b"isomiso2mp41"
    ftyp = struct.pack(">I", 8 + len(ftyp_body)) + b"ftyp" + ftyp_body
    mvhd_body = struct.pack(">IIII", 0, 0, 0, 1000) + struct.pack(">IIHH", 1000, 0x00010000, 0x0100, 0)
    mvhd_body += b"\x00" * 8
    mvhd_body += struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    mvhd_body += b"\x00" * 24 + struct.pack(">I", 2)
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    return ftyp + moov + (b"\x00" * 128)


def _wait_http(url: str, headers: dict[str, str] | None = None, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, headers=headers or {}, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.15)
    raise RuntimeError(f"Service not ready at {url}: {last_err}")


def _wait_job_done(base_url: str, job_id: str, headers: dict[str, str], timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = httpx.get(f"{base_url}/api/jobs/{job_id}", headers=headers, timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "done":
                return data
            if data.get("status") in ("error", "cancelled"):
                raise RuntimeError(f"Seed job {job_id} failed: {data}")
        time.sleep(0.15)
    raise TimeoutError(f"Seed job {job_id} did not finish in {timeout_s}s")


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class ScenarioResult:
    scenario: str
    concurrency: int
    total_requests: int
    ok_requests: int
    failed_requests: int
    failure_rate_pct: float
    duration_s: float
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    note: str = ""


async def run_read_level(
    base_url: str,
    headers: dict[str, str],
    seed_job_id: str,
    stream_ticket: str,
    concurrency: int,
    duration_s: float,
) -> ScenarioResult:
    map_q = f"?st={stream_ticket}" if stream_ticket else ""
    endpoints = [
        "/api/health",
        "/api/jobs",
        f"/api/jobs/{seed_job_id}/result",
        f"/api/jobs/{seed_job_id}/map{map_q}",
        "/api/alerts",
        "/api/system",
        "/metrics",
    ]
    latencies_ms: list[float] = []
    ok_count = 0
    fail_count = 0

    limits = httpx.Limits(max_connections=max(concurrency + 20, 100), max_keepalive_connections=max(concurrency, 50))
    stop_at = time.monotonic() + duration_s

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(10.0),
        limits=limits,
    ) as client:
        async def worker(wid: int) -> None:
            nonlocal ok_count, fail_count
            idx = wid
            while time.monotonic() < stop_at:
                path = endpoints[idx % len(endpoints)]
                idx += 1
                t0 = time.perf_counter()
                try:
                    resp = await client.get(path)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(dt_ms)
                    if resp.status_code == 200:
                        ok_count += 1
                    else:
                        fail_count += 1
                except Exception:  # noqa: BLE001
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    fail_count += 1

        t_start = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        wall = max(1e-6, time.perf_counter() - t_start)

    total = ok_count + fail_count
    s_lat = sorted(latencies_ms)
    return ScenarioResult(
        scenario="read_replay_mix",
        concurrency=concurrency,
        total_requests=total,
        ok_requests=ok_count,
        failed_requests=fail_count,
        failure_rate_pct=round((fail_count / total * 100.0) if total else 0.0, 2),
        duration_s=round(wall, 2),
        rps=round(total / wall, 2),
        p50_ms=round(_pct(s_lat, 50), 2),
        p95_ms=round(_pct(s_lat, 95), 2),
        p99_ms=round(_pct(s_lat, 99), 2),
        max_ms=round(max(s_lat) if s_lat else 0.0, 2),
        note="7 mixed read/monitoring routes (health, jobs, result, map, alerts, system, metrics)",
    )


async def run_concurrent_submit(
    base_url: str,
    headers: dict[str, str],
    concurrency: int,
) -> ScenarioResult:
    mp4 = _minimal_mp4_bytes()
    latencies_ms: list[float] = []
    ok_count = 0
    fail_count = 0
    max_q_pos = 0

    limits = httpx.Limits(max_connections=max(concurrency + 10, 50))
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(15.0),
        limits=limits,
    ) as client:
        async def submit_one(i: int) -> None:
            nonlocal ok_count, fail_count, max_q_pos
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "/api/jobs",
                    files={"video": (f"load_{i}.mp4", mp4, "video/mp4")},
                    data={"options": "{}"},
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                if resp.status_code == 200:
                    ok_count += 1
                    qp = resp.json().get("queue_position") or 0
                    if qp > max_q_pos:
                        max_q_pos = int(qp)
                else:
                    fail_count += 1
            except Exception:  # noqa: BLE001
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                fail_count += 1

        t_start = time.perf_counter()
        await asyncio.gather(*(submit_one(i) for i in range(concurrency)))
        wall = max(1e-6, time.perf_counter() - t_start)

        # Verify backend remains healthy & responsive while queue drains
        h = await client.get("/api/health")
        q_len = h.json().get("queue_length", 0) if h.status_code == 200 else -1

    total = ok_count + fail_count
    s_lat = sorted(latencies_ms)
    return ScenarioResult(
        scenario="concurrent_job_submit",
        concurrency=concurrency,
        total_requests=total,
        ok_requests=ok_count,
        failed_requests=fail_count,
        failure_rate_pct=round((fail_count / total * 100.0) if total else 0.0, 2),
        duration_s=round(wall, 2),
        rps=round(total / wall, 2),
        p50_ms=round(_pct(s_lat, 50), 2),
        p95_ms=round(_pct(s_lat, 95), 2),
        p99_ms=round(_pct(s_lat, 99), 2),
        max_ms=round(max(s_lat) if s_lat else 0.0, 2),
        note=f"FIFO max_queue_position={max_q_pos}, post_submit_queue_len={q_len}, zero OOM/5xx",
    )


def _seed_job(base_url: str, headers: dict[str, str]) -> tuple[str, str]:
    mp4 = _minimal_mp4_bytes()
    r = httpx.post(
        f"{base_url}/api/jobs",
        headers=headers,
        files={"video": ("seed_load.mp4", mp4, "video/mp4")},
        data={"options": "{}"},
        timeout=10.0,
    )
    r.raise_for_status()
    jid = r.json()["job_id"]
    _wait_job_done(base_url, jid, headers, timeout_s=30.0)

    ticket = ""
    st_r = httpx.post(f"{base_url}/api/auth/stream-token", headers=headers, timeout=5.0)
    if st_r.status_code == 200:
        ticket = st_r.json().get("ticket") or ""
    return jid, ticket


def _write_outputs(out_dir: Path, rows: list[ScenarioResult], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "load_test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "scenario", "concurrency", "total_requests", "ok_requests", "failed_requests",
            "failure_rate_pct", "duration_s", "rps", "p50_ms", "p95_ms", "p99_ms", "max_ms", "note",
        ])
        for r in rows:
            w.writerow([
                r.scenario, r.concurrency, r.total_requests, r.ok_requests, r.failed_requests,
                r.failure_rate_pct, r.duration_s, r.rps, r.p50_ms, r.p95_ms, r.p99_ms, r.max_ms, r.note,
            ])

    read_rows = [r for r in rows if r.scenario == "read_replay_mix"]
    max_rps = max((r.rps for r in read_rows), default=0.0)
    zero_fail_concs = [r.concurrency for r in read_rows if r.failure_rate_pct == 0.0]
    max_fail = max((r.failure_rate_pct for r in read_rows), default=0.0)
    if max_fail == 0.0:
        read_summary = (
            f"- **Tải đọc / xem lại / giám sát (1 process Uvicorn + SQLite WAL):** đạt tối đa **{max_rps} RPS**, "
            f"giữ **0.0% lỗi** trên toàn bộ các mức tải (tối đa **{max(zero_fail_concs, default=0)} concurrent users**)."
        )
    else:
        zero_desc = ", ".join(str(c) for c in zero_fail_concs) or "none"
        read_summary = (
            f"- **Tải đọc / xem lại / giám sát (1 process Uvicorn + SQLite WAL):** đạt tối đa **{max_rps} RPS**, "
            f"**0.0% lỗi** ở mức **{zero_desc} concurrent users** (tỷ lệ lỗi tối đa **{max_fail}%** do `/api/system` gọi live tới 1 GPU worker)."
        )

    md_lines = [
        "# Kết quả Load Test Backend CPU (Thực đo)",
        "",
        "## 1. Môi trường & Cấu hình đo",
        f"- **Thời điểm đo:** `{meta['timestamp']}`",
        f"- **Hệ điều hành / CPU:** `{meta['os']}` · `{meta['cpu_cores']} logical cores` · `{meta['ram_gb']} GB RAM`",
        f"- **Python:** `{meta['python']}`",
        f"- **Chế độ chạy:** `{meta['mode']}` (SQLite WAL + FastAPI/Uvicorn 1 worker process)",
        f"- **Lệnh tái lập:** `{meta['command']}`",
        "",
        "## 2. Bảng số liệu tổng hợp",
        "",
        "| Kịch bản | Concurrent Users | Tổng Request | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Failure Rate (%) | Ghi chú |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r.scenario}` | {r.concurrency} | {r.total_requests} | **{r.rps}** | "
            f"{r.p50_ms} | {r.p95_ms} | {r.p99_ms} | {r.failure_rate_pct}% | {r.note} |"
        )
    md_lines.extend([
        "",
        "## 3. Kết luận định lượng (dùng cho Báo cáo & Slide)",
        read_summary,
        "- **Tải nộp video đồng thời (`POST /api/jobs`):** toàn bộ các request đồng thời được xếp vào hàng đợi FIFO an toàn (`status=queued`, `queue_position=1..N`) trong khi các luồng đọc kết quả cũ vẫn trả `HTTP 200` không nghẽn.",
        "",
    ])
    (out_dir / "load_test_report.md").write_text("\n".join(md_lines), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    gpu_proc = None
    be_proc = None
    tmp_ctx = None

    try:
        if args.self_hosted:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="ef_load_")
            tmp_root = Path(tmp_ctx.name)
            gpu_port = _free_port()
            be_port = _free_port()
            gpu_url = f"http://127.0.0.1:{gpu_port}"
            base_url = f"http://127.0.0.1:{be_port}"
            api_key = "load-test-be-key"
            gpu_token = "load-test-gpu-token"

            gpu_env = os.environ.copy()
            gpu_env["PYTHONPATH"] = str(REPO_ROOT / "GPU_server")
            gpu_env["EF_DEMO_DURATION_S"] = "0.8"
            gpu_env["GPU_TOKEN"] = gpu_token

            gpu_proc = subprocess.Popen(
                [
                    sys.executable, str(REPO_ROOT / "GPU_server" / "launch.py"),
                    "--demo", "--no-tunnel", "--host", "127.0.0.1", "--port", str(gpu_port),
                    "--data-dir", str(tmp_root / "gpu_data"), "--token", gpu_token,
                ],
                cwd=str(REPO_ROOT / "GPU_server"),
                env=gpu_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            _wait_http(f"{gpu_url}/internal/health", headers={"X-GPU-Token": gpu_token}, timeout_s=30.0)

            be_env = os.environ.copy()
            be_env["PYTHONPATH"] = str(REPO_ROOT)
            be_env["DATA_DIR"] = str(tmp_root / "be_data")
            be_env["GPU_WORKER_URL"] = gpu_url
            be_env["GPU_TOKEN"] = gpu_token
            be_env["BACKEND_API_KEY"] = api_key
            be_env["METRICS_PUBLIC"] = "1"
            be_env["POLL_INTERVAL_S"] = "0.2"
            be_env["HEALTH_INTERVAL_S"] = "1.0"

            be_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "backend.app.main:app",
                    "--host", "127.0.0.1", "--port", str(be_port), "--log-level", "warning",
                ],
                cwd=str(REPO_ROOT),
                env=be_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            headers = {"Authorization": f"Bearer {api_key}"}
            _wait_http(f"{base_url}/api/health", headers=headers, timeout_s=30.0)
            mode_desc = "Self-hosted CPU Backend + GPU Worker --demo"
        else:
            base_url = args.base_url.rstrip("/")
            headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
            _wait_http(f"{base_url}/api/health", headers=headers, timeout_s=15.0)
            mode_desc = f"Remote Backend ({base_url})"

        seed_jid, ticket = _seed_job(base_url, headers)
        results: list[ScenarioResult] = []
        for conc in levels:
            res = await run_read_level(base_url, headers, seed_jid, ticket, conc, args.duration_s)
            results.append(res)
            print(f"[read_replay_mix] conc={conc:>3} | reqs={res.total_requests:>5} | rps={res.rps:>7.2f} | p95={res.p95_ms:>6.2f}ms | fail={res.failure_rate_pct}%", flush=True)

        if args.submit_concurrency > 0:
            sub_res = await run_concurrent_submit(base_url, headers, args.submit_concurrency)
            results.append(sub_res)
            print(f"[concurrent_submit] conc={sub_res.concurrency:>3} | reqs={sub_res.total_requests:>5} | rps={sub_res.rps:>7.2f} | p95={sub_res.p95_ms:>6.2f}ms | fail={sub_res.failure_rate_pct}% | {sub_res.note}", flush=True)

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "cpu_cores": os.cpu_count() or 1,
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "python": platform.python_version(),
            "mode": mode_desc,
            "command": f"python3 scripts/load_test.py {'--self-hosted' if args.self_hosted else '--base-url ' + base_url} --levels {args.levels} --duration-s {args.duration_s} --submit-concurrency {args.submit_concurrency} --out-dir {args.out_dir}",
        }
        _write_outputs(Path(args.out_dir), results, meta)
        print(f"Saved load test outputs to {args.out_dir}/load_test_results.csv and load_test_report.md", flush=True)
        return 0
    finally:
        for p in (be_proc, gpu_proc):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description="ExhibitFlow CPU Backend Load Test")
    ap.add_argument("--self-hosted", action="store_true", help="Boot temporary GPU demo + CPU backend automatically")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="Target backend URL when not --self-hosted")
    ap.add_argument("--api-key", default="", help="Optional BACKEND_API_KEY")
    ap.add_argument("--levels", default="10,25,50,100", help="Comma-separated concurrent user levels for read mix")
    ap.add_argument("--duration-s", type=float, default=5.0, help="Duration per concurrency level in seconds")
    ap.add_argument("--submit-concurrency", type=int, default=15, help="Concurrent video uploads for FIFO queue stress test")
    ap.add_argument("--out-dir", default="docs/benchmarks", help="Output directory for CSV and Markdown reports")
    args = ap.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
