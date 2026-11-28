"""Zero-dependency Prometheus metrics registry and exposition renderer."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Any

import psutil

DEFAULT_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _esc_label(val: Any) -> str:
    s = str(val if val is not None else "")
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_labels(labels: dict[str, Any]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_esc_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


class MetricsRegistry:
    """Thread-safe HTTP and ExhibitFlow runtime metrics collector."""

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_DURATION_BUCKETS) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(buckets))
        self._req_total: dict[tuple[str, str, str], int] = defaultdict(int)
        self._dur_buckets: dict[tuple[str, str], list[int]] = {}
        self._dur_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._dur_count: dict[tuple[str, str], int] = defaultdict(int)
        self._in_flight = 0

    def inc_in_flight(self, delta: int) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight + delta)

    def observe_request(
        self, method: str, path: str, status: int, duration_s: float
    ) -> None:
        m = str(method or "GET").upper()
        p = str(path or "/")
        st = str(int(status))
        dur = max(0.0, float(duration_s))
        with self._lock:
            self._req_total[(m, p, st)] += 1
            key = (m, p)
            b_counts = self._dur_buckets.get(key)
            if b_counts is None:
                b_counts = [0] * (len(self._buckets) + 1)
                self._dur_buckets[key] = b_counts
            for idx, bound in enumerate(self._buckets):
                if dur <= bound:
                    b_counts[idx] += 1
            b_counts[-1] += 1
            self._dur_sum[key] += dur
            self._dur_count[key] += 1

    def render_prometheus(
        self,
        *,
        store: Any,
        broker: Any,
        gpu_ready_fn: Any,
    ) -> str:
        lines: list[str] = []

        # 1. HTTP counters & histogram
        with self._lock:
            in_flight = self._in_flight
            req_items = sorted(self._req_total.items())
            dur_keys = sorted(self._dur_count.keys())
            dur_buckets = {k: list(self._dur_buckets[k]) for k in dur_keys}
            dur_sum = {k: self._dur_sum[k] for k in dur_keys}
            dur_count = {k: self._dur_count[k] for k in dur_keys}

        lines.extend(
            [
                "# HELP exhibitflow_http_requests_in_flight Current HTTP requests being processed.",
                "# TYPE exhibitflow_http_requests_in_flight gauge",
                f"exhibitflow_http_requests_in_flight {in_flight}",
                "# HELP exhibitflow_http_requests_total Total HTTP requests handled by ExhibitFlow backend.",
                "# TYPE exhibitflow_http_requests_total counter",
            ]
        )
        for (method, path, status), count in req_items:
            lbl = _fmt_labels({"method": method, "path": path, "status": status})
            lines.append(f"exhibitflow_http_requests_total{lbl} {count}")

        lines.extend(
            [
                "# HELP exhibitflow_http_request_duration_seconds HTTP request latency in seconds.",
                "# TYPE exhibitflow_http_request_duration_seconds histogram",
            ]
        )
        for key in dur_keys:
            method, path = key
            counts = dur_buckets[key]
            for idx, bound in enumerate(self._buckets):
                lbl = _fmt_labels({"method": method, "path": path, "le": f"{bound:g}"})
                lines.append(f"exhibitflow_http_request_duration_seconds_bucket{lbl} {counts[idx]}")
            inf_lbl = _fmt_labels({"method": method, "path": path, "le": "+Inf"})
            lines.append(f"exhibitflow_http_request_duration_seconds_bucket{inf_lbl} {counts[-1]}")
            base_lbl = _fmt_labels({"method": method, "path": path})
            lines.append(f"exhibitflow_http_request_duration_seconds_sum{base_lbl} {dur_sum[key]:.6f}")
            lines.append(f"exhibitflow_http_request_duration_seconds_count{base_lbl} {dur_count[key]}")

        # 2. Backend CPU & RAM gauges
        cpu_pct = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        try:
            load_1 = float(os.getloadavg()[0])
        except (AttributeError, OSError):
            load_1 = round(cpu_pct / 100.0 * (os.cpu_count() or 1), 2)
        lines.extend(
            [
                "# HELP exhibitflow_backend_cpu_percent Backend host CPU utilization percentage.",
                "# TYPE exhibitflow_backend_cpu_percent gauge",
                f"exhibitflow_backend_cpu_percent {cpu_pct:.2f}",
                "# HELP exhibitflow_backend_cpu_load1 Backend host 1-minute load average.",
                "# TYPE exhibitflow_backend_cpu_load1 gauge",
                f"exhibitflow_backend_cpu_load1 {load_1:.2f}",
                "# HELP exhibitflow_backend_ram_used_bytes Backend host RAM used in bytes.",
                "# TYPE exhibitflow_backend_ram_used_bytes gauge",
                f"exhibitflow_backend_ram_used_bytes {int(vm.used)}",
                "# HELP exhibitflow_backend_ram_total_bytes Backend host total RAM in bytes.",
                "# TYPE exhibitflow_backend_ram_total_bytes gauge",
                f"exhibitflow_backend_ram_total_bytes {int(vm.total)}",
            ]
        )

        # 3. Queue, Jobs, Alerts, SSE subscribers
        q_len = int(store.queue_length())
        jobs, _ = store.list_jobs(limit=10000, offset=0)
        job_counts = {"queued": 0, "running": 0, "syncing": 0, "done": 0, "error": 0, "cancelled": 0}
        for j in jobs:
            st = str(j.get("status") or "").lower()
            if st in job_counts:
                job_counts[st] += 1
            else:
                job_counts[st] = job_counts.get(st, 0) + 1

        _, _, alert_counts = store.list_alerts(limit=1, offset=0)
        sse_subs = 0
        if hasattr(broker, "_topics"):
            sse_subs = sum(len(qset) for qset in broker._topics.values())

        lines.extend(
            [
                "# HELP exhibitflow_queue_depth Current number of jobs waiting in FIFO queue.",
                "# TYPE exhibitflow_queue_depth gauge",
                f"exhibitflow_queue_depth {q_len}",
                "# HELP exhibitflow_jobs_total Total jobs stored in backend grouped by status.",
                "# TYPE exhibitflow_jobs_total gauge",
            ]
        )
        for status_name, cnt in sorted(job_counts.items()):
            lines.append(f'exhibitflow_jobs_total{{status="{_esc_label(status_name)}"}} {cnt}')

        lines.extend(
            [
                "# HELP exhibitflow_alerts_total Total stored alerts grouped by decision.",
                "# TYPE exhibitflow_alerts_total gauge",
            ]
        )
        for dec in ("ANOMALY", "CANDIDATE", "NORMAL"):
            lines.append(
                f'exhibitflow_alerts_total{{decision="{dec}"}} {int(alert_counts.get(dec, 0))}'
            )

        lines.extend(
            [
                "# HELP exhibitflow_sse_subscribers Active SSE client subscriptions.",
                "# TYPE exhibitflow_sse_subscribers gauge",
                f"exhibitflow_sse_subscribers {int(sse_subs)}",
            ]
        )

        # 4. GPU status & cached GPU utilization/VRAM
        gpu_health = store.get_setting("gpu_health") or {}
        gpu_online = 1 if gpu_health.get("online") else 0
        gpu_ready = 1 if gpu_ready_fn(gpu_health) else 0
        gpu_busy = 1 if gpu_health.get("busy") else 0
        lines.extend(
            [
                "# HELP exhibitflow_gpu_online Whether the configured GPU worker is reachable (1/0).",
                "# TYPE exhibitflow_gpu_online gauge",
                f"exhibitflow_gpu_online {gpu_online}",
                "# HELP exhibitflow_gpu_ready Whether the configured GPU worker models are ready (1/0).",
                "# TYPE exhibitflow_gpu_ready gauge",
                f"exhibitflow_gpu_ready {gpu_ready}",
                "# HELP exhibitflow_gpu_busy Whether the configured GPU worker is currently busy (1/0).",
                "# TYPE exhibitflow_gpu_busy gauge",
                f"exhibitflow_gpu_busy {gpu_busy}",
            ]
        )

        gpu_sys = store.get_setting("gpu_system") or {}
        gpus = gpu_sys.get("gpus") if isinstance(gpu_sys, dict) else None
        if isinstance(gpus, list) and gpus:
            lines.extend(
                [
                    "# HELP exhibitflow_gpu_utilization_percent Reported GPU compute utilization percentage.",
                    "# TYPE exhibitflow_gpu_utilization_percent gauge",
                    "# HELP exhibitflow_gpu_vram_used_gb Reported GPU VRAM used in GiB.",
                    "# TYPE exhibitflow_gpu_vram_used_gb gauge",
                    "# HELP exhibitflow_gpu_vram_total_gb Reported GPU total VRAM in GiB.",
                    "# TYPE exhibitflow_gpu_vram_total_gb gauge",
                ]
            )
            for idx, g in enumerate(gpus):
                if not isinstance(g, dict):
                    continue
                g_name = str(g.get("name") or f"gpu{idx}")
                lbl = _fmt_labels({"gpu_index": str(idx), "gpu_name": g_name})
                util = float(g.get("util") or 0.0)
                mu = float(g.get("mem_used_gb") or 0.0)
                mt = float(g.get("mem_total_gb") or 0.0)
                lines.append(f"exhibitflow_gpu_utilization_percent{lbl} {util:.2f}")
                lines.append(f"exhibitflow_gpu_vram_used_gb{lbl} {mu:.3f}")
                lines.append(f"exhibitflow_gpu_vram_total_gb{lbl} {mt:.3f}")

        return "\n".join(lines) + "\n"


class PrometheusASGIMiddleware:
    """Pure ASGI middleware to track HTTP metrics without buffering SSE/MJPEG streams."""

    def __init__(self, app: Any, metrics: MetricsRegistry) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return

        self.metrics.inc_in_flight(1)
        t0 = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dt = time.perf_counter() - t0
            route = scope.get("route")
            route_path = getattr(route, "path", None) or scope.get("path") or "/"
            method = scope.get("method") or "GET"
            self.metrics.observe_request(method, route_path, status_code, dt)
            self.metrics.inc_in_flight(-1)
