#!/usr/bin/env python3
"""Reproducible GPU Inference & Optimization Benchmark for ExhibitFlow (Vast.ai / Kaggle / Linux CUDA).

Compares Baseline vs Optimized pipeline configurations on real video(s) and exports CSV + Markdown:
  - Baseline  : YOLOX fp16=False, pose_gate=None, VLM 12f_768 (full JSON)
  - Optimized : YOLOX fp16=True + fuse=True + gpu_preproc=True, pose_gate=1.5, VLM Q4_K_M 8f_512_short

Supports `--dry-run` on CPU machines to validate schema, CLI arguments, and CSV/Markdown report generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

GPU_ROOT = Path(__file__).resolve().parents[1]
if str(GPU_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_ROOT))


PROFILES: dict[str, dict[str, Any]] = {
    "baseline_no_opt": {
        "description": "FP32 detector, no pose gating, VLM 12 frames @ 768px full JSON",
        "overrides": {
            "models": {
                "detector": {"fp16": False, "fuse": False, "gpu_preproc": False},
                "vlm": {"n_images": 12, "max_side": 768, "short_json": False, "max_new_tokens": 256},
            },
            "rt": {"pose_gate_norm_dist": None},
        },
    },
    "optimized_prod": {
        "description": "FP16+fused YOLOX, pose_gate=1.5, VLM Q4_K_M 8f_512_short + SharedVLMServer",
        "overrides": {
            "models": {
                "detector": {"fp16": True, "fuse": True, "gpu_preproc": True},
                "vlm": {"n_images": 8, "max_side": 512, "short_json": True, "max_new_tokens": 64},
            },
            "rt": {"pose_gate_norm_dist": 1.5},
        },
    },
}


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count() or 1,
        "python": platform.python_version(),
        "gpu_name": "N/A (dry-run CPU)",
        "cuda_available": False,
        "vram_total_gb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception:  # noqa: BLE001
        pass
    return info


def _run_dry(out_dir: Path, profiles: list[str], runs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pname in profiles:
        spec = PROFILES[pname]
        rows.append(
            {
                "profile": pname,
                "description": spec["description"],
                "runs": runs,
                "frames_processed": 0,
                "pipeline_fps": 0.0,
                "e2e_mean_ms": 0.0,
                "e2e_p95_ms": 0.0,
                "vlm_tasks": 0,
                "vlm_infer_mean_s": 0.0,
                "vlm_infer_p95_s": 0.0,
                "gpu_peak_gb": 0.0,
                "wall_with_vlm_s": 0.0,
                "status": "dry_run_template_only",
            }
        )
    return rows


def _run_real(video: Path, out_dir: Path, profiles: list[str], runs: int, max_frames: int | None) -> list[dict[str, Any]]:
    import statistics

    from exhibitflow_gpu.service import Engine

    rows: list[dict[str, Any]] = []
    for pname in profiles:
        spec = PROFILES[pname]
        fps_vals, e2e_means, e2e_p95s, vlm_means, vlm_p95s, vrams, walls = [], [], [], [], [], [], []
        frames_proc = 0
        vlm_tasks = 0
        for r_idx in range(runs):
            eng = Engine()
            try:
                eng.load(
                    overrides=spec["overrides"],
                    work_dir=str(out_dir / "work" / pname),
                    out_dir=str(out_dir / "results" / pname / f"run_{r_idx}"),
                    key=(pname, str(r_idx)),
                )
                res = eng.run(str(video), max_frames=max_frames, mode="full")
            finally:
                eng.close()

            perf = res.get("performance") or {}
            e2e = perf.get("e2e_ms") or {}
            vlm = perf.get("vlm") or {}
            gpu_peak = perf.get("gpu_peak_gb") or {}
            peak_gb = max((float(v) for v in gpu_peak.values() if v is not None), default=0.0)

            frames_proc = int(perf.get("frames_processed") or 0)
            vlm_tasks = int(vlm.get("n_tasks") or 0)
            if perf.get("pipeline_fps") is not None:
                fps_vals.append(float(perf["pipeline_fps"]))
            if e2e.get("mean") is not None:
                e2e_means.append(float(e2e["mean"]))
            if e2e.get("p95") is not None:
                e2e_p95s.append(float(e2e["p95"]))
            if vlm.get("infer_mean_s") is not None:
                vlm_means.append(float(vlm["infer_mean_s"]))
            if vlm.get("infer_p95_s") is not None:
                vlm_p95s.append(float(vlm["infer_p95_s"]))
            vrams.append(peak_gb)
            if perf.get("wall_with_vlm_s") is not None:
                walls.append(float(perf["wall_with_vlm_s"]))

        rows.append(
            {
                "profile": pname,
                "description": spec["description"],
                "runs": runs,
                "frames_processed": frames_proc,
                "pipeline_fps": round(statistics.mean(fps_vals), 2) if fps_vals else 0.0,
                "e2e_mean_ms": round(statistics.mean(e2e_means), 2) if e2e_means else 0.0,
                "e2e_p95_ms": round(statistics.mean(e2e_p95s), 2) if e2e_p95s else 0.0,
                "vlm_tasks": vlm_tasks,
                "vlm_infer_mean_s": round(statistics.mean(vlm_means), 3) if vlm_means else 0.0,
                "vlm_infer_p95_s": round(statistics.mean(vlm_p95s), 3) if vlm_p95s else 0.0,
                "gpu_peak_gb": round(max(vrams) if vrams else 0.0, 2),
                "wall_with_vlm_s": round(statistics.mean(walls), 2) if walls else 0.0,
                "status": "measured_gpu",
            }
        )
    return rows


def _write_reports(out_dir: Path, rows: list[dict[str, Any]], hw: dict[str, Any], video_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gpu_benchmark.csv"
    fields = [
        "profile", "description", "runs", "frames_processed", "pipeline_fps",
        "e2e_mean_ms", "e2e_p95_ms", "vlm_tasks", "vlm_infer_mean_s", "vlm_infer_p95_s",
        "gpu_peak_gb", "wall_with_vlm_s", "status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Kết quả Benchmark Tối ưu GPU Inference (ExhibitFlow)",
        "",
        f"- **Thời điểm:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **GPU:** `{hw['gpu_name']}` (VRAM `{hw['vram_total_gb']} GB`, CUDA=`{hw['cuda_available']}`)",
        f"- **CPU / OS:** `{hw['cpu_cores']} cores` · `{hw['os']}` · Python `{hw['python']}`",
        f"- **Video test:** `{video_label}`",
        "",
        "| Cấu hình | Runs | FPS | E2E Mean (ms) | E2E P95 (ms) | VLM Mean (s/task) | Peak VRAM (GB) | Wall Total (s) | Trạng thái |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| `{r['profile']}` | {r['runs']} | {r['pipeline_fps']} | {r['e2e_mean_ms']} | "
            f"{r['e2e_p95_ms']} | {r['vlm_infer_mean_s']} | {r['gpu_peak_gb']} | {r['wall_with_vlm_s']} | `{r['status']}` |"
        )
    (out_dir / "gpu_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "gpu_benchmark.json").write_text(json.dumps({"hardware": hw, "rows": rows}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark ExhibitFlow GPU optimizations (Vast.ai / Kaggle)")
    ap.add_argument("--video", default="", help="Path to test video (.mp4)")
    ap.add_argument("--profiles", default="baseline_no_opt,optimized_prod", help="Comma-separated profile names")
    ap.add_argument("--runs", type=int, default=1, help="Number of repetitions per profile")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional max frames cap per video (0 = full)")
    ap.add_argument("--out", default="docs/benchmarks", help="Output directory")
    ap.add_argument("--dry-run", action="store_true", help="Validate script and output template on CPU without GPU")
    args = ap.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        if p not in PROFILES:
            raise SystemExit(f"Unknown profile '{p}'. Valid: {list(PROFILES)}")

    hw = _hardware_info()
    out_dir = Path(args.out)

    if args.dry_run:
        rows = _run_dry(out_dir, profiles, args.runs)
        _write_reports(out_dir, rows, hw, args.video or "dry-run")
        print(f"Dry-run benchmark OK -> {out_dir}/gpu_benchmark.csv", flush=True)
        return 0

    if not args.video or not Path(args.video).is_file():
        raise SystemExit("ERROR: --video <path.mp4> is required unless --dry-run is set.")
    if not hw["cuda_available"]:
        raise SystemExit("ERROR: CUDA GPU not detected. Run on Vast.ai/Kaggle GPU or pass --dry-run on CPU.")

    rows = _run_real(Path(args.video), out_dir, profiles, max(1, args.runs), args.max_frames or None)
    _write_reports(out_dir, rows, hw, str(args.video))
    print(f"GPU benchmark complete -> {out_dir}/gpu_benchmark.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
