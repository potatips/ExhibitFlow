#!/usr/bin/env python3
"""Download ExhibitFlow model weights from HuggingFace into GPU_server/weights.

Usage:
    python scripts/fetch_weights.py [dst_dir] [repo_id]
Private HF repos use HF_TOKEN from environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEED = ["best_ckpt.pth", "osnet_x0_25.pth", "ltc_motion_mot17.pth", "dma_gbm_mot17.gbm"]


def main() -> None:
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "weights")
    repo = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HF_REPO", "hungnguyen190204/model_exhibitflow")
    dst.mkdir(parents=True, exist_ok=True)

    missing = lambda: [n for n in NEED if not any(dst.rglob(n))]
    if not missing():
        print(f"[weights] All 4 weight files already present in {dst}")
        return

    from huggingface_hub import snapshot_download

    print(f"[weights] Downloading {repo} -> {dst}", flush=True)
    try:
        snapshot_download(repo, local_dir=str(dst), token=os.environ.get("HF_TOKEN") or None)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[weights] Failed ({type(exc).__name__}: {exc}). Set HF_TOKEN if repo is private.")

    if missing():
        sys.exit("[weights] HuggingFace repo missing required files: " + ", ".join(missing()))
    print("[weights] Done")


if __name__ == "__main__":
    main()
