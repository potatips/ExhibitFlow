#!/usr/bin/env python3
"""Install standalone GPU Worker dependencies (preserves existing PyTorch + CUDA).

Handles the onnxruntime vs onnxruntime-gpu conflict introduced by rtmlib.
Usage:
    python scripts/install_deps.py [--cpu] [--no-uv]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cpu = "--cpu" in sys.argv


def _find_uv() -> list[str] | None:
    if "--no-uv" in sys.argv:
        return None
    exe = shutil.which("uv")
    if not exe:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
            exe = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        except Exception:  # noqa: BLE001
            return None
    return [exe] if exe and Path(exe).exists() else None


_UV = _find_uv()


def pip(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    if _UV:
        return subprocess.run([*_UV, "pip", *args, "--python", sys.executable], check=check)
    return subprocess.run([sys.executable, "-m", "pip", *args], check=check)


def main() -> None:
    print("[deps] installer:", " ".join(_UV) + " pip" if _UV else "pip", flush=True)
    gpu_pkg = "onnxruntime" if cpu else "onnxruntime-gpu[cuda,cudnn]"
    pip("install", "-r", str(ROOT / "requirements-gpu.txt"), gpu_pkg)
    if not cpu:
        from importlib import metadata

        if any((d.metadata["Name"] or "").lower() == "onnxruntime" for d in metadata.distributions()):
            if _UV:
                pip("uninstall", "-q", "onnxruntime", check=False)
                pip("install", "--reinstall", "--no-deps", "onnxruntime-gpu")
            else:
                pip("uninstall", "-y", "-q", "onnxruntime", check=False)
                pip("install", "--force-reinstall", "--no-deps", "onnxruntime-gpu")
    print("[deps] Done. Check worker with: python -m worker.app --demo --help")


if __name__ == "__main__":
    main()
