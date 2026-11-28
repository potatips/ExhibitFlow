"""Ensure backend/ remains 100% pure lightweight CPU FastAPI with zero heavy ML deps."""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

FORBIDDEN_DEPS = {
    "torch",
    "torchvision",
    "torchaudio",
    "cuda",
    "opencv",
    "cv2",
    "ultralytics",
    "numpy",
    "scipy",
    "pandas",
    "onnxruntime",
    "transformers",
}

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_requirements_has_no_heavy_deps() -> None:
    req_path = BACKEND_ROOT / "requirements.txt"
    assert req_path.is_file()
    lines = [
        line.strip().lower()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pkg_names = [re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0] for line in lines]
    for pkg in pkg_names:
        assert pkg not in FORBIDDEN_DEPS, f"Forbidden heavy dependency in requirements.txt: {pkg}"
        assert not any(f in pkg for f in ("opencv", "torch", "cuda", "ultralytics", "onnx", "transformers")), (
            f"Suspicious heavy dependency in requirements.txt: {pkg}"
        )


def test_backend_source_has_no_heavy_imports() -> None:
    app_dir = BACKEND_ROOT / "app"
    for py_file in app_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0].lower()
                    assert top not in FORBIDDEN_DEPS, f"{py_file}: forbidden import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0].lower()
                assert top not in FORBIDDEN_DEPS, f"{py_file}: forbidden import from {node.module}"


def test_importing_backend_main_loads_no_heavy_modules(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "be_data"))
    mod = importlib.import_module("backend.app.main")
    assert hasattr(mod, "create_app")
    loaded = {m.split(".")[0].lower() for m in sys.modules}
    overlap = loaded & FORBIDDEN_DEPS
    assert not overlap, f"Heavy modules loaded into sys.modules: {sorted(overlap)}"
