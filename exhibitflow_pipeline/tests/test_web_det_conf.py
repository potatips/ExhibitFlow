"""Ngưỡng phát hiện người gửi từ web: kiểm tra phần nhận / áp vào detector (không cần GPU)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'server'))
web_server = pytest.importorskip('web_server')


@pytest.mark.parametrize('raw,expected', [(0.01, 0.01), ('0.3', 0.3), (0.99, 0.99), (0.0, None), (0.5e-3, None), (1.5, None), ('abc', None), (None, None), ('', None)])
def test_parse_det_conf(raw, expected):
    assert web_server.parse_det_conf(raw) == expected


def runner_with(det):
    return SimpleNamespace(engine=SimpleNamespace(M=SimpleNamespace(detector=det)))


def test_apply_sets_yolox_conf_without_reload():
    det = SimpleNamespace(conf=0.01)
    assert web_server.apply_det_conf(runner_with(det), 0.35) is True
    assert det.conf == 0.35


def test_apply_handles_ultralytics_style_detector():
    det = SimpleNamespace(kw=dict(conf=0.1, iou=0.5))
    assert web_server.apply_det_conf(runner_with(det), 0.4) is True
    assert det.kw['conf'] == 0.4 and det.kw['iou'] == 0.5


def test_apply_noop_when_none_or_engine_not_loaded():
    det = SimpleNamespace(conf=0.01)
    assert web_server.apply_det_conf(runner_with(det), None) is False and det.conf == 0.01
    assert web_server.apply_det_conf(SimpleNamespace(engine=None), 0.3) is False
    assert web_server.apply_det_conf(SimpleNamespace(engine=SimpleNamespace(M=SimpleNamespace(detector=None))), 0.3) is False
    assert web_server.apply_det_conf(runner_with(SimpleNamespace()), 0.3) is False
