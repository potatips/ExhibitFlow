"""Máy mới cài xong: server phải tìm thấy repo map và tự tải model người còn thiếu (không cần GPU / Internet)."""
import sys
import types
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / 'server'
sys.path.insert(0, str(SERVER))
import site_runner  # noqa: E402


@pytest.mark.parametrize('where', ['external', 'exhibitflow_pipeline/external'])
def test_find_repo_in_external(tmp_path, monkeypatch, where):
    repo = tmp_path / where / 'multicam-2d-floorplan-tracker'
    (repo / 'src').mkdir(parents=True)
    (repo / 'main.py').write_text('')
    (repo / 'src' / 'map_builder.py').write_text('')
    monkeypatch.delenv('FLOORPLAN_DIR', raising=False)
    monkeypatch.setattr(site_runner, 'HERE', tmp_path / 'exhibitflow_pipeline' / 'server')
    assert site_runner.find_repo() == repo.resolve()


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """Nạp site_driver với gói `src` của repo map thay bằng bản giả."""
    for name, attr in (('src', None), ('src.dataset_loader', 'RoomDataset'), ('src.geometry', 'save_json')):
        mod = types.ModuleType(name)
        if attr:
            setattr(mod, attr, object)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'site_driver', raising=False)
    import site_driver
    monkeypatch.setattr(site_driver, 'REPO', tmp_path)
    calls = []
    monkeypatch.setattr(site_driver.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    (tmp_path / 'models').mkdir()
    yield site_driver, calls
    sys.modules.pop('site_driver', None)


@pytest.mark.parametrize('present', [[], ['yolo11n-pose.pt'], ['resnet18-f37072fd.pth']])
def test_missing_person_model_is_downloaded(driver, present):
    site_driver, calls = driver
    models = site_driver.REPO / 'models'
    for name in present:
        (models / name).write_text('x')
    site_driver.ensure_person_models(models / 'yolo11n-pose.pt')
    assert len(calls) == 1 and calls[0][1].endswith('download_person_models.py')


def test_no_download_when_both_present(driver):
    site_driver, calls = driver
    models = site_driver.REPO / 'models'
    for name in ('yolo11n-pose.pt', 'resnet18-f37072fd.pth'):
        (models / name).write_text('x')
    site_driver.ensure_person_models(models / 'yolo11n-pose.pt')
    assert calls == []
