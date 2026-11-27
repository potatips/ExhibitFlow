"""Chạy engine + VLM của nhiều camera SONG SONG cho một site job: mỗi camera một tiến trình con (exhibitflow/cam_worker.py), có llama-server riêng.

Các tiến trình nạp model LẦN LƯỢT (khoá file) rồi mới chạy cùng lúc. Mỗi camera có thư mục gốc riêng (`<root>/<cam>/runs`, `.../results`) nên kết quả không lẫn.
Trả về cho mỗi camera: timeline.json (+ `_run`, `_live`, `_result_dir`) và phiếu bầu ID cục bộ -> ID chung. Việc đối chiếu chéo camera
(ID, anomaly) làm sau ở site_fusion.py, giống hệt luồng tuần tự.
"""
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _newest(paths):
    paths = list(paths)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def collect(croot):
    """Đọc kết quả một camera từ thư mục gốc của nó. Ném RuntimeError nếu thiếu."""
    tlp = _newest((croot / 'runs').glob('*/timeline.json'))
    rp = _newest((croot / 'results').glob('*/result.json'))
    if tlp is None or rp is None:
        raise RuntimeError('camera chạy xong nhưng không có timeline.json / result.json')
    out = json.loads(tlp.read_text(encoding='utf-8'))
    try:
        out['_run'] = json.loads((tlp.parent / 'run.json').read_text(encoding='utf-8'))
    except Exception:                                          # noqa: BLE001
        pass
    out['_result_dir'] = str(rp.parent)
    if (rp.parent / 'live.mp4').is_file():
        out['_live'] = str(rp.parent / 'live.mp4')
    votes = {}
    vp = croot / 'id_votes.json'
    if vp.is_file():
        try:
            votes = json.loads(vp.read_text(encoding='utf-8'))
        except Exception:                                      # noqa: BLE001
            votes = {}
    return out, {int(t): int(max(v, key=v.get)) for t, v in votes.items() if v}


def run(videos, root, settings, setup_args, log, local=False, max_parallel=None, poll_s=2.0, timeout_s=7200):
    """videos {cam: đường dẫn}; setup_args {cam: tham số cho site_live.setup}. Trả {cam: (timeline, id_map)}.
    max_parallel: số camera chạy cùng lúc (None = tất cả). Lỗi bất kỳ -> ném RuntimeError (nơi gọi lùi về chạy tuần tự)."""
    from exhibitflow import cam_worker
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cams = list(videos)
    n = max(1, int(max_parallel or len(cams)))
    out = {}
    for i in range(0, len(cams), n):
        batch, procs = cams[i:i + n], {}
        try:
            for cam in batch:
                croot = root / cam
                cfg = dict(root=str(croot), video=str(videos[cam]), demo=False, local=bool(local), settings=dict(settings),
                           extra=dict(models=dict(vlm=dict(port=cam_worker.free_port()))),
                           flock=str(root / 'load.lock'),
                           setup=[dict(module='site_live', path=str(HERE), args=dict(setup_args[cam], root=str(croot)))])
                procs[cam] = cam_worker.CamProc(cfg)
                log(f'[parallel] {cam}: đã mở tiến trình riêng (pid {procs[cam].pid})')
            t0, last = time.time(), {}
            while True:
                views = {c: cam_worker.view_of(p.snap()) for c, p in procs.items()}
                for c, v in views.items():
                    if v['status'] == 'error':
                        raise RuntimeError(f'{c}: {v["error"]}')
                    key = (v['status'], round(v['progress'], 1))
                    if last.get(c) != key:
                        last[c] = key
                        log(f'[parallel] {c}: {v["status"]} {int(v["progress"] * 100)}%')
                if all(v['status'] == 'done' for v in views.values()):
                    break
                if time.time() - t0 > timeout_s:
                    raise RuntimeError('quá thời gian chờ các camera chạy song song')
                time.sleep(poll_s)
            for cam in batch:
                out[cam] = collect(root / cam)
        finally:
            for p in procs.values():
                try:
                    p.close()
                except Exception:                              # noqa: BLE001
                    pass
    return out
