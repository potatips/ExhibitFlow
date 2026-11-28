"""Chạy repo `multicam-2d-floorplan-tracker` cho nhiều camera ĐÃ HIỆU CHUẨN (hồ sơ địa điểm) và đọc kết quả.

Hồ sơ địa điểm: `GPU_server/sites/<site_id>.json` (K, hệ số méo, R, t từng camera trong cùng một hệ toạ độ mét).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import calibration as site_events

HERE = Path(__file__).resolve().parent
GPU_SERVER_DIR = HERE.parent.parent
SITES_DIR = GPU_SERVER_DIR / 'sites'


def find_repo():
    """Thư mục repo multicam-2d-floorplan-tracker (FLOORPLAN_DIR, GPU_server/external/, hoặc workspace external/)."""
    cands = [
        os.environ.get('FLOORPLAN_DIR'),
        GPU_SERVER_DIR / 'external' / 'multicam-2d-floorplan-tracker',
        GPU_SERVER_DIR.parent / 'external' / 'multicam-2d-floorplan-tracker',
        GPU_SERVER_DIR.parent / 'multicam-2d-floorplan-tracker',
        GPU_SERVER_DIR / 'work' / 'multicam-2d-floorplan-tracker',
        Path('/kaggle/working/multicam-2d-floorplan-tracker'),
    ]
    for c in cands:
        if c and (Path(c) / 'main.py').is_file() and (Path(c) / 'src' / 'map_builder.py').is_file():
            return Path(c).resolve()
    return None


def list_sites():
    out = {}
    if not SITES_DIR.is_dir():
        return out
    for p in sorted(SITES_DIR.glob('*.json')):
        try:
            s = json.loads(p.read_text(encoding='utf-8'))
            out[s['site_id']] = dict(site_id=s['site_id'], name=s.get('name', s['site_id']), cameras=list(s['cameras']),
                                     labels={k: v.get('label', k) for k, v in s['cameras'].items()})
        except Exception:
            continue
    return out


def load_site(site_id):
    p = SITES_DIR / f'{Path(site_id).name}.json'
    if not p.is_file():
        raise FileNotFoundError(f'Không có hồ sơ địa điểm "{site_id}"')
    return json.loads(p.read_text(encoding='utf-8'))


def prepare_dataset(site, videos, work_dir, seconds=None, log=print):
    """Cắt khung đã khử méo của từng camera ở site['sample_fps'] rồi ghi manifest.json. Trả (thư mục dataset, số mẫu)."""
    import cv2
    import numpy as np
    work = Path(work_dir) / site['room_id']
    work.mkdir(parents=True, exist_ok=True)
    sfps = float(site.get('sample_fps', 4.0))
    caps = {}
    for cam, path in videos.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'Không mở được video của {cam}: {path}')
        caps[cam] = cap
    secs = [c.get(cv2.CAP_PROP_FRAME_COUNT) / max(c.get(cv2.CAP_PROP_FPS), 1) for c in caps.values()]
    total = min(secs) if seconds is None else min(float(seconds), min(secs))
    n = max(1, int(total * sfps))
    cameras = []
    for cam, cap in caps.items():
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            spec = site_events.fit_to_video(site['cameras'][cam], w, h)
        except RuntimeError as e:
            raise RuntimeError(f'{cam}: {e}')
        K = np.asarray(spec['K'], float)
        dist = np.asarray(spec.get('dist') or [0, 0, 0, 0, 0], float)
        undist = bool(np.any(dist))
        folder = work / cam / 'frames'
        folder.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        want = [int(round(i * fps / sfps)) for i in range(n)]
        idx, k = 0, 0
        while k < n:
            ok, frame = cap.read()
            if not ok:
                n = k
                break
            if idx == want[k]:
                if undist:
                    frame = cv2.undistort(frame, K, dist)
                cv2.imwrite(str(folder / f'{k:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                k += 1
            idx += 1
        cap.release()
        log(f'[site] {cam}: {k} khung ({sfps:g}/s)')
        cameras.append(dict(id=cam, room_id=site['room_id'], width=w, height=h, fps=sfps, frames=n, frames_dir=f'{cam}/frames',
                            frame_pattern='{index:06d}.jpg', K=spec['K'], R_world_to_camera=spec['R_world_to_camera'],
                            t_world_to_camera=spec['t_world_to_camera']))
    for c in cameras:
        c['frames'] = n
    scale = max(640, min(1920, int(max(c['width'] for c in cameras) * 1.5)))
    manifest = dict(room_id=site['room_id'], coordinate_units='meters', map_y_axis='up', allow_open_ends=True,
                    source=f"ExhibitFlow GPU: {site.get('name', site['site_id'])}", pose_scales=[scale],
                    timing=dict(clock_id=f"site-{site['site_id']}", time_offset_seconds=0.0), cameras=cameras)
    if site.get('room_bounds'):
        manifest['room_bounds'] = site['room_bounds']
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return Path(work_dir), n


def run_repo(repo, data_dir, out_dir, frames, site_room, device='cpu', log=print, timeout=3600):
    """Chạy site_driver.py (collect + process) trên repo multicam-2d-floorplan-tracker. Trả thư mục kết quả."""
    repo = Path(repo).resolve()
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = repo / 'models' / 'yolo11n-pose.pt'
    cmd = [sys.executable, '-u', str(HERE / 'site_driver.py'), '--data-dir', str(Path(data_dir) / site_room), '--output-dir', str(out_dir),
           '--frames', str(frames), '--device', device, '--weights', str(weights)]
    env = dict(
        os.environ,
        YOLO_CONFIG_DIR=str(out_dir / 'yolo_config'),
        PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join([str(GPU_SERVER_DIR), os.environ.get('PYTHONPATH', '')]),
    )
    p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    tail = []
    try:
        for line in p.stdout:
            line = line.rstrip()
            tail.append(line)
            if line and not line.startswith(('WARNING', 'Warning')):
                log('[map2d] ' + line[:200])
        p.wait(timeout=timeout)
    except BaseException:
        p.kill()
        raise
    if p.returncode != 0:
        raise RuntimeError('map2d lỗi: ' + ' | '.join(tail[-5:])[-600:])
    return out_dir
