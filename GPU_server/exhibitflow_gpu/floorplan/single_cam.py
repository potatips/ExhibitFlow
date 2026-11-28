"""Tạo map 2D từ 1 video bằng multicam-2d-floorplan-tracker (external/).

Chạy trong tiến trình riêng (CPU):
    python -m exhibitflow_gpu.floorplan.single_cam --video v.mp4 --out DIR [--tool PATH_TOOL]

Kết quả: DIR/map.png và DIR/map_info.json (homography 3x3 chuẩn hoá: (x/W, y/H) video -> (x/w, y/h) map).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
GPU_SERVER_DIR = PACKAGE_DIR.parent
TOOL_NAME = 'multicam-2d-floorplan-tracker'


def find_tool(explicit=None):
    cands = [
        explicit,
        os.environ.get('FLOORPLAN_DIR'),
        GPU_SERVER_DIR / 'external' / TOOL_NAME,
        GPU_SERVER_DIR.parent / 'external' / TOOL_NAME,
        GPU_SERVER_DIR / 'work' / TOOL_NAME,
    ]
    for c in cands:
        if c and (Path(c) / 'src' / 'map_builder.py').is_file():
            return Path(c).resolve()
    return None


def models_ready(tool):
    m = Path(tool) / 'models'
    return ((m / 'depth_anything_v2_metric_hypersim_vits.pth').is_file()
            and (m / 'Depth-Anything-V2' / 'metric_depth').is_dir()
            and (m / 'segformer-b2' / 'pytorch_model.bin').is_file())


def ensure_models(tool):
    if not models_ready(tool):
        print('[mapgen] tải model của tool (Depth-Anything V2 + SegFormer, lần đầu)...', flush=True)
        subprocess.run([sys.executable, str(Path(tool) / 'scripts' / 'download_structural_models.py')], check=True, cwd=str(tool))


def _grab_frame(video, size_max=1280):
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(n - 1, 0), int(fps * 0.5)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Không đọc được video {video}')
    h, w = frame.shape[:2]
    if max(h, w) > size_max:
        s = size_max / max(h, w)
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return frame, fps


def _fallback_map(image, h_metric, out_png):
    import cv2
    import numpy as np
    H, W = image.shape[:2]
    corners = np.array([[[0, H * 0.55], [W, H * 0.55], [W, H], [0, H], [W * 0.5, H * 0.55]]], np.float32)
    g = cv2.perspectiveTransform(corners, h_metric)[0]
    g[:, 1] = np.clip(g[:, 1], -1, 25)
    x0, x1 = min(g[:, 0].min(), -1), max(g[:, 0].max(), 1)
    y0, y1 = min(g[:, 1].min(), -1), max(g[:, 1].max(), 1)
    span = max(x1 - x0, y1 - y0) * 1.1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    S = 900
    px = S / span
    to_canvas = np.array([[px, 0, S / 2 - cx * px], [0, -px, S / 2 + cy * px], [0, 0, 1.0]])
    img = np.full((S, S, 3), (22, 18, 11), np.uint8)
    step = 1 if span < 20 else 2
    for m in np.arange(np.floor((cx - span / 2) / step) * step, cx + span / 2, step):
        x = int(round(S / 2 + (m - cx) * px))
        cv2.line(img, (x, 0), (x, S), (60, 48, 36), 1)
    for m in np.arange(np.floor((cy - span / 2) / step) * step, cy + span / 2, step):
        y = int(round(S / 2 - (m - cy) * px))
        cv2.line(img, (0, y), (S, y), (60, 48, 36), 1)
    poly = cv2.perspectiveTransform(np.array([g[:4]], np.float32), to_canvas)[0].astype(np.int32)
    ov = img.copy()
    cv2.fillPoly(ov, [poly], (90, 70, 40))
    img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
    cam = tuple(int(v) for v in (to_canvas @ np.array([0, 0, 1.0]))[:2])
    cv2.circle(img, cam, 9, (80, 160, 255), -1)
    cv2.imwrite(str(out_png), img)
    return to_canvas, (S, S)


def _run(video, out, tool):
    import cv2
    import numpy as np
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'room'
    (work / 'frames').mkdir(parents=True, exist_ok=True)
    frame, fps = _grab_frame(video)
    H, W = frame.shape[:2]
    cv2.imwrite(str(work / 'frames' / '000000.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    sys.path.insert(0, str(tool))
    os.chdir(tool)
    from src.boundaries import floor_rectification
    from src.semantics import SemanticBackend
    from src.structure import MetricDepthBackend

    structure = work / 'structure'
    structure.mkdir(exist_ok=True)
    print('[mapgen] ước lượng độ sâu + mặt sàn...', flush=True)
    depth = MetricDepthBackend(structure).infer(frame, 'cam1')
    labels = SemanticBackend(structure).infer(frame, 'cam1')
    rect = floor_rectification(frame, depth, labels, np.eye(3))
    h_metric = np.asarray(rect['T_published_to_depth_metric'], float)
    print(f"[mapgen] mặt sàn: inlier {rect['floor_inlier_ratio']:.2f}", flush=True)

    manifest = dict(source='ExhibitFlow GPU video', room_id='scene', fps=fps, map_y_axis='down',
                    coordinate_units='meters', rectification='none',
                    cameras=[dict(id='cam1', room_id='scene', frames_dir='frames', frame_pattern='{index:06d}.jpg',
                                  width=W, height=H, fps=fps, frames=1, H_image_to_map=h_metric.tolist())])
    (work / 'manifest.json').write_text(json.dumps(manifest))
    info = dict(method='architectural', floor_inlier_ratio=rect['floor_inlier_ratio'], frame_size=[W, H])
    try:
        from main import build
        from src.dataset_loader import RoomDataset
        out_tool = work / 'out'
        out_tool.mkdir(exist_ok=True)
        build(RoomDataset(work), out_tool)
        tr = json.loads((out_tool / 'transforms.json').read_text())
        cfg = json.loads((out_tool / 'map_config.json').read_text())
        h_img_canvas = np.asarray(tr['cameras']['cam1']['H_image_to_canvas'], float)
        img = cv2.imread(str(out_tool / 'global_2d_map.png'))
        cv2.imwrite(str(out / 'map.png'), img)
        canvas = (img.shape[1], img.shape[0])
        info['rooms'] = [r.get('id') for r in cfg.get('rooms', [])]
    except Exception:
        print('[mapgen] dựng tường lỗi -> dùng map lưới mét:\n' + traceback.format_exc()[-800:], flush=True)
        h_img_canvas, canvas = _fallback_map(frame, h_metric, out / 'map.png')
        h_img_canvas = h_img_canvas @ h_metric
        info['method'] = 'floor_grid'
    cw, ch = canvas
    h_norm = np.diag([1 / cw, 1 / ch, 1.0]) @ h_img_canvas @ np.diag([W, H, 1.0])
    h_norm /= h_norm[2, 2]
    info.update(homography=h_norm.tolist(), canvas=[cw, ch])
    (out / 'map_info.json').write_text(json.dumps(info, indent=2))
    print('[mapgen] xong:', info['method'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tool')
    a = ap.parse_args()
    tool = find_tool(a.tool)
    if tool is None:
        sys.exit(f'Không tìm thấy {TOOL_NAME} (đặt FLOORPLAN_DIR hoặc để trong external/)')
    ensure_models(tool)
    _run(Path(a.video).resolve(), Path(a.out).resolve(), tool)


def generate(video, out_dir, cache_root=None, log=print, timeout=900):
    """Chạy tiến trình con tạo map 2D từ 1 video, trả dict map_info (hoặc ném lỗi)."""
    video_p = Path(video)
    cache_dir = None
    if cache_root is not None:
        from ..common import video_fingerprint
        cache_dir = Path(cache_root) / f'v{video_fingerprint(video_p)}'
        if out_dir is None:
            out_dir = cache_dir
        if (cache_dir / 'map.png').is_file() and (cache_dir / 'map_info.json').is_file():
            out_p = Path(out_dir)
            if out_p.resolve() != cache_dir.resolve():
                out_p.mkdir(parents=True, exist_ok=True)
                shutil.copy(cache_dir / 'map.png', out_p / 'map.png')
                shutil.copy(cache_dir / 'map_info.json', out_p / 'map_info.json')
            return json.loads((cache_dir / 'map_info.json').read_text())

    out_p = Path(out_dir)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES='',
        PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join([str(GPU_SERVER_DIR), os.environ.get('PYTHONPATH', '')]),
    )
    p = subprocess.Popen(
        [sys.executable, '-m', 'exhibitflow_gpu.floorplan.single_cam', '--video', str(video_p), '--out', str(out_p)],
        cwd=str(GPU_SERVER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = []
    try:
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith('[mapgen]'):
                log(line)
            tail.append(line)
        p.wait(timeout=timeout)
    except BaseException:
        p.kill()
        raise
    if p.returncode != 0 or not (out_p / 'map_info.json').is_file():
        raise RuntimeError('mapgen lỗi: ' + ' | '.join(tail[-4:])[-500:])
    if cache_dir is not None and out_p.resolve() != cache_dir.resolve():
        cache_dir.mkdir(parents=True, exist_ok=True)
        if (out_p / 'map.png').is_file():
            shutil.copy(out_p / 'map.png', cache_dir / 'map.png')
        shutil.copy(out_p / 'map_info.json', cache_dir / 'map_info.json')
    return json.loads((out_p / 'map_info.json').read_text())


if __name__ == '__main__':
    main()
