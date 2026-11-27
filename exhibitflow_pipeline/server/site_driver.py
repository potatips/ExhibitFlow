"""Chạy pipeline của repo multicam-2d-floorplan-tracker trên camera đã hiệu chuẩn (K, R, t) và gom kết quả cho web.

Đường chính (đúng như `main.py --mode all` của repo, chỉ bỏ phần xuất video kiểm chứng):
  build                         dựng MẶT BẰNG từ tường (Depth-Anything + SegFormer, ArchitecturalMapBuilder), transforms.json, map_config.json
  src.observations.collect      phát hiện người (YOLO pose đa kích thước, ngoại suy điểm chân theo tỉ lệ cơ thể)
  src.projection_pipeline.process   tinh chỉnh ma trận chiếu, hợp nhất người giữa các camera, ID chung
  draw_floorplan                ảnh mặt bằng global_2d_map.png (web dùng làm nền Map 2D)
Toạ độ trong kết quả là toạ độ MAP của repo (mét, theo ảnh mặt bằng); web vẽ người / hành vi bất thường lên chính ảnh đó.

Đường dự phòng: nếu bước dựng mặt bằng của repo lỗi (cảnh không có tường: "Insufficient points for a structural plane") thì bỏ qua nó,
tự dựng transforms/map_config tối thiểu từ K/R/t (không có ảnh mặt bằng, web vẽ lưới + vùng nhìn), vẫn chạy collect + process.

Chạy từ thư mục repo:  python site_driver.py --data-dir D --output-dir O --frames N [--weights W] [--device cpu] [--map repo|fallback]
Kết quả: O/site_result.json (map, camera, người theo thời gian) + các file của repo (projections.json, global_2d_map.png...).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))       # site_events.py (cùng thư mục, chạy được ngoài repo)

import site_events  # noqa: E402
from src.dataset_loader import RoomDataset  # noqa: E402
from src.geometry import save_json  # noqa: E402


def ground_homography(cam):
    K = np.asarray(cam['K'], float)
    R = np.asarray(cam['R_world_to_camera'], float)
    t = np.asarray(cam['t_world_to_camera'], float)
    h = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))      # ảnh -> mặt đất Z=0 (mét)
    return h / h[2, 2], K, R, t


def ground_point(h, u, v):
    p = h @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def visible_polygon(K, R, t, w, hgt, max_dist=25.0, n=40):
    """Vùng mặt đất camera nhìn thấy: lấy mẫu dày CẢ VIỀN khung ảnh (n điểm mỗi cạnh). Tia chạm sàn thì lấy điểm chạm; tia hướng lên
    hoặc chạm sàn quá xa thì kéo ra đúng max_dist theo hướng ngang. (Bản cũ chỉ lấy 6 điểm với hàng trên cố định ở 40-45% chiều cao
    ảnh nên cắt mất phần sàn xa: vùng nhìn chỉ bằng một nửa thực tế và nhiều người nằm ngoài vùng.)"""
    C = -R.T @ t
    Ki = np.linalg.inv(K)
    edge = [(w * i / n, 0) for i in range(n)] + [(w, hgt * i / n) for i in range(n)] + \
           [(w * (1 - i / n), hgt) for i in range(n)] + [(0, hgt * (1 - i / n)) for i in range(n)]
    pts = []
    for u, v in edge:
        r = R.T @ (Ki @ np.array([u, v, 1.0]))
        hor = r[:2] / max(np.hypot(*r[:2]), 1e-9)
        p = C[:2] + hor * max_dist
        if r[2] < -1e-3:
            q = C[:2] + (-C[2] / r[2]) * r[:2]
            if np.hypot(*(q - C[:2])) <= max_dist:
                p = q
        pts.append(p)
    return np.array(pts)


def clip_rect(poly, b):
    """Cắt đa giác theo hình chữ nhật b=[x0,y0,x1,y1] (Sutherland-Hodgman)."""
    x0, y0, x1, y1 = b
    for inside, inter in [
        (lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))),
        (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1]) * (x1 - p[0]) / (q[0] - p[0]))),
        (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)),
        (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0]) * (y1 - p[1]) / (q[1] - p[1]), y1))]:
        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            if inside(p):
                out.append(tuple(p))
                if not inside(q):
                    out.append(inter(p, q))
            elif inside(q):
                out.append(inter(p, q))
        poly = out
        if not poly:
            break
    return np.array(poly) if poly else np.zeros((0, 2))


def build_inputs(dataset, output):
    """transforms.json, map_config.json, dataset_profile.json tối thiểu mà collect/process của repo đọc."""
    cams, polys = {}, []
    rb = dataset.manifest.get('room_bounds')                     # [x0,y0,x1,y1] mét nếu file hiệu chuẩn ghi giới hạn phòng
    for c in dataset.cameras:
        h, K, R, t = ground_homography(c)
        center = -R.T @ t
        poly = visible_polygon(K, R, t, c['width'], c['height'])
        if rb:
            poly = clip_rect(poly, rb)
            if len(poly) < 3:
                raise RuntimeError(f"Camera {c['id']} không nhìn thấy phần sàn nào trong giới hạn phòng {rb}: kiểm tra hiệu chuẩn.")
        polys.append(poly)
        axis = R.T @ np.array([0.0, 0.0, 1.0])
        cams[c['id']] = dict(
            room_id=c['room_id'], H_image_to_map=h.tolist(), H_map_to_image=np.linalg.inv(h).tolist(), K=K.tolist(),
            R_world_to_camera=R.tolist(), t_world_to_camera=t.tolist(), camera_center_world=center.tolist(),
            camera_xy_estimate=center[:2].tolist(), optical_axis_world=axis.tolist(), extrinsics_available=True,
            calibration_image_size=[c['width'], c['height']], calibration_source='site calibration (K, R, t)',
            camera_position_status='calibrated', visibility_polygon=poly.tolist())
    allp = np.vstack(polys + [np.array([v['camera_center_world'][:2] for v in cams.values()])])
    if rb:
        m = 0.3                                                               # biết giới hạn phòng: vùng hợp lệ = phòng (lề nhỏ)
        x0, y0, x1, y1 = rb[0] - m, rb[1] - m, rb[2] + m, rb[3] + m
    else:
        pad = 12.0                                                            # lề rộng: người ở xa vẫn nằm trong vùng hợp lệ
        x0, y0 = allp.min(0) - pad
        x1, y1 = allp.max(0) + pad
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]                       # CCW, hình chữ nhật bao vùng quan sát
    ppm = 20.0                                                                # điểm ảnh trên mét của "canvas"
    T = np.array([[ppm, 0, -x0 * ppm], [0, -ppm, y1 * ppm], [0, 0, 1.0]])    # bản đồ (X,Y mét) -> canvas (Y hướng lên)
    transforms = dict(schema_version=2, dataset='site', units='meters', convention='world X,Y metres, Z up; img->ground via K[r1 r2 t]',
                      gauge=dict(method='site calibration (no recentring)'), cameras=cams)
    room = dict(id=dataset.manifest['room_id'], polygon=polygon, corners=polygon, walls=[], doors=[], geometry_source='open space bounds',
                area_m2=float((x1 - x0) * (y1 - y0)))
    config = dict(schema_version=2, units='meters', rooms=[room], minimum_display_clearance_m=0.0, canvas_size=[int((x1 - x0) * ppm), int((y1 - y0) * ppm)],
                  T_map_to_canvas=T.tolist(), pixels_per_meter=ppm, source='site calibration')
    save_json(output / 'transforms.json', transforms)
    save_json(output / 'map_config.json', config)
    save_json(output / 'dataset_profile.json', dataset.profile())
    return transforms, config


def collect_people(records, cam_polys=None, gate_m=1.6, keep_frames=8, room_bounds=None):
    """Người theo từng mẫu thời gian: người ĐÃ HỢP NHẤT của repo (>=2 camera thấy cùng người, giữ ID của repo) cộng với
    những người CHỈ MỘT camera thấy. `fuse_and_track` của repo chỉ tạo người từ nhóm >=2 camera, nên với 1 camera (hoặc người
    nằm ngoài vùng chồng lấp) nó không ra ai; các phát hiện đơn đã chiếu xuống mặt đất (display_accepted) được theo dõi
    bằng ghép gần nhất theo vị trí (Hungarian, ngưỡng gate_m mét) để có ID ổn định."""
    import cv2
    from scipy.optimize import linear_sum_assignment
    polys = {c: np.asarray(p, np.float32) for c, p in (cam_polys or {}).items()}

    def covered_by(x, y):                          # có bao nhiêu camera nhìn thấy điểm mặt đất này
        return sum(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in polys.values())

    min_cover = 0 if len(polys) <= 1 else 1        # 1 camera: nhận cả người ngoài vùng nhìn gần (xa hơn); nhiều camera: chỉ vùng đúng một camera
    rb = room_bounds

    def in_room(x, y):                             # biết giới hạn phòng: bỏ người nằm ngoài phòng (bóng, phản chiếu, nhiễu)
        return rb is None or (rb[0] - 0.3 <= x <= rb[2] + 0.3 and rb[1] - 0.3 <= y <= rb[3] + 0.3)
    ids, nxt = {}, [1]
    singles = {}                                   # khoá theo dõi -> (xy cuối, chỉ số mẫu cuối thấy)

    def new_id():
        v = nxt[0]
        nxt[0] += 1
        return v

    frames = []
    for k, r in enumerate(records):
        people, used = [], set()
        for p in r.get('fused_people', []):
            if p.get('xy') is None:
                continue
            used.update(p.get('member_detection_ids', []))
            used.update(p.get('candidate_detection_ids', []))
            if not in_room(p['xy'][0], p['xy'][1]):
                continue
            pid = ids.setdefault(('f', p.get('id')), new_id())
            people.append(dict(id=pid, x=round(float(p['xy'][0]), 3), y=round(float(p['xy'][1]), 3), cams=list(p.get('member_cameras', []))))
        # người đơn lẻ: chỉ nhận ở vùng đúng MỘT camera nhìn thấy (ngoài mọi vùng nhìn thì điểm chiếu không đáng tin, vùng chồng lấp thì tin kết quả hợp nhất của repo)
        dets = [d for d in r.get('detections', []) if d.get('display_accepted') and d.get('detection_id') not in used
                and in_room(d['xy'][0], d['xy'][1]) and min_cover <= covered_by(d['xy'][0], d['xy'][1]) <= 1]
        keys = [key for key, (_, last) in singles.items() if k - last <= keep_frames]
        if dets and keys:
            cost = np.array([[np.hypot(d['xy'][0] - singles[key][0][0], d['xy'][1] - singles[key][0][1]) for key in keys] for d in dets])
            rows, cols = linear_sum_assignment(cost)
            match = {int(i): keys[j] for i, j in zip(rows, cols) if cost[i, j] <= gate_m}
        else:
            match = {}
        for i, d in enumerate(dets):
            key = match.get(i)
            if key is None:
                key = ('s', len(singles))
            singles[key] = ((float(d['xy'][0]), float(d['xy'][1])), k)
            pid = ids.setdefault(key, new_id())
            people.append(dict(id=pid, x=round(float(d['xy'][0]), 3), y=round(float(d['xy'][1]), 3), cams=[d['camera']]))
        frames.append(dict(t=round(r['time_seconds'], 3), people=people))
    # bỏ các "người" chỉ nhấp nháy dưới 3 mẫu (nhiễu phát hiện), giữ ID còn lại
    seen = {}
    for fr in frames:
        for pr in fr['people']:
            seen[pr['id']] = seen.get(pr['id'], 0) + 1
    if len(frames) >= 4:
        for fr in frames:
            fr['people'] = [pr for pr in fr['people'] if seen[pr['id']] >= 3]
    return frames


def ensure_structural_models():
    """Model dựng mặt bằng của repo (Depth-Anything V2 + SegFormer, vài trăm MB): thiếu thì tải bằng script của repo (cần Internet)."""
    m = REPO / 'models'
    if (m / 'depth_anything_v2_metric_hypersim_vits.pth').is_file() and (m / 'segformer-b2').exists() and (m / 'Depth-Anything-V2').exists():
        return
    print('Tải model dựng mặt bằng của repo...', flush=True)
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'download_structural_models.py')], check=True, cwd=str(REPO))


def ensure_person_models(weights):
    """Model người của repo (YOLO11n-pose + ResNet18), cả hai đường chạy đều cần: thiếu thì tải bằng script của repo (cần Internet và curl)."""
    if Path(weights).is_file() and (REPO / 'models' / 'resnet18-f37072fd.pth').is_file():
        return
    print('Tải model người của repo...', flush=True)
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'download_person_models.py')], check=True, cwd=str(REPO))


def repo_pipeline(dataset, out, weights, frames, device):
    """Pipeline của repo: dựng mặt bằng + phát hiện / chiếu / hợp nhất người + vẽ ảnh mặt bằng. Trả (records, map_config, transforms_refined)."""
    import cv2
    ensure_structural_models()
    from main import build                                    # main.py của repo (cwd = thư mục repo)
    from src.map_builder import draw_floorplan
    from src.observations import collect
    from src.projection_pipeline import process
    build(dataset, out)
    collect(dataset, out, weights, dataset.manifest.get('pose_scales', [1280]), device, frame_count=frames)
    records = process(out)
    config = json.loads((out / 'map_config.json').read_text())
    transforms = json.loads((out / 'transforms_refined.json').read_text())
    if not cv2.imwrite(str(out / 'global_2d_map.png'), draw_floorplan(config, transforms['cameras'])):
        raise RuntimeError('Không ghi được global_2d_map.png')
    return records, config, transforms


def map_polygons(transforms, config, dataset):
    """Vùng sàn mỗi camera nhìn thấy, theo toạ độ map của repo (cắt theo phòng); dùng để chọn người chỉ một camera thấy."""
    polys = {}
    try:
        from shapely.geometry import Polygon
        room = Polygon(config['rooms'][0]['polygon'])
    except Exception:                                          # noqa: BLE001
        room = None
    for c in dataset.cameras:
        cam = transforms['cameras'][c['id']]
        try:
            P = visible_polygon(np.asarray(cam['K_estimated'], float), np.asarray(cam['R_world_to_camera'], float),
                                np.asarray(cam['t_world_to_camera'], float), c['width'], c['height'])
            if room is not None:
                g = Polygon(P).buffer(0).intersection(room)
                if g.is_empty:
                    continue
                P = np.asarray(g.exterior.coords) if g.geom_type == 'Polygon' else np.asarray(max(g.geoms, key=lambda q: q.area).exterior.coords)
            polys[c['id']] = np.asarray(P, float)
        except Exception:                                      # noqa: BLE001
            continue
    return polys


def fix_depth_device():
    """Repo map đặt Depth Anything lên CPU/MPS, nhưng infer_image của code gốc đẩy ảnh lên CUDA nếu có -> trên GPU server lỗi
    'Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)'. Đưa model về cùng thiết bị với ảnh trước khi suy luận."""
    try:
        import torch
        vendor = REPO / 'models' / 'Depth-Anything-V2' / 'metric_depth'
        if not vendor.is_dir() or not torch.cuda.is_available():
            return
        sys.path.insert(0, str(vendor))
        from depth_anything_v2.dpt import DepthAnythingV2
        orig = DepthAnythingV2.infer_image

        def infer_image(self, *a, **k):
            self.to('cuda')
            return orig(self, *a, **k)

        DepthAnythingV2.infer_image = infer_image
    except Exception as e:                                     # noqa: BLE001
        print('fix_depth_device bỏ qua:', e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--weights', type=Path, default=REPO / 'models' / 'yolo11n-pose.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--map', choices=['repo', 'fallback'], default='repo', help='repo: mặt bằng + chiếu của repo; fallback: bỏ qua bước dựng mặt bằng')
    a = ap.parse_args()
    fix_depth_device()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dataset = RoomDataset(a.data_dir)
    ensure_person_models(a.weights)                            # ngoài try bên dưới: tải lỗi thì báo đúng lỗi tải
    fps = dataset.cameras[0]['fps']
    records = config = transforms = None
    note = None
    if a.map == 'repo':
        try:
            records, config, transforms = repo_pipeline(dataset, out, a.weights, a.frames, a.device)
        except Exception as e:                                 # noqa: BLE001
            note = f'{type(e).__name__}: {e}'
            print('REPO_MAP_FAILED:', note[:300], flush=True)
    if records is not None:
        polys = map_polygons(transforms, config, dataset)
        frames = site_events.stitch_ids(collect_people(records, polys))
        T = config['T_map_to_canvas']
        cw, ch = config['canvas_size']
        if abs(T[0][1]) > 1e-6 or abs(T[1][0]) > 1e-6:
            raise RuntimeError('Phép biến đổi map->ảnh của repo có xoay, web chưa vẽ được')
        bounds = [(0 - T[0][2]) / T[0][0], (ch - T[1][2]) / T[1][1], (cw - T[0][2]) / T[0][0], (0 - T[1][2]) / T[1][1]]
        cams = [dict(id=c['id'], center=transforms['cameras'][c['id']]['camera_xy_estimate'], H=transforms['cameras'][c['id']]['H_image_to_map'],
                     fov=(polys[c['id']].tolist() if c['id'] in polys else []), K_map=transforms['cameras'][c['id']].get('K_estimated'))
                for c in dataset.cameras]
        result = dict(room_id=config['rooms'][0]['id'], bounds=bounds, cameras=cams, fps=fps, frames=frames, map_source='repo',
                      map=dict(file='global_2d_map.png', T=T, canvas=[cw, ch], metric=config.get('metric_status')))
    else:                                                      # dự phòng: không có ảnh mặt bằng
        from src.observations import collect
        from src.projection_pipeline import process
        transforms, config = build_inputs(dataset, out)
        collect(dataset, out, a.weights, dataset.manifest.get('pose_scales', [1280]), a.device, frame_count=a.frames)
        records = process(out)
        cams = []
        for c in dataset.cameras:
            cam = transforms['cameras'][c['id']]
            cams.append(dict(id=c['id'], center=cam['camera_center_world'][:2], fov=cam['visibility_polygon'], H=cam['H_image_to_map']))
        room = config['rooms'][0]
        xs, ys = [p[0] for p in room['polygon']], [p[1] for p in room['polygon']]
        frames = site_events.stitch_ids(collect_people(records, {cid: c['visibility_polygon'] for cid, c in transforms['cameras'].items()},
                                                       room_bounds=dataset.manifest.get('room_bounds')))
        result = dict(room_id=room['id'], bounds=[min(xs), min(ys), max(xs), max(ys)], cameras=cams, fps=fps, frames=frames,
                      map_source='fallback', map=None, map_note=note)
    save_json(out / 'site_result.json', result)
    print('SITE_RESULT_OK', len(frames), 'mau', result['map_source'], flush=True)


if __name__ == '__main__':
    main()
