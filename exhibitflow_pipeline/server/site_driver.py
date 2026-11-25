"""Chạy các bước LÕI của repo multicam-2d-floorplan-tracker trên camera đã hiệu chuẩn, KHÔNG dựng tường từ độ sâu.

Dành cho địa điểm không có tường (sân ngoài trời như WILDTRACK) hoặc đã có hiệu chuẩn K/R/t chính xác: bộ dựng mặt bằng
(`ArchitecturalMapBuilder`) của repo cần tường nên bị bỏ qua. Dùng lại nguyên của repo:
  src.observations.collect   phát hiện người (YOLO pose đa kích thước, ngoại suy điểm chân theo tỉ lệ cơ thể)
  src.projection_pipeline.process   tinh chỉnh ma trận chiếu, hợp nhất người giữa camera, ID chung (src.fusion)

Chạy từ thư mục repo:  python site_driver.py --data-dir D --output-dir O --frames N [--weights W] [--device cpu]
Kết quả: O/projections.json (fused_people theo từng mẫu thời gian), O/site_result.json (khung map, camera, FOV, người theo thời gian).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--weights', type=Path, default=REPO / 'models' / 'yolo11n-pose.pt')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dataset = RoomDataset(a.data_dir)
    transforms, config = build_inputs(dataset, out)
    from src.observations import collect
    from src.projection_pipeline import process
    scales = dataset.manifest.get('pose_scales', [1280])
    collect(dataset, out, a.weights, scales, a.device, frame_count=a.frames)
    records = process(out)
    # kết quả gọn cho web
    cams = [dict(id=cid, center=c['camera_center_world'][:2], fov=c['visibility_polygon']) for cid, c in transforms['cameras'].items()]
    room = config['rooms'][0]
    xs, ys = [p[0] for p in room['polygon']], [p[1] for p in room['polygon']]
    fps = dataset.cameras[0]['fps']
    frames = collect_people(records, {cid: c['visibility_polygon'] for cid, c in transforms['cameras'].items()},
                            room_bounds=dataset.manifest.get('room_bounds'))
    save_json(out / 'site_result.json', dict(room_id=room['id'], bounds=[min(xs), min(ys), max(xs), max(ys)], cameras=cams, fps=fps, frames=frames))
    print('SITE_RESULT_OK', len(frames), 'mau', flush=True)


if __name__ == '__main__':
    main()
