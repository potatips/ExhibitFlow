"""Hiệu chuẩn tự nhập cho từng camera + chiếu hành vi bất thường lên sàn (mét).

Engine của Hưng phát hiện hành vi bất thường (chạm / cầm / nghi lấy cắp) trong TOẠ ĐỘ ẢNH của từng video. Có hiệu chuẩn camera
(K, R, t) thì điểm chân của người gây ra sự kiện (giữa cạnh dưới hộp bao) đổi được sang toạ độ sàn Z=0 (mét), cùng hệ toạ độ với
bản đồ ghép camera. Module này làm 3 việc, không cần GPU / model:

  normalize_calibration / split_calibration   đọc file hiệu chuẩn người dùng nhập (nhiều kiểu khoá), kiểm tra hợp lý
  fit_to_video                                video có độ phân giải khác hiệu chuẩn (vd. đã hạ 240p) -> co K theo tỉ lệ
  engine_events / build_incidents             sự kiện từng camera -> sàn (mét), gộp các camera cùng thấy một sự kiện, nối đường đi
"""
import math

import numpy as np

SEVERITY = {'stealing': 3, 'holding': 2, 'touching': 1}
MAX_RANGE_M = 120.0              # điểm chiếu xa hơn mức này (gần đường chân trời) coi là không đáng tin


class CalibrationError(ValueError):
    pass


def _pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _matrix(v, shape, what):
    try:
        a = np.asarray(v, float).reshape(shape)
    except Exception:                                   # noqa: BLE001
        raise CalibrationError(f'{what}: cần ma trận {shape[0]}x{shape[1]} (9 số), nhận được dạng khác')
    if not np.all(np.isfinite(a)):
        raise CalibrationError(f'{what}: có giá trị không hợp lệ')
    return a


def normalize_calibration(raw, label='camera'):
    """raw = dict hiệu chuẩn của MỘT camera. Chấp nhận:
         K | camera_matrix | intrinsics (3x3) hoặc {fx, fy, cx, cy}
         R | R_world_to_camera | rotation (3x3)  hoặc  rvec (Rodrigues, 3 số)
         t | t_world_to_camera | tvec | translation (3 số)
         dist | distortion (tuỳ chọn) ; image_width / width, image_height / height (tuỳ chọn) ; units: m (mặc định) | cm | mm
       Quy ước OpenCV: Xcam = R @ Xworld + t, world Z hướng lên, mặt sàn Z=0. Trả dict chuẩn hoá."""
    import cv2
    if not isinstance(raw, dict):
        raise CalibrationError(f'{label}: hiệu chuẩn phải là một đối tượng JSON')
    Kraw = _pick(raw, 'K', 'camera_matrix', 'intrinsics', 'intrinsic')
    if isinstance(Kraw, dict) and 'fx' in Kraw:
        Kraw = [[Kraw['fx'], 0, Kraw.get('cx', 0)], [0, Kraw.get('fy', Kraw['fx']), Kraw.get('cy', 0)], [0, 0, 1]]
    if Kraw is None and 'fx' in raw:
        Kraw = [[raw['fx'], 0, raw.get('cx', 0)], [0, raw.get('fy', raw['fx']), raw.get('cy', 0)], [0, 0, 1]]
    if Kraw is None:
        raise CalibrationError(f'{label}: thiếu ma trận nội tham số K')
    K = _matrix(Kraw, (3, 3), f'{label}.K')
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise CalibrationError(f'{label}: tiêu cự trong K phải dương')
    Rraw = _pick(raw, 'R', 'R_world_to_camera', 'rotation')
    if Rraw is None and _pick(raw, 'rvec') is not None:
        R = cv2.Rodrigues(_matrix(raw['rvec'], (3, 1), f'{label}.rvec'))[0]
    elif Rraw is None:
        raise CalibrationError(f'{label}: thiếu ma trận xoay R (hoặc rvec)')
    else:
        R = _matrix(Rraw, (3, 3), f'{label}.R')
    traw = _pick(raw, 't', 't_world_to_camera', 'tvec', 'translation')
    if traw is None:
        raise CalibrationError(f'{label}: thiếu vector tịnh tiến t')
    t = _matrix(traw, (3,), f'{label}.t')
    unit = str(raw.get('units') or 'm').lower()
    t = t / {'m': 1.0, 'meter': 1.0, 'meters': 1.0, 'cm': 100.0, 'mm': 1000.0}.get(unit, 1.0)
    if abs(np.linalg.det(R) - 1) > 0.05 or np.abs(R @ R.T - np.eye(3)).max() > 0.05:
        raise CalibrationError(f'{label}: R không phải ma trận xoay hợp lệ (cần trực chuẩn, định thức 1)')
    center = -R.T @ t
    if center[2] < 0.2:
        raise CalibrationError(f'{label}: camera nằm ở độ cao Z={center[2]:.2f} m (sát hoặc dưới sàn). Thường do R, t là camera→world '
                               f'thay vì world→camera, hoặc trục Z của hệ toạ độ không hướng lên.')
    if (R.T @ np.array([0.0, 0.0, 1.0]))[2] > 0.3:
        raise CalibrationError(f'{label}: camera hướng lên trên, không nhìn thấy sàn. Kiểm tra lại R, t (quy ước OpenCV, world→camera).')
    if center[2] > 40:
        raise CalibrationError(f'{label}: camera ở độ cao {center[2]:.1f} m, không hợp lý. Kiểm tra đơn vị (cm/mm) hoặc quy ước R, t.')
    # rìa dưới ảnh phải nhìn xuống sàn, cách camera một khoảng hợp lý (bắt nhầm quy ước, nhầm đơn vị)
    w0 = float(_pick(raw, 'image_width', 'width') or 2 * K[0, 2])
    h0 = float(_pick(raw, 'image_height', 'height') or 2 * K[1, 2])
    ray = R.T @ np.linalg.solve(K, np.array([w0 / 2, h0 - 1, 1.0]))
    if ray[2] >= -1e-3:
        raise CalibrationError(f'{label}: tia nhìn ở rìa dưới ảnh không chạm sàn. R, t có thể sai quy ước (cần world→camera, OpenCV).')
    if (center[2] / -ray[2]) * np.hypot(ray[0], ray[1]) > 150:
        raise CalibrationError(f'{label}: rìa dưới ảnh chiếu xuống sàn cách camera quá xa (>150 m). Kiểm tra đơn vị hoặc R, t.')
    dist = _pick(raw, 'dist', 'distortion', 'dist_coeffs')
    dist = [] if dist is None else [float(v) for v in np.asarray(dist, float).reshape(-1)]
    w = _pick(raw, 'image_width', 'width')
    h = _pick(raw, 'image_height', 'height')
    size = _pick(raw, 'image_size')
    if (w is None or h is None) and size and len(size) == 2:
        w, h = size
    return dict(K=K.tolist(), dist=dist, R_world_to_camera=R.tolist(), t_world_to_camera=t.tolist(),
                width=int(w) if w else None, height=int(h) if h else None, label=raw.get('label') or label)


def split_calibration(raw):
    """File hiệu chuẩn -> danh sách (id hoặc None, dict camera). Hiểu: một camera; {"cameras": {id: {...}}} (kiểu transforms.json);
    {"cameras": [{id, ...}]}; hoặc dict id -> hiệu chuẩn."""
    if isinstance(raw, dict) and isinstance(raw.get('cameras'), dict):
        return [(str(k), v) for k, v in raw['cameras'].items()]
    if isinstance(raw, dict) and isinstance(raw.get('cameras'), list):
        return [(str(c.get('id') or c.get('name') or i), c) for i, c in enumerate(raw['cameras'])]
    if isinstance(raw, list):
        return [(str(c.get('id') or c.get('name') or i) if isinstance(c, dict) else str(i), c) for i, c in enumerate(raw)]
    if isinstance(raw, dict) and raw and all(isinstance(v, dict) for v in raw.values()) and not any(k in raw for k in ('K', 'R', 't', 'rvec')):
        return [(str(k), v) for k, v in raw.items()]
    return [(None, raw)]


def fit_to_video(spec, w, h):
    """Co K theo kích thước video thật (cùng tỉ lệ khung). spec chưa ghi kích thước thì coi là đúng bằng video."""
    sw, sh = spec.get('width'), spec.get('height')
    out = dict(spec)
    if not sw or not sh:
        out['width'], out['height'] = int(w), int(h)
        return out
    if (sw, sh) == (w, h):
        return out
    if abs((w / h) - (sw / sh)) / (sw / sh) > 0.02:
        raise RuntimeError(f'Video {w}x{h} khác tỉ lệ khung của hiệu chuẩn {sw}x{sh}: không thể quy đổi. Xuất video đúng tỉ lệ hoặc sửa hiệu chuẩn.')
    S = np.diag([w / sw, h / sh, 1.0])
    out['K'] = (S @ np.asarray(spec['K'], float)).tolist()
    out['width'], out['height'] = int(w), int(h)
    return out


def floor_mapper(spec):
    """Hàm (N,2) điểm ảnh của video GỐC (chưa khử méo) -> (N,2) toạ độ sàn (X, Y mét)."""
    import cv2
    K = np.asarray(spec['K'], float)
    R = np.asarray(spec['R_world_to_camera'], float)
    t = np.asarray(spec['t_world_to_camera'], float)
    dist = np.asarray(spec.get('dist') or [], float)
    Hinv = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))
    center = -R.T @ t

    def to_floor(pts):
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if dist.size and dist.any():
            p = cv2.undistortPoints(p, K, dist, P=K)
        p = p.reshape(-1, 2)
        h = np.c_[p, np.ones(len(p))] @ Hinv.T
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = h[:, :2] / h[:, 2:3]
        bad = ~np.isfinite(xy).all(1) | (np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1]) > MAX_RANGE_M)
        xy[bad] = np.nan
        return xy
    return to_floor


def _smooth(a, k=5):
    """Lọc trung vị trượt (giảm rung điểm chân), giữ nguyên độ dài."""
    if len(a) < k:
        return a
    pad = k // 2
    p = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    return np.stack([np.median(p[i:i + k], axis=0) for i in range(len(a))])


def engine_events(timeline, to_floor, cam='C1'):
    """timeline.json do dashboard ghi (frames, alerts, fps, width, height) -> sự kiện bất thường của MỘT camera trên sàn.
    frames[<khung>] = [[[id, x1, y1, x2, y2, chạm], ...], [vị trí map...]] với hộp chuẩn hoá 0-1. Điểm chân = giữa cạnh dưới hộp."""
    W, H, fps = float(timeline['width']), float(timeline['height']), float(timeline.get('fps') or 25.0)
    alerts = timeline.get('alerts') or []
    want = {int(a['tid']) for a in alerts}
    series = {tid: [] for tid in want}
    for f in sorted(timeline.get('frames', {}), key=int):
        for b in timeline['frames'][f][0]:
            tid = int(b[0])
            if tid in series:
                series[tid].append((int(f), (b[1] + b[3]) / 2 * W, b[4] * H))
    out = []
    for i, a in enumerate(alerts):
        s = series.get(int(a['tid']))
        if not s:
            continue
        fr = np.array([q[0] for q in s], float)
        xy = to_floor([(q[1], q[2]) for q in s])
        ok = np.isfinite(xy).all(1)
        if ok.sum() < 2:
            continue
        fr, xy = fr[ok], _smooth(xy[ok])
        t = fr / fps
        t0, t1 = a['start'] / fps, a['end'] / fps
        sel = (t >= t0) & (t <= t1)
        pos = np.median(xy[sel], axis=0) if sel.any() else xy[int(np.argmin(np.abs(t - (t0 + t1) / 2)))]
        step = max(1, len(t) // 220)
        out.append(dict(cam=cam, track=int(a['tid']), behavior=str(a.get('behavior') or 'stealing'), conf=a.get('conf'),
                        artifact=a.get('artifact_id'), evidence=a.get('evidence') or '', t0=round(t0, 2), t1=round(t1, 2),
                        x=round(float(pos[0]), 3), y=round(float(pos[1]), 3),
                        path=[[round(float(t[j]), 2), round(float(xy[j, 0]), 3), round(float(xy[j, 1]), 3)] for j in range(0, len(t), step)]))
    return out


def build_incidents(cam_events, frames, gate_m=2.0, time_gate=2.5, match_m=2.0):
    """Gộp sự kiện của các camera cùng thấy một sự kiện (trùng thời gian, gần nhau trên sàn); gắn với người ghép chung (ID của
    bản đồ ghép) gần nhất để lấy đường đi đầy đủ qua nhiều camera. Không khớp người nào thì dùng đường đi từ camera tin cậy nhất."""
    evs = sorted(cam_events, key=lambda e: e['t0'])
    groups = []
    for e in evs:
        for g in groups:
            if e['t0'] <= g['t1'] + time_gate and e['t1'] >= g['t0'] - time_gate and \
                    math.hypot(e['x'] - g['x'], e['y'] - g['y']) <= gate_m:
                g['items'].append(e)
                g['t0'], g['t1'] = min(g['t0'], e['t0']), max(g['t1'], e['t1'])
                g['x'] = float(np.mean([q['x'] for q in g['items']]))
                g['y'] = float(np.mean([q['y'] for q in g['items']]))
                break
        else:
            groups.append(dict(items=[e], t0=e['t0'], t1=e['t1'], x=e['x'], y=e['y']))
    out = []
    for n, g in enumerate(groups, 1):
        items = g['items']
        best = max(items, key=lambda q: (q['conf'] or 0, SEVERITY.get(q['behavior'], 0)))
        beh = max((q['behavior'] for q in items), key=lambda b: SEVERITY.get(b, 0))
        pid, path, bd = None, best['path'], match_m
        for fr in frames:
            if fr['t'] < g['t0'] - 1 or fr['t'] > g['t1'] + 1:
                continue
            for p in fr['people']:
                d = math.hypot(p['x'] - g['x'], p['y'] - g['y'])
                if d < bd:
                    bd, pid = d, p['id']
        if pid is not None:
            fused = [[fr['t'], p['x'], p['y']] for fr in frames for p in fr['people'] if p['id'] == pid]
            if len(fused) >= 2:
                path = fused
        out.append(dict(id=f'inc{n}', behavior=beh, conf=best['conf'], t0=round(g['t0'], 2), t1=round(g['t1'], 2),
                        x=round(g['x'], 3), y=round(g['y'], 3), person_id=pid, cams=sorted({q['cam'] for q in items}),
                        evidence=best['evidence'], artifact=best['artifact'], path=path,
                        per_cam=[dict(cam=q['cam'], track=q['track'], t0=q['t0'], t1=q['t1']) for q in items]))
    return out
