"""Hiệu chuẩn tự nhập cho từng camera + chiếu hành vi bất thường lên sàn (mét).

- `normalize_calibration` / `split_calibration`: đọc hiệu chuẩn người dùng nhập, kiểm tra hợp lý.
- `fit_to_video`: co K theo độ phân giải video thật.
- `floor_mapper` / `homography_mapper`: điểm ảnh -> toạ độ sàn (mét) hoặc map.
- `stitch_ids` / `engine_summary` / `engine_events` / `build_incidents`: gom sự kiện từ từng camera -> sự cố hợp nhất trên sàn.
"""
import math

try:
    import numpy as np
except ImportError:
    np = None

SEVERITY = {'stealing': 3, 'holding': 2, 'touching': 1}
MAX_RANGE_M = 120.0


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
    except Exception:
        raise CalibrationError(f'{what}: cần ma trận {shape[0]}x{shape[1]} (9 số), nhận được dạng khác')
    if not np.all(np.isfinite(a)):
        raise CalibrationError(f'{what}: có giá trị không hợp lệ')
    return a


def normalize_calibration(raw, label='camera'):
    """raw = dict hiệu chuẩn của MỘT camera -> dict chuẩn hoá OpenCV (world->camera, Z-up, mét)."""
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
    up = str(_pick(raw, 'up_axis', 'world_up') or 'z').lower().lstrip('+')
    if up == 'y':
        R = R @ np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    elif up != 'z':
        raise CalibrationError(f'{label}: up_axis chỉ nhận "z" hoặc "y", nhận được "{up}"')
    center = -R.T @ t
    if center[2] < 0.2:
        raise CalibrationError(f'{label}: camera nằm ở độ cao Z={center[2]:.2f} m (sát hoặc dưới sàn). Thường do R, t là camera→world '
                               f'thay vì world→camera, hoặc trục Z của hệ toạ độ không hướng lên.')
    if (R.T @ np.array([0.0, 0.0, 1.0]))[2] > 0.3:
        raise CalibrationError(f'{label}: camera hướng lên trên, không nhìn thấy sàn. Kiểm tra lại R, t (quy ước OpenCV, world→camera).')
    if center[2] > 40:
        raise CalibrationError(f'{label}: camera ở độ cao {center[2]:.1f} m, không hợp lý. Kiểm tra đơn vị (cm/mm) hoặc quy ước R, t.')
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


def _file_is_y_up(filedict):
    cs = filedict.get('coordinate_system') if isinstance(filedict, dict) else None
    txt = (cs.get('axes') if isinstance(cs, dict) else cs if isinstance(cs, str) else '') or ''
    flat = str(txt).lower().replace(' ', '').replace('_', '-')
    return 'y-up' in flat or 'y=up' in flat


def _inherit_axes(cam, filedict):
    if not isinstance(cam, dict) or 'up_axis' in cam or 'world_up' in cam:
        return cam
    return dict(cam, up_axis='y') if _file_is_y_up(filedict) else cam


def room_bounds_from(raw):
    """Giới hạn sàn phòng [x0, y0, x1, y1] (mét, X đông, Y bắc) nếu file hiệu chuẩn có ghi, không thì None."""
    if not isinstance(raw, dict):
        return None
    try:
        rb = raw.get('room_bounds')
        if isinstance(rb, dict):
            rb = [rb['x_min'], rb['y_min'], rb['x_max'], rb['y_max']]
        if rb is None:
            lm = raw.get('room_landmarks_m') or {}
            if 'floor_bounds_X' in lm and 'floor_bounds_Z' in lm:
                (x0, x1), (z0, z1) = lm['floor_bounds_X'], lm['floor_bounds_Z']
                rb = [x0, -z1, x1, -z0] if _file_is_y_up(raw) else [x0, z0, x1, z1]
        if rb is None:
            return None
        x0, y0, x1, y1 = (float(v) for v in rb)
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)] if x1 != x0 and y1 != y0 else None
    except (KeyError, TypeError, ValueError):
        return None


def split_calibration(raw):
    """File hiệu chuẩn -> danh sách (id hoặc None, dict camera)."""
    if isinstance(raw, dict) and isinstance(raw.get('cameras'), dict):
        return [(str(k), _inherit_axes(v, raw)) for k, v in raw['cameras'].items()]
    if isinstance(raw, dict) and isinstance(raw.get('cameras'), list):
        return [(str(c.get('id') or c.get('name') or i), _inherit_axes(c, raw)) for i, c in enumerate(raw['cameras'])]
    if isinstance(raw, list):
        return [(str(c.get('id') or c.get('name') or i) if isinstance(c, dict) else str(i), c) for i, c in enumerate(raw)]
    if isinstance(raw, dict) and raw and all(isinstance(v, dict) for v in raw.values()) and not any(k in raw for k in ('K', 'R', 't', 'rvec')):
        return [(str(k), v) for k, v in raw.items()]
    return [(None, raw)]


def fit_to_video(spec, w, h):
    """Co K theo kích thước video thật (cùng tỉ lệ khung)."""
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
    """Hàm (N,2) điểm ảnh của video GỐC -> (N,2) toạ độ sàn (X, Y mét)."""
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


def homography_mapper(H, K, dist=None):
    """Dùng ma trận chiếu ảnh -> map (H_image_to_map) cùng hệ toạ độ với ảnh mặt bằng."""
    import cv2
    H = np.asarray(H, float)
    K = np.asarray(K, float)
    d = np.asarray(dist or [], float)

    def to_map(pts):
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if d.size and d.any():
            p = cv2.undistortPoints(p, K, d, P=K)
        p = p.reshape(-1, 2)
        h = np.c_[p, np.ones(len(p))] @ H.T
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = h[:, :2] / h[:, 2:3]
        xy[~np.isfinite(xy).all(1)] = np.nan
        return xy
    return to_map


def appearance_embedding(img, box):
    """Đặc trưng ngoại hình nhẹ của một người: histogram màu HSV của thân trên và thân dưới (căn Hellinger, chuẩn hoá L2). Không cần trọng số/GPU."""
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return None
    parts = []
    for a, b in ((0.15, 0.5), (0.5, 0.95)):
        c = img[y1 + int(bh * a):y1 + int(bh * b), x1 + int(bw * 0.2):x2 - int(bw * 0.2)]
        if c.size == 0:
            return None
        hist = cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        parts.append(np.sqrt(hist / max(hist.sum(), 1.0)))
    v = np.concatenate(parts)
    return v / max(float(np.linalg.norm(v)), 1e-9)


class GlobalIdLabeler:
    """Ánh xạ ID cục bộ theo từng camera sang ID chung của map ghép dựa trên vị trí sàn và đặc trưng ngoại hình chia sẻ."""

    def __init__(self, frames, to_floor, fps, video_wh, gate_m=1.5, max_dt=1.0, gallery=None, app_weight=0.8, app_max=0.6):
        self.frames = [f for f in frames if f.get('people')]
        self.times = np.array([f['t'] for f in self.frames], float)
        self.to_floor, self.fps, self.vw, self.vh = to_floor, float(fps or 25.0), float(video_wh[0]), float(video_wh[1])
        self.gate, self.max_dt, self.last = gate_m, max_dt, {}
        self.gallery = gallery if gallery is not None else {}
        self.app_weight, self.app_max = app_weight, app_max

    def labels(self, idx, recs, work_wh, image=None):
        """Trả {id cục bộ: ID chung} cho một khung hình."""
        if not len(self.times) or not recs:
            return {}
        t = idx / self.fps
        j = int(np.argmin(np.abs(self.times - t)))
        people = self.frames[j]['people'] if abs(self.times[j] - t) <= self.max_dt else []
        sx, sy = self.vw / float(work_wh[0]), self.vh / float(work_wh[1])
        feet = [(((r['box'][0] + r['box'][2]) / 2) * sx, r['box'][3] * sy) for r in recs]
        xy = np.asarray(self.to_floor(feet), float).reshape(-1, 2)
        embs = [appearance_embedding(image, r['box']) if image is not None else None for r in recs]
        pairs = []
        for i, r in enumerate(recs):
            if not np.isfinite(xy[i]).all():
                continue
            for q in people:
                d = math.hypot(xy[i][0] - q['x'], xy[i][1] - q['y'])
                if d > self.gate:
                    continue
                g = self.gallery.get(q['id'])
                da = float(1.0 - embs[i] @ g) if embs[i] is not None and g is not None else None
                if da is not None and da > self.app_max:
                    continue
                pairs.append((d / self.gate + (self.app_weight * da if da is not None else 0.0), i, q['id']))
        pairs.sort()
        out, used = {}, set()
        for _, i, gid in pairs:
            tid = recs[i]['id']
            if tid not in out and gid not in used:
                out[tid] = gid
                used.add(gid)
        for r in recs:
            tid = r['id']
            if tid not in out and self.last.get(tid) is not None and self.last[tid] not in used:
                out[tid] = self.last[tid]
                used.add(self.last[tid])
        for i, r in enumerate(recs):
            gid = out.get(r['id'])
            if gid is not None and embs[i] is not None:
                g = self.gallery.get(gid)
                m = embs[i] if g is None else 0.9 * g + 0.1 * embs[i]
                self.gallery[gid] = m / max(float(np.linalg.norm(m)), 1e-9)
        self.last.update(out)
        return out


def stitch_ids(frames, max_gap_s=3.0, gate_m=0.8, speed_mps=0.8):
    """Nối các ID bị đứt thành một và đánh số lại gọn 1..N theo thời điểm xuất hiện."""
    tracks = {}
    for fr in frames:
        for p in fr['people']:
            tracks.setdefault(p['id'], []).append((fr['t'], p['x'], p['y']))
    info = {}
    for i, q in tracks.items():
        q.sort()
        k = min(2, len(q))
        info[i] = dict(t0=q[0][0], t1=q[-1][0], a=np.median([[x, y] for _, x, y in q[:k]], axis=0), b=np.median([[x, y] for _, x, y in q[-k:]], axis=0))
    cands = []
    for A, ia in info.items():
        for B, ib in info.items():
            gap = ib['t0'] - ia['t1']
            if A == B or gap <= 0 or gap > max_gap_s:
                continue
            d = float(np.hypot(*(ia['b'] - ib['a'])))
            allowed = gate_m + speed_mps * gap
            if d <= allowed:
                cands.append((d / allowed, gap, A, B))
    cands.sort()
    succ, pred = {}, {}
    for _, _, A, B in cands:
        if A not in succ and B not in pred:
            succ[A], pred[B] = B, A
    root = {}
    for i in info:
        r = i
        while r in pred:
            r = pred[r]
        root[i] = r
    new = {r: k + 1 for k, r in enumerate(sorted(set(root.values()), key=lambda r: info[r]['t0']))}
    out = []
    for fr in frames:
        seen, people = set(), []
        for p in fr['people']:
            nid = new[root[p['id']]]
            if nid in seen:
                continue
            seen.add(nid)
            people.append(dict(p, id=nid))
        out.append(dict(fr, people=people))
    return out


def _smooth(a, k=5):
    """Lọc trung vị trượt (giảm rung điểm chân), giữ nguyên độ dài."""
    if len(a) < k:
        return a
    pad = k // 2
    p = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    return np.stack([np.median(p[i:i + k], axis=0) for i in range(len(a))])


def engine_summary(timeline, run=None):
    """Tóm tắt kết quả engine ở một camera."""
    fps = float(timeline.get('fps') or 25.0)
    ids, contact = set(), 0
    for v in timeline.get('frames', {}).values():
        boxes = v[0]
        ids.update(int(b[0]) for b in boxes)
        contact += any(b[5] for b in boxes if len(b) > 5)
    out = dict(tracks=len(ids), contact_s=round(contact / fps, 1), alerts=len(timeline.get('alerts') or []), events=None, normal=None, uncertain=None)
    if run:
        s = run.get('summary') or {}
        out['events'] = s.get('n_events')
        d = [t.get('decision') for t in run.get('vlm_tasks') or []]
        out['normal'], out['uncertain'] = d.count('NORMAL'), d.count('CANDIDATE')
    return out


def engine_events(timeline, to_floor, cam='C1'):
    """timeline.json -> sự kiện bất thường của MỘT camera trên sàn."""
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
    for a in alerts:
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
    """Gộp sự kiện của các camera cùng thấy một sự kiện trên sàn và gắn với person_id trên map ghép."""
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
