"""Số hiện vật CHUNG cho các camera đã hiệu chuẩn (mỗi camera chạy engine riêng nên tự đánh số riêng).

Từ mỗi camera kẻ tia qua tâm hộp hiện vật tới mặt đầu tiên của phòng (4 tường hoặc sàn); hai camera trỏ vào cùng một chỗ
(<= MATCH_M mét) là cùng một hiện vật. Cần giới hạn phòng [x0, y0, x1, y1] trong file hiệu chuẩn; thiếu thì không ghép.
Camera đầu giữ nguyên số của nó; hiện vật chỉ camera sau thấy nhận số mới tiếp theo. Hàm thuần, không cần GPU.
"""
import copy

import numpy as np

MATCH_M = 0.8
MAX_HEIGHT_M = 8.0


def room_point(spec, u, v, room):
    """Điểm 3D (m) mà tia qua điểm ảnh (u, v) chạm tường / sàn đầu tiên, None nếu không chạm trong phòng."""
    K, R, t = (np.asarray(spec[k], float) for k in ('K', 'R_world_to_camera', 't_world_to_camera'))
    C = -R.T @ t
    d = R.T @ np.linalg.solve(K, np.array([u, v, 1.0]))
    x0, y0, x1, y1 = room
    best = None
    for axis, value in ((0, x0), (0, x1), (1, y0), (1, y1), (2, 0.0)):
        if abs(d[axis]) < 1e-9:
            continue
        s = (value - C[axis]) / d[axis]
        if s <= 0 or (best is not None and s >= best[0]):
            continue
        p = C + s * d
        if x0 - .05 <= p[0] <= x1 + .05 and y0 - .05 <= p[1] <= y1 + .05 and -.05 <= p[2] <= MAX_HEIGHT_M:
            best = (s, p)
    return None if best is None else best[1]


def camera_points(res, spec, room):
    """{id hiện vật của camera: điểm 3D}. Hộp trong result.json theo ảnh làm việc (rộng tối đa 960 px)."""
    video = res.get('video') or {}
    width = float(video.get('width') or spec.get('width') or 960)
    cfg = (((res.get('config') or {}).get('thresholds') or {}).get('CFG') or {})
    scale = width / min(float(cfg.get('width') or 960), width)
    out = {}
    for a in res.get('artifacts') or []:
        box = a.get('box')
        if not box:
            continue
        p = room_point(spec, (box[0] + box[2]) / 2 * scale, (box[1] + box[3]) / 2 * scale, room)
        if p is not None:
            out[a['id']] = p
    return out


def match(results, specs, room):
    """results {cam: result.json}, specs {cam: hiệu chuẩn đã quy về đúng kích thước video} -> {cam: {id riêng: id chung}}."""
    cams = [c for c in results if c in specs and (results[c].get('artifacts') or [])]
    if len(cams) < 2 or not room:
        return {}
    first = cams[0]
    pts = camera_points(results[first], specs[first], room)
    out = {first: {a['id']: a['id'] for a in results[first].get('artifacts') or []}}
    known = {i: [p] for i, p in pts.items()}
    nxt = max(out[first].values(), default=-1) + 1
    for cam in cams[1:]:
        mine = camera_points(results[cam], specs[cam], room)
        pairs = sorted((float(np.linalg.norm(p - np.mean(q, axis=0))), i, g) for i, p in mine.items() for g, q in known.items())
        got, used = {}, set()
        for d, i, g in pairs:
            if d <= MATCH_M and i not in got and g not in used:
                got[i] = g
                used.add(g)
        for a in results[cam].get('artifacts') or []:
            i = a['id']
            if i not in got:
                got[i] = nxt
                nxt += 1
            if i in mine:
                known.setdefault(got[i], []).append(mine[i])
        out[cam] = got
    return out


def renumber_result(res, ids):
    """result.json với id hiện vật đổi theo `ids` (bản sao)."""
    if not ids:
        return res
    res = copy.deepcopy(res)
    new = lambda i: ids.get(i, i)  # noqa: E731
    for a in res.get('artifacts') or []:
        a['id'] = new(a['id'])
    for key in ('events', 'alerts', 'vlm_tasks'):
        for e in res.get(key) or []:
            if e.get('artifact_id') is not None:
                e['artifact_id'] = new(e['artifact_id'])
            if e.get('artifact_ids'):
                e['artifact_ids'] = [new(i) for i in e['artifact_ids']]
    res['artifact_global'] = True
    return res


def renumber_timeline(tl, ids):
    """Cảnh báo trong timeline.json đổi sang id hiện vật chung (để map ghép dùng cùng số)."""
    if not ids:
        return tl
    return dict(tl, alerts=[dict(a, artifact_id=ids.get(a.get('artifact_id'), a.get('artifact_id'))) for a in tl.get('alerts') or []])


def map_exhibits(results, specs, art_map, room, camera_h):
    """Vị trí các hiện vật trên map ghép [{id, label, pedestal, x, y}] để thống kê dừng chân (số hiện vật chung)."""
    if not room or not results or (len(results) > 1 and not art_map):
        return []
    found = {}
    for cam, res in results.items():
        spec = specs.get(cam)
        if spec is None:
            continue
        points = camera_points(res, spec, room)
        K, R, t = (np.asarray(spec[k], float) for k in ('K', 'R_world_to_camera', 't_world_to_camera'))
        ground = K @ np.column_stack([R[:, 0], R[:, 1], t])
        H = camera_h.get(cam)
        for a in res.get('artifacts') or []:
            p = points.get(a['id'])
            if p is None:
                continue
            xy = p[:2]
            if H is not None:
                q = np.asarray(H, float) @ (ground @ np.array([p[0], p[1], 1.0]))
                if abs(q[2]) < 1e-9:
                    continue
                xy = q[:2] / q[2]
            gid = (art_map.get(cam) or {}).get(a['id'], a['id'])
            item = found.setdefault(gid, dict(id=gid, label=a.get('label') or '', pedestal=bool(a.get('pedestal')), points=[]))
            item['points'].append(xy)
    out = []
    for gid in sorted(found):
        item = found[gid]
        x, y = np.mean(item.pop('points'), axis=0)
        out.append(dict(item, x=round(float(x), 3), y=round(float(y), 3)))
    return out
