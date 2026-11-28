"""Cảnh báo "chạm" chỉ tính từ lúc người đó ĐỨNG LẠI (hậu xử lý cho bản web, không đụng engine).

Bộ lọc chạm 2D coi tay che lên khung tranh trong ảnh là chạm, nên người chỉ ĐI NGANG cũng thành sự kiện; clip gửi VLM lại chứa
cả cú chạm xảy ra sau đó nên VLM gán "touching" cho cả đoạn đi ngang. Ở đây dùng hộp người trong timeline.json:
  * đoạn người đó còn đang đi (kể từ đầu sự kiện)  -> CANDIDATE ("cần xác nhận");
  * từ lúc đứng lại đủ lâu                          -> giữ ANOMALY;
  * suốt sự kiện không lúc nào đứng lại             -> cả sự kiện là CANDIDATE.
Chỉ áp cho `touching`: cầm / lấy trộm vốn diễn ra khi đang đi. Hàm thuần trên dict, không cần GPU.
Tắt: EXHIBITFLOW_STILL_GATE=0. Ngưỡng: EXHIBITFLOW_STILL_SPEED (số lần chiều cao người đi được mỗi giây, mặc định 0.08).
"""
import copy
import os

WINDOW_S = 1.0        # đo tốc độ trên cửa sổ 1 giây
MIN_STILL_S = 0.7     # phải đứng liên tục ít nhất chừng này
MIN_PIECE_S = 0.5     # phần "đang đi" ngắn hơn mức này không tách thành sự kiện riêng
NOTE = '[Người còn đang đi lúc này: cần xác nhận]'


def enabled():
    return os.environ.get('EXHIBITFLOW_STILL_GATE', '1') != '0'


def max_speed():
    return float(os.environ.get('EXHIBITFLOW_STILL_SPEED') or 0.08)


def _track(tl, tid):
    """{khung: (x chân, y chân, chiều cao)} theo điểm ảnh của một track."""
    W, H = float(tl.get('width') or 1), float(tl.get('height') or 1)
    out = {}
    for f, v in (tl.get('frames') or {}).items():
        for b in v[0]:
            if int(b[0]) == tid:
                out[int(f)] = ((b[1] + b[3]) / 2 * W, b[4] * H, (b[4] - b[2]) * H)
    return out


def still_from(tl, tid, f0, f1, speed=None):
    """Khung đầu tiên trong [f0, f1] bắt đầu một đoạn đứng yên đủ lâu. None: không lúc nào đứng yên. -1: thiếu dữ liệu để kết luận."""
    fps = float(tl.get('fps') or 25.0)
    tr = _track(tl, tid)
    half, need = int(WINDOW_S * fps / 2), max(1, int(MIN_STILL_S * fps))
    speed = max_speed() if speed is None else speed
    run, known = None, 0
    for f in range(f0, f1 + 1):
        near = [k for k in range(f - half, f + half + 1) if k in tr]
        if len(near) <= half:                                  # track hụt quá nhiều quanh khung này
            run = None
            continue
        known += 1
        a, b = near[0], near[-1]
        h = sorted(tr[k][2] for k in near)[len(near) // 2] or 1.0
        v = ((tr[b][0] - tr[a][0]) ** 2 + (tr[b][1] - tr[a][1]) ** 2) ** .5 / h / (max(1, b - a) / fps)
        if v > speed:
            run = None
            continue
        run = f if run is None else run
        if f - run + 1 >= need:
            return run
    return None if known >= need else -1


def apply(res, tl):
    """(result.json, timeline.json) -> bản sao đã áp quy tắc đứng yên. Dữ liệu vào không bị sửa."""
    if not enabled() or not res or not tl:
        return res, tl
    res = copy.deepcopy(res)
    tl = dict(tl, alerts=[dict(a) for a in tl.get('alerts') or []])
    fps = float(tl.get('fps') or (res.get('video') or {}).get('fps') or 25.0)
    events = res.get('events') or []
    ids = [e.get('event_id') for e in events]
    alerts = {a.get('event_id'): a for a in res.get('alerts') or []}
    out, changed = [], False
    for ev in events:
        if ev.get('decision') != 'ANOMALY' or (ev.get('behavior') or 'touching') != 'touching' or ev.get('track_id') is None:
            out.append(ev)
            continue
        al = alerts.get(ev.get('event_id')) or {}
        e0 = int(ev.get('start_frame', round(float(ev.get('start_time_s') or 0) * fps)))
        e1 = int(ev.get('end_frame', round(float(ev.get('end_time_s') or 0) * fps)))
        a0 = int(al.get('alarm_start_frame') if al.get('alarm_start_frame') is not None else e0)
        a1 = int(al.get('alarm_end_frame') if al.get('alarm_end_frame') is not None else e1)
        tid = int(ev['track_id'])
        onset = still_from(tl, tid, a0, a1)
        if onset == -1 or onset == a0 == e0:
            out.append(ev)
            continue
        changed = True
        mine = [a for a in tl['alerts'] if int(a['tid']) == tid and a.get('artifact_id') in (None, ev.get('artifact_id'))
                and a['start'] <= a1 and a['end'] >= a0]
        why = f"{NOTE} {ev.get('vlm_evidence') or ''}".strip()
        if onset is None:                                      # đi suốt: cả sự kiện chỉ là "cần xác nhận"
            ev.update(decision='CANDIDATE', behavior='uncertain', vlm_evidence=why, alarm_spans=[])
            alerts.pop(ev.get('event_id'), None)
            tl['alerts'] = [a for a in tl['alerts'] if not any(a is m for m in mine)]
            out.append(ev)
            continue
        t = onset / fps
        if t - e0 / fps > MIN_PIECE_S:                         # tách phần đang đi thành một sự kiện "cần xác nhận"
            head = copy.deepcopy(ev)
            head.update(decision='CANDIDATE', behavior='uncertain', vlm_evidence=why, alarm_spans=[],
                        end_frame=onset, end_time_s=t)
            new = max(ids) + 1 if all(isinstance(i, int) for i in ids) else f"{ev.get('event_id')}-s"
            ids.append(new)
            ev.update(start_frame=onset, start_time_s=t)
            if al:
                alerts.pop(ev.get('event_id'), None)
                al['event_id'] = new
                alerts[new] = al
            head_id, ev['event_id'] = ev.get('event_id'), new
            head['event_id'] = head_id
            out.append(head)
        ev['alarm_spans'] = [dict(start_frame=onset, end_frame=a1, start_time_s=t, end_time_s=a1 / fps)]
        if al:
            al.update(alarm_start_frame=onset, alarm_start_time_s=t)
        for a in mine:
            a['start'] = max(int(a['start']), onset)
        out.append(ev)
    if not changed:
        return res, tl
    res['events'] = out
    res['alerts'] = [a for a in alerts.values()]
    s = res.get('summary')
    if isinstance(s, dict):
        by = {}
        for a in res['alerts']:
            by[a.get('behavior') or 'touching'] = by.get(a.get('behavior') or 'touching', 0) + 1
        s['n_alerts'], s['alerts_by_behavior'], s['has_anomaly'] = len(res['alerts']), by, bool(res['alerts'])
    return res, tl
