"""Đối chiếu chéo camera sau khi mỗi camera đã chạy xong engine + VLM (song song hay tuần tự đều được).

Mỗi camera tự quyết định "bất thường" từ góc nhìn của riêng nó, nên cùng một người ở cùng thời điểm có thể bị camera 1 báo `touching`
còn camera 2 (nhìn từ hướng khác) thấy `normal`. Module này gom các quyết định đó theo NGƯỜI CHUNG (ID của map ghép) + thời gian:

  * ANOMALY của một camera được XÁC NHẬN nếu không camera nào khác phản đối (hoặc số camera đồng ý > số phản đối).
  * Có phản đối (camera khác thấy rõ người đó mà VLM nói `normal`, hoặc, chính sách `strict`, camera khác thấy người đó nhưng engine
    không thấy chạm) và không đủ đa số: HẠ xuống CANDIDATE ("cần xem lại"), KHÔNG xoá. Camera khác không nhìn thấy người đó: giữ nguyên.

Chính sách (EXHIBITFLOW_XCAM_POLICY): `strict` (mặc định) | `normal` (chỉ tính phản đối khi camera khác có VLM `normal` rõ ràng) | `off`.
Mọi hàm ở đây là hàm thuần trên dữ liệu (dict), không đụng engine, nên kiểm thử được không cần GPU.
"""
import copy
import os

PAD_S = 1.0                    # dung sai thời gian khi coi hai sự kiện ở hai camera là cùng một lúc
GROUP_GAP_S = 2.0              # hai sự kiện cùng người + cùng hiện vật cách nhau <= 2 s coi là một lần tiếp xúc
MIN_CONF = float(os.environ.get('EXHIBITFLOW_XCAM_MIN_CONF', '0.5'))
SEEN_FRAC = 0.5                # camera khác "nhìn thấy" người đó nếu có hộp của người đó ở >= 50% số khung của sự kiện
ANOMALY, NORMAL, CANDIDATE = 'ANOMALY', 'NORMAL', 'CANDIDATE'


def policy():
    p = (os.environ.get('EXHIBITFLOW_XCAM_POLICY') or 'strict').strip().lower()
    return p if p in ('strict', 'normal', 'off') else 'strict'


def _presence(tl, id_map):
    """{ID chung: mảng chỉ số khung (đã sắp xếp) mà người đó có hộp trong camera này}."""
    out = {}
    for f, v in (tl.get('frames') or {}).items():
        for b in v[0]:
            g = id_map.get(int(b[0]))
            if g is not None:
                out.setdefault(g, []).append(int(f))
    return {g: sorted(set(fs)) for g, fs in out.items()}


def _seen_frac(frames, f0, f1):
    n = f1 - f0 + 1
    if n <= 0:
        return 0.0
    return sum(1 for f in frames if f0 <= f <= f1) / n


def _events(cam, res, id_map):
    out = []
    for e in res.get('events') or []:
        tid = e.get('track_id')
        out.append(dict(cam=cam, id=e.get('event_id'), tid=tid, gid=id_map.get(int(tid)) if tid is not None else None,
                        t0=float(e.get('start_time_s') or 0), t1=float(e.get('end_time_s') or 0),
                        decision=e.get('decision'), art=e.get('artifact_id'), conf=e.get('vlm_confidence')))
    return out


def _strong(e):
    """ANOMALY đủ mạnh để làm bằng chứng cho camera khác: VLM nói chạm và độ tin cậy >= XCAM_MIN_CONF (conf=0.0 là "không chắc")."""
    return e['decision'] == ANOMALY and (e['conf'] is None or e['conf'] >= MIN_CONF)


def _verdict(o, e, evs, pres, fps):
    """Camera `o` nói gì về sự kiện `e` (của camera khác) - cùng người chung, cùng lúc."""
    if e['gid'] is None:
        return 'UNKNOWN_PERSON'
    near = [x for x in evs[o] if x['gid'] == e['gid'] and x['t0'] <= e['t1'] + PAD_S and x['t1'] >= e['t0'] - PAD_S]
    if any(_strong(x) for x in near):
        return ANOMALY
    if any(x['decision'] == NORMAL for x in near):
        return NORMAL
    if near:
        return CANDIDATE                                     # chỉ có ANOMALY yếu / CANDIDATE: không đủ để đồng ý hay phản đối
    if _seen_frac(pres[o].get(e['gid'], []), int(e['t0'] * fps[o]), int(e['t1'] * fps[o])) >= SEEN_FRAC:
        return 'NO_CONTACT'
    return 'NOT_SEEN'


def fuse(cams, pol=None):
    """cams = {cam: dict(tl=timeline.json, res=result.json, id_map={ID cục bộ: ID chung})}.
    Trả ({cam: dict(res=result.json đã sửa, drop=[(tid, artifact_id, f0, f1)] cảnh báo của timeline cần bỏ)}, info) với `info` là tóm tắt
    ghi vào site_result.json. Dữ liệu vào không bị sửa tại chỗ."""
    pol = pol or policy()
    fused = {c: dict(res=copy.deepcopy(v['res']), drop=[]) for c, v in cams.items()}
    info = dict(policy=pol, cameras=sorted(cams), events=[])
    if pol == 'off' or len(cams) < 2:
        return fused, info
    pres = {c: _presence(v['tl'], v['id_map']) for c, v in cams.items()}
    evs = {c: _events(c, v['res'], v['id_map']) for c, v in cams.items()}
    fps = {c: float(v['tl'].get('fps') or 25.0) for c, v in cams.items()}
    todo = []
    for c, lst in evs.items():
        for e in lst:
            if e['decision'] != ANOMALY:
                continue
            verd = {o: _verdict(o, e, evs, pres, fps) for o in cams if o != c}
            agree = 1 + sum(1 for v in verd.values() if v == ANOMALY)
            against = sum(1 for v in verd.values() if v == NORMAL or (pol == 'strict' and v == 'NO_CONTACT'))
            status = ('confirmed' if agree > 1 else 'single_view') if against == 0 else ('confirmed_majority' if agree > against else 'conflict')
            todo.append(dict(e=e, cam=c, verd=verd, agree=agree, against=against, status=status))
    for t in todo:                                           # cùng người + cùng hiện vật + liền kề: một lần tiếp xúc, chung một kết luận
        if t['status'] != 'single_view':
            continue
        for u in todo:
            e, f = t['e'], u['e']
            if u['status'] == 'conflict' and u['cam'] == t['cam'] and f['gid'] == e['gid'] and f['art'] == e['art'] and e['gid'] is not None \
                    and e['t0'] <= f['t1'] + GROUP_GAP_S and e['t1'] >= f['t0'] - GROUP_GAP_S:
                t['status'], t['verd'] = 'conflict', dict(f_ for f_ in u['verd'].items())
                break
    for t in todo:
        e, c, verd, agree, against, status = t['e'], t['cam'], t['verd'], t['agree'], t['against'], t['status']
        others = ', '.join(f'{o}: {v}' for o, v in sorted(verd.items()))
        note = {'confirmed': f'Xác nhận bởi {agree}/{len(cams)} camera', 'confirmed_majority': f'Đa số camera đồng ý ({agree}/{len(cams)})',
                'single_view': 'Chỉ một camera xác nhận', 'conflict': f'Camera khác không xác nhận ({others}): cần xem lại'}[status]
        info['events'].append(dict(cam=c, event_id=e['id'], person=e['gid'], artifact=e['art'], t0=e['t0'], t1=e['t1'], status=status,
                                   agree=agree, against=against, verdicts=verd))
        res = fused[c]['res']
        for ev in res.get('events') or []:
            if ev.get('event_id') != e['id']:
                continue
            ev['xcam'] = dict(status=status, agree=agree, against=against, verdicts=verd, person=e['gid'])
            if status == 'conflict':
                ev['decision'], ev['behavior'] = CANDIDATE, 'uncertain'
            ev['vlm_evidence'] = f"[{note}] {ev.get('vlm_evidence') or ''}".strip()
        if status == 'conflict':
            res['alerts'] = [a for a in (res.get('alerts') or []) if a.get('event_id') != e['id']]
            fused[c]['drop'].append((e['tid'], e['art'], int(e['t0'] * fps[c]) - 1, int(e['t1'] * fps[c]) + 1))
    for c, v in fused.items():                               # đếm lại tóm tắt theo quyết định cuối
        res = v['res']
        s = res.get('summary')
        if isinstance(s, dict):
            by = {}
            for a in res.get('alerts') or []:
                ev = next((x for x in res.get('events') or [] if x.get('event_id') == a.get('event_id')), {})
                b = ev.get('behavior') or a.get('behavior') or 'touching'
                by[b] = by.get(b, 0) + 1
            s['n_alerts'], s['alerts_by_behavior'], s['has_anomaly'] = len(res.get('alerts') or []), by, bool(res.get('alerts'))
    return fused, info


def drop_demoted_alerts(tl, drop):
    """Bỏ khỏi timeline['alerts'] những cảnh báo vừa bị hạ xuống CANDIDATE (khớp theo track, hiện vật và khoảng khung)."""
    if not drop:
        return tl
    keep = []
    for a in tl.get('alerts') or []:
        bad = any(int(a['tid']) == int(t) and (art is None or a.get('artifact_id') in (None, art)) and a['start'] <= f1 and a['end'] >= f0
                  for t, art, f0, f1 in drop)
        if not bad:
            keep.append(a)
    return dict(tl, alerts=keep)
