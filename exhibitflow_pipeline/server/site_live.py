"""Chạy TRONG tiến trình con của một camera (xem exhibitflow/cam_worker.py, khoá cfg['setup']).

Tiến trình con không có các bản vá mà `create_app` chỉ áp ở tiến trình chính, nên ở đây áp lại cho đúng một camera:
  * ghi live.mp4 (video đã vẽ khung) và bỏ 3 dòng HUD;
  * đổi nhãn "ID n" trong live.mp4 sang ID CHUNG của map ghép (cùng một người = cùng một số ở mọi camera);
  * ghi phiếu bầu "ID cục bộ -> ID chung" ra <root>/id_votes.json khi chạy xong, để bảng cảnh báo dùng cùng ID với video;
  * áp ngưỡng detector (det_conf) nếu có.
Ngoại hình (gallery) chỉ dùng trong camera này: các tiến trình không chia sẻ bộ nhớ, ghép ID dựa vào vị trí sàn so với map chung.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _apply_det_conf(runner, conf):
    if conf is None:
        return
    det = getattr(getattr(getattr(runner, 'engine', None), 'M', None), 'detector', None)
    if det is None:
        return
    if hasattr(det, 'conf'):
        det.conf = float(conf)
    elif isinstance(getattr(det, 'kw', None), dict):
        det.kw['conf'] = float(conf)


def setup(args, cam):
    import exhibitflow.engine as eng_mod
    import exhibitflow.service as svc
    import site_events

    mapper = (site_events.homography_mapper(args['H'], args['spec']['K'], args['spec'].get('dist')) if args.get('H')
              else site_events.floor_mapper(args['spec']))
    lab = site_events.GlobalIdLabeler(args.get('frames') or [], mapper, args['fps'], tuple(args['wh']), gallery={})
    ids = dict(map={}, votes={})
    root = Path(args['root'])

    orig_run_video = svc.run_video

    def run_video(*a, **k):
        svc.RT['render_live'] = True
        _apply_det_conf(getattr(cam, 'runner', None), args.get('det_conf'))
        try:
            return orig_run_video(*a, **k)
        finally:
            try:
                root.mkdir(parents=True, exist_ok=True)
                (root / 'id_votes.json').write_text(json.dumps(ids['votes']), encoding='utf-8')
            except Exception:                                  # noqa: BLE001
                pass

    svc.run_video = run_video

    orig_render = eng_mod.render_live

    def render_live(ctx, p, *a, **k):
        ids['map'] = {}
        try:
            h, w = p.work.shape[:2]
            ids['map'] = lab.labels(p.idx, [dict(id=r['id'], box=r['box']) for r in (p.tracks or [])], (w, h), image=p.work)
            for tid, gid in ids['map'].items():
                v = ids['votes'].setdefault(str(int(tid)), {})
                v[str(gid)] = v.get(str(gid), 0) + 1
        except Exception:                                      # noqa: BLE001 - lỗi ghép ID không được làm hỏng video
            ids['map'] = {}
        return orig_render(ctx, p, *a, **k)

    eng_mod.render_live = render_live

    orig_put = eng_mod.put_label

    def put_label(img, text, *a, **k):
        if isinstance(text, str) and text.startswith(('frame ', 'pipeline ', 'VLM queue')):
            return None
        if ids['map'] and isinstance(text, str):
            m = re.match(r'ID (\d+)(.*)$', text)
            if m and int(m.group(1)) in ids['map']:
                text = f'ID {ids["map"][int(m.group(1))]}' + m.group(2)
        return orig_put(img, text, *a, **k)

    eng_mod.put_label = put_label


def paint_promoted(live, tl, add, id_map=None, pad_frames=0):
    """Vẽ lại khung ĐỎ + nhãn "ID n | TOUCHING" lên live.mp4 cho người bị lan truyền từ camera khác.

    live.mp4 do engine vẽ trước khi đối chiếu chéo nên không có khung đỏ cho người này; ở đây vẽ đè lên đúng các khung của
    cảnh báo được lan sang (`add` = site_fusion.fuse()[..]['add']), dùng hộp trong timeline (toạ độ 0-1). Trả True nếu đã ghi lại video.
    Lỗi ở bất kỳ bước nào giữ nguyên video cũ."""
    import cv2
    live = Path(live)
    if not add or not live.is_file():
        return False
    spans = []
    for a in add:
        for s, e in a.get('spans') or [[a['start'], a['end']]]:
            spans.append((int(a['tid']), int(s) - pad_frames, int(e) + pad_frames, str(a.get('behavior') or 'touching').upper()))
    frames = tl.get('frames') or {}
    tmp = live.with_name(live.stem + '.fused_tmp.mp4')
    cap = cv2.VideoCapture(str(live))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or float(tl.get('fps') or 25.0)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        if not out.isOpened():
            return False
        red, n = (0, 0, 255), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for b in (frames.get(str(n)) or [[]])[0]:
                tid = int(b[0])
                hit = next((s for s in spans if s[0] == tid and s[1] <= n <= s[2]), None)
                if hit is None:
                    continue
                x1, y1, x2, y2 = int(b[1] * W), int(b[2] * H), int(b[3] * W), int(b[4] * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), red, 3)
                gid = (id_map or {}).get(tid, tid)
                txt = f'ID {gid} | {hit[3]}'
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(0, y1 - th - 8)
                cv2.rectangle(img, (x1, top), (x1 + tw + 8, top + th + 8), (0, 0, 0), -1)
                cv2.putText(img, txt, (x1 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1, cv2.LINE_AA)
            out.write(img)
            n += 1
        out.release()
    finally:
        cap.release()
    if n == 0:
        tmp.unlink(missing_ok=True)
        return False
    from exhibitflow.common import encode_browser_mp4
    encode_browser_mp4(tmp, live)                              # tmp -> live (H.264 nếu có ffmpeg; lỗi thì giữ mp4v)
    return True
