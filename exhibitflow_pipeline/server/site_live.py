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
