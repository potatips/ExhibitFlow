"""ExhibitFlow — server nối giao diện web (thư mục `web/`) với engine realtime của dashboard Hưng.

Không sửa file nào của `dashboard/`: dùng lại `State` + `PipelineRunner` (model nạp sẵn một lần, chạy nhiều video)
và trả lời bằng API "job" mà web đang dùng:

    GET  /api/health                      kiểm tra sống + trạng thái nạp model
    GET  /api/state                       trạng thái dashboard (dùng cho launcher chờ nạp model)
    GET  /api/system                      CPU / RAM / GPU
    POST /api/jobs                        multipart: video (upload) | source (đường dẫn trên server / URL) + options -> {job_id}
    GET  /api/jobs/{id}                   {status, stage, stages, progress, key, error}
    POST /api/jobs/{id}/cancel            dừng (bỏ kết quả)
    POST /api/jobs/{id}/finish            camera: dừng thu và xử lý nốt để có kết quả
    POST /api/multi-jobs                  nhiều camera (2-8): song song (options.multi_parallel, tối đa multi_max_parallel camera cùng lúc,
                                          mỗi camera một tiến trình riêng) hoặc tuần tự. Form gồm options (JSON), cameras (JSON [{name, source}]),
                                          và video_<i> (file upload của camera i) -> {multi_id}
    GET  /api/multi-jobs/{id}             trạng thái từng camera (job_id, status), cảnh báo gộp (có nhãn camera), camera đang chạy (cur)
    POST /api/multi-jobs/{id}/finish      dừng: job con đang chạy xử lý nốt, các camera còn lại bỏ qua
    POST /api/camera/session              tạo phiên webcam của trình duyệt -> {sid}; rồi mở WebSocket /api/camera/{sid}/push gửi từng khung JPEG
                                          và tạo job với source=push://<sid>?fps=15. source=rtsp://... để đọc camera IP
    GET  /api/jobs/{id}/live              khung hình mới nhất (JPEG, đã vẽ khung người / hiện vật)
    GET  /api/jobs/{id}/live/events       meta + trạng thái + cảnh báo đang chạy (?after=<seq>)
    GET  /api/jobs/{id}/result            result.json của pipeline
    GET  /api/jobs/{id}/video             live.mp4 (có khung); ?kind=original là video gốc
    GET  /api/jobs/{id}/files/<path>      file trong thư mục kết quả (ảnh bằng chứng)

Chạy:
    python server/web_server.py --port 8000             # pipeline thật (cần GPU + trọng số)
    python server/web_server.py --port 8000 --demo      # dữ liệu giả của dashboard (không cần GPU), để thử giao diện
Server cũng phục vụ giao diện `web/` tại `/` (cùng cổng): mở http://<ip>:<port>/ là dùng được, không cần URL backend riêng.
Biến môi trường: EXHIBITFLOW_WEB_DIR (thư mục giao diện, mặc định <repo>/web), EF_WEIGHTS_DIR (thư mục trọng số), EXHIBITFLOW_SOURCE_ROOTS (thư mục được phép đọc, ngăn bằng dấu ':'
trên Linux), EXHIBITFLOW_MAX_MB (giới hạn upload, mặc định 600), EXHIBITFLOW_SAVE_VIDEO=0 để không ghi live.mp4.
"""
import argparse
import copy
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # trước khi torch dùng CUDA

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # site_runner.py cùng thư mục

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.runner import DemoRunner, PipelineRunner, State  # noqa: E402
from exhibitflow import cam_worker  # noqa: E402
from exhibitflow import multicam  # noqa: E402
from exhibitflow import source as efsource  # noqa: E402
from exhibitflow import tuning as _tuning  # noqa: E402
import site_events  # noqa: E402
import site_fusion  # noqa: E402
import site_parallel  # noqa: E402
import site_runner  # noqa: E402
import still_gate  # noqa: E402

ALLOWED = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def sniff_ext(head):
    """Đoán đuôi video từ vài chục byte đầu (khi tên file / đường dẫn không có đuôi). Trả None nếu không giống video."""
    if len(head) >= 12 and head[4:8] == b'ftyp':
        return '.mov' if head[8:12] == b'qt  ' else '.mp4'
    if head[:4] == bytes.fromhex('1a45dfa3'):
        return '.mkv'                              # Matroska / WebM
    if head[:4] == b'RIFF' and head[8:12] == b'AVI ':
        return '.avi'
    return None


MAX_BYTES = int(os.environ.get('EXHIBITFLOW_MAX_MB', '600')) * 1024 * 1024
STAGES = ['Khởi động & nạp model', 'Tách hiện vật', 'Nạp VLM', 'Phân tích video', 'Ghi kết quả']
SOURCE_ROOTS = [Path(p).resolve() for p in (os.environ.get('EXHIBITFLOW_SOURCE_ROOTS')
                                            or os.pathsep.join(['/workspace', '/kaggle/input', '/kaggle/working', '/content', str(PROJ)])).split(os.pathsep) if p]
WEB_DIR = Path(os.environ.get('EXHIBITFLOW_WEB_DIR') or PROJ.parent / 'web')   # giao diện web, phục vụ chung cổng với API
# Cấu hình pipeline web được phép đặt (khoá + kiểu) = các mục tương ứng trong Settings của dashboard cũ
PIPE_MODES = ('tracking', 'tracking_seg', 'full')
PIPE_OPTS = dict(pipeline_mode=lambda v: v if v in PIPE_MODES else 'full', tracker_process=bool, hand_process=bool, pose_process=bool, pose_gpu_preproc=bool, hand_batch=bool, drop_frames=bool, limit_cpu_threads=bool, alarm_show_sec=float,
                  tracker_threads=int, hand_threads=int, pose_threads=int, queue_size=int, pose_gate_norm_dist=float,
                  multi_parallel=bool, multi_max_parallel=int, tuning=_tuning.validate_tuning)
NW = 960


def pipe_of(opts):
    """Cấu hình pipeline lấy từ options của request (chỉ các khoá trong PIPE_OPTS)."""
    return {k: t(opts[k]) for k, t in PIPE_OPTS.items() if opts.get(k) is not None}                                   # web vẽ hộp hiện vật theo khung rộng 960 px
MAKE_MAP = os.environ.get('EXHIBITFLOW_MAKE_MAP', '1') != '0'   # tạo map 2D từ video (multicam-2d-floorplan-tracker), mất ~15-25 s/video


def det_conf_state(web_cfg):
    """Ngưỡng detector: đang đặt (None = chưa đặt), mặc định của engine, và giá trị thực sự đang dùng."""
    from exhibitflow import settings as efs
    default = float(efs.MODELS.get('detector', {}).get('conf', 0.01))
    cur = web_cfg.get('det_conf')
    return dict(det_conf=cur, det_conf_default=default, det_conf_effective=default if cur is None else cur)


def parse_det_conf(v):
    """Ngưỡng phát hiện người từ web: số trong [0.001, 0.99], không hợp lệ thì None (giữ mặc định)."""
    try:
        c = float(v)
    except (TypeError, ValueError):
        return None
    return c if 0.001 <= c <= 0.99 else None


def apply_det_conf(runner, conf):
    """Đặt ngưỡng cho detector đang nạp. YOLOX đọc `self.conf` ở mỗi lần detect, nên có hiệu lực ngay, không phải nạp lại model."""
    if conf is None:
        return False
    det = getattr(getattr(getattr(runner, 'engine', None), 'M', None), 'detector', None)
    if det is None:
        return False
    if hasattr(det, 'conf'):
        det.conf = float(conf)
    elif isinstance(getattr(det, 'kw', None), dict):
        det.kw['conf'] = float(conf)                  # detector kiểu ultralytics
    else:
        return False
    return True


def create_app(data_dir, demo=False, preload=True, mode=None, local=False):
    state = State(data_dir)
    runner = (DemoRunner if demo else PipelineRunner)(state, local=local)
    app_site_ids = {}                                   # {'labeler': GlobalIdLabeler|None, 'map': {id cục bộ: ID chung}}; do site_events_stage đặt
    web_cfg = dict(det_conf=parse_det_conf(os.environ.get('EXHIBITFLOW_DET_CONF')))   # ngưỡng phát hiện người (None = giữ mặc định 0.01 của engine)
    if mode:
        state.save_settings(dict(pipeline_mode=mode))
    # mặc định bật khi khởi động (preload nạp model đúng cấu hình ngay, không phải nạp lại ở lần chạy đầu); web gửi lại mỗi lần chạy
    state.save_settings(dict(tracker_process=True, hand_process=True, pose_process=True, hand_batch=True, drop_frames=True))
    if not demo and os.environ.get('EXHIBITFLOW_SAVE_VIDEO', '1') != '0':
        # Engine.run đặt RT['render_live']=False trước mỗi lần chạy, nên bật lại ngay trước run_video (không sửa file của Hưng):
        # engine ghi live.mp4 (video đã vẽ khung người / hiện vật) để web xem lại sau khi chạy xong.
        import exhibitflow.service as svc
        _orig_run_video = svc.run_video

        def _run_video_with_live(*a, **k):
            svc.RT['render_live'] = True
            if apply_det_conf(runner, web_cfg.get('det_conf')):
                state.log(f"Ngưỡng phát hiện người (detector conf) = {web_cfg['det_conf']}")
            return _orig_run_video(*a, **k)

        svc.run_video = _run_video_with_live

        # engine vẽ 3 dòng thông số (frame / FPS / VLM queue) vào live.mp4: bỏ chúng đi, giữ nguyên khung người, ID và nhãn
        import exhibitflow.engine as eng_mod
        _orig_put_label = eng_mod.put_label

        # nhiều camera đã hiệu chuẩn: nhãn "ID n" trong live.mp4 đổi sang ID chung của map (cùng một người = cùng một số ở mọi camera)
        site_ids = {'labeler': None, 'map': {}}
        app_site_ids.update(site_ids)
        _orig_render_live = eng_mod.render_live

        def _render_live_global_ids(ctx, p, *a, **k):
            lab = app_site_ids.get('labeler')
            app_site_ids['map'] = {}
            if lab is not None:
                try:
                    h, w = p.work.shape[:2]
                    app_site_ids['map'] = lab.labels(p.idx, [dict(id=r['id'], box=r['box']) for r in (p.tracks or [])], (w, h), image=p.work)
                    for tid, gid in app_site_ids['map'].items():          # phiếu bầu để bảng cảnh báo dùng cùng ID chung với video
                        v = app_site_ids.setdefault('votes', {}).setdefault(int(tid), {})
                        v[gid] = v.get(gid, 0) + 1
                except Exception:                      # noqa: BLE001 - không để lỗi ghép ID làm hỏng video
                    app_site_ids['map'] = {}
            return _orig_render_live(ctx, p, *a, **k)

        eng_mod.render_live = _render_live_global_ids

        def _put_label_no_hud(img, text, *a, **k):
            if isinstance(text, str) and text.startswith(('frame ', 'pipeline ', 'VLM queue')):
                return None
            gm = app_site_ids.get('map')
            if gm and isinstance(text, str):
                m = re.match(r'ID (\d+)(.*)$', text)
                if m and int(m.group(1)) in gm:
                    text = f'ID {gm[int(m.group(1))]}' + m.group(2)
            return _orig_put_label(img, text, *a, **k)

        eng_mod.put_label = _put_label_no_hud
    if preload:
        runner.preload()                   # nạp model một lần ngay khi server lên

    app = FastAPI(title='ExhibitFlow Web API')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
    jobs, cur, multis = {}, {'id': None}, {}
    uploads = state.root / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ tiện ích
    def check_ext(ext):
        if ext not in ALLOWED:
            raise HTTPException(400, f'Định dạng không hỗ trợ ({ext or "không có đuôi"}). Dùng: {", ".join(sorted(ALLOWED))}')

    async def save_upload(video):
        """Ghi file upload vào thư mục uploads -> (đường dẫn, tên gốc)."""
        ext, head = Path(video.filename).suffix.lower(), b''
        if ext not in ALLOWED:                 # tên file không có đuôi (hoặc đuôi lạ): xem nội dung
            head = await video.read(1 << 16)
            ext = sniff_ext(head) or ext
        check_ext(ext)
        vpath, size = uploads / f'{uuid.uuid4().hex[:8]}{ext}', 0
        with open(vpath, 'wb') as f:
            f.write(head)
            size = len(head)
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    f.close()
                    vpath.unlink(missing_ok=True)
                    raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                f.write(chunk)
        return vpath, video.filename

    def resolve_source(src, dst_dir):
        """URL http(s) (tải về) hoặc đường dẫn file / thư mục trên server (chỉ trong SOURCE_ROOTS)."""
        if re.match(r'^(rtsps?|rtmps?)://', src, re.I):          # camera IP: file mô tả .efstream thay cho đường dẫn video
            return efsource.write_stream_file(dst_dir / f'cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}', url=src), 'Camera ' + efsource._mask(src)
        m = re.match(r'^push://([\w-]+)(?:\?fps=(\d+))?$', src)
        if m:                                                      # webcam của trình duyệt (khung đẩy qua WebSocket)
            if m.group(1) not in efsource.PUSH:
                raise HTTPException(400, 'Phiên webcam không tồn tại hoặc đã đóng: mở lại camera rồi thử lại.')
            return efsource.write_stream_file(dst_dir / f'cam_{uuid.uuid4().hex[:8]}{efsource.STREAM_EXT}', push=m.group(1),
                                              fps=int(m.group(2) or 15)), 'Camera máy'
        if re.match(r'^https?://', src, re.I):
            ext = Path(urllib.parse.urlparse(src).path).suffix.lower() or '.mp4'
            check_ext(ext)
            dst, size = dst_dir / f'{uuid.uuid4().hex[:8]}{ext}', 0
            try:
                with urllib.request.urlopen(src, timeout=30) as r, open(dst, 'wb') as f:
                    while chunk := r.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise HTTPException(413, f'Video quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                        f.write(chunk)
            except HTTPException:
                raise
            except Exception as e:             # noqa: BLE001
                raise HTTPException(400, f'Không tải được URL: {e}')
            return dst, Path(urllib.parse.urlparse(src).path).name or dst.name
        p = Path(src).expanduser().resolve()
        if not any(r == p or r in p.parents for r in SOURCE_ROOTS):
            raise HTTPException(403, 'Đường dẫn nằm ngoài thư mục cho phép: ' + ', '.join(map(str, SOURCE_ROOTS)))
        if p.is_dir():
            vids = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED)
            if not vids:
                raise HTTPException(404, f'Không có video trong thư mục {p}')
            p = vids[0]
        if not p.is_file():
            raise HTTPException(404, f'Không thấy file {p}')
        ext = p.suffix.lower()
        if ext not in ALLOWED:                     # đường dẫn không có đuôi: xem nội dung file
            with open(p, 'rb') as fh:
                ext = sniff_ext(fh.read(64)) or ext
        check_ext(ext)
        return p, p.name

    def video_info(path):
        if efsource.is_stream(path):             # camera: chưa mở (mở ở luồng pipeline); kích thước lấy từ efsource.INFO khi có
            return dict(fps=25.0, n_frames=0, width=0, height=0)
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            info = dict(fps=fps if fps > 1 else 25.0, n_frames=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)),
                        width=max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)), height=max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
            cap.release()
            return info
        except Exception:                      # noqa: BLE001
            return dict(fps=25.0, n_frames=0, width=0, height=0)

    def vinfo_of(j):
        """Thông tin video của job; với camera thì cập nhật từ kích thước / fps pipeline đã đo ở lần mở đầu."""
        if efsource.is_stream(j['path']) and (m := efsource.INFO.get(str(j['path']))):
            j['vinfo'] = dict(fps=m['fps'], n_frames=0, width=m['width'], height=m['height'])
        return j['vinfo']

    def get_job(jid):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, 'Không có job này')
        return j

    def result_dir(j):
        """Thư mục kết quả (chứa result.json) của job: tìm result.json mới nhất sinh ra sau khi job bắt đầu."""
        if j.get('rdir') is not None:
            return j['rdir']
        found = [p for p in ((j.get('root') or state.root) / 'results').rglob('result.json') if p.stat().st_mtime >= j['t0'] - 2]
        if not found:
            return None
        j['rdir'] = max(found, key=lambda p: p.stat().st_mtime).parent
        return j['rdir']

    def gated_result(j, rd):
        """result.json đã áp quy tắc đứng yên (still_gate.py) bằng timeline.json của cùng lượt chạy."""
        res = json.loads((rd / 'result.json').read_text(encoding='utf-8'))
        try:
            runs = sorted(((j.get('root') or state.root) / 'runs').glob(f'*_{rd.name}/timeline.json'), key=lambda p: p.stat().st_mtime)
            if runs:
                res = still_gate.apply(res, json.loads(runs[-1].read_text(encoding='utf-8')))[0]
        except Exception as e:                                 # noqa: BLE001 - lỗi hậu xử lý: giữ kết quả gốc
            state.log(f'Không áp được quy tắc đứng yên: {e}')
        return res

    def finished(j):
        """Job đã có kết quả cuối (trạng thái chốt, không đổi nữa)."""
        return j.get('final') is not None

    def remote_view(j):
        """Job con chạy trong tiến trình riêng (multi-camera song song): trạng thái đọc từ tiến trình đó."""
        if finished(j):
            return j['final']
        try:
            v = cam_worker.view_of(j['remote'].snap(), cancelled=j['cancelled'])
        except Exception as e:  # noqa: BLE001 - tiến trình con chết / lỗi IPC: chỉ camera này lỗi
            v = dict(status='error', stage=0, progress=0, error=f'Tiến trình camera lỗi: {e}'[:300])
        if v['status'] in ('done', 'error', 'cancelled'):
            j['final'] = v
        return v

    def remote_frame(j, last):
        try:
            return j['remote'].frame(last)
        except Exception:  # noqa: BLE001
            return None

    def stop_job(j):
        """Dừng một job: job con (multi song song) hoặc pipeline chung của server."""
        if j.get('remote') is not None:
            j['remote'].stop()
        else:
            runner.stop()

    def view(j):
        """Chuyển trạng thái `State` của dashboard sang trạng thái job mà web hiểu."""
        if j.get('remote') is not None:
            return remote_view(j)
        if finished(j):
            return j['final']
        if cur['id'] != j['id']:                # job cũ bị job mới thay: coi như đã dừng
            j['final'] = dict(status='cancelled', stage=0, progress=0, error=None)
            return j['final']
        with state.lock:
            st, phase, msg, fi, nf = state.status, state.phase, state.message, state.frame_idx, state.n_frames
        alive = runner.running()
        if j['cancelled'] and not alive:
            j['final'] = dict(status='cancelled', stage=4, progress=0, error=None)
        elif st == 'error':
            j['final'] = dict(status='error', stage=0, progress=0, error=msg or 'Pipeline lỗi')
        elif st == 'done' and not alive:
            j['final'] = dict(status='done', stage=4, progress=1.0, error=None)
        elif st in ('idle', 'loading'):
            stage = 1 if 'map' in (phase or '').lower() else 0
            return dict(status='running', stage=stage, progress=0.0, error=None)
        elif st == 'running':
            return dict(status='running', stage=3, progress=round(fi / nf, 3) if nf else 0.0, error=None)
        else:                                   # finishing / done nhưng luồng chưa tắt
            return dict(status='running', stage=4, progress=0.5, error=None)
        return j['final']

    # ------------------------------------------------------------------ cơ bản
    @app.get('/api/health')
    def health():
        with state.lock:
            ms = state.models_state
        return {'ok': True, 'version': 'ef-web-1', 'demo': demo, 'models': ms, 'jobs': len(jobs), **det_conf_state(web_cfg)}

    @app.post('/api/config')
    async def set_config(request: Request):
        """Đặt ngưỡng phát hiện người (detector conf) lúc đang chạy: {"det_conf": 0.3} hoặc null = mặc định của engine.
        Áp ngay cho model đang nạp và cho mọi lượt phân tích sau (kể cả tiến trình camera song song)."""
        try:
            body = await request.json()
        except Exception:                                  # noqa: BLE001
            raise HTTPException(400, 'Body phải là JSON')
        v = body.get('det_conf') if isinstance(body, dict) else None
        if v is not None and parse_det_conf(v) is None:
            raise HTTPException(400, 'det_conf phải nằm trong [0.001, 0.99] hoặc null')
        web_cfg['det_conf'] = parse_det_conf(v)
        applied = apply_det_conf(runner, web_cfg['det_conf'])
        return {**det_conf_state(web_cfg), 'applied_now': applied}

    @app.get('/api/state')
    def get_state():
        return state.snapshot()

    @app.get('/api/tuning/schema')
    def get_tuning_schema():
        return _tuning.schema()

    def cpu_cores():
        """Số nhân mà tiến trình thực sự dùng được: min(affinity, hạn mức cgroup). Container Vast: os.cpu_count() thường báo cả máy chủ."""
        info = {'logical': os.cpu_count() or 1}
        try:
            info['usable'] = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            info['usable'] = info['logical']
        quota = None
        try:
            q, per = open('/sys/fs/cgroup/cpu.max').read().split()[:2]          # cgroup v2: "<quota> <period>" hoặc "max <period>"
            quota = None if q == 'max' else int(q) / int(per)
        except Exception:                      # noqa: BLE001
            try:                               # cgroup v1
                q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
                per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
                quota = q / per if q > 0 else None
            except Exception:                  # noqa: BLE001
                pass
        if quota:
            info['quota'] = round(quota, 1)
            info['usable'] = max(1, min(info['usable'], int(quota)))
        return info

    @app.get('/api/system')
    def system():
        out = {'cpu': None, 'ram': None, 'gpus': [], 'cores': cpu_cores()}
        try:
            la, n = os.getloadavg(), os.cpu_count() or 1
            out['cpu'] = {'load': round(la[0], 2), 'pct': round(min(100.0, la[0] / n * 100), 1)}
        except (AttributeError, OSError):
            try:
                import psutil
                out['cpu'] = {'load': 0, 'pct': psutil.cpu_percent()}
            except Exception:                  # noqa: BLE001
                pass
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    info[k] = int(v.split()[0])
            tot = info['MemTotal'] / 1048576
            out['ram'] = {'used_gb': round(tot - info['MemAvailable'] / 1048576, 1), 'total_gb': round(tot, 1)}
        except Exception:                      # noqa: BLE001
            try:
                import psutil
                vm = psutil.virtual_memory()
                out['ram'] = {'used_gb': round(vm.used / 1e9, 1), 'total_gb': round(vm.total / 1e9, 1)}
            except Exception:                  # noqa: BLE001
                pass
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
            for line in r.stdout.strip().splitlines():
                name, util, mu, mt = [x.strip() for x in line.split(',')]
                out['gpus'].append({'name': name, 'util': float(util), 'mem_used_gb': round(float(mu) / 1024, 1),
                                    'mem_total_gb': round(float(mt) / 1024, 1)})
        except Exception:                      # noqa: BLE001
            pass
        return out

    def demo_map():
        """--demo: tạo ảnh map lưới giả để thử khối Map 2D (pipeline thật tạo map bằng multicam-2d-floorplan-tracker)."""
        import cv2
        import numpy as np
        real = os.environ.get('EXHIBITFLOW_DEMO_MAP')           # tuỳ chọn: nạp ảnh map có sẵn (vd map do mapgen tạo) để thử giao diện
        if real and Path(real).is_file():
            out = state.root / 'maps' / 'demo'
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(real, out / 'map.png')
            with state.lock:
                state.map_path = out / 'map.png'
                state.map_version += 1
            return
        img = np.full((600, 800, 3), (34, 30, 26), np.uint8)
        for x in range(0, 800, 50):
            cv2.line(img, (x, 0), (x, 600), (52, 46, 40), 1)
        for y in range(0, 600, 50):
            cv2.line(img, (0, y), (800, y), (52, 46, 40), 1)
        cv2.rectangle(img, (20, 20), (780, 580), (170, 165, 160), 3)
        out = state.root / 'maps' / 'demo'
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / 'map.png'), img)
        with state.lock:
            state.map_path = out / 'map.png'
            state.map_version += 1

    # ------------------------------------------------------------------ job
    @app.post('/api/jobs')
    async def create_job(video: UploadFile | None = File(None), options: str = Form('{}'), source: str = Form('')):
        try:
            opts = json.loads(options or '{}')
        except json.JSONDecodeError:
            opts = {}
        try:
            pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy video khác, đợi xong hoặc bấm Dừng.')
        if runner.loading():
            raise HTTPException(409, 'Model đang nạp (lần đầu mất vài phút), thử lại sau.')
        if video is not None and video.filename:
            vpath, name = await save_upload(video)
        elif source.strip():
            vpath, name = resolve_source(source.strip(), uploads)
        else:
            raise HTTPException(400, 'Cần gửi video hoặc đường dẫn / URL')
        return {'job_id': start_job(vpath, name, opts, realtime=bool(opts.get('realtime')))}

    def start_job(vpath, name, opts, realtime):
        """Khởi chạy một lần pipeline cho một nguồn; dùng chung cho job đơn và job con của multi-camera."""
        # realtime: đọc đúng nhịp fps như camera, bỏ frame nếu không kịp; theo lô (Video): xử lý mọi frame
        try:
            pipe = pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if efsource.is_stream(vpath):             # camera: thời lượng tối đa (phút) + không giả lập realtime
            from exhibitflow import settings as efs
            efs.RT['stream_max_sec'] = max(10.0, float(opts.get('camera_max_min') or 30) * 60)
        state.save_settings(dict(pipe, realtime=realtime, make_map=MAKE_MAP, map_image='auto'))
        try:
            runner.start(str(vpath))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = dict(id=jid, path=Path(vpath), name=name, t0=time.time(), cancelled=False, final=None, rdir=None,
                         realtime=realtime, vinfo=video_info(vpath))
        cur['id'] = jid
        if demo:
            demo_map()
        return jid

    # ------------------------------------------------------------------ multi-camera (chạy tuần tự, xem exhibitflow/multicam.py)
    def get_multi(mid):
        mr = multis.get(mid)
        if mr is None:
            raise HTTPException(404, 'Không có lần multi-camera này')
        return mr

    @app.post('/api/multi-jobs')
    async def create_multi(request: Request):
        form = await request.form()
        try:
            opts = json.loads(form.get('options') or '{}')
            specs = json.loads(form.get('cameras') or '[]')
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(400, 'Danh sách camera không hợp lệ')
        try:
            pipe_opts = pipe_of(opts)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if not isinstance(specs, list) or not (multicam.MIN_CAMS <= len(specs) <= multicam.MAX_CAMS):
            raise HTTPException(400, f'Cần từ {multicam.MIN_CAMS} đến {multicam.MAX_CAMS} camera')
        if runner.running():
            raise HTTPException(409, 'Pipeline đang chạy, đợi xong hoặc bấm Dừng.')
        if runner.loading():
            raise HTTPException(409, 'Model đang nạp (lần đầu mất vài phút), thử lại sau.')
        cams = []
        for i, sp in enumerate(specs):
            sp = sp if isinstance(sp, dict) else {}
            up = form.get(f'video_{i}')                # UploadFile của Starlette (không phải lớp con của fastapi.UploadFile)
            if up is not None and getattr(up, 'filename', None):
                vpath, label = await save_upload(up)
            else:
                src = str(sp.get('source') or '').strip()
                if not src:
                    raise HTTPException(400, f'Camera {i + 1} chưa có nguồn (URL RTSP, đường dẫn hoặc file)')
                if src.lower().startswith('push://'):
                    raise HTTPException(400, 'Multi-camera không nhận webcam của trình duyệt: dùng RTSP hoặc file video.')
                vpath, label = resolve_source(src, uploads)
            cams.append(dict(name=str(sp.get('name') or '').strip()[:40] or label, path=vpath))

        maxp = min(max(1, int(opts.get('multi_max_parallel') or 2)), len(cams))
        par = bool(opts.get('multi_parallel', True)) and maxp > 1
        run_root = state.root / 'multi' / uuid.uuid4().hex[:12]     # thư mục của lần multi này (mỗi camera một thư mục con)
        if par:
            for i, c in enumerate(cams):
                c['root'] = run_root / f'cam{i + 1}'
            if runner.engine is not None:               # model của tiến trình chính nhường VRAM cho các tiến trình camera
                runner.engine.close()
                runner.engine = None
                with state.lock:
                    state.models_state = 'idle'

        def launch_seq(c):
            while runner.loading() or runner.running():     # luồng pipeline của camera trước phải hết hẳn mới mở camera kế
                time.sleep(0.3)
            return start_job(c['path'], c['name'], opts, realtime=efsource.is_stream(c['path']))

        def launch_par(c):
            """Camera chạy trong tiến trình riêng (cam_worker): cấu hình = cài đặt hiện tại + tuỳ chọn của lần này."""
            realtime = efsource.is_stream(c['path'])
            cfg = dict(root=str(c['root']), video=str(c['path']), demo=demo, local=local,
                       settings=dict(state.settings, **pipe_opts, realtime=realtime),
                       extra=dict(models=dict(vlm=dict(port=cam_worker.free_port()))),   # llama-server riêng cho từng camera
                       flock=str(run_root / 'load.lock'),
                       stream_max_sec=max(10.0, float(opts.get('camera_max_min') or 30) * 60) if realtime else None)
            proc = cam_worker.CamProc(cfg)
            jid = uuid.uuid4().hex[:12]
            jobs[jid] = dict(id=jid, path=Path(c['path']), name=c['name'], t0=time.time(), cancelled=False, final=None, rdir=None,
                             realtime=realtime, vinfo=video_info(c['path']), root=c['root'], remote=proc)
            return jid

        def release_par(jid):
            j = jobs.get(jid)
            if j is not None and j.get('remote') is not None:
                j['remote'].close()

        def poll(jid):
            j = get_job(jid)
            v = view(j)
            key = None
            if v['status'] == 'done':
                rd = result_dir(j)
                key = rd.name if rd else (state.cam_name if j.get('remote') is None else j['path'].stem) or j['path'].stem
            return dict(status=v['status'], key=key, error=v['error'])

        def alerts_of(jid, final):
            j = get_job(jid)
            if not final:
                if j.get('remote') is not None:
                    return multicam.from_live_alerts(j['remote'].snap()['alerts'])
                with state.lock:
                    return multicam.from_live_alerts(list(state.alerts))
            rd = result_dir(j)
            if rd is None or not (rd / 'result.json').is_file():
                return None
            return multicam.from_result_alerts(json.loads((rd / 'result.json').read_text(encoding='utf-8')))

        def stop_cam(jid):
            stop_job(get_job(jid))

        mr = multicam.MultiRun(cams, launch=launch_par if par else launch_seq, poll=poll, stop=stop_cam, alerts_of=alerts_of,
                               parallel=maxp if par else 1, release=release_par if par else None).start()
        multis[mr.id] = mr
        return {'multi_id': mr.id, 'cameras': [c['name'] for c in mr.cams], 'parallel': maxp if par else 1}

    @app.get('/api/multi-jobs/{mid}')
    def multi_status(mid: str):
        return get_multi(mid).snapshot()

    @app.get('/api/multi-jobs/{mid}/results')
    def multi_results(mid: str):
        """result.json của từng camera đã xong. Đọc thẳng từ thư mục kết quả (không qua get_result), vì job cũ đã bị coi là 'cancelled'
        khi job con sau bắt đầu. Camera chưa có kết quả (lỗi / bỏ qua / demo) không có trong danh sách."""
        out = []
        for c in get_multi(mid).snapshot()['cams']:
            j = jobs.get(c['job_id']) if c['job_id'] else None
            rd = result_dir(j) if (j is not None and c['status'] == 'done') else None
            if rd is not None and (rd / 'result.json').is_file():
                out.append(dict(idx=c['idx'], name=c['name'], job_id=c['job_id'], key=rd.name,
                                result=gated_result(j, rd)))
        return out

    @app.post('/api/multi-jobs/{mid}/finish')
    def multi_finish(mid: str):
        get_multi(mid).finish()
        return {'status': 'finishing'}

    @app.get('/api/jobs/{jid}')
    def job_status(jid: str):
        j = get_job(jid)
        v = view(j)
        key = None
        if v['status'] == 'done':
            rd = result_dir(j)
            key = rd.name if rd else (state.cam_name or j['path'].stem)
        return dict(status=v['status'], stage=v['stage'], stages=STAGES, progress=v['progress'], log=[], key=key,
                    error=v['error'], name=j['name'], realtime=j['realtime'])

    @app.post('/api/camera/session')
    def camera_session():
        """Phiên webcam: trình duyệt mở WebSocket /api/camera/{sid}/push và gửi từng khung JPEG (binary)."""
        sid = uuid.uuid4().hex[:12]
        efsource.PUSH[sid] = efsource.PushBuffer()
        return {'sid': sid, 'ws': f'/api/camera/{sid}/push'}

    @app.websocket('/api/camera/{sid}/push')
    async def camera_push(ws: WebSocket, sid: str):
        buf = efsource.PUSH.get(sid)
        if buf is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        try:
            while True:
                buf.put(await ws.receive_bytes())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            buf.close()                           # đóng luồng: pipeline đọc hết khung còn lại rồi kết thúc và ghi kết quả
            efsource.PUSH.pop(sid, None)

    @app.post('/api/jobs/{jid}/finish')
    def finish(jid: str):
        """Camera: dừng thu rồi xử lý nốt (VLM các đoạn còn lại) để ra kết quả. Khác /cancel: không bỏ kết quả."""
        j = get_job(jid)
        v = view(j)
        if v['status'] in ('done', 'error', 'cancelled'):
            return {'status': v['status']}
        stop_job(j)
        return {'status': 'finishing'}

    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid: str):
        j = get_job(jid)
        v = view(j)
        if v['status'] in ('done', 'error', 'cancelled'):
            return {'status': v['status']}
        j['cancelled'] = True
        stop_job(j)
        return {'status': 'cancelling'}

    # ------------------------------------------------------------------ phát trực tiếp
    @app.get('/api/jobs/{jid}/live')
    def live_frame(jid: str):
        j = get_job(jid)
        if j.get('remote') is not None:
            r = remote_frame(j, -1)
            jpg = r['jpg'] if r else None
        else:
            with state.lock:
                jpg = state.frame_jpeg
        if jpg is None:
            raise HTTPException(404, 'Chưa có khung hình')
        return Response(jpg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/jobs/{jid}/live/stream')
    async def live_stream(jid: str):
        """MJPEG (multipart/x-mixed-replace) như /api/stream của dashboard cũ: server đẩy mỗi khung mới ngay khi có, không bị trễ do web poll.
        Kết thúc khi pipeline xong và không còn khung mới (để <img> dừng)."""
        j = get_job(jid)
        if j.get('remote') is not None:                 # camera chạy trong tiến trình riêng: lấy khung từ tiến trình đó
            async def gen_remote():
                last, idle = -1, 0
                while True:
                    r = await asyncio.to_thread(remote_frame, j, last)
                    if r is None:
                        return
                    if r['jpg'] is not None:
                        last, idle = r['seq'], 0
                        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(r['jpg'])).encode() + b'\r\n\r\n' + r['jpg'] + b'\r\n'
                    else:
                        idle += 1
                        if r['st'] in ('done', 'error') and not r['alive'] and idle > 60:
                            return
                        await asyncio.sleep(0.03)
            return StreamingResponse(gen_remote(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

        async def gen():
            last, idle = -1, 0
            while True:
                with state.lock:
                    seq, jpg, status = state.frame_seq, state.frame_jpeg, state.status
                if jpg is not None and seq != last:
                    last, idle = seq, 0
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
                else:
                    idle += 1
                    if (finished(j) or status in ('done', 'error', 'idle')) and idle > 60:      # ~2 s không có khung mới sau khi xong
                        return
                await asyncio.sleep(0.03)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

    def art_boxes(j):
        """Hộp hiện vật (toạ độ khung rộng 960 px) lấy từ đường viền cảnh đầu tiên của dashboard."""
        vi = vinfo_of(j)
        ratio = (vi['height'] / vi['width']) if vi['width'] else 0.5625
        with state.lock:
            scenes = list(state.scene_contours)
        out = []
        for a in (scenes[0]['arts'] if scenes else []):
            pts = [pt for poly in a['c'] for pt in poly]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(dict(id=a['id'], label=f'Hiện vật {a["id"]}', pedestal=False,
                            box=[round(min(xs) * NW, 1), round(min(ys) * NW * ratio, 1), round(max(xs) * NW, 1), round(max(ys) * NW * ratio, 1)]))
        return out

    @app.get('/api/jobs/{jid}/live/events')
    def live_events(jid: str, after: int = 0):
        j = get_job(jid)
        if j.get('remote') is not None:                 # job con song song: không có sự kiện chi tiết theo khung (dùng trạng thái multi)
            return {'meta': None, 'status': None, 'events': [], 'job': view(j)['status']}
        vi = vinfo_of(j)
        with state.lock:
            fi, nf, fps_now, alerts = state.frame_idx, state.n_frames, state.fps, list(state.alerts)
            running = state.status in ('running', 'finishing', 'done')
        nf = nf or vi['n_frames']
        vfps = vi['fps'] or 25.0
        meta = dict(artifacts=art_boxes(j), n_frames=nf, fps=vfps, width=vi['width'], height=vi['height']) if (running and nf) else None
        status = dict(frame=fi, n_frames=nf, t_s=round(fi / vfps, 2), live_fps=round(fps_now, 1), dropped=0, vlm_pending=None) if running else None
        events = []
        for a in alerts:
            if a['id'] <= after:
                continue
            vt = float(a.get('video_time') or 0)
            events.append(dict(seq=a['id'], kind='vlm', track=a['track_id'], artifacts=a.get('artifact_ids') or [],
                               decision='ANOMALY', behavior=a['behavior'], confidence=a['confidence'], evidence=a.get('evidence') or '',
                               label=a['behavior'], start_s=max(0.0, vt - 3), end_s=vt + 1))
        return {'meta': meta, 'status': status, 'events': events, 'job': view(j)['status']}

    # ------------------------------------------------------------------ kết quả
    def synth_result(j):
        """Không có result.json (chế độ demo): dựng từ danh sách cảnh báo để web vẫn hiện được."""
        vi = vinfo_of(j)
        with state.lock:
            alerts = list(state.alerts)
        fps = vi['fps'] or 25.0
        events = []
        for a in alerts:
            vt = float(a.get('video_time') or 0)
            events.append(dict(event_id=f'e{a["id"]}', track_id=a['track_id'], artifact_id=(a.get('artifact_ids') or [None])[0],
                               start_time_s=max(0.0, vt - 3), end_time_s=vt + 1, decision='ANOMALY', behavior=a['behavior'],
                               vlm_confidence=a['confidence'], vlm_evidence=a.get('evidence') or ''))
        return dict(video=dict(key=j['path'].stem, name=j['name'], fps=fps, n_frames=vi['n_frames'] or int(fps * 60),
                               width=vi['width'] or 960, height=vi['height'] or 540),
                    artifacts=[], alerts=[], events=events)

    @app.get('/api/jobs/{jid}/result')
    def get_result(jid: str):
        j = get_job(jid)
        if view(j)['status'] != 'done':
            raise HTTPException(409, 'Job chưa xong')
        rd = result_dir(j)
        if rd is not None and (rd / 'result.json').is_file():
            return gated_result(j, rd)
        return synth_result(j)

    @app.get('/api/jobs/{jid}/video')
    def get_video(jid: str, kind: str = 'live'):
        j = get_job(jid)
        rd = result_dir(j) if finished(j) else None
        if kind != 'original' and rd is not None and (rd / 'live.mp4').is_file():
            return FileResponse(rd / 'live.mp4', media_type='video/mp4')
        if efsource.is_stream(j['path']) or not j['path'].is_file():
            raise HTTPException(404, 'Không còn video')
        return FileResponse(j['path'])

    # ------------------------------------------------------------------ nhiều camera đã hiệu chuẩn (hồ sơ địa điểm)
    site_jobs = {}
    SITE_MAX_SECONDS = float(os.environ.get('EXHIBITFLOW_SITE_MAX_SECONDS', '30'))     # cắt tối đa N giây đầu của mỗi video
    SITE_STAGES = ['Cắt khung đã khử méo', 'Phát hiện người + ghép các camera', 'Hoàn tất']
    SITE_STAGES_EV = ['Cắt khung đã khử méo', 'Phát hiện người + ghép các camera', 'Phát hiện hành vi bất thường (từng camera)',
                      'Chiếu hành vi bất thường lên map', 'Hoàn tất']
    # chạy engine của Hưng trên từng camera để lấy hành vi bất thường (cần model + GPU); --demo không có nên tắt
    app_flags = dict(site_events=(not demo) and os.environ.get('EXHIBITFLOW_SITE_EVENTS', '1') != '0')

    def analyze_video(job, path):
        """Chạy engine (State + PipelineRunner của Hưng) trên MỘT video, trả timeline.json (hộp người theo khung + cảnh báo)."""
        end = time.time() + 900
        while runner.loading() and time.time() < end:
            time.sleep(1)
        if runner.running():
            raise RuntimeError('Engine đang chạy video khác')
        info = video_info(path)
        t_start = time.time()
        state.save_settings(dict(realtime=False, make_map=False, map_image='auto', max_frames=int(SITE_MAX_SECONDS * info['fps'])))
        runner.start(str(path))
        while runner.running():
            time.sleep(0.5)
        with state.lock:
            st, msg, rid = state.status, state.message, getattr(state, 'run_id', None)
        tl = (state.root / 'runs' / str(rid) / 'timeline.json') if rid else None
        if st != 'done' or tl is None or not tl.is_file() or tl.stat().st_mtime < t_start - 1:
            raise RuntimeError(msg or 'Engine không trả kết quả cho video này')
        out = json.loads(tl.read_text(encoding='utf-8'))
        try:
            out['_run'] = json.loads((tl.parent / 'run.json').read_text(encoding='utf-8'))
        except Exception:                                  # noqa: BLE001
            pass
        # live.mp4 (video đã vẽ khung) do engine ghi vào results/<tên video>/, KHÔNG nằm trong runs/<rid>/
        res = state.root / 'results'
        cands = [res / Path(path).stem / 'live.mp4'] + sorted(res.rglob('live.mp4'), key=lambda q: q.stat().st_mtime, reverse=True)
        for lv in cands:
            if lv.is_file() and lv.stat().st_mtime >= t_start - 1:
                out['_live'], out['_result_dir'] = str(lv), str(lv.parent)
                break
        return out

    app.state.analyze_video = analyze_video            # thay được khi kiểm thử (không cần GPU)
    app.state.web_cfg, app.state.runner = web_cfg, runner
    app.state.site_events = app_flags['site_events']

    def site_device():
        dev = os.environ.get('EXHIBITFLOW_SITE_DEVICE')
        if dev:
            return dev
        try:
            import torch
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except Exception:                                  # noqa: BLE001
            return 'cpu'

    def site_events_stage(job, site):
        """Engine trên từng camera (SONG SONG nếu được) -> đối chiếu chéo camera (ID người, hành vi bất thường) -> hành vi theo toạ độ ảnh ->
        sàn (mét) -> gộp các camera -> ghi vào site_result.json. Lỗi ở đây KHÔNG làm hỏng bản đồ ghép: kết quả vẫn trả, kèm `events_error`."""
        rpath = job['out'] / 'site_result.json'
        result = json.loads(rpath.read_text(encoding='utf-8'))
        errors, seen, cam_events = [], [], []
        tls, id_maps = {}, job.setdefault('id_map', {})
        job['stage'] = 2

        def cam_ctx(cam, path):
            """K, R, t / ma trận chiếu của camera + thông số video, dùng cho nhãn ID chung (cả luồng tuần tự lẫn song song)."""
            vi = video_info(path)
            spec = site_events.fit_to_video(site['cameras'][cam], int(vi['width']), int(vi['height']))
            H = next((c.get('H') for c in result.get('cameras', []) if c.get('id') == cam), None)
            return vi, spec, H

        par_n = int(os.environ.get('EXHIBITFLOW_SITE_PARALLEL', '0') or 0) or len(job['videos'])     # 1 = chạy tuần tự
        if not demo and len(job['videos']) > 1 and par_n > 1:
            try:
                args = {}
                for cam, path in job['videos'].items():
                    vi, spec, H = cam_ctx(cam, path)
                    args[cam] = dict(spec=spec, H=H, frames=result.get('frames', []), fps=vi['fps'], wh=(vi['width'], vi['height']),
                                     det_conf=web_cfg.get('det_conf'))
                if runner.engine is not None:           # model của tiến trình chính nhường VRAM cho các tiến trình camera
                    runner.engine.close()
                    runner.engine = None
                    with state.lock:
                        state.models_state = 'idle'
                maxf = int(SITE_MAX_SECONDS * max(video_info(p)['fps'] for p in job['videos'].values()))
                job['log'].append(f'[events] chạy song song {len(job["videos"])} camera (mỗi camera một tiến trình + VLM riêng)...')
                got = site_parallel.run(job['videos'], state.root / 'sites' / job['id'] / 'cams',
                                        dict(state.settings, realtime=False, make_map=False, map_image='auto', max_frames=maxf),
                                        args, job['log'].append, local=local, max_parallel=par_n,
                                        on_progress=lambda f: job.__setitem__('par_progress', f))
                for cam, (tl, idm) in got.items():
                    tls[cam], id_maps[cam] = tl, idm
            except Exception as e:                      # noqa: BLE001 - song song lỗi (vd. thiếu VRAM): lùi về chạy tuần tự, kết quả vẫn đúng
                job['log'].append(f'[events] chạy song song lỗi ({type(e).__name__}: {str(e)[:200]}): chuyển sang chạy tuần tự')
                job.pop('par_progress', None)
                tls.clear()
                runner.preload()
        if not tls:                                     # luồng tuần tự (và đường lùi)
            gallery = {}                                # ngoại hình theo ID chung, dùng chung các camera
            try:
                for k, (cam, path) in enumerate(job['videos'].items()):
                    job['seq'] = (k, len(job['videos']))
                    job['log'].append(f'[events] {site["cameras"][cam].get("label", cam)}: phân tích hành vi bất thường...')
                    try:
                        app_site_ids['labeler'], app_site_ids['votes'] = None, {}
                        try:
                            vi, spec0, H0 = cam_ctx(cam, path)
                            mp0 = site_events.homography_mapper(H0, spec0['K'], spec0.get('dist')) if H0 else site_events.floor_mapper(spec0)
                            app_site_ids['labeler'] = site_events.GlobalIdLabeler(result.get('frames', []), mp0, vi['fps'], (vi['width'], vi['height']), gallery=gallery)
                        except Exception as e:         # noqa: BLE001 - không ghép được ID chung: video vẫn có ID cục bộ
                            job['log'].append(f'[events] {cam}: không đổi được sang ID chung ({e})')
                        tls[cam] = app.state.analyze_video(job, path)
                        id_maps[cam] = {t: max(v, key=v.get) for t, v in app_site_ids.get('votes', {}).items() if v}
                    except Exception as e:              # noqa: BLE001
                        errors.append(f'{cam}: {type(e).__name__}: {e}')
                        job['log'].append(f'[events] {cam} lỗi: {e}')
            finally:
                app_site_ids['labeler'] = None
                if not demo:
                    state.save_settings(dict(max_frames=0))     # trả lại cài đặt, không cắt video của lượt phân tích sau
        for cam, tl in tls.items():
            if tl.get('_live'):
                job.setdefault('live', {})[cam] = tl['_live']
            if tl.get('_result_dir'):
                job.setdefault('result_dirs', {})[cam] = tl['_result_dir']
        # "chạm" chỉ tính từ lúc người đó đứng lại: đoạn còn đang đi là "cần xác nhận" (xem still_gate.py), trước khi đối chiếu chéo
        gated = {}
        for cam, tl in list(tls.items()):
            rd = (job.get('result_dirs') or {}).get(cam)
            if rd and (Path(rd) / 'result.json').is_file():
                try:
                    gated[cam], tls[cam] = still_gate.apply(json.loads((Path(rd) / 'result.json').read_text(encoding='utf-8')), tl)
                    job.setdefault('fused', {})[cam] = gated[cam]
                except Exception as e:                  # noqa: BLE001 - lỗi hậu xử lý: giữ kết quả gốc
                    job['log'].append(f'[events] {cam}: không áp được quy tắc đứng yên: {e}')
        # đối chiếu chéo: cùng một người (ID chung) ở cùng thời điểm phải có kết luận khớp nhau giữa các camera
        fx = {}
        try:
            for cam, tl in tls.items():
                rd = (job.get('result_dirs') or {}).get(cam)
                if rd and (Path(rd) / 'result.json').is_file():
                    fx[cam] = dict(tl=tls[cam], res=gated.get(cam) or json.loads((Path(rd) / 'result.json').read_text(encoding='utf-8')),
                                   id_map=id_maps.get(cam) or {})
            if len(fx) > 1:
                fused, info = site_fusion.fuse(fx)
                result['fusion'] = info
                for cam, v in fused.items():
                    job.setdefault('fused', {})[cam] = v['res']
                    tls[cam] = site_fusion.add_promoted_alerts(site_fusion.drop_demoted_alerts(tls[cam], v['drop']), v.get('add'))
                    lv = (job.get('live') or {}).get(cam)
                    if v.get('add') and lv:                      # video do engine vẽ trước đối chiếu: vẽ thêm khung đỏ cho người bị lan truyền
                        try:
                            import site_live
                            if site_live.paint_promoted(lv, tls[cam], v['add'], id_maps.get(cam) or {}):
                                job['log'].append(f'[events] {cam}: đã vẽ khung đỏ cho {len(v["add"])} sự kiện lan truyền vào live.mp4')
                        except Exception as e:                      # noqa: BLE001 - lỗi vẽ video không làm hỏng kết quả
                            job['log'].append(f'[events] {cam}: không vẽ được khung đỏ vào video ({type(e).__name__}: {e})')
                n_dem = sum(1 for e in info['events'] if e['status'] == 'conflict')
                job['log'].append(f'[events] đối chiếu chéo camera ({info["policy"]}): người bất thường {info.get("anomaly_persons") or []}, '
                                  f'{len(info.get("propagated") or [])} sự kiện được lan sang camera khác, {n_dem} bị hạ xuống "cần xem lại"')
        except Exception as e:                          # noqa: BLE001 - đối chiếu lỗi: giữ kết quả riêng từng camera
            job['log'].append(f'[events] đối chiếu chéo lỗi: {type(e).__name__}: {e}')
        for cam, tl in tls.items():
            try:
                vi, spec, H = cam_ctx(cam, job['videos'][cam])
                spec = site_events.fit_to_video(site['cameras'][cam], int(tl['width']), int(tl['height']))
                # có ma trận chiếu của repo map: dùng nó (cùng hệ toạ độ với ảnh mặt bằng); không thì tự tính từ K, R, t
                mapper = site_events.homography_mapper(H, spec['K'], spec.get('dist')) if H else site_events.floor_mapper(spec)
                cam_events += site_events.engine_events(tl, mapper, cam)
                seen.append(dict(cam=cam, **site_events.engine_summary(tl, tl.get('_run'))))
            except Exception as e:                      # noqa: BLE001
                errors.append(f'{cam}: {type(e).__name__}: {e}')
                job['log'].append(f'[events] {cam} lỗi: {e}')
        job['stage'] = 3
        result['events'] = site_events.build_incidents(cam_events, result.get('frames', []))
        result['engine'] = seen
        result['events_error'] = '; '.join(errors) or None
        rpath.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')

    def site_worker(job):
        try:
            site = job.get('site_def') or site_runner.load_site(job['site'])
            repo = site_runner.find_repo()
            if repo is None:
                raise RuntimeError('Không thấy repo multicam-2d-floorplan-tracker (đặt FLOORPLAN_DIR).')
            job['stage'] = 0
            work = state.root / 'sites' / job['id']
            data_dir, n = site_runner.prepare_dataset(site, job['videos'], work / 'data', seconds=SITE_MAX_SECONDS, log=job['log'].append)
            job['stage'], job['n'] = 1, n
            out = site_runner.run_repo(repo, data_dir, work / 'out', n, site['room_id'], device=site_device(), log=job['log'].append)
            if not (out / 'site_result.json').is_file():
                raise RuntimeError('Pipeline map2d không tạo ra site_result.json')
            job['out'] = out
            if job['events']:
                site_events_stage(job, site)
            job['stage'], job['status'] = len(job['stages']) - 1, 'done'
        except BaseException as e:                         # noqa: BLE001
            job['status'], job['error'] = 'error', f'{type(e).__name__}: {e}'
            try:
                print('[site] LỖI: ' + job['error'], flush=True)
            except UnicodeError:                           # console Windows không in được tiếng Việt
                print('[site] ERROR: ' + job['error'].encode('ascii', 'replace').decode(), flush=True)

    @app.get('/api/sites')
    def sites():
        return list(site_runner.list_sites().values())

    async def save_video(up, dst_base, label):
        ext, head = Path(up.filename).suffix.lower(), b''
        if ext not in ALLOWED:
            head = await up.read(1 << 16)
            ext = sniff_ext(head) or ext
        check_ext(ext)
        dst, size = Path(f'{dst_base}{ext}'), 0
        with open(dst, 'wb') as f:
            f.write(head)
            size = len(head)
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    f.close()
                    dst.unlink(missing_ok=True)
                    raise HTTPException(413, f'Video {label} quá lớn (tối đa {MAX_BYTES >> 20} MB)')
                f.write(chunk)
        return dst

    def cam_match(cid, k, label, filename):
        """Id camera trong file hiệu chuẩn khớp với video: trùng khoá / nhãn, hoặc nằm trong tên file (camera_A ~ camera_A_cctv.mp4)."""
        if not cid:
            return False
        c = cid.lower()
        return c in (k.lower(), label.lower()) or (len(c) >= 3 and c in Path(filename).stem.lower())

    async def read_json_field(v, label):
        if v is None:
            return None
        text = (await v.read()).decode('utf-8-sig', errors='replace') if hasattr(v, 'read') else str(v)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f'{label}: file hiệu chuẩn không phải JSON hợp lệ ({e.msg}, dòng {e.lineno})')

    @app.post('/api/site-jobs')
    async def create_site_job(request: Request):
        """Hai kiểu form.
        1) Địa điểm có sẵn: site=<id>; mỗi camera một tệp video, tên trường = id camera (C1, C2...).
        2) Hiệu chuẩn tự nhập: site=__custom__; mỗi camera: cam_<k> (video), calib_<k> (JSON K,R,t), label_<k> (tên hiển thị);
           hoặc calib_all = một JSON chứa cả các camera (kiểu transforms.json). events=0 để bỏ bước hành vi bất thường."""
        form = await request.form()
        sid = str(form.get('site') or '')
        custom = sid == '__custom__'
        if not custom:
            try:
                site = site_runner.load_site(sid)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
        if site_runner.find_repo() is None:
            raise HTTPException(501, 'Chưa cài repo multicam-2d-floorplan-tracker trên server.')
        if any(j['status'] == 'running' for j in site_jobs.values()) or runner.running():
            raise HTTPException(409, 'Server đang phân tích video khác, đợi xong hoặc bấm Dừng.')
        jid = uuid.uuid4().hex[:12]
        videos = {}
        if custom:
            keys = [k[4:] for k in form.keys() if k.startswith('cam_') and getattr(form.get(k), 'filename', None)]
            if not keys:
                raise HTTPException(400, 'Cần ít nhất một video (trường cam_<k>)')
            allcal = await read_json_field(form.get('calib_all'), 'Hiệu chuẩn chung')
            pool = site_events.split_calibration(allcal) if allcal is not None else []
            cams, missing = {}, []
            room_bounds = site_events.room_bounds_from(allcal)
            for pos, k in enumerate(keys):
                up = form.get(f'cam_{k}')
                label = str(form.get(f'label_{k}') or up.filename)
                raw = await read_json_field(form.get(f'calib_{k}'), f'Camera {label}')
                if raw is not None:
                    room_bounds = room_bounds or site_events.room_bounds_from(raw)
                    items = site_events.split_calibration(raw)
                    raw = next((c for cid, c in items if cam_match(cid, k, label, up.filename)), items[0][1])
                elif pool:
                    raw = next((c for cid, c in pool if cam_match(cid, k, label, up.filename)), pool[pos][1] if pos < len(pool) else None)
                if raw is None:
                    missing.append(label)
                    continue
                try:
                    cams[f'C{pos + 1}'] = site_events.normalize_calibration(raw, label)
                except site_events.CalibrationError as e:
                    raise HTTPException(400, str(e))
                cams[f'C{pos + 1}']['label'] = label
                videos[f'C{pos + 1}'] = await save_video(up, uploads / f'{jid}_C{pos + 1}', label)
            if missing:
                for v in videos.values():
                    v.unlink(missing_ok=True)
                raise HTTPException(400, 'Thiếu hiệu chuẩn cho: ' + ', '.join(missing))
            site = dict(site_id='custom', name='Hiệu chuẩn tự nhập', room_id='custom', sample_fps=4.0, cameras=cams, room_bounds=room_bounds)
        else:
            for cam in site['cameras']:
                up = form.get(cam)
                if up is None or not getattr(up, 'filename', None):
                    continue
                videos[cam] = await save_video(up, uploads / f'{jid}_{cam}', cam)
            if len(videos) < 1:
                raise HTTPException(400, 'Cần ít nhất một video, đặt tên trường theo id camera: ' + ', '.join(site['cameras']))
        ev = bool(app.state.site_events) and str(form.get('events', '1')) != '0'
        job = dict(id=jid, site=sid, site_def=site if custom else None, videos=videos, status='running', stage=0, error=None, log=[], out=None,
                   n=0, t0=time.time(), events=ev, stages=SITE_STAGES_EV if ev else SITE_STAGES)
        site_jobs[jid] = job
        threading.Thread(target=site_worker, args=(job,), daemon=True, name='site-job').start()
        return {'job_id': jid, 'cameras': list(videos), 'events': ev}

    def get_site_job(jid):
        j = site_jobs.get(jid)
        if j is None:
            raise HTTPException(404, 'Không có job này')
        return j

    def site_progress(j):
        """Phần video (0-1) engine đã phân tích xong ở bước hành vi bất thường."""
        if j['stage'] != 2:
            return 1.0 if j['status'] == 'done' else 0.0
        if 'par_progress' in j:                      # chạy song song: camera chậm nhất
            return round(j['par_progress'], 3)
        k, n = j.get('seq') or (0, 1)                # chạy tuần tự: camera đã xong + phần của camera đang chạy
        part = 0.0
        if runner.running():
            with state.lock:
                if state.status == 'running' and state.n_frames:
                    part = min(1.0, max(0.0, state.frame_idx / state.n_frames))
        return round((k + part) / n, 3)

    @app.get('/api/site-jobs/{jid}')
    def site_job_status(jid: str):
        j = get_site_job(jid)
        return dict(status=j['status'], stage=j['stage'], stages=j['stages'], error=j['error'], cameras=list(j['videos']),
                    log=[l for l in j['log'][-8:]], elapsed=round(time.time() - j['t0']),
                    progress=site_progress(j))

    @app.get('/api/site-jobs/{jid}/result')
    def site_job_result(jid: str):
        j = get_site_job(jid)
        if j['status'] != 'done':
            raise HTTPException(409, 'Job chưa xong')
        return FileResponse(j['out'] / 'site_result.json', media_type='application/json')

    @app.get('/api/site-jobs/{jid}/result/{cam}')
    def site_job_cam_result(jid: str, cam: str):
        """result.json của engine cho một camera (hộp, hiện vật, cảnh báo) - cùng dạng với chế độ 1 camera."""
        rd = (get_site_job(jid).get('result_dirs') or {}).get(cam)
        if not rd or not (Path(rd) / 'result.json').is_file():
            raise HTTPException(404, 'Chưa có kết quả engine cho camera này')
        job = get_site_job(jid)
        res = copy.deepcopy((job.get('fused') or {}).get(cam)) or json.loads((Path(rd) / 'result.json').read_text(encoding='utf-8'))
        m = (job.get('id_map') or {}).get(cam) or {}
        if m:                                              # ID cục bộ của tracker camera -> ID chung của map ghép (khớp nhãn trên live.mp4)
            g = lambda t: m.get(int(t), t) if t is not None else t      # noqa: E731
            for k in ('alerts', 'events', 'vlm_tasks'):
                for e in res.get(k) or []:
                    if e.get('track_id') is not None:
                        e['track_id'] = g(e['track_id'])
            res['track_summary'] = {str(g(t)): v for t, v in (res.get('track_summary') or {}).items()}
            dg = res.get('diagnostics') or {}
            if isinstance(dg.get('tracks'), dict):
                dg['tracks'] = {str(g(t)): dict(v, track_id=g(t)) if isinstance(v, dict) else v for t, v in dg['tracks'].items()}
        return res

    @app.get('/api/site-jobs/{jid}/files/{cam}/{path:path}')
    def site_job_cam_file(jid: str, cam: str, path: str):
        """File trong thư mục kết quả của camera (vlm_inputs/<task>/sheet.jpg ...)."""
        rd = (get_site_job(jid).get('result_dirs') or {}).get(cam)
        if not rd:
            raise HTTPException(404, 'Chưa có kết quả')
        base = Path(rd).resolve()
        target = (base / path).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(404, 'Không có file')
        return FileResponse(target)

    @app.get('/api/site-jobs/{jid}/map')
    def site_job_map(jid: str):
        j = get_site_job(jid)
        p = Path(j['out']) / 'global_2d_map.png' if j.get('out') else None
        if p is None or not p.is_file():
            raise HTTPException(404, 'Không có ảnh mặt bằng (đang dùng đường dự phòng, không có map của repo)')
        return FileResponse(p, media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.get('/api/site-jobs/{jid}/video/{cam}')
    def site_job_video(jid: str, cam: str, kind: str = 'live'):
        j = get_site_job(jid)
        lv = (j.get('live') or {}).get(cam)
        if kind != 'original' and lv and Path(lv).is_file():
            return FileResponse(lv, media_type='video/mp4')
        p = j['videos'].get(cam)
        if p is None or not Path(p).is_file():
            raise HTTPException(404, 'Không có video camera này')
        return FileResponse(p)

    # ------------------------------------------------------------------ Map 2D (multicam-2d-floorplan-tracker)
    @app.get('/api/jobs/{jid}/map')
    def map_image(jid: str):
        get_job(jid)
        with state.lock:
            p = state.map_path
        if p is None or not Path(p).is_file():
            raise HTTPException(404, 'Chưa có map')
        return FileResponse(p, media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.get('/api/jobs/{jid}/map/data')
    def map_data(jid: str, full: int = 0):
        """Vị trí người / hiện vật trên map (toạ độ 0-1 của ảnh map). full=1 (khi xong): kèm vị trí theo từng khung để phát lại."""
        j = get_job(jid)
        vfps = j['vinfo']['fps'] or 25.0
        with state.lock:
            has = state.map_path is not None and Path(state.map_path).is_file()
            out = dict(has_map=has, version=state.map_version, fps=vfps, frame=state.frame_idx,
                       people=[dict(id=p['id'], x=round(p['x'], 4), y=round(p['y'], 4), state=p.get('state', 'idle')) for p in state.people],
                       artifacts=[dict(id=a['id'], x=round(a['x'], 4), y=round(a['y'], 4)) for a in state.artifacts])
            if full and finished(j):
                out['frames'] = {int(k): [[q[0], q[1], q[2]] for q in v[1]] for k, v in state.timeline.items() if v[1]}
        return out

    @app.get('/api/jobs/{jid}/files/{path:path}')
    def get_file(jid: str, path: str):
        j = get_job(jid)
        rd = result_dir(j)
        if rd is None:
            raise HTTPException(404, 'Chưa có kết quả')
        base = rd.parent.resolve()
        target = (base / path).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(404, 'Không có file')
        return FileResponse(target)

    # Giao diện web cùng origin với API (chạy thẳng trên server GPU, không cần dán URL backend). Phải mount SAU các route /api.
    if WEB_DIR.is_dir():
        app.mount('/', StaticFiles(directory=str(WEB_DIR), html=True), name='web')
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--data-dir', default=str(PROJ / 'dashboard_data'))
    ap.add_argument('--demo', action='store_true', help='dữ liệu giả của dashboard, không cần GPU / trọng số')
    ap.add_argument('--mode', choices=('tracking', 'tracking_seg', 'full'), default=None)
    ap.add_argument('--no-preload', action='store_true', help='không nạp model lúc khởi động')
    ap.add_argument('--local', action='store_true', help='máy ít VRAM (xem dashboard.server --local)')
    a = ap.parse_args()
    uvicorn.run(create_app(a.data_dir, a.demo, preload=not a.no_preload, mode=a.mode, local=a.local),
                host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
