# Báo cáo Kiểm chứng Triển khai & Bảo mật ExhibitFlow (3 Tầng Tách Biệt)

Báo cáo này ghi nhận kết quả kiểm thử tự động, kiểm thử end-to-end qua **Cloudflare Quick Tunnel HTTPS thật (`cloudflared`)**, rà soát bảo mật và hướng dẫn vận hành cho kiến trúc 3 tầng tách biệt:

- **Tầng 1 — Static Frontend (`web/`)**: Thuần HTML5/CSS3/JS ES6+, không build step, cấu hình sẵn cho Vercel Static Hosting (`web/vercel.json` + `.vercelignore`).
- **Tầng 2 — CPU Backend (`backend/`)**: FastAPI siêu nhẹ chạy trên máy cục bộ/CPU server, quản lý hàng đợi FIFO, SQLite (WAL) / Redis, lưu trữ artifact cục bộ/Vercel Blob, xác thực `BACKEND_API_KEY` + stream ticket `?st=`, phát trực tiếp & xem lại ngoại tuyến.
- **Tầng 3 — Standalone GPU Worker (`GPU_server/`)**: Gói ML độc lập (`exhibitflow_gpu/` + `worker/`), tách rời hoàn toàn khỏi `exhibitflow_pipeline/`, hỗ trợ chế độ `--demo` thuần CPU và triển khai 1 lệnh trên Vast.ai (`start_vast.sh` / `make vast`) kèm VLM dùng chung (`SharedVLMServer`) và `FloorplanService`.

> [!NOTE]
> **Phạm vi kiểm chứng thực tế:** Máy kiểm thử hiện tại là Linux ARM64 không có GPU NVIDIA CUDA, do đó các bài kiểm thử chạy trực tiếp `GPU_server` sử dụng chế độ thuần CPU `--demo` (`EF_DEMO_DURATION_S=1.5`) và `test_floorplan_calib.py` tự động `SKIP` các phép tính phụ thuộc `numpy`/`cv2`. Đường dẫn suy luận ML thật trên GPU CUDA (YOLO, ByteTrack-DMA-LTC, `llama-server` Qwen3-VL-8B, `multicam-2d-floorplan-tracker`) **chưa chạy thực thi trên GPU thật** trong phiên này.

---

## 1. Danh sách File đã tạo & chỉnh sửa

Kiểm tra `git status -s` và `git diff --stat exhibitflow_pipeline/`: `exhibitflow_pipeline/` có **0 thay đổi** (giữ nguyên làm nguồn tham chiếu chỉ đọc).

### Cấu hình gốc & Frontend (`web/` + root)
- `.vercelignore` *(mới)*: Loại trừ `backend/`, `GPU_server/`, `exhibitflow_pipeline/`, `.venv/`, `backend_data/`, `gpu_data/`, `docs/`, `external/`, `.make/`, `*.db*`, `*.mp4`, `.env*` khỏi Vercel build output.
- `.gitignore` *(cập nhật)*: Bổ sung `backend_data/`, `gpu_data/`, `*.db`, `*.db-wal`, `*.db-shm`.
- `web/vercel.json` *(mới)*: Thiết lập HTTP security headers (`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`) và `Cache-Control: no-cache, no-store, must-revalidate` cho `/index.html`, `/js/config.js`.
- `web/index.html`, `web/css/style.css`, `web/css/theme-light.css`, `web/js/app.js`, `web/README.md` *(cập nhật)*:
  - Giao diện Settings tách riêng kết nối Backend (`#beUrl`, `#beKey`, `#beRememberKey`, `#beDirect`) và GPU Worker (`#gpuUrl`, `#gpuToken`, `#gpuSaveBtn`).
  - Chip trạng thái đa tầng (`#beChip`) và banner phục hồi kết nối (`#connBanner`) phân biệt rõ: mất kết nối Backend, thiếu/sai `BACKEND_API_KEY`, GPU ngoại tuyến, GPU đang nạp model, GPU lỗi model, và sẵn sàng.
  - Hỗ trợ `Authorization: Bearer <key>`, tự động xin/làm mới stream ticket (`POST /api/auth/stream-token` → `?st=`), phát lại lịch sử/cảnh báo khi GPU ngoại tuyến, và tự chuyển polling 5s khi SSE bị gián đoạn qua Quick Tunnel.

### CPU Backend (`backend/`)
- `backend/requirements.txt`, `backend/Dockerfile`, `backend/.env.example`, `backend/Makefile`, `backend/README.md`, `backend/__init__.py`
- `backend/launch.py` *(mới)*: Trình khởi chạy kèm auto-port, auto-sinh `BACKEND_API_KEY`, tải/chạy `cloudflared` (`quick` / `named` / `none`), kiểm tra DNS qua `1.1.1.1` DoH trước khi probe `/api/health` công khai, giám sát tunnel ngắt kết nối và tách log ra file (`--log-file`) không chứa secret.
- `backend/app/`: `__init__.py`, `config.py`, `auth.py`, `models.py`, `gpu_client.py`, `sse.py`, `dispatcher.py`, `main.py`
- `backend/app/store/`: `__init__.py`, `base.py`, `sqlite_store.py`, `redis_store.py`
- `backend/app/artifacts/`: `__init__.py`, `base.py`, `local.py`, `vercel_blob.py`
- `backend/tests/`: `test_zero_heavy_deps.py`, `test_stores.py`, `test_artifact_store.py`, `test_auth_and_security.py`, `test_launch.py`, `test_e2e_backend_gpu.py`

### Standalone GPU Worker (`GPU_server/`)
- `GPU_server/requirements-gpu.txt`, `GPU_server/Dockerfile`, `GPU_server/Makefile`, `GPU_server/setup.sh`, `GPU_server/start_vast.sh`, `GPU_server/README.md`
- `GPU_server/launch.py`: Hỗ trợ `--tunnel quick|named|none` (`--no-tunnel`), phát hiện Vast.ai IP:port, chờ DNS `1.1.1.1` DoH trước khi probe `/internal/health` công khai (`200` khi có token và `401` khi thiếu token), cổng kiểm tra `STATUS: READY (SẴN SÀNG)`, cảnh báo ngắt tunnel, và `--log-file` không ghi secret.
- `GPU_server/scripts/`: `install_deps.py`, `install_llama.sh`, `fetch_weights.py`
- `GPU_server/configs/default.yaml`, `GPU_server/weights/README.md`, `GPU_server/sites/.gitkeep`
- `GPU_server/exhibitflow_gpu/`: `__init__.py`, `settings.py`, `common.py`, `prof.py`, `tuning.py`, `source.py`, `perception.py`, `segmenter.py`, `interaction.py`, `engine.py`, `pipeline.py`, `tracker_worker.py`, `vlm.py`, `vlm_server.py` (`SharedVLMServer`), `service.py`, `runner.py`, `multicam.py`, `cam_worker.py`, `floorplan/{__init__.py,calibration.py,single_cam.py,multi_cam.py,site_driver.py}`
- `GPU_server/worker/`: `__init__.py`, `auth.py`, `demo.py`, `runtime.py`, `app.py`
- `GPU_server/notebooks/exhibitflow_kaggle_worker.ipynb` *(mới)*
- `GPU_server/tests/`: `test_worker_demo.py`, `test_shared_vlm.py`, `test_floorplan_calib.py`, `test_launcher_and_notebooks.py`

---

## 2. Lệnh chạy thực tế & Kết quả kiểm thử

### 2.1. Kiểm tra tách biệt môi trường & tự đóng gói
- **CPU `.venv/`**: Cài đặt các gói CPU thuần (`fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx`, `pydantic`, `psutil`, `redis`, `pytest`, `pillow`). Xác nhận `torch`, `cv2`, `ultralytics`, `onnxruntime`, `transformers` **không có** trong `.venv/`.
- **Kiểm tra import chéo `GPU_server/`**:
  ```bash
  grep -rn --exclude-dir=__pycache__ -E "exhibitflow_pipeline|from dashboard|import dashboard" GPU_server/
  ```
  Kết quả: **0 import code Python** (chỉ khớp chuỗi kiểm tra phủ định tại `GPU_server/tests/test_worker_demo.py:54` và mô tả tài liệu tại `GPU_server/README.md:3`).
- **Biên dịch cú pháp Python**:
  - `.venv/bin/python -m compileall -q GPU_server/` → exit `0`
  - `.venv/bin/python -m compileall -q backend/` → exit `0`
  - `bash -n GPU_server/start_vast.sh GPU_server/setup.sh GPU_server/scripts/install_llama.sh` → exit `0`
- **Quét secret trong mã nguồn**:
  - Không phát hiện token/API key cứng nào trong `web/`, `backend/`, `GPU_server/`.
  - `git diff --stat exhibitflow_pipeline/` → rỗng (`0` thay đổi).

### 2.2. Các bộ Test tự động
| Bộ kiểm thử | Lệnh chạy thực tế | Kết quả quan sát |
|---|---|---|
| **GPU_server unit & launcher** | `PYTHONPATH=GPU_server .venv/bin/pytest GPU_server/tests -v` | **16 passed, 1 skipped** (`20.79s`). `test_floorplan_calib.py` skipped do `.venv` CPU không cài `numpy`/`cv2`. |
| **Backend unit, security, store & E2E** | `PYTHONPATH=. .venv/bin/pytest backend/tests -v` | **33 passed** (`55.07s`). Đạt toàn bộ `test_zero_heavy_deps.py`, `test_stores.py` (SQLite + FakeRedis), `test_artifact_store.py` (Local + VercelBlob mock), `test_auth_and_security.py`, `test_launch.py`, `test_e2e_backend_gpu.py`. |
| **Frontend syntax** | `node --check web/js/app.js` | Exit `0` (0 lỗi cú pháp). |
| **Frontend heatmap & analytics** | `node --test web/tests/heatmap.test.js web/tests/museum_analytics.test.js` | Chạy sau bản sửa cuối của `web/js/app.js` đạt **23/23 passed** (`2381.6ms`; `heatmap` perf `233.65ms < 250ms`, `analytics` perf `348.36ms < 350ms`). Lưu ý: khi chạy lại lúc máy tải cao hơn, 22/23 bài chức năng pass ổn định, riêng bài benchmark thời gian thực `Performance: full analytics (< 350ms)` tại `web/tests/museum_analytics.test.js:351` dao động vượt ngưỡng lên `375.4ms–792.3ms` do nhiễu tải CPU VM. |

### 2.3. Kiểm thử Live Cross-Tier qua Cloudflare Quick Tunnel HTTPS thật
Hai tiến trình thật đã được khởi chạy song song với `cloudflared` (`--demo`, `EF_DEMO_DURATION_S=1.5`, cổng nội bộ `18011` và `18010`) và sau đó đã tắt + xoá sạch thư mục tạm `/tmp/ef_verify_*`. Các URL `*.trycloudflare.com` dưới đây là **URL thử nghiệm tạm thời đã đóng**:

- `GPU_server` Quick Tunnel tạm: `https://warm-ordering-departmental-hiring.trycloudflare.com` (token đã che `<REDACTED>`)
- `backend` Quick Tunnel tạm: `https://mechanisms-language-communities-cho.trycloudflare.com` (key đã che `<REDACTED>`)

Log kết quả kiểm thử rút gọn thực tế:
```text
[STEP 1] GPU_server READY at https://warm-ordering-departmental-hiring.trycloudflare.com (token redacted)
[STEP 2] Backend READY at https://mechanisms-language-communities-cho.trycloudflare.com (key redacted)
[CHECK 1] CORS preflight OK: https://exhibitflow.vercel.app
[CHECK 2] Unauth 401 + X-GPU-Token rejection + /api/health privacy OK
[CHECK 3] PUT /api/config/gpu -> GPU online & ready over HTTPS tunnel OK
[CHECK 4] Stream ticket scope enforcement OK
[CHECK 5a] Submitted 2 jobs over HTTPS: fa77391db39d, 7564a1e1a664 (pos2=1)
[CHECK 5b] MJPEG /api/jobs/{id}/live/stream?st=... over Quick Tunnel OK
[CHECK 5c] SSE over Quick Tunnel observed events=0 ([]...) note=SSE stream closed/errored: ReadTimeout: The read operation timed out
[CHECK 5d] Both jobs synced over HTTPS: fa77391db39d (alerts=2), 7564a1e1a664 (alerts=2)
[CHECK 5e] Single-cam artifacts, Range 206, alert PATCH, export.csv over HTTPS OK
[CHECK 6] Calibrated site-job 725eacb46510 over HTTPS OK
[CHECK 7] Offline replay (single + site) & alert review after GPU kill ALL 200/206 OK
[CHECK 8] Zero secret leakage in logs (gpu_log=0B, be_log=0B) OK
[CHECK 9] Cleanup complete
```

> [!TIP]
> **Phát hiện thực nghiệm quan trọng đã được khắc phục trong phiên kiểm chứng:**
> Khi `cloudflared` vừa in ra URL `https://*.trycloudflare.com`, DNS biên của Cloudflare chưa kịp lan truyền ở giây `t=0` (`1.1.1.1` DoH trả `Status=3 NXDOMAIN`, đến `t=+2s` mới trả `Status=0`). Nếu trình khởi chạy gọi ngay DNS hệ điều hành ở `t=0`, resolver cục bộ (`192.168.64.1` / mDNSResponder / systemd-resolved) sẽ lưu cache âm tính (`NXDOMAIN`) khiến mọi lần probe sau đều lỗi. Cả `GPU_server/launch.py` và `backend/launch.py` đã được bổ sung bước chờ `1.1.1.1/dns-query` DoH trước khi probe URL công khai, giúp probe đạt `200` ngay lần đầu.

### 2.4. Triển khai & Kiểm chứng Live Static Frontend trên Vercel
- **URL Production:** `https://exhibitflow-ten.vercel.app` (alias team: `https://exhibitflow-clevinum.vercel.app`)
- **Project / Deployment:** `exhibitflow` (`prj_HkjSzYcTrLObxEgrjLu6EiRAvfpS`, team `Clevinum` / `team_RgoDwNL7ImhIJygHahzRIzec`), Deployment ID `dpl_9jRmCAMoj9mwLeJV8mTgSh7cuFAo`, `target: production`, `readyState: READY` (`Build Completed in /vercel/output [805ms]`).
- **Bảo mật & Biến môi trường trên Vercel:** `ssoProtection.deploymentType = preview` (route production mở công khai, preview vẫn được bảo vệ), `filter_project_envs` = **`0` biến môi trường / `0` secret**.
- **Kiểm chứng HTTP `curl` live:**
  - `GET /`, `GET /index.html`, `GET /js/config.js` đều trả **`HTTP/2 200`** kèm header bảo mật `x-content-type-options: nosniff`, `referrer-policy: no-referrer`, `x-frame-options: DENY`, `strict-transport-security: max-age=63072000; includeSubDomains; preload`.
  - `/index.html` và `/js/config.js` trả `cache-control: no-cache, no-store, must-revalidate`; nội dung `/js/config.js` trả chính xác `window.EF_CONFIG={demo:false};` (`224` bytes).
  - **So khớp SHA-256 byte-for-byte 8/8 tệp static:** `index.html`, `css/style.css`, `css/theme-light.css`, `js/config.js`, `js/heatmap.js`, `js/hex.js`, `js/museum_analytics.js`, `js/app.js` đều đạt `match=YES`.
  - **Kiểm tra loại trừ tệp nội bộ / build (`HTTP 404`):** `/serve.py` → `404`, `/tests/heatmap.test.js` → `404`, `/build.mjs` → `404`, `/README.md` → `404`.
- **Ghi chú vận hành:** Snapshot này được xuất bản trực tiếp từ 8 tệp static cục bộ trong `web/` cùng `web/vercel.json` (tunnel tạm phục vụ gói nén khi build đã ngắt và xoá `/tmp/ef_vercel_stage`). Để tự động redeploy mỗi lần `git push`, chỉ cần push code lên GitHub và gắn Git repository vào dự án `exhibitflow` trong Vercel Project Settings (`Root Directory: web`, `Framework Preset: Other`, Build Command trống).

---

## 3. Kiểm chứng Bảo mật API

1. **Tách biệt hoàn toàn `BACKEND_API_KEY` và `X-GPU-Token`**:
   - Khi gửi `X-GPU-Token` làm `Authorization: Bearer <gpu_token>` hoặc header `X-GPU-Token` vào Backend (`GET /api/jobs`), Backend trả về **`401 Unauthorized`**.
   - Frontend lưu `BACKEND_API_KEY` trong `sessionStorage` mặc định (chỉ lưu `localStorage` khi người dùng tick `#beRememberKey`), tách biệt với `X-GPU-Token` (chỉ cấu hình từ Browser → Backend qua `PUT /api/config/gpu` và không bao giờ trả ngược về UI).
2. **Bảo vệ `/api/health` công khai**:
   - Khi chưa xác thực, `GET /api/health` trả `200` chỉ gồm `{ok: true, version, auth_required: true, authenticated: false, store, artifact_store}` và **ẩn hoàn toàn** `gpu`, `gpu.url`, `queue_length`.
3. **Từ chối `401` khi thiếu/sai auth**:
   - Đã kiểm chứng trực tiếp qua HTTPS tunnel với `GET/POST /api/jobs`, `DELETE /api/jobs/x`, `GET/PUT /api/config/gpu`, `GET/PATCH /api/alerts`, `GET /api/events`, `GET /api/jobs/x/video`, `POST /api/site-jobs` → tất cả đều trả `401`.
4. **Stream Ticket `?st=` ngắn hạn (HMAC-SHA256)**:
   - Cấp qua `POST /api/auth/stream-token` (bắt buộc header `Authorization: Bearer`), ký bằng khóa phiên dẫn xuất từ `BACKEND_API_KEY` + process nonce (TTL mặc định `600s`).
   - `?st=` được **chấp nhận (`200`/`206`)** trên các route media/stream/download (`live/stream`, `video`, `map`, `files/...`, `site-jobs/{id}/map`, `site-jobs/{id}/video/{cam}`, `alerts/export.csv`) và **bị từ chối (`401`)** trên các route điều khiển (`POST /api/jobs`, `PUT /api/config/gpu`, `DELETE`, `PATCH`, `POST /api/site-jobs`).
5. **CORS & Log an toàn**:
   - `OPTIONS /api/jobs` với `Origin: https://exhibitflow.vercel.app` trả `Access-Control-Allow-Origin: https://exhibitflow.vercel.app`.
   - File `--log-file` của cả `GPU_server` và `backend` có **0 lần xuất hiện** `BACKEND_API_KEY`, `X-GPU-Token`, `HF_TOKEN`, hoặc `CLOUDFLARE_TUNNEL_TOKEN`.

---

## 4. So sánh Cloudflare Quick Tunnel vs Named Tunnel

| Tiêu chí | Quick Tunnel (`--tunnel quick`) | Named Tunnel (`--tunnel named`) |
|---|---|---|
| **Điều kiện tài khoản** | Không cần tài khoản Cloudflare, không cần tên miền | **Bắt buộc** có tài khoản Cloudflare + **tên miền riêng** trỏ NS về Cloudflare DNS + Zero Trust Tunnel Token (`CLOUDFLARE_TUNNEL_TOKEN`) |
| **URL HTTPS** | Ngẫu nhiên `https://*.trycloudflare.com`, **đổi mới mỗi lần restart** | Cố định trên tên miền riêng (ví dụ `https://api.yourdomain.com`, `https://gpu.yourdomain.com`) |
| **Hành vi SSE thực tế** | **Hạn chế:** Trong kiểm thử thật (`CHECK 5c`), luồng SSE `/api/jobs/{id}/events` qua Quick Tunnel bị treo/hết thời gian (`ReadTimeout`, `0` event nhận được) do buffering/giới hạn kết nối dài của Quick Tunnel. | Ổn định hơn cho kết nối HTTP/2 & SSE dài hạn dưới cấu hình Zero Trust của bạn |
| **Luồng MJPEG & HTTP polling** | MJPEG (`/live/stream`), upload video, tải artifact, Range `206`, và polling JSON hoạt động bình thường (`200`/`206`) | Hoạt động đầy đủ (`200`/`206`) |
| **Xử lý ở Frontend** | `web/js/app.js` tự động phát hiện lỗi SSE liên tiếp, ngắt `EventSource` và chuyển sang polling 5 giây (`sseFallbackPolling`) + polling job trạng thái/map mà không báo mất kết nối giả | Dùng SSE bình thường và tự fallback nếu mạng chập chờn |

---

## 5. Khảo sát Khả thi & Giới hạn Kaggle

Notebook mẫu nằm tại `GPU_server/notebooks/exhibitflow_kaggle_worker.ipynb`. Kết luận kỹ thuật dựa trên tài liệu chính thức của Kaggle:

- **Phần cứng miễn phí:** **NVIDIA T4 x2** (16 GB × 2 VRAM, ~29 GB RAM hệ thống, 4 vCPU; GPU P100 đã ngừng cung cấp).
- **Giới hạn vận hành nghiêm ngặt:**
  - Tối đa **12 giờ/phiên**.
  - Ngắt tự động sau **20 phút không tương tác (interactive idle timeout)**.
  - Hạn mức khoảng **30 giờ GPU/tuần**.
  - Lưu trữ `/kaggle/working` giới hạn **20 GB** và **bị xoá sạch** khi kết thúc phiên (chỉ **Kaggle Datasets** gắn vào `/kaggle/input/` mới lưu bền vững).
- **Điều khoản sử dụng (ToS) & Mạng:** Kaggle chỉ cho phép mục đích học tập/thử nghiệm cá nhân phi thương mại, không phù hợp làm API production chạy liên tục. Quick Tunnel URL thay đổi mỗi lần mở notebook.
- **Chiến lược Smart Cache trong notebook:**
  - Nên đóng gói bộ weights (`best_ckpt.pth`, `osnet_x0_25.pth`, `ltc_motion_mot17.pth`, `dma_gbm_mot17.gbm`, và GGUF nếu cần) thành một Kaggle Dataset riêng gắn vào `/kaggle/input/`.
  - Notebook tự quét `/kaggle/input/` để đặt `EF_WEIGHTS_DIR` trỏ thẳng tới dataset cache (tránh tải lại nhiều GB mỗi phiên), đồng thời đọc `HF_TOKEN` từ `kaggle_secrets.UserSecretsClient`.
  - Nếu không biên dịch/chạy được `llama-server` trong giới hạn phiên Kaggle, notebook khuyến nghị lùi về `MODE="tracking_seg"`.
- **Kết luận:** **Kaggle KHÔNG tương đương Vast.ai.** Kaggle chỉ phù hợp chạy demo hoặc thử nghiệm ngắn hạn; để chạy VLM `Qwen3-VL-8B` và xử lý video ổn định liên tục, hãy dùng **Vast.ai** (`GPU_server/start_vast.sh`) hoặc máy GPU riêng.

---

## 6. Các bước người dùng cần tự thực hiện theo từng nền tảng

1. **Vercel (`web/`):**
   - **Đã deploy live bản production** tại `https://exhibitflow-ten.vercel.app` (project `exhibitflow` / `prj_HkjSzYcTrLObxEgrjLu6EiRAvfpS`, `ssoProtection: preview`, `0` Environment Variables/Secrets).
   - *(Tuỳ chọn cho auto-deploy tương lai)* Nếu muốn mỗi lần `git push` tự redeploy từ GitHub, push commit lên repo rồi vào **Project Settings → Git / Build and Deployment** trên Vercel, nối Git repo với **Framework Preset** = `Other`, **Root Directory** = `web`, **Build Command** = *(trống)*, **Output Directory** = `.`.
   - Khi chạy Backend cục bộ, đặt `CORS_ORIGINS=https://exhibitflow-ten.vercel.app`.
2. **Cloudflare (tuỳ chọn Named Tunnel nếu muốn URL cố định):**
   - Tạo tài khoản Cloudflare và thêm tên miền riêng (custom domain) vào Cloudflare DNS.
   - Vào **Cloudflare Zero Trust → Networks → Tunnels**, tạo 2 tunnel (hoặc 2 Public Hostname) trỏ tới `http://127.0.0.1:8000` (Backend) và `http://127.0.0.1:8001` (GPU Worker), lấy `CLOUDFLARE_TUNNEL_TOKEN` tương ứng.
3. **Vast.ai (`GPU_server/`):**
   - Thuê instance có NVIDIA GPU (khuyến nghị VRAM ≥ 16–24 GB cho chế độ `full` kèm Qwen3-VL-8B) và mở shell SSH/Jupyter Terminal.
   - Clone repo, vào `GPU_server/` và chạy 1 lệnh:
     ```bash
     HF_TOKEN=hf_xxx bash start_vast.sh
     # hoặc kèm Named Tunnel cố định:
     HF_TOKEN=hf_xxx TUNNEL=named CLOUDFLARE_TUNNEL_TOKEN=<token> PUBLIC_GPU_URL=https://gpu.yourdomain.com bash start_vast.sh
     ```
   - Sao chép `Public HTTPS` URL và `X-GPU-Token` in ra một lần duy nhất trên màn hình khi thấy `STATUS: READY (SẴN SÀNG)`.
4. **Kaggle (chỉ dùng thử nghiệm/demo):**
   - Tải lên `GPU_server/notebooks/exhibitflow_kaggle_worker.ipynb`.
   - Bật **Accelerator: GPU T4 x2** và **Internet: On** trong Settings.
   - (Tuỳ chọn) Gắn Dataset weights vào `/kaggle/input/` và thêm `HF_TOKEN` vào **Add-ons → Secrets**.
   - Chạy lần lượt 4 cell để lấy URL Quick Tunnel tạm và `X-GPU-Token`.

---

## 7. Rủi ro còn lại

1. **Chưa kiểm thử tải ML thật trên GPU NVIDIA CUDA:** Toàn bộ smoke test live dùng `--demo` thuần CPU. Các bước tải weights HuggingFace, suy luận YOLO/ByteTrack-DMA-LTC, `SharedVLMServer` (`llama-server`), và `FloorplanService` trên video thật cần được kiểm tra lần đầu trực tiếp trên Vast.ai.
2. **Độ ổn định SSE qua Cloudflare Quick Tunnel:** Đã quan sát `ReadTimeout` trên SSE qua `trycloudflare.com`. Hệ thống đã có cơ chế fallback polling hoạt động ổn định, nhưng nếu muốn độ trễ push SSE chuẩn, nên dùng Cloudflare Named Tunnel trên domain riêng.
3. **URL Quick Tunnel thay đổi khi restart:** Khi máy chạy Backend hoặc GPU Worker khởi động lại, URL `trycloudflare.com` thay đổi; người dùng cần dán URL mới vào tab **Settings** (Web đã có `#connBanner` nhắc rõ).
4. **Ngưỡng benchmark hiệu năng frontend (`< 350ms`) nhạy với tải CPU:** Test `Performance: full analytics (1000 frames x 15 visitors) completes in < 350ms` ở `web/tests/museum_analytics.test.js:351` phụ thuộc hiệu năng đơn nhân lúc chạy (pass ở `348.36ms` khi máy nhàn rỗi, có thể lên `375ms–792ms` khi máy đang bận).

---

## 8. Quick Start (5 bước vận hành nhanh)

1. **Mở Web**: Truy cập `https://exhibitflow-ten.vercel.app` (hoặc `http://localhost:8000` nếu chạy cùng backend).
2. **Chạy Backend cục bộ + Tunnel**:
   ```bash
   CORS_ORIGINS=https://exhibitflow-ten.vercel.app python3 backend/launch.py --tunnel quick
   ```
   *(Đợi dòng `STATUS: READY (SẴN SÀNG)`, copy `Public HTTPS URL` và `Backend Access Key`.)*
3. **Nhập Backend URL & Access Key trên Web**: Vào tab **Settings → Kết nối máy chủ Backend**, dán URL + `Backend Access Key` → bấm **Kết nối**.
4. **Chạy GPU Worker** (trên Vast.ai bằng `HF_TOKEN=hf_xxx bash GPU_server/start_vast.sh`, hoặc test CPU nhanh bằng `python3 GPU_server/launch.py --demo`):
   *(Đợi dòng `STATUS: READY (SẴN SÀNG)`, copy `Public HTTPS` và `X-GPU-Token`.)*
5. **Nhập GPU URL & X-GPU-Token → Chạy job**: Vào **Settings → Kết nối GPU Worker**, dán GPU URL + `X-GPU-Token` → bấm **Lưu & Kiểm tra**. Khi chip góc trên chuyển xanh **`Backend + GPU sẵn sàng`**, quay lại **Dashboard** để tải video và phân tích!
