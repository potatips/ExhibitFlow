# ExhibitFlow Web Dashboard

Dashboard giao diện người dùng cho hệ thống phân tích hành vi khách tham quan bảo tàng ExhibitFlow. Được xây dựng theo kiến trúc **3 tầng tách biệt (Decoupled 3-Tier Architecture)**:
1. **Frontend (`web/`)**: Thuần HTML5, CSS3, JavaScript ES6+, zero build step, zero framework.
2. **Backend (`backend/`)**: Máy chủ CPU FastAPI siêu nhẹ, quản lý hàng đợi job FIFO, lưu trữ dữ liệu (SQLite/Redis), ArtifactStore (Local/Vercel Blob), phân phối sự kiện Server-Sent Events (SSE), và phục vụ xem lại ngoại tuyến.
3. **GPU Worker (`GPU_server/`)**: Worker tính toán AI độc lập (Vast.ai, Kaggle, Cloud GPU), chạy các mô hình nặng (YOLO, ByteTrack-DMA-LTC, Qwen3-VL-8B shared server, 2D Floorplan) và giao tiếp qua API nội bộ với token xác thực `X-GPU-Token`.

---

## Kiến trúc 3 tầng (3-Tier Architecture)

```
┌─────────────────────────────────┐
│     Trình duyệt (web/)          │
│  - Static HTML/CSS/JS           │
│  - SSE đa người dùng live       │
│  - Xem lại lịch sử offline      │
└────────────────┬────────────────┘
                 │ HTTP / SSE (/api/*)
┌────────────────▼────────────────┐
│   Backend Server (backend/)     │
│  - CPU FastAPI (không cần GPU)  │
│  - SQLite (WAL) / Redis Store   │
│  - Local / Cloud Artifact Store │
│  - SSE Broker & Job Dispatcher  │
└────────────────┬────────────────┘
                 │ HTTP (/internal/*) + X-GPU-Token
┌────────────────▼────────────────┐
│   GPU Worker (GPU_server/)      │
│  - YOLOv11 & ByteTrack-DMA-LTC  │
│  - Shared Qwen3-VL-8B Server    │
│  - 2D Floorplan Service         │
└─────────────────────────────────┘
```

---

## Các tính năng chính

### 1. Trạng thái kết nối trung thực & Cấu hình bảo mật
- **Tự động nhận diện backend**: Khi mở dashboard qua backend (cùng host/origin), trang tự động kết nối `API.base = ''`. Khi triển khai tĩnh trên Vercel hoặc server riêng, người dùng cấu hình URL Backend trong **Settings**.
- **Chip trạng thái đa tầng (`#beChip`)**:
  - `Backend + GPU sẵn sàng` (`.ok`): Cả backend và GPU worker đều trực tuyến, sẵn sàng nhận job mới (kèm chú thích nếu GPU đang bận hoặc chạy Demo).
  - `Backend sẵn sàng · GPU đang nạp model` (`.warn`): Backend trực tuyến, GPU đang tải checkpoint mô hình AI.
  - `Backend sẵn sàng · GPU ngoại tuyến` (`.warn`): Backend online, GPU worker đang tắt hoặc chưa kết nối tunnel. Toàn bộ tính năng xem lại lịch sử video, bản đồ 2D và duyệt cảnh báo vẫn hoạt động bình thường mà không cần GPU.
  - `Cần xác thực Backend` (`.bad`): Backend yêu cầu Mã truy cập (API Key) nhưng chưa nhập hoặc mã không chính xác.
  - `Mất kết nối Backend` (`.bad`): Không kết nối được tới Backend (hoặc URL Cloudflare Tunnel đã thay đổi).
- **Banner phục hồi kết nối (`#connBanner`)**:
  - Xuất hiện trên đầu Dashboard khi mất kết nối hoặc URL thay đổi, kèm thông điệp tiếng Việt rõ ràng và nút 1-click **Mở Settings** để cập nhật nhanh.
  - Tự động ẩn khi kết nối phục hồi ổn định.
- **Xác thực Backend & Stream Tickets ngắn hạn**:
  - Hỗ trợ nhập `Mã truy cập Backend (API Key)` và tuỳ chọn `Ghi nhớ mã truy cập` (`localStorage` vs `sessionStorage`).
  - Frontend tự động gắn header `Authorization: Bearer <key>` vào các API call.
  - Với các phần tử trình duyệt không gửi được custom header (`<video src>`, `<img src>`, `EventSource`, xuất file CSV), frontend tự động lấy HMAC stream ticket ngắn hạn (`POST /api/auth/stream-token`) và gắn query param `?st=<ticket>`, đảm bảo dữ liệu luôn được bảo vệ an toàn.
- **Khả năng chịu lỗi Quick Tunnel (SSE Resilience)**:
  - Khi chạy qua Cloudflare Quick Tunnel, nếu luồng SSE toàn cục bị ngắt kết nối nhiều lần, frontend tự động chuyển sang chế độ polling chu kỳ 5 giây mà không gây cảnh báo giả "Mất kết nối".
- **Cấu hình GPU Worker trong Settings**:
  - Nhập `GPU Worker URL` và `X-GPU-Token`.
  - Token được bảo mật tuyệt đối trên máy chủ backend, không bao giờ gửi ngược lại client (chỉ hiện placeholder đã lưu).
  - Kiểm tra trạng thái GPU trực tiếp với nút **Lưu & Kiểm tra**.

### 2. Live Streaming & Server-Sent Events (SSE)
- **Đa người dùng thời gian thực**: Kết nối tới `/api/events` để đồng bộ trạng thái GPU, danh sách job và trạng thái duyệt cảnh báo giữa nhiều tab/người dùng cùng lúc.
- **Xếp hàng & Đồng bộ mượt mà**: Hỗ trợ đầy đủ các trạng thái `queued` (hiển thị vị trí hàng đợi `#N`), `running` (tiến độ theo từng bước), `syncing` (đồng bộ dữ liệu về máy chủ), `done`, `error`, `cancelled`.
- **Luồng MJPEG độ trễ thấp**: Chế độ Realtime phát luồng trực tiếp từ `/api/jobs/{id}/live/stream`, tự động chuyển sang chế độ polling nếu kết nối gián đoạn.

### 3. Lịch sử phân tích & Xem lại ngoại tuyến (Past Jobs & Offline Replay)
- **Bảng lịch sử phân tích**: Lọc theo *Tất cả*, *Hoàn tất*, *Đang chạy / Hàng đợi*, *Lỗi*; hiển thị tên job, loại (1 camera / Đa camera), trạng thái, số cảnh báo, thời gian.
- **Xem lại không phụ thuộc GPU**:
  - Video phát trực tiếp qua HTTP Range (`206 Partial Content`), hỗ trợ tua mượt mà không cần tải toàn bộ file về bộ nhớ.
  - Khôi phục đầy đủ bản đồ 2D, luồng di chuyển, heatmap mật độ, ảnh bằng chứng VLM và quyết định đánh giá của nhân viên kiểm duyệt.
  - Hỗ trợ xem lại cả phân tích đơn camera và phân tích ghép mặt bằng đa camera (`site-jobs`).

### 4. Kiểm duyệt cảnh báo bền vững (Persistent Alert Review)
- Nhân viên có thể bấm **Xác nhận** hoặc **Bỏ qua** cho từng cảnh báo; trạng thái được lưu bền vững vào cơ sở dữ liệu qua `PATCH /api/alerts/{id}` và đồng bộ tức thời tới các phiên làm việc khác.
- **Chế độ xem toàn cục (Tất cả lịch sử)**: Chuyển đổi giữa *Video hiện tại* và *Tất cả lịch sử* để quản lý và lọc cảnh báo trên toàn hệ thống.
- **Xuất báo cáo CSV**: Tải file CSV chứa danh sách cảnh báo chi tiết theo bộ lọc hiện tại.

---

## Hướng dẫn khởi chạy

### Chạy cùng backend (Khuyến nghị khi phát triển cục bộ)
Khi backend FastAPI chạy, nó tự động phục vụ thư mục `web/` tại root `/`:
```bash
# Tại thư mục gốc repository
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Truy cập trình duyệt tại: `http://localhost:8000/`

### Triển khai tĩnh trên Vercel (Vercel Static Deployment)
ExhibitFlow frontend hoàn toàn là static HTML/CSS/JS, sẵn sàng deploy lên Vercel Edge Network chỉ trong 1 phút:

1. **Kết nối Git repo vào Vercel**.
2. **Cài đặt cấu hình dự án (Project Settings)**:
   - **Framework Preset**: `Other`
   - **Root Directory**: `web` (hoặc để root và Vercel tự đọc file `web/vercel.json` khi build)
   - **Build Command**: Để trống (không cần build)
   - **Output Directory**: Để trống hoặc `.`
3. **Cơ chế bảo mật & cấu hình Edge**:
   - File `web/vercel.json` thiết lập đầy đủ security headers: `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, cùng chính sách cache `no-cache` cho `index.html` và `config.js`.
   - File `.vercelignore` ở thư mục gốc tự động loại trừ mọi backend code, GPU code, models, venv và database — **tuyệt đối không có secret hay credential nào bị lộ lên frontend**.
4. **Kết nối Backend từ Vercel**:
   - Khởi chạy backend trên máy chủ hoặc cloud với Cloudflare Tunnel (`python backend/launch.py --tunnel quick` hoặc `named`).
   - Mở domain Vercel trong trình duyệt, vào tab **Settings**, dán URL HTTPS Cloudflare Tunnel vào ô **URL backend** (và nhập **Mã truy cập Backend** nếu backend bật auth) rồi bấm **Kết nối**.

### Chạy web độc lập (Static Dev Server)
```bash
cd web
python -m http.server 8765 --bind 127.0.0.1
```
Mở `http://127.0.0.1:8765/index.html`, vào **Settings** và nhập URL backend (ví dụ: `http://localhost:8000`).

---

## Cấu trúc thư mục

```
web/
├── index.html              # Trang giao diện chính (Dashboard + Settings)
├── README.md               # Tài liệu giao diện và tích hợp API
├── css/
│   ├── style.css           # Định kiểu cơ sở
│   └── theme-light.css     # Chủ đề trắng - đen - cam (chuẩn nhận diện ExhibitFlow)
├── js/
│   ├── app.js              # Xử lý logic dashboard, API client, SSE, render bản đồ & cảnh báo
│   ├── config.js           # Cấu hình môi trường (demo: false/true)
│   ├── heatmap.js          # Thuật toán tính toán heatmap & mật độ không gian
│   └── museum_analytics.js # Thống kê chuyên sâu bảo tàng (Holding/Attraction Power, mỏi bảo tàng, archetypes)
└── tests/
    ├── heatmap.test.js     # Kiểm thử thuật toán heatmap
    └── museum_analytics.test.js # Kiểm thử thống kê chuyên sâu bảo tàng
```

---

## Kiểm tra chất lượng (Verification)

Chạy kiểm tra cú pháp và bộ kiểm thử tự động:
```bash
# Kiểm tra cú pháp JavaScript
node --check web/js/app.js

# Chạy toàn bộ 23 bài kiểm thử unit/regression
node --test web/tests/heatmap.test.js web/tests/museum_analytics.test.js
```
