# ExhibitFlow Web + Backend: hướng dẫn chạy

Giao diện web (thư mục `web/`) gửi video lên **backend** chạy engine của dashboard (model nạp một lần, chạy nhiều video) rồi hiện video có khung người, danh sách cảnh báo (chạm / cầm / nghi lấy cắp) và trạng thái máy.

```
 Máy người dùng                         Máy có GPU (Kaggle / Vast.ai / server riêng)
┌─────────────────┐   URL (cloudflared)  ┌────────────────────────────────────────┐
│ web/ (serve.py) │ ───────────────────► │ server/web_launch.py                   │
│ trình duyệt     │ ◄─────────────────── │  └─ server/web_server.py (FastAPI)     │
└─────────────────┘   API /api/jobs ...  │      └─ dashboard.runner + Engine      │
                                          └────────────────────────────────────────┘
```

## File nằm ở đâu (nhánh `Thuan`)

| Đường dẫn | Việc |
|---|---|
| `web/` | Giao diện đen. Chạy trên **máy người dùng** (`python serve.py`). |
| `exhibitflow_pipeline/server/web_server.py` | Backend nối web với engine của dashboard (`State`, `PipelineRunner`). Không sửa file nào của `dashboard/`. |
| `exhibitflow_pipeline/server/web_launch.py` | Chạy `web_server.py` + mở URL công khai (cloudflared), in URL để dán vào web. |
| `exhibitflow_pipeline/notebooks/exhibitflow_kaggle_server.ipynb` | Notebook Kaggle: clone, cài thư viện, tìm trọng số, chạy backend. |

## Cách 1: Thử nhanh không cần GPU (dữ liệu giả)

Dùng để xem giao diện và luồng hoạt động. Không chạy pipeline thật.

```bash
pip install fastapi uvicorn python-multipart psutil
cd exhibitflow_pipeline
python server/web_server.py --port 8000 --demo
```

Cửa sổ khác, chạy web:

```bash
cd web
python serve.py
```

Mở link `serve.py` in ra (`http://127.0.0.1:8765/index.html`), vào **Settings → URL backend**, dán `http://127.0.0.1:8000` → **Kết nối** → Upload một video bất kỳ → **Phân tích**.

## Cách 2: Kaggle (GPU thật)

1. Kaggle → New Notebook → **File → Import Notebook** → chọn `exhibitflow_kaggle_server.ipynb`.
2. Settings: **Accelerator = GPU T4 x2**, **Internet = On**. **Add Input**: dataset chứa 4 file trọng số `best_ckpt.pth`, `osnet_x0_25.pth`, `ltc_motion_mot17.pth`, `dma_gbm_mot17.gbm`.
3. Ô 1: `BRANCH = 'Thuan'` (đổi nếu code nằm ở nhánh khác).
4. Chạy lần lượt các ô. Ô 5 hỏi `HF_TOKEN` (Enter để bỏ qua), nạp model rồi in **URL BACKEND**.
5. Giữ ô 5 chạy. Dán URL vào web (như Cách 1) → **Kết nối**.

Ô 5 không bao giờ "chạy xong"; nó chạy mãi để giữ server sống. URL chỉ in sau dòng `Model đã sẵn sàng.`

## Cách 3: Server thuê (Vast.ai) hoặc máy Linux có GPU

Dùng `setup.sh` có sẵn (lấy đúng nhánh):

```bash
BRANCH=Thuan HF_TOKEN=hf_xxx bash setup.sh
cd /workspace/ExhibitFlow/exhibitflow_pipeline
python server/web_launch.py --port 8000 --data-dir /workspace/dashboard_data
```

Lệnh cuối in **URL BACKEND**. Chạy web trên máy của bạn rồi dán URL như trên. Muốn không dùng cloudflared (trình duyệt truy cập được thẳng IP máy), chạy
`python server/web_server.py --host 0.0.0.0 --port 8000` rồi nhập `http://<ip>:8000` ở Settings (trang web chạy `https://` thì backend cũng phải `https://`).

## Dùng giao diện

1. Menu bên trái (nút ở góc trên) → **Settings** → **URL backend** → **Kết nối**. Chip đầu trang đổi thành "Đã kết nối".
2. **Dashboard**: chọn **Video**, bấm **Upload** (hoặc nhập đường dẫn trên server, vd. `/kaggle/input/<dataset>/4.mp4`) → **Phân tích**. **Dừng** để hủy.
3. Dòng thông báo cho biết bước đang chạy (`[1/5] ... [4/5] Phân tích video`). Xong hiện "Hoàn tất", cảnh báo, KPI.
4. Danh sách cảnh báo: lọc Tất cả / Bất thường / Cần xác nhận / Đã loại, **Xem lại / Xác nhận / Bỏ qua**, **Xuất CSV**.

## Map 2D (multicam-2d-floorplan-tracker)

Khối **Map 2D** trên web hiện **ảnh map + vị trí người + hiện vật** do repo `multicam-2d-floorplan-tracker` tạo (qua `dashboard/mapgen.py` của Hưng).

- **Cách hoạt động:** khi bấm Phân tích, backend lấy một khung hình của video, ước lượng sàn bằng Depth-Anything V2 + SegFormer, dựng ảnh map, rồi chiếu chân từng người vào map. Web vẽ chấm người (xanh = bình thường, vàng = đang chạm, đỏ = cảnh báo), hiện vật (vuông xanh), **Vệt đi** (6 giây gần nhất) và **Vùng gần** (vòng quanh hiện vật). Sau khi chạy xong, kéo/tua video thì chấm người chạy theo thời gian video.
- **Cần có:** repo `multicam-2d-floorplan-tracker` (notebook Kaggle tự clone và đặt `FLOORPLAN_DIR`; trên Vast, `make setup` của Hưng cài vào `external/`). Lần đầu tool tự tải model (Depth-Anything V2, SegFormer, vài trăm MB), nên **video đầu tiên chậm hơn khoảng 15 đến 25 giây** (hoặc hơn nếu phải tải model).
- **Tắt tạo map** (nhanh hơn, khối Map 2D giữ chỗ như cũ): đặt biến môi trường `EXHIBITFLOW_MAKE_MAP=0`.
- **Giới hạn:** tool vốn làm cho 5 camera nhiều phòng; ở đây chỉ dùng **1 khung hình của 1 camera** nên map là một phòng suy ra từ ảnh. Nếu cảnh không có tường rõ, map chỉ là lưới mét. Tool ghim `torch 2.11`, `transformers 4.57.3`; chạy chung môi trường với pipeline nên có thể xung đột phiên bản. **Chưa thử với map thật**, mới thử bằng `--demo` (map giả).
- Nếu tạo map lỗi, pipeline vẫn chạy bình thường và khối Map 2D chỉ hiện khung giữ chỗ (xem log của server có dòng `Không tạo được map 2D`).

### Nhiều camera

Chọn **nhiều video cùng lúc** khi bấm **Upload** (giữ `Ctrl` hoặc `Shift`). Có hai cách chạy, tuỳ ô **Địa điểm** bên cạnh ô nguồn:

**1. Địa điểm: "không" (mặc định)**: mỗi video chạy **riêng, lần lượt** như một camera với pipeline của Hưng (cảnh báo hành vi + map ước lượng từ chính video đó). Xong hiện các tab **CAM 1, CAM 2...** (kèm số cảnh báo). Mỗi camera có map riêng, **chưa ghép chung**.

**2. Chọn một địa điểm đã hiệu chuẩn (ví dụ WILDTRACK)**: các video được **ghép thành MỘT map chung** bằng repo `multicam-2d-floorplan-tracker`:
- Web tự gán video vào camera theo số trong tên file (`cam1...`, `C2...`), file nào không có số thì gán theo thứ tự.
- Server cắt khung đã khử méo (4 khung/giây), rồi chạy các bước lõi của repo: phát hiện người đa kích thước (YOLO pose, ngoại suy điểm chân theo tỉ lệ cơ thể), tinh chỉnh ma trận chiếu giữa các camera, **hợp nhất người giữa các camera và gán ID chung**.
- Kết quả: khối **Map 2D** vẽ mặt phẳng chung với **vùng nhìn của từng camera**, **người với ID chung**, **vệt đi** (6 giây gần nhất, bật/tắt bằng nút Vệt đi). Bấm tab **CAM 1, CAM 2...** để xem video từng camera (camera đang chọn sáng hơn trên map); tua video thì người trên map chạy theo.
- Mặc định chế độ này còn chạy **engine của Hưng trên từng camera** để phát hiện hành vi bất thường rồi chiếu lên map chung (xem mục **Hiệu chuẩn tự nhập, hành vi bất thường trên map, heatmap** ngay dưới). Muốn tắt (chỉ ghép map và người, nhanh hơn) thì đặt biến môi trường `EXHIBITFLOW_SITE_EVENTS=0` trên server.

#### Hiệu chuẩn tự nhập, hành vi bất thường trên map, heatmap

Không có sẵn địa điểm trong danh sách? Chọn video rồi dùng ô **Hiệu chuẩn camera** (hiện ngay dưới ô nguồn khi đã chọn video, ô Địa điểm để "không"):
- Mỗi video một file JSON, hoặc **một file chung** cho cả các camera (kiểu `transforms.json`: `{"cameras": {"C01": {...}, "C02": {...}}}`, gán theo tên/thứ tự). Mỗi camera gồm `K` (3×3), `R` (3×3, hoặc `rvec` Rodrigues) và `t` (3 số) theo quy ước OpenCV **world→camera** (`Xcam = R·Xworld + t`), đơn vị **mét** (có thể ghi `"units": "cm"`), trục Z hướng lên, mặt sàn Z = 0; tuỳ chọn `dist` (5 hệ số méo) và `image_width` / `image_height`. Các tên khoá khác cũng được nhận (`camera_matrix`, `R_world_to_camera`, `t_world_to_camera`, `tvec`, `fx/fy/cx/cy`...).
- File hiệu chuẩn có **giới hạn sàn phòng** thì server dùng nó: cắt **vùng nhìn của camera** theo phòng, đặt vùng hợp lệ của map đúng bằng phòng và **bỏ người nằm ngoài phòng** (bóng, phản chiếu, nhiễu). Ghi bằng `room_bounds` (`[x0, y0, x1, y1]` mét, X đông, Y bắc) hoặc `room_landmarks_m.floor_bounds_X` + `floor_bounds_Z` (hệ Y-up). Không ghi thì vùng nhìn chỉ giới hạn theo tầm xa 25 m.
- Video **khác độ phân giải hiệu chuẩn** (ví dụ đã hạ xuống 240p) thì server tự co `K` theo tỉ lệ (cùng tỉ lệ khung). Sai tỉ lệ khung thì báo lỗi.
- Server kiểm tra hiệu chuẩn và báo lỗi rõ khi thiếu `K`/`R`/`t`, `R` không phải ma trận xoay, camera nằm sát/dưới sàn, hoặc tia nhìn không chạm sàn. **Không bắt được mọi sai sót**: hiệu chuẩn sai quy ước (ví dụ camera→world) đôi khi vẫn lọt và map sẽ lệch mà không báo lỗi.
- Server chạy engine của Hưng lần lượt từng camera (GPU), lấy hộp người theo từng khung và cảnh báo, đổi **điểm chân** (giữa cạnh dưới hộp) sang toạ độ sàn bằng hiệu chuẩn, gộp các camera cùng thấy một sự kiện (trùng thời gian, gần nhau dưới 2 m), rồi gắn với người ghép chung gần nhất để lấy **đường đi** qua nhiều camera. Thời gian chạy khoảng **gấp đôi** so với chỉ ghép map.
- **Map 2D** mặc định chỉ hiện **hành vi bất thường** (vòng đỏ có nhãn, giây xảy ra) và **đường đi của người gây ra** (nét đứt mờ = cả đường đi, nét liền sáng = đoạn đã đi tới thời điểm đang xem, chấm đặc = vị trí hiện tại). Nút **Mọi người** bật thêm người bình thường, **Vệt đi** bật/tắt đường đi. Danh sách cảnh báo của từng tab camera hiện các hành vi mà camera đó thấy.
- Tab **Heatmap** (dưới map): mật độ **thời gian có mặt** của mọi người trong cả video (ô khoảng 0,25 đến 1 m, làm mịn), đánh dấu **khu đông nhất** kèm vùng camera thấy khu đó. Video ngắn hoặc ít người thì heatmap thưa. Với một camera chạy cách 1, tab Heatmap vẽ theo toạ độ map ước lượng (tỉ lệ phần trăm, không phải mét).
- **Giới hạn:** vị trí sự kiện là chân người ước lượng từ hộp bao nên **sai số khoảng 0,3 đến 1 m** khi người ở xa hoặc bị che; người bị cắt ngang đáy khung hình cho điểm chân kém. Bước chiếu này **chưa thử với engine thật** (chỉ thử bằng engine giả trên 3 video WILDTRACK, kiểm tra được phần toán, hiệu chuẩn tự nhập, gộp camera và giao diện).

Yêu cầu video cho cách 2: đúng độ phân giải của hiệu chuẩn (WILDTRACK: 1920×1080), **các video cùng một khoảng thời gian** (cùng giây bắt đầu), **H.264** (trình duyệt không phát được MPEG-4 Part 2 `mp4v`, video sẽ đen dù map vẫn đúng). Server chỉ dùng **tối đa 30 giây đầu** mỗi video (đổi bằng `EXHIBITFLOW_SITE_MAX_SECONDS`). Thiết bị chạy người: GPU nếu có, không thì CPU (`EXHIBITFLOW_SITE_DEVICE` để ép). Tham khảo: 3 video × 8 giây mất khoảng 90 giây trên CPU máy cá nhân.

Cần có trên server (notebook Kaggle đã làm sẵn ở ô 3): repo `multicam-2d-floorplan-tracker` (`FLOORPLAN_DIR`), `pip install ultralytics shapely`, và model của repo (`python scripts/download_person_models.py`: `yolo11n-pose.pt`, `resnet18`).

**Vì sao không dùng bộ dựng tường của repo:** `ArchitecturalMapBuilder` dựng mặt bằng từ tường (độ sâu) nên không chạy được ở nơi **không có tường** như sân WILDTRACK (báo `Insufficient points for a structural plane`). Với địa điểm đã hiệu chuẩn, web dùng thẳng hiệu chuẩn rồi gọi `collect` + `process` của repo (`server/site_driver.py`), và vẽ map (vùng nhìn camera, người) ngay trên giao diện.

**Thêm địa điểm mới:** tạo `exhibitflow_pipeline/server/sites/<id>.json` (xem `wildtrack.json`): `site_id`, `name`, `room_id`, `sample_fps`, `cameras` với mỗi camera gồm `K` (3×3), `dist` (5 hệ số méo; để 0 nếu video đã khử méo), `R_world_to_camera` (3×3), `t_world_to_camera` (mét), `width`, `height`, `label`. Hệ toạ độ thế giới: mét, mặt đất Z = 0. Khởi động lại server để thấy địa điểm mới.

**Dữ liệu WILDTRACK dùng để thử:** hiệu chuẩn camera 1 đến 3 (`CVLab1..3`, ngoại tại đổi từ cm sang mét, nội tại bản gốc có khử méo) lấy từ bộ [WILDTRACK](https://epfl.ch/labs/cvlab/data/data-wildtrack) của EPFL CVLab (trang không nêu rõ giấy phép; chỉ dùng để thử nghiệm, trích dẫn: Chavdarova et al., CVPR 2018). Video thử là camera 1 đến 3, đoạn giây 60 đến 68, đã nén H.264.

## Giới hạn hiện tại

- **Chỉ chế độ Video.** Realtime bằng camera (RTSP) chưa chạy: engine mới đọc file video, server trả lỗi 501.
- Lúc đang chạy, giao diện chỉ hiện cảnh báo VLM đã kết luận **bất thường**; danh sách đầy đủ hiện khi chạy xong.
- Map 2D xem mục **Map 2D** bên dưới. Chất lượng map phụ thuộc cảnh (xem lưu ý ở đó).
- Server chỉ chạy **một job một lúc**. Gửi job khi model đang nạp hoặc đang có job khác → lỗi 409 kèm lý do.
- Đường dẫn trên server chỉ lấy trong `/kaggle/input`, `/kaggle/working`, `/content` (đổi bằng biến `EXHIBITFLOW_SOURCE_ROOTS`).
- Backend này **chưa được thử** với GPU thật trên Kaggle / Vast.ai tại thời điểm viết; mới thử với `--demo`.

## Biến môi trường (tuỳ chọn)

| Biến | Tác dụng |
|---|---|
| `EF_WEIGHTS_DIR` | Thư mục trọng số (notebook tự tìm trong `/kaggle/input`) |
| `HF_TOKEN` | Token Hugging Face để tải model nhanh hơn. **Không ghi vào file.** |
| `EXHIBITFLOW_SOURCE_ROOTS` | Thư mục server được phép đọc video |
| `EXHIBITFLOW_MAX_MB` | Giới hạn upload (mặc định 600) |
| `EXHIBITFLOW_SAVE_VIDEO=0` | Không ghi `live.mp4` (video có khung) |
| `EXHIBITFLOW_SITE_EVENTS=0` | Chế độ nhiều camera chỉ ghép map + người, không chạy engine hành vi bất thường |

## Gặp lỗi?

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| Ô 2 Kaggle báo "Không thấy server/web_server.py" | Nhánh trong `BRANCH` chưa có file này (chưa push). Kiểm tra tên nhánh. |
| Chip "Mất kết nối" | URL sai hoặc đã đổi (URL cloudflared đổi mỗi lần chạy lại ô cuối); ô 5 đã dừng. |
| Lỗi 409 khi Phân tích | Model đang nạp hoặc đang có job khác: đợi hoặc bấm Dừng. |
| Chọn địa điểm mà báo lỗi `Chưa cài repo multicam-2d-floorplan-tracker` | Server không thấy repo map2d: đặt `FLOORPLAN_DIR` (notebook Kaggle tự đặt ở ô 2) |
| Lỗi 501 | Nguồn camera (RTSP/RTMP) chưa hỗ trợ; dùng file video. |
| Log có `CUDNN_STATUS_NOT_SUPPORTED`, `Falling back to CPUExecutionProvider` | Model bàn tay (RTMW) rơi về CPU vì cuDNN không khớp trên máy này (hay gặp ở Kaggle T4). Vẫn chạy, chỉ **chậm hơn**. |
| Video xem lại sau khi chạy xong không có khung người | Server không có `live.mp4` để phát (đang `--demo`, hoặc `EXHIBITFLOW_SAVE_VIDEO=0`, hoặc ghi video lỗi) nên web phát video gốc. |

## An toàn

- URL backend **không có mật khẩu**: ai có URL đều upload và chạy được trên máy của bạn. Không đăng công khai; tắt máy thuê khi dùng xong.
- Không ghi key / token (HF, cloudflared...) vào file rồi push.
