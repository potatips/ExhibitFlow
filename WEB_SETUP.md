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

## Cách 3 (khuyến nghị): chạy tất cả trên Vast.ai, không cần URL backend

`web_server.py` phục vụ luôn giao diện `web/` ở `/` (cùng cổng với API), nên mở một địa chỉ là dùng được.

```bash
# trên instance Vast (template mở port 8000, xem VAST_SETUP.md)
BRANCH=main HF_TOKEN=hf_xxx START=1 bash setup.sh     # hoặc: make run
make url                                              # in http://<IP>:<port-ngoài>
```

`make run` chạy `server/web_server.py` nền (log: `make logs`). Mở `http://<IP>:<port-ngoài>/` trên trình duyệt, đợi chip trạng thái báo **Đã kết nối**
(lần đầu nạp model vài phút), rồi Upload video hoặc nhập đường dẫn trên server (vd `/workspace/...`) và bấm Phân tích.
Ô URL trong Settings để trống. Không cần tunnel, Kaggle hay `serve.py`.  Chưa mở được port ngoài: thêm `-p 8000:8000` vào Docker options của template.
Chỉ dùng cách cũ (`serve.py` + dán URL) khi muốn mở web ở máy khác.

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

### Ba cách dùng (video có hoặc không có hiệu chuẩn)

Bấm **Upload** (giữ `Ctrl` để chọn nhiều video). Có hiệu chuẩn hay không quyết định đường chạy:

| Bạn upload | Server chạy | Map 2D |
|---|---|---|
| **1 video, không hiệu chuẩn** | Engine của Hưng + map ước lượng từ chính video (`mapgen`, dùng bộ dựng mặt bằng của repo trên 1 khung hình) | Map ước lượng, chưa chính xác |
| **1 video, có hiệu chuẩn** | Pipeline đầy đủ của repo (xem dưới) + engine của Hưng | Mặt bằng của repo, người và hành vi bất thường vẽ lên đó |
| **Nhiều video, có hiệu chuẩn** | Như trên, các camera ghép thành MỘT map, người có ID chung | Một mặt bằng chung |
| Nhiều video, không hiệu chuẩn | Mỗi video chạy riêng, lần lượt như cách 1 | Mỗi camera một map, các tab CAM 1, CAM 2... |

**Pipeline của repo `multicam-2d-floorplan-tracker` (khi có hiệu chuẩn)**, chạy bằng `server/site_driver.py` đúng các bước của `main.py --mode all`:
1. Cắt khung đã khử méo (4 khung/giây) và ghi `manifest.json` từ K, R, t của bạn.
2. `build`: **dựng mặt bằng từ tường** (Depth-Anything V2 + SegFormer, `ArchitecturalMapBuilder`), ghi `transforms.json`, `map_config.json`. Kích thước phòng là **ước lượng từ độ sâu** (ghi rõ "model-estimated" trên ảnh).
3. `collect` + `process`: phát hiện người (YOLO pose đa kích thước, ngoại suy điểm chân theo tỉ lệ cơ thể), tinh chỉnh ma trận chiếu, **hợp nhất người giữa các camera và gán ID chung**.
4. `draw_floorplan`: ảnh `global_2d_map.png`. Web dùng ảnh này làm nền Map 2D.
4b. **Nối ID bị đứt** (`stitch_ids` trong `site_events.py`): người bị che hoặc không phát hiện vài giây thì bộ theo dõi cấp ID mới nên cùng một người đổi số. Sau khi có toàn bộ kết quả, track nào bắt đầu trong vòng 3 giây sau khi một track khác kết thúc và nằm gần điểm cuối của nó (0,8 m + 0,8 m/s × khoảng hở) được nối thành một ID; hai người cùng có mặt một lúc thì không bao giờ bị nối. ID cuối được đánh số lại gọn 1..N. Trong lần thử người ngồi băng ghế từ 2 ID (#12, #77) thành 1.
5. Engine của Hưng chạy lần lượt từng camera lấy hành vi bất thường; **điểm chân** người gây ra được đổi sang toạ độ map bằng chính ma trận `H_image_to_map` của repo (nên nằm đúng chỗ trên ảnh mặt bằng), gộp các camera cùng thấy một sự kiện, rồi nối với người ghép chung để vẽ **đường đi**.

Lần đầu repo tự tải model dựng mặt bằng (vài trăm MB, cần Internet). Notebook Kaggle đã cài sẵn `ultralytics`, `shapely` và model người.

**Dự phòng:** cảnh **không có tường** (sân ngoài trời) thì bước dựng mặt bằng của repo báo `Insufficient points for a structural plane`. Khi đó server **vẫn chạy**: bỏ qua bước dựng mặt bằng, tự dựng `transforms`/`map_config` tối thiểu từ K, R, t, vẫn phát hiện / hợp nhất người và chiếu hành vi, nhưng **không có ảnh mặt bằng**: web vẽ lưới 5 m và vùng nhìn camera, và ghi chú lý do ở dòng trạng thái.

#### Hiệu chuẩn camera

Ô **Hiệu chuẩn camera** hiện ngay dưới ô nguồn khi đã chọn video:
- Mỗi video một file JSON, hoặc **một file chung** cho cả các camera (kiểu `transforms.json`: `{"cameras": {"C01": {...}, "C02": {...}}}`, gán theo tên file hoặc thứ tự). Mỗi camera gồm `K` (3×3), `R` (3×3, hoặc `rvec` Rodrigues) và `t` (3 số) theo quy ước OpenCV **world→camera** (`Xcam = R·Xworld + t`), đơn vị **mét** (có thể ghi `"units": "cm"`), mặt sàn là mặt phẳng Z = 0; tuỳ chọn `dist` (5 hệ số méo), `image_width` / `image_height`. Nhận nhiều tên khoá (`camera_matrix`, `R_world_to_camera`, `t_world_to_camera`, `tvec`, `fx/fy/cx/cy`, `dist_coeffs`...).
- Hệ **Y-up** (trục Y hướng lên, ví dụ Unity / three.js) được nhận khi file ghi `coordinate_system.axes` có "Y-up" hoặc từng camera có `"up_axis": "y"`; server tự đổi sang hệ Z hướng lên.
- Video **khác độ phân giải hiệu chuẩn** (ví dụ đã hạ 240p) thì server tự co `K` theo tỉ lệ khung; sai tỉ lệ khung thì báo lỗi.
- Server báo lỗi rõ khi thiếu `K`/`R`/`t`, `R` không phải ma trận xoay, camera nằm sát/dưới sàn, hoặc tia nhìn không chạm sàn. **Không bắt được mọi sai sót**: hiệu chuẩn sai quy ước (ví dụ camera→world) đôi khi vẫn lọt và map sẽ lệch mà không báo lỗi.

#### Map 2D, hành vi bất thường, heatmap
- **Map 2D** (tab Hành vi) chỉ hiện **người chạm hiện vật** (người có hành vi bất thường): **cả đường đi trong suốt video** và **một chấm ở vị trí cuối cùng**, màu theo hành vi nặng nhất; một người chạm nhiều lần chỉ vẽ một đường. Chạm mà VLM kết luận bình thường thì **không** hiện.
- Tab **Heatmap** (dưới map): mật độ **thời gian có mặt** của mọi người trong cả video, đánh dấu **khu đông nhất**. Video ngắn hoặc ít người thì heatmap thưa.
- **3 khung map hiện sẵn:** **Hành vi** (cạnh video), **Luồng đi** (hướng di chuyển chính) và **Heatmap** với hai chế độ **Mật độ** (thời gian có mặt), **Lượt khách** (số người khác nhau đi qua). Tính ngay trên trình duyệt (`js/heatmap.js`, `js/museum_analytics.js`). Kiểm thử: `node --test web/tests/heatmap.test.js web/tests/museum_analytics.test.js`.
- **Giao diện** trắng, đen, cam (`css/theme-light.css`, nạp sau `style.css`).
- Dòng trạng thái cuối cho biết engine đã thấy gì ở mỗi camera (số người theo dõi, số giây chạm, số bất thường) để biết vì sao chưa có hành vi nào.
- Muốn tắt bước hành vi (chỉ ghép map và người, nhanh hơn): biến môi trường `EXHIBITFLOW_SITE_EVENTS=0` trên server.
- **Giới hạn:** vị trí sự kiện là chân người ước lượng từ hộp bao nên **sai số khoảng 0,3 đến 1 m** khi người ở xa hoặc bị che. Kích thước phòng trên mặt bằng là ước lượng từ độ sâu (lệch cỡ 10 đến 15% so với số đo thật trong lần thử). Bước chiếu hành vi **chưa thử với engine thật** (chỉ thử bằng engine giả: kiểm tra được phần toán, hiệu chuẩn, gộp camera, giao diện).

Yêu cầu video: các video **cùng một khoảng thời gian** (cùng giây bắt đầu), **H.264** (trình duyệt không phát được MPEG-4 Part 2 `mp4v`, video sẽ đen dù map vẫn đúng). Server chỉ dùng **tối đa 30 giây đầu** mỗi video (`EXHIBITFLOW_SITE_MAX_SECONDS`). Thiết bị chạy người: GPU nếu có, không thì CPU (`EXHIBITFLOW_SITE_DEVICE` để ép). Tham khảo trên CPU máy cá nhân: 2 video × 10 giây khoảng 1 đến 2 phút (chưa kể engine).

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
| `EXHIBITFLOW_DET_CONF` | Ngưỡng phát hiện người của detector (YOLOX `conf`). Không đặt = mặc định **0.01** của engine (giữ cả khung điểm thấp để bộ theo dõi nối ID). Tăng (vd `0.3`) để bớt phát hiện nhầm, nhưng ID dễ đứt hơn. Áp dụng từ lần phân tích sau, không phải nạp lại model. Trên Kaggle: đặt `DET_CONF` ở ô cấu hình của notebook. Xem giá trị đang dùng ở `/api/health` (`det_conf`) |
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
