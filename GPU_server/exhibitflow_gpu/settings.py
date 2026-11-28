"""Cấu hình mặc định của pipeline ExhibitFlow GPU.

Mọi hằng số nằm ở đây. Các giá trị kiểu dict / list được cập nhật TẠI CHỖ bởi `apply_overrides()` (file YAML / JSON),
nên module khác có thể `from .settings import CFG` mà vẫn thấy giá trị mới.
Các giá trị đơn (đường dẫn, chuỗi, số) phải đọc qua `settings.X` (không `from .settings import X`).
"""
import copy
import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent                      # thư mục GPU_server/
VERSION = '1.0.0'

# ---------------- Thư mục làm việc ----------------
WORK = Path(os.environ.get('EXHIBITFLOW_WORK') or PROJECT_DIR / 'work')
OUT = WORK / 'results'
CACHE = WORK / 'cache'

# Thư mục tìm trọng số (đệ quy, độ sâu hạn chế).
WEIGHT_ROOTS = [Path(p) for p in os.environ.get('EXHIBITFLOW_WEIGHTS', '').split(os.pathsep) if p]
WEIGHT_ROOTS += [PROJECT_DIR / 'weights', Path('/kaggle/input'), Path('/content'), Path('/content/drive/MyDrive')]

# Repo tracker (ByteTrack-DMA-LTC): cần cho detector 'yolox' và tracker 'bytetrack_dma_ltc'.
_TRK_ENV = os.environ.get('EXHIBITFLOW_TRACKER_REPO')
if _TRK_ENV:
    TRACKER_REPO = Path(_TRK_ENV)
else:
    _TRK_CANDS = [
        PROJECT_DIR / 'external' / 'ByteTrack-DMA-LTC-Motion-Tracker',
        PROJECT_DIR.parent / 'external' / 'ByteTrack-DMA-LTC-Motion-Tracker',
    ]
    TRACKER_REPO = next((c for c in _TRK_CANDS if (c / 'yolox').is_dir()), None)
TRACKER_REPO_URL = 'https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker'

try:
    import torch
    N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
    GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(N_GPU)]
except Exception:                                      # chưa cài torch (chỉ đọc cấu hình)
    N_GPU, GPU_NAMES = 0, []
CUDA_VISIBLE_DEVICES_0 = os.environ.get('CUDA_VISIBLE_DEVICES')   # Ultralytics sửa biến này khi predict(device=...)
GPU_A = 'cuda:0' if N_GPU >= 1 else 'cpu'              # luồng perception theo frame
GPU_B = 'cuda:1' if N_GPU >= 2 else GPU_A              # riêng VLM; chỉ 1 GPU thì dùng chung GPU_A

# ---------------- Model ----------------
MODELS = dict(
    detector=dict(backend='yolox', device=GPU_A, conf=0.01, nms=0.7, fp16=True, fuse=True, gpu_preproc=True,
                  exp=None, depth=0.33, width=0.5, num_classes=1, test_size=(608, 1088), ckpt='best_ckpt.pth',
                  low_light_boost=dict(enabled=True, luma_thr=75.0, clip_limit=2.0, tile_grid=8)),
    tracker=dict(backend='bytetrack_dma_ltc', device=GPU_A, with_reid=True,
                 track_thresh=0.5, track_buffer=30, match_thresh=0.9,
                 reid_model='osnet_x0_25', reid_ckpt='osnet_x0_25.pth', reid_weight=0.5, reid_thresh=0.6,
                 ltc_ckpt='ltc_motion_mot17.pth', ltc_min_history=10,
                 dma_ckpt='dma_gbm_mot17.gbm', min_box_area=100, max_aspect=1.6,
                 keep_predicted_frames=8, attach_min_iou=0.30, attach_stable_min_iou=0.15,
                 attach_stable_frames=10, predicted_score_decay=0.9),
    pose=dict(backend='ultralytics_pose', device=GPU_A, weights='yolov8n-pose.pt', batch=True),
    hand=dict(backend='rtmw', device='cuda' if N_GPU else 'cpu', mode='lightweight'),          # | dict(backend='none')
    segmenter=dict(backend='gdino_sam', device=GPU_A, fp16=True,
                   dino='openmmlab-community/mm_grounding_dino_large_all', sam='sam2.1_b.pt'),
    vlm=dict(backend='llama_cpp', repo_id='unsloth/Qwen3-VL-8B-Instruct-GGUF', quant_file='Q4_K_M', mmproj='F16',
             device=GPU_B, n_gpu_layers=99, ctx=16384, max_new_tokens=64, max_pixels=768 * 768, flash_attn='on',
             n_images=8, max_side=512, short_json=True),
)

# ---------------- Engine realtime ----------------
RT = dict(
    mode='parallel',
    realtime=False,
    drop_frames=True,
    queue_size=8,
    pose_gate_norm_dist=2.0,
    artifact_person_filter=True,
    artifact_person_min_score=0.45,
    person_filter=dict(
        enabled=True,
        display_min_score=0.5,
        predicted_min_score=0.05,
        min_hits=3,
        min_box_area=600,
        aspect_range=(0.2, 1.0),
        min_valid_kpts=5,
        pose_min_valid_frac=0.5,
        pose_window=10,
    ),
    seg_ref_default='auto',
    seg_multi_frame=dict(
        enabled=True,
        step_sec=2.0,
        max_frames_per_scene=4,
    ),
    seg_merge_iou=0.5,
    seg_merge_min_votes=1,
    seg_merge_keep_ref=True,
    seg_merge_min_votes_extra=1,
    scene_cut_diff=25.0,
    scene_color_cut=True,
    scene_color_hist_thr=0.35,
    scene_step_sec=0.0,
    scene_min_sec=1.0,
    scene_ref_offset_sec=0.5,
    artifact_cache=True,
    ring_seconds=20.0,
    store_long=1280,
    vlm_early_sec=None,
    periodic_vlm=dict(
        enabled=True,
        min_near_sec=1.0,
        cooldown_sec=3.0,
        max_near_gap_frames=5,
        flush_on_end=True,
        flush_min_sec=1.0,
        flush_max_per_track=2,
    ),
    missing_detect=dict(
        enabled=True,
        check_sec=1.0,
        sustain_sec=1.5,
        max_side=160,
        change_thr=0.35,
        edge_weight=0.5,
        edge_diff_scale=64.0,
        color_diff_scale=48.0,
        max_person_occlude=0.25,
        person_range_norm=2.0,
        global_luma_diff=25.0,
        global_hist_thr=0.35,
        cooldown_sec=4.0,
        carry_detect=True,
        carry_change_thr=0.42,
        carry_sustain_sec=1.5,
    ),
    limit_cpu_threads=False,
    render_live=False,
    alarm_show_sec=4.0,
    alarm_persist=True,
    flag_inherit=dict(
        enabled=True,
        reid_cos=0.7,
        max_gap_sec=3.0,
        max_center_dist_norm=2.0,
    ),
    stream_max_sec=1800.0,
    vlm_sync=True,
    vlm_sync_timeout_s=180.0,
)

# ---------------- Chạy ----------------
RUN = dict(
    max_frames=None,
    wait_vlm_loaded=True,
    wait_vlm_per_video=True,
    gpu_monitor=False,
    gpu_monitor_interval=0.5,
)

# ---------------- Ngưỡng thuật toán ----------------
CFG = dict(
    width=960,
    pose_crop_pad_frac=0.30, pose_predicted_expand=True, pose_min_crop_px=16, pose_conf_thr=0.30, pose_iou_match_thr=0.30, pose_margin_thr=0.15,
    hand_conf_thr=0.30, hand_min_bbox_diag_px=18, hand_small_use_center=True, hand_small_min_conf=0.5, hand_small_max_reach=0.6,
    pose_wide_retry=True, pose_wide_edge_frac=0.08, pose_wide_pad_x=0.7, pose_wide_pad_down=0.6,
    artifact_max_person_overlap=0.6, artifact_min_person_iou=0.4, artifact_max_motion=0.35, artifact_motion_samples=8,
    pairing_max_norm_dist=3.0,
    state_near_thr=0.9, state_contact_thr=0.35, state_hold_seconds=0.25, state_history_seconds=2.0,
    state_lost_grace_frames=12,
    state_hysteresis=True, state_hysteresis_mult=1.3,
    pose_smooth=True, pose_smooth_min_cutoff=1.5, pose_smooth_beta=0.02, pose_smooth_d_cutoff=1.0,
    pose_wrist_interp_max_gap=8,
    state_min_person_h_px=80, state_reject_rest_arm=True, rest_arm_max_dx=0.10, rest_arm_min_dy=-0.12,
    self_touch_head_frac=0.10, state_ground_max_gap=0.30,
    foot_from_pose=True, foot_extend_min_frac=0.15, foot_extend_max_frac=0.8,
    event_merge_gap_seconds=0.5, event_min_frames=12, max_tasks_per_video=40,
    ambiguous_policy='nearest',
    ambiguous_diff_thr=0.15,
    relaxed_gate=True,
    relaxed_event_min_frames=8, relaxed_min_person_h_px=60,
    relaxed_hold_seconds=0.15, relaxed_pedestal_contact_thr=0.45,
)
SEGMENTATION_CFG = dict(
    sculpture_phrases=['statue', 'sculpture', 'bust'],
    painting_phrases=['painting', 'framed artwork', 'picture frame', 'artwork on wall', 'wall art', 'mounted painting'],
    pedestal_phrases=['pedestal'],
    full_thr=0.30, text_thr=0.25, mask_overlap=0.80, pedestal_merge=True,
    pedestal_point_frac=0.25, pedestal_max_area_ratio=4.0, pedestal_max_person_frac=0.30, min_mask_fill=0.15,
    merge_standalone_pedestal=True, pedestal_cover_merge=0.5,
    fallback_when_no_3d=True,
    fallback_object_phrases=['stone', 'rock', 'artifact', 'small object', 'vase', 'display case', 'glass cabinet', 'shelf'],
    fallback_pedestal_phrases=['pedestal', 'plinth', 'display stand', 'white box'],
    fallback_thr=0.20, fallback_empty_thr=0.15, fallback_lower_when_empty=True,
    fallback_text_thr=0.20, fallback_max_area_frac=0.20, fallback_max_keep=4,
)
EXCLUDE_ARTIFACTS = {}
MANUAL_ARTIFACTS = {}
SEG_REF_FRAMES = {}

EXPORT = dict(pre_sec=1.0, post_sec=3.0, max_window_sec=10.0, n_sample_frames=12, crop_pad=0.4, crop_out_long=768,
              before_ref=True)
MASK_REGIONS = [[0.855, 0.925, 0.995, 0.990]]
VLM_IMAGE_SET = 'marked'

BEFORE_NOTE = ('Image 1 is a reference taken BEFORE (exhibit intact). '
               'If the exhibit visible in Image 1 is missing, moved or carried away in the later frames, answer stealing.\n')


class _PromptTemplate(str):
    """Chuỗi prompt hỗ trợ tuỳ chọn {before_note} mặc định rỗng khi caller cũ chỉ truyền (n, dt, tid)."""

    def format(self, *args, **kwargs):
        kwargs.setdefault('before_note', '')
        return super().format(*args, **kwargs)


PROMPT_TEMPLATE = _PromptTemplate('''You are a museum security analyst. The {n} images are consecutive frames \
(about {dt:.2f}s apart, in order) cropped around ONE visitor marked with a RED box labeled "ID {tid}". \
BLUE outline(s) mark exhibit(s).
{before_note}Classify ONLY the red-box person's behavior toward the exhibit:
- "normal": looking, walking, pointing, standing close without contact
- "touching": hand makes contact with the exhibit, exhibit stays in place
- "holding": person lifts / holds / moves the exhibit
- "stealing": person takes the exhibit away, hides it (bag/clothes), or the exhibit is missing afterwards
- "uncertain": view is occluded or too small to decide
Answer JSON only:
{{"label": "...", "confidence": 0.0, "contact_frames": [], "evidence": "<one sentence>"}}
"contact_frames" = the Frame numbers (1..{n}) in which the red-box person's hand is VISIBLY touching/holding the exhibit (empty list if never). Do not include frames where the hand is only approaching or near.''')
SYSTEM_PROMPT = ('You are a careful CCTV analyst for a museum. Judge only what is visible in the frames. '
                 'Touching means the hand visibly reaches the exhibit or its pedestal; being near is not touching. '
                 'Always answer with a single JSON object and nothing else.')

DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATE, SYSTEM_PROMPT
PROMPT_FIELDS = ('n', 'dt', 'tid', 'before_note')

ANOMALY_LABELS = ('touching', 'holding', 'stealing')
VLM_MIN_CONF = 0.0
ALARM_ONSET = 'vlm'

# ---------------- Ghi đè cấu hình ----------------
_DICTS = ('MODELS', 'RT', 'RUN', 'CFG', 'SEGMENTATION_CFG', 'EXPORT', 'EXCLUDE_ARTIFACTS', 'MANUAL_ARTIFACTS', 'SEG_REF_FRAMES')
_SCALARS = ('ALARM_ONSET', 'VLM_MIN_CONF', 'VLM_IMAGE_SET', 'MASK_REGIONS', 'ANOMALY_LABELS', 'PROMPT_TEMPLATE', 'SYSTEM_PROMPT')


def _merge_inplace(dst, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_inplace(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply_overrides(over):
    """over: dict {TÊN_HẰNG: giá trị}. Tên dict (MODELS, RT, CFG...) được gộp tại chỗ; tên đơn thì thay thế."""
    g = globals()
    for key, val in (over or {}).items():
        name = key.upper()
        if name in _DICTS:
            if not isinstance(val, dict):
                raise TypeError(f'{key} phải là dict')
            _merge_inplace(g[name], val)
        elif name in _SCALARS:
            if name == 'MASK_REGIONS':
                g[name][:] = [list(r) for r in val]
            elif name == 'ANOMALY_LABELS':
                g[name] = tuple(val)
            else:
                g[name] = val
        elif name in ('WORK', 'OUT', 'CACHE'):
            g[name] = Path(val)
        elif name == 'WEIGHT_ROOTS':
            WEIGHT_ROOTS[:0] = [Path(p) for p in val]
        else:
            raise KeyError(f'Khoá cấu hình không hợp lệ: {key}. Hợp lệ: {list(_DICTS) + list(_SCALARS)}')


def load_config_file(path):
    """YAML (.yaml / .yml) hoặc JSON -> dict, rồi apply_overrides."""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    apply_overrides(data)
    return data


def set_paths(work=None, out=None):
    global WORK, OUT, CACHE
    if work:
        WORK = Path(work)
        CACHE = WORK / 'cache'
        OUT = WORK / 'results'
    if out:
        OUT = Path(out)
    for d in (WORK, OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
