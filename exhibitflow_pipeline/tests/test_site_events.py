"""Kiểm thử hiệu chuẩn tự nhập + chiếu hành vi bất thường lên sàn (server/site_events.py). Không cần GPU / model."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import site_events as se  # noqa: E402


def make_cam(pos=(0.0, -6.0, 3.0), yaw_deg=90.0, pitch_down_deg=30.0, w=1280, h=720, hfov=90.0, dist=None):
    """Camera nhìn ra phía +Y, chúc xuống; trả dict hiệu chuẩn (K, R, t world->camera, OpenCV)."""
    f = (w / 2) / np.tan(np.radians(hfov / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_down_deg)
    fwd = np.array([np.cos(yaw) * np.cos(pit), np.sin(yaw) * np.cos(pit), -np.sin(pit)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ np.asarray(pos, float)
    return dict(K=K.tolist(), R=R.tolist(), t=t.tolist(), width=w, height=h, dist=dist or [])


def project(spec, X):
    p, _ = cv2.projectPoints(np.asarray(X, float).reshape(-1, 1, 3), cv2.Rodrigues(np.array(spec['R_world_to_camera']))[0],
                             np.array(spec['t_world_to_camera']), np.array(spec['K']), np.array(spec.get('dist') or [0, 0, 0, 0, 0.0]))
    return p.reshape(-1, 2)


def test_normalize_accepts_common_key_styles():
    raw = make_cam()
    a = se.normalize_calibration(raw, 'a')
    b = se.normalize_calibration({'camera_matrix': raw['K'], 'rotation': raw['R'], 'tvec': raw['t'], 'image_width': 1280, 'image_height': 720}, 'b')
    assert np.allclose(a['K'], b['K']) and np.allclose(a['R_world_to_camera'], b['R_world_to_camera'])
    c = se.normalize_calibration({'fx': 640, 'fy': 640, 'cx': 640, 'cy': 360, 'rvec': cv2.Rodrigues(np.array(raw['R']))[0].ravel().tolist(), 't': raw['t']}, 'c')
    assert np.allclose(c['R_world_to_camera'], raw['R'], atol=1e-6)


def test_units_cm_converted_to_meters():
    raw = make_cam()
    cm = dict(raw, t=(np.array(raw['t']) * 100).tolist(), units='cm')
    assert np.allclose(se.normalize_calibration(cm, 'cm')['t_world_to_camera'], raw['t'])


@pytest.mark.parametrize('mutate,needle', [
    (lambda r: r.pop('K'), 'K'),
    (lambda r: r.pop('R'), 'R'),
    (lambda r: r.pop('t'), 't'),
    (lambda r: r.update(R=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]), 'xoay'),
    (lambda r: r.update(t=[0, 0, 0.1]), 'độ cao'),
])
def test_bad_calibration_rejected_with_reason(mutate, needle):
    raw = make_cam()
    mutate(raw)
    with pytest.raises(se.CalibrationError) as e:
        se.normalize_calibration(raw, 'cam')
    assert needle in str(e.value)


def test_split_calibration_shapes():
    one = make_cam()
    assert se.split_calibration(one)[0][0] is None
    multi = {'cameras': {'C01': one, 'C02': one}}
    assert [k for k, _ in se.split_calibration(multi)] == ['C01', 'C02']
    assert [k for k, _ in se.split_calibration({'A': one, 'B': one})] == ['A', 'B']
    assert [k for k, _ in se.split_calibration({'cameras': [dict(one, id='X'), dict(one, id='Y')]})] == ['X', 'Y']


def test_fit_to_video_scales_intrinsics_and_rejects_wrong_aspect():
    spec = se.normalize_calibration(make_cam(), 'a')
    small = se.fit_to_video(spec, 426, 240)
    assert small['width'] == 426 and abs(small['K'][0][0] - spec['K'][0][0] * 426 / 1280) < 1e-9
    X = [1.0, 4.0, 0.0]
    assert np.allclose(se.floor_mapper(spec)(project(spec, X))[0], X[:2], atol=1e-6)
    assert np.allclose(se.floor_mapper(small)(project(small, X))[0], X[:2], atol=1e-6)
    with pytest.raises(RuntimeError):
        se.fit_to_video(spec, 640, 640)


def test_floor_mapper_roundtrip_with_lens_distortion():
    spec = se.normalize_calibration(make_cam(dist=[-0.25, 0.08, 0.0005, 0.0003, 0.0]), 'd')
    for X in ([0.0, 3.0, 0.0], [-2.5, 5.0, 0.0], [3.0, 8.0, 0.0]):
        assert np.allclose(se.floor_mapper(spec)(project(spec, X))[0], X[:2], atol=1e-3)


def timeline_for(spec, path_xy, fps=24, alert=(48, 96)):
    W, H = spec['width'], spec['height']
    frames = {}
    for i, (x, y) in enumerate(path_xy):
        foot, head = project(spec, [[x, y, 0], [x, y, 1.7]])
        hh = abs(foot[1] - head[1])
        frames[str(i)] = [[[7, (foot[0] - hh * .175) / W, (foot[1] - hh) / H, (foot[0] + hh * .175) / W, foot[1] / H, 0]], []]
    return dict(width=W, height=H, fps=fps, frames=frames,
                alerts=[dict(tid=7, behavior='stealing', conf=0.9, artifact_id=2, start=alert[0], end=alert[1], evidence='lấy vật')])


def test_engine_event_lands_on_true_floor_position():
    spec = se.normalize_calibration(make_cam(), 'a')
    n = 144
    path = [(-1 + 2 * i / n, 3 + 2 * i / n) for i in range(n)]
    ev = se.engine_events(timeline_for(spec, path), se.floor_mapper(spec), 'C1')
    assert len(ev) == 1
    truth = np.mean(path[48:97], axis=0)
    assert np.hypot(ev[0]['x'] - truth[0], ev[0]['y'] - truth[1]) < 0.05
    assert ev[0]['t0'] == 2.0 and ev[0]['t1'] == 4.0 and len(ev[0]['path']) > 10


def test_incidents_merge_across_cameras_and_link_fused_person():
    a = dict(cam='C1', track=7, behavior='touching', conf=0.7, artifact=1, evidence='a', t0=2.0, t1=4.0, x=4.0, y=3.0, path=[[2.0, 4.0, 3.0], [4.0, 4.1, 3.1]])
    b = dict(cam='C2', track=3, behavior='stealing', conf=0.9, artifact=1, evidence='b', t0=2.5, t1=4.5, x=4.3, y=3.2, path=[[2.5, 4.3, 3.2], [4.5, 4.4, 3.3]])
    far = dict(cam='C2', track=9, behavior='touching', conf=0.6, artifact=2, evidence='c', t0=10.0, t1=11.0, x=-8.0, y=9.0, path=[[10.0, -8.0, 9.0], [11.0, -8.0, 9.1]])
    frames = [dict(t=t / 2, people=[dict(id=5, x=3.0 + t * 0.15, y=2.5 + t * 0.1, cams=['C1', 'C2'])]) for t in range(0, 12)]
    inc = se.build_incidents([a, b, far], frames)
    assert len(inc) == 2
    first = next(i for i in inc if i['t0'] < 5)
    assert first['behavior'] == 'stealing' and first['cams'] == ['C1', 'C2'] and first['conf'] == 0.9
    assert first['person_id'] == 5 and len(first['path']) == len(frames)          # đường đi lấy từ người ghép chung
    other = next(i for i in inc if i['t0'] > 5)
    assert other['person_id'] is None and other['path'] == far['path']             # không có người ghép gần: giữ đường đi từ camera


def to_y_up(raw):
    """Biểu diễn lại cùng camera trong hệ Y-up (X đông, Y lên, Z nam): R_cũ = R_mới @ A^-1."""
    A = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    return dict(raw, R=(np.array(raw['R']) @ A.T).tolist())


def test_y_up_calibration_is_converted_to_z_up():
    z_up = make_cam()
    expected = se.normalize_calibration(z_up, 'z')
    got = se.normalize_calibration(dict(to_y_up(z_up), up_axis='y'), 'y')
    assert np.allclose(got['R_world_to_camera'], expected['R_world_to_camera'], atol=1e-9)
    X = [1.5, 4.0, 0.0]
    assert np.allclose(se.floor_mapper(got)(project(got, X))[0], X[:2], atol=1e-6)


def test_y_up_inherited_from_file_level_coordinate_system():
    cam = to_y_up(make_cam())
    raw = {'coordinate_system': {'axes': 'Right-handed Y-up: +X = East, +Y = Up'}, 'cameras': {'camera_A': cam}}
    (cid, c), = se.split_calibration(raw)
    assert cid == 'camera_A' and c['up_axis'] == 'y'
    assert np.allclose(se.normalize_calibration(c, 'a')['R_world_to_camera'], se.normalize_calibration(make_cam(), 'a')['R_world_to_camera'], atol=1e-9)


def test_unknown_up_axis_rejected():
    with pytest.raises(se.CalibrationError):
        se.normalize_calibration(dict(make_cam(), up_axis='x'), 'a')


def test_room_bounds_from_calibration_file():
    y_up = {'coordinate_system': {'axes': 'Right-handed Y-up: +X = East'}, 'room_landmarks_m': {'floor_bounds_X': [-5.2, 5.2], 'floor_bounds_Z': [-5.15, 4.9]}}
    assert se.room_bounds_from(y_up) == [-5.2, -4.9, 5.2, 5.15]                    # Y bắc = -Z
    assert se.room_bounds_from({'room_bounds': [8, 6, 0, 1]}) == [0.0, 1.0, 8.0, 6.0]
    assert se.room_bounds_from({'room_bounds': {'x_min': 0, 'x_max': 8, 'y_min': 1, 'y_max': 6}}) == [0.0, 1.0, 8.0, 6.0]
    assert se.room_bounds_from({'cameras': {}}) is None
    assert se.room_bounds_from({'room_bounds': [1, 1, 1, 5]}) is None               # suy biến


def test_engine_summary_counts_tracks_contact_and_decisions():
    tl = dict(fps=10, frames={'0': [[[1, .1, .1, .2, .5, 0], [2, .3, .1, .4, .5, 0]], []],
                              '1': [[[1, .1, .1, .2, .5, 1]], []], '2': [[[1, .1, .1, .2, .5, 1]], []]},
              alerts=[dict(tid=1, behavior='touching', conf=.8, start=1, end=2)])
    s = se.engine_summary(tl, dict(summary=dict(n_events=3), vlm_tasks=[dict(decision='NORMAL'), dict(decision='CANDIDATE'), dict(decision='ANOMALY')]))
    assert (s['tracks'], s['contact_s'], s['alerts'], s['events'], s['normal'], s['uncertain']) == (2, 0.2, 1, 3, 1, 1)
    assert se.engine_summary(tl)['events'] is None


def test_homography_mapper_matches_floor_mapper_and_handles_distortion():
    spec = se.normalize_calibration(make_cam(dist=[-0.2, 0.05, 0.0004, 0.0002, 0.0]), 'a')
    K, R, t = (np.array(spec[k]) for k in ('K', 'R_world_to_camera', 't_world_to_camera'))
    H = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))
    for X in ([0.5, 3.0, 0.0], [-2.0, 6.0, 0.0]):
        px = project(spec, X)
        assert np.allclose(se.homography_mapper(H, K, spec['dist'])(px), se.floor_mapper(spec)(px), atol=1e-6)
        assert np.allclose(se.homography_mapper(H, K, spec['dist'])(px)[0], X[:2], atol=1e-3)
