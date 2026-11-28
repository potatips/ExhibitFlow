"""Số hiện vật chung giữa các camera (server/art_match.py): hình học thuần, không cần GPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import art_match as am  # noqa: E402

ROOM = [-5.0, -5.0, 5.0, 5.0]


def camera(center, target):
    """Hiệu chuẩn (OpenCV) của camera đặt ở `center` nhìn vào `target`."""
    c, f = np.array(center, float), np.array(target, float) - np.array(center, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return dict(K=[[600, 0, 480], [0, 600, 270], [0, 0, 1]], R_world_to_camera=R.tolist(), t_world_to_camera=(-R @ c).tolist(), width=960, height=540)


def box(spec, p):
    x = np.array(spec['K'], float) @ (np.array(spec['R_world_to_camera']) @ np.array(p, float) + np.array(spec['t_world_to_camera']))
    u, v = x[:2] / x[2]
    return [u - 10, v - 10, u + 10, v + 10]


A, B = camera((-4.5, -4.5, 3), (0, 5, 1.5)), camera((4.5, -4.5, 3), (0, 5, 1.5))
WALL = dict(left=(-2.0, 5.0, 1.6), middle=(0.0, 5.0, 1.6), right=(2.0, 5.0, 1.6), only_b=(-5.0, 2.0, 1.6))


def result(spec, ids):
    return dict(video=dict(width=960, height=540), artifacts=[dict(id=i, box=box(spec, WALL[n])) for n, i in ids.items()],
                events=[dict(event_id=0, artifact_id=list(ids.values())[0])], alerts=[], vlm_tasks=[dict(artifact_ids=list(ids.values()))])


def test_the_same_wall_object_gets_one_number():
    res = dict(C1=result(A, dict(left=4, middle=7, right=3)), C2=result(B, dict(right=4, left=7, middle=5, only_b=2)))
    ids = am.match(res, dict(C1=A, C2=B), ROOM)
    assert ids['C1'] == {4: 4, 7: 7, 3: 3}
    assert ids['C2'] == {4: 3, 7: 4, 5: 7, 2: 8}, ids['C2']          # chỉ camera 2 thấy: số mới sau số lớn nhất
    new = am.renumber_result(res['C2'], ids['C2'])
    assert [a['id'] for a in new['artifacts']] == [3, 4, 7, 8] and new['events'][0]['artifact_id'] == 3
    assert new['vlm_tasks'][0]['artifact_ids'] == [3, 4, 7, 8] and new['artifact_global']
    assert res['C2']['artifacts'][0]['id'] == 4                      # dữ liệu vào không bị sửa
    tl = am.renumber_timeline(dict(alerts=[dict(tid=1, artifact_id=5)]), ids['C2'])
    assert tl['alerts'][0]['artifact_id'] == 7


def test_nothing_changes_without_room_bounds_or_second_camera():
    res = dict(C1=result(A, dict(left=4)), C2=result(B, dict(left=7)))
    assert am.match(res, dict(C1=A, C2=B), None) == {}
    assert am.match(dict(C1=res['C1']), dict(C1=A), ROOM) == {}


def test_exhibit_positions_on_the_map():
    res = dict(C1=result(A, dict(left=4, middle=7)), C2=result(B, dict(middle=5)))
    specs = dict(C1=A, C2=B)
    ids = am.match(res, specs, ROOM)
    got = am.map_exhibits(res, specs, ids, ROOM, {})
    assert [e['id'] for e in got] == [4, 7]
    assert abs(got[1]['x']) < .05 and abs(got[1]['y'] - 5) < .05
    shift = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1.0]])           # map lệch 10 m so với toạ độ sàn
    ground = {c: np.array(s['K'], float) @ np.column_stack([np.array(s['R_world_to_camera'])[:, 0], np.array(s['R_world_to_camera'])[:, 1], s['t_world_to_camera']])
              for c, s in specs.items()}
    moved = am.map_exhibits(res, specs, ids, ROOM, {c: shift @ np.linalg.inv(g) for c, g in ground.items()})
    assert abs(moved[1]['x'] - 10) < .05 and abs(moved[1]['y'] - 5) < .05
    assert am.map_exhibits(res, specs, ids, None, {}) == [] and am.map_exhibits(res, specs, {}, ROOM, {}) == []

