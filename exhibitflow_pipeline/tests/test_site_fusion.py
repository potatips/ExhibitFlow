"""Đối chiếu chéo camera (server/site_fusion.py): thuần dữ liệu, không cần GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import site_fusion as sf  # noqa: E402

FPS = 10.0


def cam(events, tids, n=100):
    """events: [(event_id, local_tid, t0, t1, decision, conf)]; tids: các track cục bộ có mặt ở mọi khung."""
    frames = {str(f): [[[t, .1, .1, .2, .2, 0] for t in tids], []] for f in range(n)}
    res = dict(events=[dict(event_id=i, track_id=t, artifact_id=1, start_time_s=a, end_time_s=b, decision=d, behavior='touching' if d == 'ANOMALY' else 'normal',
                            vlm_confidence=c, vlm_evidence=None) for i, t, a, b, d, c in events],
               alerts=[dict(event_id=i, track_id=t) for i, t, a, b, d, c in events if d == 'ANOMALY'],
               summary=dict(n_alerts=0, alerts_by_behavior={}, has_anomaly=False))
    return dict(tl=dict(fps=FPS, frames=frames, alerts=[dict(tid=t, artifact_id=1, start=int(a * FPS), end=int(b * FPS)) for i, t, a, b, d, c in events if d == 'ANOMALY']), res=res)


def decisions(f, c):
    return [e['decision'] for e in f[c]['res']['events']]


def test_conflict_demotes_to_candidate():
    # người chung 7 là local 2 ở A, local 5 ở B: A báo chạm, B (thấy rõ, cùng lúc) nói normal
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1, 3, 'NORMAL', 0.0)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['CANDIDATE'] and f['A']['res']['alerts'] == [] and f['A']['res']['summary']['has_anomaly'] is False
    assert decisions(f, 'B') == ['NORMAL'] and f['A']['drop'] and info['events'][0]['status'] == 'conflict'
    assert 'cần xem lại' in f['A']['res']['events'][0]['vlm_evidence']
    assert a['res']['events'][0]['decision'] == 'ANOMALY'                        # dữ liệu vào không bị sửa
    assert sf.drop_demoted_alerts(a['tl'], f['A']['drop'])['alerts'] == []


def test_both_anomaly_confirmed():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1.5, 3.5, 'ANOMALY', .99)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['ANOMALY'] and decisions(f, 'B') == ['ANOMALY'] and {e['status'] for e in info['events']} == {'confirmed'}


def test_weak_anomaly_is_not_support():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1, 3, 'ANOMALY', 0.0)], [5]), id_map={5: 7})            # touching nhưng conf 0.0 = không chắc
    _, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert next(e for e in info['events'] if e['cam'] == 'A')['status'] == 'single_view'


def test_other_camera_does_not_see_person_keeps_anomaly():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([], [9], n=100), id_map={9: 8})                                 # B chỉ thấy người khác
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['ANOMALY'] and info['events'][0]['status'] == 'single_view'


def test_no_contact_counts_only_in_strict():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([], [5]), id_map={5: 7})                                        # B thấy người đó cả đoạn nhưng không có sự kiện chạm
    assert decisions(sf.fuse(dict(A=a, B=b), 'strict')[0], 'A') == ['CANDIDATE']
    assert decisions(sf.fuse(dict(A=a, B=b), 'normal')[0], 'A') == ['ANOMALY']
    assert decisions(sf.fuse(dict(A=a, B=b), 'off')[0], 'A') == ['ANOMALY']


def test_adjacent_segments_share_the_conflict():
    a = dict(cam([(0, 2, 0.5, 1.5, 'ANOMALY', .99), (1, 2, 2.0, 4.0, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 2.0, 4.0, 'NORMAL', 0.0)], [5], n=100), id_map={5: 7})
    b['tl']['frames'] = {k: v for k, v in b['tl']['frames'].items() if int(k) >= 15}   # B chưa thấy người ở đoạn đầu
    f, _ = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['CANDIDATE', 'CANDIDATE']


if __name__ == '__main__':
    for n, fn in list(globals().items()):
        if n.startswith('test_'):
            fn()
            print('ok', n)
