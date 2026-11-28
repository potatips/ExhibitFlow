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


def test_person_policy_propagates_anomaly_to_other_camera():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1.2, 3.2, 'NORMAL', 0.0), (1, 6, 1.2, 3.2, 'NORMAL', 0.0)], [5, 6]), id_map={5: 7, 6: 8})   # 5 = người 7, 6 = người khác
    f, info = sf.fuse(dict(A=a, B=b), 'person')
    assert decisions(f, 'B') == ['ANOMALY', 'NORMAL']                              # chỉ người 7 bị đánh dấu
    assert f['B']['res']['events'][0]['xcam']['status'] == 'propagated' and f['B']['res']['summary']['has_anomaly']
    assert [al['track_id'] for al in f['B']['res']['alerts']] == [5] and f['B']['add'][0]['tid'] == 5
    assert info['anomaly_persons'] == [7] and decisions(f, 'A') == ['ANOMALY']
    assert b['res']['events'][0]['decision'] == 'NORMAL'                           # dữ liệu vào không bị sửa
    assert sf.add_promoted_alerts(b['tl'], f['B']['add'])['alerts'][0]['behavior'] == 'touching'


def test_person_policy_ignores_weak_anomaly_and_no_overlap():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', 0.0)], [2]), id_map={2: 7})             # conf 0.0: không đủ để lan
    b = dict(cam([(0, 5, 1, 3, 'NORMAL', 0.0)], [5]), id_map={5: 7})
    assert decisions(sf.fuse(dict(A=a, B=b), 'person')[0], 'B') == ['NORMAL']
    a2 = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b2 = dict(cam([(0, 5, 8, 9, 'NORMAL', 0.0)], [5]), id_map={5: 7})             # cùng người nhưng thời điểm khác hẳn
    assert decisions(sf.fuse(dict(A=a2, B=b2), 'person')[0], 'B') == ['NORMAL']


def test_default_policy_is_person(monkeypatch=None):
    import os
    os.environ.pop('EXHIBITFLOW_XCAM_POLICY', None)
    assert sf.policy() == 'person'


def test_person_policy_only_raises_the_time_of_the_touch():
    # A không chắc suốt 0-7,5 s; B chỉ thấy chạm ở cuối (7-10 s): đoạn đầu của A phải giữ nguyên CANDIDATE
    a = dict(cam([(0, 2, 0.0, 7.5, 'CANDIDATE', .4)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 7.0, 10.0, 'ANOMALY', .9)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'person')
    got = [(e['decision'], e['start_time_s'], e['end_time_s']) for e in f['A']['res']['events']]
    assert got == [('CANDIDATE', 0.0, 7.0), ('ANOMALY', 7.0, 7.5)], got
    assert len({e['event_id'] for e in f['A']['res']['events']}) == 2
    assert [al['alarm_start_time_s'] for al in f['A']['res']['alerts']] == [7.0]
    assert f['A']['add'][0]['spans'] == [[70, 75]] and info['propagated'][0]['t0'] == 7.0
    assert a['res']['events'][0]['end_time_s'] == 7.5                              # dữ liệu vào không bị sửa


def test_person_policy_keeps_both_sides_of_a_short_touch():
    a = dict(cam([(0, 2, 0.0, 9.0, 'NORMAL', 0.0)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 4.0, 5.0, 'ANOMALY', .9)], [5]), id_map={5: 7})
    f, _ = sf.fuse(dict(A=a, B=b), 'person')
    got = [(e['decision'], e['start_time_s'], e['end_time_s']) for e in f['A']['res']['events']]
    assert got == [('NORMAL', 0.0, 4.0), ('ANOMALY', 4.0, 5.0), ('NORMAL', 5.0, 9.0)], got
    assert len({e['event_id'] for e in f['A']['res']['events']}) == 3


def test_person_policy_trims_the_end_when_the_touch_is_first():
    a = dict(cam([(0, 2, 2.0, 9.0, 'CANDIDATE', .4)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1.0, 4.0, 'ANOMALY', .9)], [5]), id_map={5: 7})
    f, _ = sf.fuse(dict(A=a, B=b), 'person')
    got = [(e['decision'], e['start_time_s'], e['end_time_s']) for e in f['A']['res']['events']]
    assert got == [('ANOMALY', 2.0, 4.0), ('CANDIDATE', 4.0, 9.0)], got
    assert len({e['event_id'] for e in f['A']['res']['events']}) == 2

