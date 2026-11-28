"""Đối chiếu chéo camera (exhibitflow_gpu.floorplan.site_fusion): thuần dữ liệu, không cần GPU."""
import os

from exhibitflow_gpu.floorplan import site_fusion as sf

FPS = 10.0


def cam(events, tids, n=100):
    frames = {str(f): [[[t, .1, .1, .2, .2, 0] for t in tids], []] for f in range(n)}
    res = dict(events=[dict(event_id=i, track_id=t, artifact_id=1, start_time_s=a, end_time_s=b, decision=d, behavior='touching' if d == 'ANOMALY' else 'normal',
                            vlm_confidence=c, vlm_evidence=None) for i, t, a, b, d, c in events],
               alerts=[dict(event_id=i, track_id=t) for i, t, a, b, d, c in events if d == 'ANOMALY'],
               summary=dict(n_alerts=0, alerts_by_behavior={}, has_anomaly=False))
    return dict(tl=dict(fps=FPS, frames=frames, alerts=[dict(tid=t, artifact_id=1, start=int(a * FPS), end=int(b * FPS)) for i, t, a, b, d, c in events if d == 'ANOMALY']), res=res)


def decisions(f, c):
    return [e['decision'] for e in f[c]['res']['events']]


def test_conflict_demotes_to_candidate():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1, 3, 'NORMAL', 0.0)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['CANDIDATE'] and f['A']['res']['alerts'] == [] and f['A']['res']['summary']['has_anomaly'] is False
    assert decisions(f, 'B') == ['NORMAL'] and f['A']['drop'] and info['events'][0]['status'] == 'conflict'
    assert 'cần xem lại' in f['A']['res']['events'][0]['vlm_evidence']
    assert a['res']['events'][0]['decision'] == 'ANOMALY'
    assert sf.drop_demoted_alerts(a['tl'], f['A']['drop'])['alerts'] == []


def test_both_anomaly_confirmed():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1.5, 3.5, 'ANOMALY', .99)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['ANOMALY'] and decisions(f, 'B') == ['ANOMALY'] and {e['status'] for e in info['events']} == {'confirmed'}


def test_weak_anomaly_is_not_support():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1, 3, 'ANOMALY', 0.0)], [5]), id_map={5: 7})
    _, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert next(e for e in info['events'] if e['cam'] == 'A')['status'] == 'single_view'


def test_other_camera_does_not_see_person_keeps_anomaly():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([], [9], n=100), id_map={9: 8})
    f, info = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['ANOMALY'] and info['events'][0]['status'] == 'single_view'


def test_no_contact_counts_only_in_strict():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([], [5]), id_map={5: 7})
    assert decisions(sf.fuse(dict(A=a, B=b), 'strict')[0], 'A') == ['CANDIDATE']
    assert decisions(sf.fuse(dict(A=a, B=b), 'normal')[0], 'A') == ['ANOMALY']
    assert decisions(sf.fuse(dict(A=a, B=b), 'off')[0], 'A') == ['ANOMALY']


def test_adjacent_segments_share_the_conflict():
    a = dict(cam([(0, 2, 0.5, 1.5, 'ANOMALY', .99), (1, 2, 2.0, 4.0, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 2.0, 4.0, 'NORMAL', 0.0)], [5], n=100), id_map={5: 7})
    b['tl']['frames'] = {k: v for k, v in b['tl']['frames'].items() if int(k) >= 15}
    f, _ = sf.fuse(dict(A=a, B=b), 'strict')
    assert decisions(f, 'A') == ['CANDIDATE', 'CANDIDATE']


def test_person_policy_propagates_anomaly_to_other_camera():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1.2, 3.2, 'NORMAL', 0.0), (1, 6, 1.2, 3.2, 'NORMAL', 0.0)], [5, 6]), id_map={5: 7, 6: 8})
    f, info = sf.fuse(dict(A=a, B=b), 'person')
    assert decisions(f, 'B') == ['ANOMALY', 'NORMAL']
    assert f['B']['res']['events'][0]['xcam']['status'] == 'propagated' and f['B']['res']['summary']['has_anomaly']
    assert [al['track_id'] for al in f['B']['res']['alerts']] == [5] and f['B']['add'][0]['tid'] == 5
    assert info['anomaly_persons'] == [7] and decisions(f, 'A') == ['ANOMALY']
    assert b['res']['events'][0]['decision'] == 'NORMAL'
    assert sf.add_promoted_alerts(b['tl'], f['B']['add'])['alerts'][0]['behavior'] == 'touching'


def test_person_policy_ignores_weak_anomaly_and_no_overlap():
    a = dict(cam([(0, 2, 1, 3, 'ANOMALY', 0.0)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 1, 3, 'NORMAL', 0.0)], [5]), id_map={5: 7})
    assert decisions(sf.fuse(dict(A=a, B=b), 'person')[0], 'B') == ['NORMAL']
    a2 = dict(cam([(0, 2, 1, 3, 'ANOMALY', .99)], [2]), id_map={2: 7})
    b2 = dict(cam([(0, 5, 8, 9, 'NORMAL', 0.0)], [5]), id_map={5: 7})
    assert decisions(sf.fuse(dict(A=a2, B=b2), 'person')[0], 'B') == ['NORMAL']


def test_default_policy_is_person():
    old = os.environ.pop('EXHIBITFLOW_XCAM_POLICY', None)
    try:
        assert sf.policy() == 'person'
    finally:
        if old is not None:
            os.environ['EXHIBITFLOW_XCAM_POLICY'] = old


def test_person_policy_only_raises_the_time_of_the_touch():
    a = dict(cam([(0, 2, 0.0, 7.5, 'CANDIDATE', .4)], [2]), id_map={2: 7})
    b = dict(cam([(0, 5, 7.0, 10.0, 'ANOMALY', .9)], [5]), id_map={5: 7})
    f, info = sf.fuse(dict(A=a, B=b), 'person')
    got = [(e['decision'], e['start_time_s'], e['end_time_s']) for e in f['A']['res']['events']]
    assert got == [('CANDIDATE', 0.0, 7.0), ('ANOMALY', 7.0, 7.5)], got
    assert len({e['event_id'] for e in f['A']['res']['events']}) == 2
    assert [al['alarm_start_time_s'] for al in f['A']['res']['alerts']] == [7.0]
    assert f['A']['add'][0]['spans'] == [[70, 75]] and info['propagated'][0]['t0'] == 7.0
    assert a['res']['events'][0]['end_time_s'] == 7.5


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
