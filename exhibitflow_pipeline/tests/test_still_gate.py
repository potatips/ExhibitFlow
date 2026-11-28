"""Quy tắc "chạm chỉ tính từ lúc đứng lại" (server/still_gate.py): thuần dữ liệu, không cần GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import still_gate as sg  # noqa: E402

FPS = 10.0


def data(xs, events, behavior='touching'):
    """xs[khung] = vị trí chân theo chiều ngang (px); người cao 100 px."""
    frames = {str(f): [[[3, (x - 20) / 1000, .1, (x + 20) / 1000, .2, 0]], []] for f, x in enumerate(xs)}
    tl = dict(fps=FPS, width=1000, height=1000, frames=frames,
              alerts=[dict(tid=3, artifact_id=1, start=a, end=b, behavior=behavior) for a, b in events])
    res = dict(events=[dict(event_id=i, track_id=3, artifact_id=1, start_frame=a, end_frame=b, start_time_s=a / FPS, end_time_s=b / FPS,
                            decision='ANOMALY', behavior=behavior, vlm_confidence=.9, vlm_evidence=None) for i, (a, b) in enumerate(events)],
               alerts=[dict(event_id=i, track_id=3, behavior=behavior, alarm_start_frame=a, alarm_end_frame=b, alarm_start_time_s=a / FPS,
                            alarm_end_time_s=b / FPS) for i, (a, b) in enumerate(events)],
               summary=dict(n_alerts=len(events), alerts_by_behavior={}, has_anomaly=True))
    return res, tl


WALK_THEN_STAND = [100 + 3 * min(f, 40) for f in range(80)]          # đi 30 px/s trong 4 s rồi đứng yên


def test_walking_past_is_only_a_candidate():
    res, tl = data(WALK_THEN_STAND, [(2, 27)])
    new, ntl = sg.apply(res, tl)
    assert [e['decision'] for e in new['events']] == ['CANDIDATE'] and new['alerts'] == [] and ntl['alerts'] == []
    assert new['summary']['has_anomaly'] is False
    assert res['events'][0]['decision'] == 'ANOMALY' and len(tl['alerts']) == 1      # dữ liệu vào không bị sửa


def test_alarm_starts_when_the_person_stops():
    res, tl = data(WALK_THEN_STAND, [(10, 70)])
    new, ntl = sg.apply(res, tl)
    assert [e['decision'] for e in new['events']] == ['CANDIDATE', 'ANOMALY']
    head, touch = new['events']
    assert head['end_frame'] == touch['start_frame'] and 36 <= touch['start_frame'] <= 48, touch['start_frame']
    assert head['event_id'] != touch['event_id'] and new['alerts'][0]['event_id'] == touch['event_id']
    assert new['alerts'][0]['alarm_start_frame'] == touch['start_frame'] == ntl['alerts'][0]['start']


def test_standing_still_and_other_behaviours_are_unchanged():
    res, tl = data([500] * 60, [(0, 50)])
    assert sg.apply(res, tl)[0]['events'] == res['events']
    res, tl = data(WALK_THEN_STAND, [(2, 27)], behavior='stealing')
    assert sg.apply(res, tl)[0]['events'] == res['events']


def test_missing_track_data_keeps_the_engine_decision():
    res, tl = data(WALK_THEN_STAND, [(2, 27)])
    assert sg.apply(res, dict(tl, frames={}))[0]['events'] == res['events']
