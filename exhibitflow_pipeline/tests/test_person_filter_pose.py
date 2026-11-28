"""Bộ lọc sau pose không được xoá track chỉ vì pose không trả keypoint (xem PersonTrackFilter.filter_post_pose)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
perception = pytest.importorskip('exhibitflow.perception')


def rec(tid, valid):
    r = dict(id=tid, box=[0, 0, 50, 100])
    if valid is not None:
        r['kp'] = [[1.0, 1.0]] * 17
        r['kc'] = [0.9] * valid + [0.0] * (17 - valid)
    return r


def run(valid, frames=30):
    f = perception.PersonTrackFilter()
    kept = 0
    for i in range(frames):
        kept += len(f.filter_post_pose([rec(1, valid)], posed_ids=[1], frame_idx=i))
    return kept


def test_pose_attempted_but_no_keypoints_keeps_track():
    assert run(None) == 30


def test_pose_with_enough_keypoints_keeps_track():
    assert run(12) == 30


def test_pose_with_too_few_keypoints_still_dropped():            # box giả (ít keypoint hợp lệ) vẫn bị loại
    assert run(2) == 0
