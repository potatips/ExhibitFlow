"""Unit tests for FloorplanService and calibration math."""
from __future__ import annotations

import math

import pytest


def test_floorplan_service_imports_and_calibration_math() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    from exhibitflow_gpu.floorplan import (
        CalibrationError,
        FloorplanService,
        build_incidents,
        fit_to_video,
        floor_mapper,
        homography_mapper,
        normalize_calibration,
        room_bounds_from,
        split_calibration,
    )

    assert hasattr(FloorplanService, "generate_single_cam")
    assert hasattr(FloorplanService, "run_calibrated_multicam")
    assert isinstance(FloorplanService.list_sites(), dict)

    # Valid overhead camera looking down at Z=0 floor from Z=4m
    # Xcam = R @ Xworld + t ; camera at (0, 0, 4) looking down (-Z world -> +Z cam)
    R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    t = [0.0, 0.0, 4.0]
    K = [[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]]
    raw = {
        "K": K,
        "R_world_to_camera": R,
        "t_world_to_camera": t,
        "image_width": 960,
        "image_height": 540,
        "label": "CAM_TEST",
    }

    norm = normalize_calibration(raw, "CAM_TEST")
    assert norm["label"] == "CAM_TEST"
    assert norm["width"] == 960
    assert norm["height"] == 540

    # Principal point (480, 270) maps straight down to (0, 0) on the floor
    mapper = floor_mapper(norm)
    pts = mapper([(480.0, 270.0), (880.0, 270.0)])
    assert pts.shape == (2, 2)
    assert math.isclose(float(pts[0, 0]), 0.0, abs_tol=1e-4)
    assert math.isclose(float(pts[0, 1]), 0.0, abs_tol=1e-4)
    # Offset by fx/2 = 400px at height 4m -> X = 2.0m
    assert math.isclose(float(pts[1, 0]), 2.0, abs_tol=1e-4)

    # Fit to half-resolution video (480x270) scales K by 0.5
    scaled = fit_to_video(norm, 480, 270)
    assert math.isclose(scaled["K"][0][0], 400.0, rel_tol=1e-6)
    assert math.isclose(scaled["K"][1][2], 135.0, rel_tol=1e-6)

    # Homography mapper with identity
    hm = homography_mapper(np.eye(3).tolist())
    h_pts = hm([(1.5, 2.5)])
    assert math.isclose(float(h_pts[0, 0]), 1.5, abs_tol=1e-4)
    assert math.isclose(float(h_pts[0, 1]), 2.5, abs_tol=1e-4)

    # Split calibration + room bounds
    items = split_calibration({"cameras": {"C1": raw}, "room_bounds": [-5, -4, 5, 4]})
    assert len(items) == 1 and items[0][0] == "C1"
    assert room_bounds_from({"room_bounds": [-5, -4, 5, 4]}) == [-5.0, -4.0, 5.0, 4.0]

    # Invalid R (non-orthonormal) raises CalibrationError
    bad = dict(raw, R_world_to_camera=[[2.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(CalibrationError):
        normalize_calibration(bad, "BAD_CAM")

    # Incident clustering across 2 cameras
    inc = build_incidents(
        [
            {"cam": "C1", "track": 1, "behavior": "touching", "conf": 0.7, "artifact": 1, "evidence": "a", "t0": 1.0, "t1": 2.0, "x": 1.0, "y": 1.0, "path": [[1.0, 1.0, 1.0], [2.0, 1.1, 1.0]]},
            {"cam": "C2", "track": 3, "behavior": "stealing", "conf": 0.9, "artifact": 1, "evidence": "b", "t0": 1.2, "t1": 2.2, "x": 1.2, "y": 1.1, "path": [[1.2, 1.2, 1.1], [2.2, 1.2, 1.1]]},
        ],
        frames=[{"t": 1.5, "people": [{"id": 7, "x": 1.1, "y": 1.05}]}],
    )
    assert len(inc) == 1
    assert inc[0]["behavior"] == "stealing"
    assert inc[0]["person_id"] == 7
    assert inc[0]["cams"] == ["C1", "C2"]
