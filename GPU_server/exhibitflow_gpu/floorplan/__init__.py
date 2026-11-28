"""Unified Floorplan Service for ExhibitFlow GPU.

Unifies:
- Uncalibrated single-camera floorplan estimation (`single_cam`)
- Calibrated multi-camera floorplan tracking & event projection (`multi_cam`, `calibration`, `site_driver`)
"""
import json
from pathlib import Path

from . import calibration, multi_cam, single_cam
from .calibration import (  # noqa: F401
    MAX_RANGE_M,
    SEVERITY,
    CalibrationError,
    GlobalIdLabeler,
    appearance_embedding,
    build_incidents,
    engine_events,
    engine_summary,
    fit_to_video,
    floor_mapper,
    homography_mapper,
    normalize_calibration,
    room_bounds_from,
    split_calibration,
    stitch_ids,
)
from .multi_cam import (  # noqa: F401
    SITES_DIR,
    find_repo,
    list_sites,
    load_site,
    prepare_dataset,
    run_repo,
)


class FloorplanService:
    """Unified interface for single-camera and calibrated multi-camera 2D floorplan pipelines."""

    @staticmethod
    def generate_single_cam(video, out_dir, cache_root=None, log=print, timeout=900):
        return single_cam.generate(video, out_dir, cache_root=cache_root, log=log, timeout=timeout)

    @staticmethod
    def find_repo():
        return multi_cam.find_repo()

    @staticmethod
    def list_sites():
        return multi_cam.list_sites()

    @staticmethod
    def load_site(site_id):
        return multi_cam.load_site(site_id)

    @staticmethod
    def prepare_dataset(site, videos, work_dir, seconds=None, log=print):
        return multi_cam.prepare_dataset(site, videos, work_dir, seconds=seconds, log=log)

    @staticmethod
    def run_repo(repo, data_dir, out_dir, frames, site_room, device='cpu', log=print, timeout=3600):
        return multi_cam.run_repo(repo, data_dir, out_dir, frames, site_room, device=device, log=log, timeout=timeout)

    @classmethod
    def run_calibrated_multicam(cls, site, videos, work_dir, out_dir=None, seconds=None, device='cpu', log=print, timeout=3600):
        site_def = cls.load_site(site) if isinstance(site, str) else dict(site)
        repo = cls.find_repo()
        if repo is None:
            raise RuntimeError('Không thấy repo multicam-2d-floorplan-tracker (đặt FLOORPLAN_DIR).')
        work_p = Path(work_dir)
        data_dir, n = cls.prepare_dataset(site_def, videos, work_p / 'data', seconds=seconds, log=log)
        out_p = Path(out_dir) if out_dir is not None else (work_p / 'out')
        cls.run_repo(repo, data_dir, out_p, n, site_def['room_id'], device=device, log=log, timeout=timeout)
        res_file = out_p / 'site_result.json'
        if not res_file.is_file():
            raise RuntimeError('Pipeline map2d không tạo ra site_result.json')
        return dict(
            out_dir=str(out_p),
            n_samples=n,
            result_path=str(res_file),
            result=json.loads(res_file.read_text(encoding='utf-8')),
        )


__all__ = [
    'FloorplanService',
    'calibration',
    'single_cam',
    'multi_cam',
    'SITES_DIR',
    'SEVERITY',
    'MAX_RANGE_M',
    'CalibrationError',
    'GlobalIdLabeler',
    'appearance_embedding',
    'normalize_calibration',
    'split_calibration',
    'room_bounds_from',
    'fit_to_video',
    'floor_mapper',
    'homography_mapper',
    'stitch_ids',
    'engine_summary',
    'engine_events',
    'build_incidents',
    'find_repo',
    'list_sites',
    'load_site',
    'prepare_dataset',
    'run_repo',
]
