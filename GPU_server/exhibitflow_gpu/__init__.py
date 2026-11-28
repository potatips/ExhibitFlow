"""ExhibitFlow GPU ML package: standalone anomaly detection (touching / holding / stealing) in museum video.

    from exhibitflow_gpu import run_pipeline
    run_pipeline(['video.mp4'], out_dir='results')
"""
from .settings import VERSION as __version__  # noqa: F401


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run  # lazy import: avoid loading torch / cv2 when only __version__ is needed
    return _run(*args, **kwargs)
