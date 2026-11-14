"""Vision pipeline: detection, alignment, embedding, liveness checks. Independent of Flask.

Usage:
    from app.config import VisionConfig
    from app.vision import VisionModels, run_pipeline
    models = VisionModels.load()                     # once, at startup
    result = run_pipeline(frames, models, template=student_template_or_None)
"""
from .pipeline import FAIL, PASS, RETRY, PipelineResult, VisionModels, run_pipeline

__all__ = ["FAIL", "PASS", "RETRY", "PipelineResult", "VisionModels", "run_pipeline"]
