"""Automatic model-annotation agent built on existing AAAIM tools."""

from core.agent.controller import auto_annotate_model
from core.agent.schemas import AutoAnnotateResult, ExecutionLimits, AnnotationScope

__all__ = [
    "auto_annotate_model",
    "AutoAnnotateResult",
    "ExecutionLimits",
    "AnnotationScope",
]
