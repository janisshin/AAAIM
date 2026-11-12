"""LLM backends for the automatic annotation agent."""

from core.agent.backends.mock import MockBackend
from core.agent.backends.openai_planner import OpenAIPlannerBackend

__all__ = ["MockBackend", "OpenAIPlannerBackend"]

try:
    from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

    __all__ += ["MetaboLLMBackend", "ensure_gpu_idle_enough"]
except Exception:  # pragma: no cover - optional heavy deps
    MetaboLLMBackend = None  # type: ignore
    ensure_gpu_idle_enough = None  # type: ignore
