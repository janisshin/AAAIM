"""Pytest bootstrap: prefer local chromadb stub when the real package DLL is blocked."""
from __future__ import annotations

import sys
from pathlib import Path

_STUBS = Path(__file__).resolve().parent / "tests" / "_stubs"
if _STUBS.is_dir() and str(_STUBS) not in sys.path:
    sys.path.insert(0, str(_STUBS))
