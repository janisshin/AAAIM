"""Atomic filesystem helpers for agent checkpoints and exports."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _replace_with_retry(tmp: Path, path: Path, *, attempts: int = 12, delay_s: float = 0.05) -> None:
    """Atomically replace ``path`` with ``tmp`` (retry briefly on Windows locks)."""
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay_s * (i + 1))
    if last_exc is not None:
        raise last_exc
    raise PermissionError(f"could not replace {path}")


def atomic_write_json(obj: Any, path: Path | str) -> str:
    """Write JSON via temp file + replace so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    _replace_with_retry(tmp, path)
    return str(path)
