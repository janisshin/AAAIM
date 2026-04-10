"""Parse KEGG-style definition / classification text into cleaned strings."""

from __future__ import annotations

import functools
import re


@functools.lru_cache(maxsize=4096)
def extract_classifications(raw_text: str, classification: str) -> str:
    """Extract and clean classification text based on type (brite, orthology, definition)."""
    lines = raw_text.splitlines()
    clean_lines: list[str] = []

    if classification == "brite":
        for line in lines:
            stripped = line.strip()
            if not stripped or "[BR:" in stripped:
                continue
            if re.fullmatch(r"(\d+\.)+\d+", stripped):
                continue
            if re.match(r"R\d{5}", stripped):
                continue

            parts = stripped.split(maxsplit=1)
            if len(parts) > 1:
                clean_lines.append(parts[1].strip())
            else:
                clean_lines.append(stripped)

    elif classification == "orthology":
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)

    elif classification == "definition":
        parts: list[str] = []
        buf = ""
        paren_level = 0

        i = 0
        while i < len(raw_text):
            c = raw_text[i]

            if c == "(":
                paren_level += 1
            elif c == ")":
                paren_level -= 1

            if c == "+" and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
            elif raw_text[i : i + 3] == "<=>" and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 2
            elif raw_text[i : i + 2] == "->" and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 1
            else:
                buf += c

            i += 1

        if buf:
            parts.append(buf.strip())

        strip_dollars = [p.lstrip("$") for p in parts if p]
        clean_lines = [
            re.sub(r"^(?:\(?[0-9nmt+\-*/]+\)?\s+)+", "", p).strip() for p in strip_dollars
        ]

    return "; ".join(set(clean_lines))
