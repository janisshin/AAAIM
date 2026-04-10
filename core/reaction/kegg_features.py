"""Load and query KEGG reaction feature payloads (definitions, equations)."""

from __future__ import annotations

import logging
import lzma
import pickle
import re
from typing import Dict

from utils.constants import REF_KEGG_REACTION_FEATURES

from .kegg_definition import extract_classifications

logger = logging.getLogger(__name__)


def _normalize_kegg_reaction_id(annotation) -> str:
    """Resolve KEGG reaction id (R#####) from table/URI values like ``KEGG:R01600``."""
    if annotation is None:
        return ""
    if isinstance(annotation, float) and annotation != annotation:
        return ""
    s = str(annotation).strip()
    if not s or s.lower() == "nan":
        return ""
    if "KEGG:" in s.upper():
        s = s.split("KEGG:")[-1].strip()
    m = re.search(r"\b(R\d{5})\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"R\d{5}", s, flags=re.IGNORECASE):
        return s.upper()
    return ""


class KEGGReactionFeatures:
    """Encapsulates KEGG reaction feature data and operations."""

    def __init__(self, features_dict: Dict):
        self._features = features_dict

    def get_participants(self, annotation: str) -> str:
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            return ""
        definition = self._features.get(kegg_id, {}).get("DEFINITION", "")
        return extract_classifications(definition, "definition")

    def get_participant_ids(self, annotation: str) -> str:
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            return ""
        definition = self._features.get(kegg_id, {}).get("EQUATION", "")
        return extract_classifications(definition, "definition")

    def get_definition(self, annotation: str) -> str:
        """KEGG ``DEFINITION`` (human-readable reaction string) for the reaction."""
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            return ""
        raw = (self._features.get(kegg_id, {}) or {}).get("DEFINITION", "") or ""
        return " ".join(ln.strip() for ln in str(raw).splitlines() if ln.strip())

    @classmethod
    def load_from_file(cls, data_path: str) -> "KEGGReactionFeatures":
        try:
            with lzma.open(data_path, "rb") as f:
                features_dict = pickle.load(f)
            logger.info("Loaded KEGG reaction features from %s", data_path)
            return cls(features_dict)
        except (FileNotFoundError, lzma.LZMAError) as e:
            logger.error("Error loading KEGG reaction features: %s", e)
            return cls({})
