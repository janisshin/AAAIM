"""Load and query KEGG reaction feature payloads (definitions, equations)."""

from __future__ import annotations

import logging
import lzma
import pickle
from typing import Dict

from .kegg_definition import extract_classifications

logger = logging.getLogger(__name__)


class KEGGReactionFeatures:
    """Encapsulates KEGG reaction feature data and operations."""

    def __init__(self, features_dict: Dict):
        self._features = features_dict

    def get_participants(self, annotation: str) -> str:
        kegg_id = annotation.split(":")[1] if ":" in annotation else annotation
        definition = self._features.get(kegg_id, {}).get("DEFINITION", "")
        return extract_classifications(definition, "definition")

    def get_participant_ids(self, annotation: str) -> str:
        kegg_id = annotation.split(":")[1] if ":" in annotation else annotation
        definition = self._features.get(kegg_id, {}).get("EQUATION", "")
        return extract_classifications(definition, "definition")

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
