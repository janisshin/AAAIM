"""
Reaction eligibility classification for scoring/coverage accounting.
"""

from __future__ import annotations

from typing import Iterable


def classify_reaction(
    reaction,
    filtered_species: Iterable[str],
    candidates: Iterable[str],
) -> str:
    """
    Classify a reaction by score eligibility.

    Returns:
        - "non_mappable": no species remain after filtering
        - "failed_mapping": mappable species exist but no KEGG candidates
        - "mappable": eligible and has at least one candidate
    """
    _ = reaction
    filtered = list(filtered_species) if filtered_species is not None else []
    cand = list(candidates) if candidates is not None else []
    if len(filtered) == 0:
        return "non_mappable"
    if len(cand) == 0:
        return "failed_mapping"
    return "mappable"

