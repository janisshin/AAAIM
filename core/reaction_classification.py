"""
Reaction eligibility classification for scoring/coverage accounting.
"""

from __future__ import annotations

from typing import Iterable
from core.reaction_amendment_config import CofactorConfig


def classify_reaction(
    reaction,
    filtered_species: Iterable[str],
    candidates: Iterable[str],
) -> str:
    """
    Classify a reaction by score eligibility.

    Returns:
        - "non_mappable": no species remain after filtering
        - "ambiguous_mapping": mappable species exist but no KEGG candidates
        - "mappable": eligible and has at least one candidate
    """
    filtered = list(filtered_species) if filtered_species is not None else []
    cand = list(candidates) if candidates is not None else []
    if all(species in CofactorConfig().kegg_ids for species in filtered):
        return "ambiguous_mapping"
    if "=>" in reaction:
        lhs, rhs = (s.strip() for s in reaction.split("=>", 1))
    elif "->" in reaction:
        lhs, rhs = (s.strip() for s in reaction.split("->", 1))
    else:
        return "non_mappable"
    if not lhs or not rhs:
        return "non_mappable"
    if len(cand) == 0:
        return "non_mappable"
    return "mappable"

