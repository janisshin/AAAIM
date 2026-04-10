"""Scoring and similarity helpers for reaction annotation."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Set, Iterable

import pandas as pd
from rapidfuzz import fuzz

from .amendment_config import MatchingConfig

if TYPE_CHECKING:
    from .amendment import ParticipantFilter


class TextNormalizer:
    """Handles text normalization for species name comparison."""
    
    @staticmethod
    def standardize_name(name: str) -> str:
        """Standardize species names for comparison."""
        name = name.lower()
        name = name.replace('α-', 'alpha-').replace('β-', 'beta-')
        name = name.replace('α', 'alpha').replace('β', 'beta')
        name = name.replace('-', ' ')
        return name


class SimilarityCalculator:
    """Handles similarity calculations between species and reactions."""
    
    def __init__(self, config: MatchingConfig):
        self.config = config
    
    def is_plausible_match(self, query_species: str, cand_species: str) -> bool:
        """Check if two species names are plausibly similar."""
        max_score = fuzz.partial_ratio(query_species.lower(), cand_species.lower())
        return max_score >= self.config.similarity_threshold
    
    def fuzzy_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute fuzzy Jaccard similarity between two sets of strings."""
        if not set_a or not set_b:
            return 0.0
            
        overlap = 0
        for a in set_a:
            best = max((_cached_fuzz_ratio(a, b) for b in set_b), default=0)
            if best * 100 >= self.config.jaccard_threshold:
                overlap += best
        
        denom = len(set_a) + len(set_b) - overlap
        return overlap / denom if denom > 0 else 0.0


def softmax_normalize(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Apply softmax normalization to convert scores to probabilities.
    
    Args:
        scores: Dictionary mapping keys to scores
        temperature: Temperature parameter for softmax (default=1.0)
        
    Returns:
        Dictionary with softmax-normalized probabilities that sum to 1
    """
    import numpy as np
    
    if not scores:
        return {}
    
    # Extract values and apply softmax
    keys = list(scores.keys())
    values = np.array([scores[k] for k in keys])
    
    # Subtract max for numerical stability
    values_shifted = values - np.max(values)
    exp_values = np.exp(values_shifted / temperature)
    probabilities = exp_values / np.sum(exp_values)
    
    return {k: float(p) for k, p in zip(keys, probabilities)}


@lru_cache(maxsize=200_000)
def _cached_fuzz_ratio(a: str, b: str) -> float:
    """Cached symmetric rapidfuzz ratio scaled to [0, 1]."""
    a = str(a)
    b = str(b)
    if a == b:
        return 1.0
    # Symmetry: ratio(a,b) == ratio(b,a)
    if a > b:
        a, b = b, a
    return fuzz.ratio(a, b) / 100.0


def _parse_participant_ids_field(value) -> Set[str]:
    if value is None or (isinstance(value, float) and value != value):
        return set()
    s = str(value)
    if not s or s.lower() == "nan":
        return set()
    return {p.strip() for p in s.split(";") if p and p.strip()}


def compute_rscore_from_participant_ids(
    reference_participant_ids_value,
    participant_annotations: Dict[str, str],
    participant_filter: "ParticipantFilter",
    similarity_calc: "SimilarityCalculator",
    *,
    prefiltered_ref_participant_ids: Set[str] | None = None,
    prefiltered_query_kegg_ids: Set[str] | None = None,
) -> float:
    """
    Compute reaction match score (rScore) for a query-reference reaction pair.
    
    The rScore is computed as a weighted combination of:
    - Number of matched participants
    - Similarity of matched participants (formula, charge, annotation consistency)
    - Weighted by match scores
    
    Args:
        query_reaction_id: ID of the query reaction
        reference_reaction: Series containing reference reaction data
        participant_annotations: Dict mapping participant IDs to their current annotations
        participant_filter: Filter for removing cofactors
        similarity_calc: Calculator for similarity metrics
        
    Returns:
        rScore value between 0 and 1
    """
    if prefiltered_ref_participant_ids is None:
        ref_participant_ids = _parse_participant_ids_field(reference_participant_ids_value)
        ref_participant_ids = participant_filter.filter_cofactors(ref_participant_ids)
    else:
        ref_participant_ids = prefiltered_ref_participant_ids
    
    if not ref_participant_ids:
        return 0.0
    
    # Get query participants with current annotations (optionally prefiltered/cached by caller).
    if prefiltered_query_kegg_ids is None:
        query_kegg_ids = set(participant_annotations.values())
        query_kegg_ids = participant_filter.filter_cofactors(query_kegg_ids)
    else:
        query_kegg_ids = prefiltered_query_kegg_ids
    
    if not query_kegg_ids:
        return 0.0
    
    # Count matched participants
    matched_participants = ref_participant_ids.intersection(query_kegg_ids)
    num_matched = len(matched_participants)
    
    if num_matched == 0:
        return 0.0
    
    # Compute Jaccard similarity for matched participants
    jaccard_score = similarity_calc.fuzzy_jaccard(query_kegg_ids, ref_participant_ids)
    
    # Combine metrics: weighted average of match ratio and Jaccard similarity
    match_ratio = num_matched / max(len(ref_participant_ids), len(query_kegg_ids))
    rscore = 0.5 * match_ratio + 0.5 * jaccard_score
    
    return rscore


def compute_rscore_from_sets(
    query_kegg_ids: Set[str],
    ref_participant_ids: Set[str],
    similarity_calc: "SimilarityCalculator",
) -> float:
    """
    Compute rScore given already-filtered query and reference KEGG participant-id sets.

    This is the lowest-overhead scoring path and is intended for callers that
    cache cofactor filtering and participant-id parsing.
    """
    if not query_kegg_ids or not ref_participant_ids:
        return 0.0

    matched_participants = ref_participant_ids.intersection(query_kegg_ids)
    num_matched = len(matched_participants)
    if num_matched == 0:
        return 0.0

    jaccard_score = similarity_calc.fuzzy_jaccard(query_kegg_ids, ref_participant_ids)
    match_ratio = num_matched / max(len(ref_participant_ids), len(query_kegg_ids))
    return 0.5 * match_ratio + 0.5 * jaccard_score
