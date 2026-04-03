"""Scoring and similarity helpers for reaction annotation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Set

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
            best = max((fuzz.ratio(a, b) / 100 for b in set_b), default=0)
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


def compute_rscore(
    reference_reaction: pd.Series,
    participant_annotations: Dict[str, str],
    participant_filter: 'ParticipantFilter',
    similarity_calc: 'SimilarityCalculator'
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
    # Extract reference reaction participants
    if pd.isna(reference_reaction.get('participant_ids')) or not reference_reaction['participant_ids']:
        return 0.0
    
    ref_participant_ids = set(
        p.strip() for p in str(reference_reaction['participant_ids']).split(';') if p.strip()
    )
    
    # Filter cofactors from reference participants
    ref_participant_ids = participant_filter.filter_cofactors(ref_participant_ids)
    
    if not ref_participant_ids:
        return 0.0
    
    # Get query participants with current annotations
    query_participant_ids = set(participant_annotations.keys())
    query_kegg_ids = set(participant_annotations.values())
    
    # Filter cofactors from query participants
    query_kegg_ids = participant_filter.filter_cofactors(query_kegg_ids)
    
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


def _species_ids_from_reaction_string(reaction_str: str) -> Set[str]:
    """
    Extract candidate species ids from an Antimony-like reaction string.

    Intended to map model species ids (as they appear in `participant_df['id']`)
    to the reaction ids in `reaction_ids`.
    """
    if not reaction_str:
        return set()
    s = str(reaction_str)
    # Remove comments / ids prefixes if any, keep only equation-ish content.
    # Tokenize on word boundaries (SBML ids are typically [A-Za-z0-9_]).
    toks = re.findall(r"\b[A-Za-z0-9_]+\b", s)
    if not toks:
        return set()
    # Filter out obvious non-species tokens.
    stop = {"to", "and", "or"}
    out: Set[str] = set()
    for t in toks:
        if not t:
            continue
        if t.isdigit():
            continue
        if t.lower() in stop:
            continue
        out.add(t.lstrip("$"))
    return out
