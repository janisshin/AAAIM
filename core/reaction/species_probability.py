""" Species–candidate probability helpers for reaction annotation workflows (EM-style updates)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def normalize_probability_dict(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize values in ``prob_dict`` in place so they sum to 1."""
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict


def exact_ci_species_name_match(query_species: str, cand_species: str) -> bool:
    """Case-insensitive equality match for species names (simple baseline)."""
    return query_species.lower() == cand_species.lower()


def init_species_probs_for_reactions(
    query_reaction: Any,
    candidate_reactions: List[Any],
    is_plausible_match: Callable[[str, str], bool],
) -> Dict[str, Dict[str, float]]:
    """
    Initialize species match probabilities for a query reaction given candidate
    reference reactions. Each query species maps to candidate species and uniform
    initial probabilities over plausible matches.
    """
    species_match_probs: Dict[str, Dict[str, float]] = {}

    for query_species in query_reaction.participants:
        possible_matches = set()
        for candidate in candidate_reactions:
            for cand_species in candidate.participants:
                if is_plausible_match(query_species, cand_species):
                    possible_matches.add(cand_species)

        if possible_matches:
            prob = 1.0 / len(possible_matches)
            species_match_probs[query_species] = {s: prob for s in possible_matches}
        else:
            species_match_probs[query_species] = {}

    return species_match_probs


def init_species_probs_from_dict(
    reaction_participants: Dict[str, List[str]],
    counters: pd.Series,
    is_plausible_match: Callable[[str, str], bool],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Initialize species match probabilities from participant lists and count counters."""
    species_match_probs: Dict[str, Dict[str, Dict[str, float]]] = {}

    for rxn_id, query_species_list in reaction_participants.items():
        if rxn_id not in counters:
            continue

        candidate_counter = counters[rxn_id]
        species_probs_for_rxn: Dict[str, Dict[str, float]] = {}

        for query_species in query_species_list:
            plausible = {
                cand: count
                for cand, count in candidate_counter.items()
                if is_plausible_match(query_species, cand)
            }

            if plausible:
                total = sum(plausible.values())
                species_probs_for_rxn[query_species] = {
                    cand: count / total for cand, count in plausible.items()
                }
            else:
                species_probs_for_rxn[query_species] = {}

        species_match_probs[rxn_id] = species_probs_for_rxn

    return species_match_probs


def update_species_probs(
    query_species: str,
    candidate_reactions: List[Any],
    candidate_probs: Dict[Any, float],
    is_plausible_match: Callable[[str, str], bool],
) -> Dict[str, float]:
    """Update species probabilities from candidate reaction weights."""
    updated_probs: Dict[str, float] = {}

    for candidate in candidate_reactions:
        prob_candidate = candidate_probs.get(candidate, 0.0)

        for cand_species in candidate.participants:
            if is_plausible_match(query_species, cand_species):
                if cand_species not in updated_probs:
                    updated_probs[cand_species] = 0.0
                updated_probs[cand_species] += prob_candidate

    return normalize_probability_dict(updated_probs)


def choose_best_annotation(species_probs: Dict[str, float]) -> Optional[str]:
    """Return the candidate species with highest probability, or None if empty."""
    if not species_probs:
        return None
    return max(species_probs, key=species_probs.get)


def has_converged(
    updated_annotations: Dict[str, str],
    previous_annotations: Dict[str, str],
) -> bool:
    """True if every species annotation is unchanged from the previous iteration."""
    if not previous_annotations:
        return False

    for species, new_annotation in updated_annotations.items():
        old_annotation = previous_annotations.get(species)
        if new_annotation != old_annotation:
            return False

    return True
