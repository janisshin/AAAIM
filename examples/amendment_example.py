#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KEGG Reaction Annotation Example

This script demonstrates the workflow for annotating reactions in SBML models
using KEGG database references. It includes methods for mapping ChEBI IDs to KEGG
compound IDs, extracting reaction participants, and computing likelihood scores
for candidate reactions.

"""

import os
import sys
import re
import logging
import lzma
import pickle
from pathlib import Path
from collections import Counter
from itertools import chain
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    annotate_model, 
    get_available_databases, 
    database_search,
    load_chebi2kegg_dict, 
)
from core.model_info import (
    extract_reactions_from_sbml, 
    extract_model_info,
    get_all_reaction_ids
)
from core.annotation_workflow import map_reactions_to_kegg_with_relaxation, _generate_recommendation_table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


#------------------------------------------------------------------------------
# Configuration Classes
#------------------------------------------------------------------------------

@dataclass
class CofactorConfig:
    """Configuration for cofactors to ignore in reaction matching."""
    cofactors_dict: Dict[str, str] = field(default_factory=lambda: {
        'C00001': 'H2O',
        'C00080': 'H+',
        'C00007': 'O2',
        'C00027': 'H2O2',
        'C00009': 'Phosphate',
        'C00013': 'Diphosphate',
        'C00008': 'ADP',
        'C00002': 'ATP',
        'C00003': 'NAD+',
        'C00004': 'NADH',
        'C00005': 'NADPH',
        'C00006': 'NADP+'
    })
    
    @property
    def kegg_ids(self) -> Set[str]:
        """Get set of KEGG IDs for cofactors."""
        return set(self.cofactors_dict.keys())
    
    @property
    def name_patterns(self) -> List[str]:
        """Get list of name patterns for cofactors."""
        return list(self.cofactors_dict.values())
    
    def should_filter(self, participant: str) -> bool:
        """Check if a participant should be filtered as a cofactor."""
        return any(pattern in participant for pattern in self.name_patterns)


@dataclass
class ConvergenceConfig:
    """Configuration for iterative convergence algorithm."""
    max_iterations: int = 5
    threshold: float = 0.001
    stable_count: int = 3
    
    # EM algorithm parameters
    match_score_cutoff: float = 0.1  # Filter reactions by match score
    convergence_threshold: float = 0.001  # For rScore changes
    
    # Blending parameters
    reaction_alpha_start: float = 0.1
    reaction_alpha_increment: float = 0.1
    reaction_alpha_max: float = 0.9
    participant_alpha: float = 0.7
    
    # Participant discovery parameters
    participant_confidence_threshold: float = 0.3
    enable_participant_discovery: bool = True
    min_reaction_likelihood_for_discovery: float = 0.1
    
    def get_reaction_alpha(self, iteration: int) -> float:
        """Calculate alpha for reaction likelihood blending based on iteration."""
        return min(
            self.reaction_alpha_start + (iteration * self.reaction_alpha_increment),
            self.reaction_alpha_max
        )


@dataclass
class MatchingConfig:
    """Configuration for fuzzy matching."""
    similarity_threshold: int = 80
    jaccard_threshold: int = 70
    default_low_probability: float = 1e-6


#------------------------------------------------------------------------------
# Utility Classes
#------------------------------------------------------------------------------

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


class KEGGReactionFeatures:
    """Encapsulates KEGG reaction feature data and operations."""
    
    def __init__(self, features_dict: Dict):
        self._features = features_dict
    
    def get_participants(self, annotation: str) -> str:
        """Extract participant names from a KEGG annotation."""
        kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
        definition = self._features.get(kegg_id, {}).get("DEFINITION", "")
        return extract_classifications(definition, 'definition')
    
    def get_participant_ids(self, annotation: str) -> str:
        """Extract participant KEGG IDs from a KEGG annotation."""
        kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
        definition = self._features.get(kegg_id, {}).get("EQUATION", "")
        return extract_classifications(definition, 'definition')
    
    @classmethod
    def load_from_file(cls, data_path: str) -> 'KEGGReactionFeatures':
        """Load KEGG reaction features from compressed file."""
        try:
            with lzma.open(data_path, 'rb') as f:
                features_dict = pickle.load(f)
            logger.info(f"Loaded KEGG reaction features from {data_path}")
            return cls(features_dict)
        except (FileNotFoundError, lzma.LZMAError) as e:
            logger.error(f"Error loading KEGG reaction features: {e}")
            return cls({})


#------------------------------------------------------------------------------
# Text Processing Functions
#------------------------------------------------------------------------------

def extract_classifications(raw_text: str, classification: str) -> str:
    """Extract and clean classification text based on type."""
    lines = raw_text.splitlines()
    clean_lines = []

    if classification == 'brite':
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
    
    elif classification == 'orthology':
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)

    elif classification == 'definition':
        parts = []
        buf = ""
        paren_level = 0

        i = 0
        while i < len(raw_text):
            c = raw_text[i]

            if c == '(':
                paren_level += 1
            elif c == ')':
                paren_level -= 1

            if c == '+' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
            elif raw_text[i:i+3] == '<=>' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 2
            elif raw_text[i:i+2] == '->' and paren_level == 0:
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
            re.sub(r'^(?:\(?[0-9nmt+\-*/]+\)?\s+)+', '', p).strip()
            for p in strip_dollars
        ]
    
    return "; ".join(set(clean_lines))


#------------------------------------------------------------------------------
# Probability and Matching Functions
#------------------------------------------------------------------------------

def normalize(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize probability dictionary to sum to 1."""
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict


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
    query_reaction_id: str,
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


def update_species_probs(
    query_species: str, 
    candidate_reactions: List, 
    candidate_probs: Dict,
    similarity_calc: SimilarityCalculator
) -> Dict[str, float]:
    """Update species probabilities based on candidate reactions."""
    updated_probs = {}

    for candidate in candidate_reactions:
        prob_candidate = candidate_probs.get(candidate, 0.0)
        
        for cand_species in candidate.participants:
            if similarity_calc.is_plausible_match(query_species, cand_species):
                if cand_species not in updated_probs:
                    updated_probs[cand_species] = 0.0
                updated_probs[cand_species] += prob_candidate

    return normalize(updated_probs)


def choose_best_annotation(species_probs: Dict[str, float]) -> Optional[str]:
    """Select the best annotation based on probability scores."""
    if not species_probs:
        return None
    return max(species_probs, key=species_probs.get)


def has_converged(
    updated_annotations: Dict[str, str], 
    previous_annotations: Dict[str, str]
) -> bool:
    """Check if annotations have converged."""
    if not previous_annotations:
        return False
    
    for species, new_annotation in updated_annotations.items():
        old_annotation = previous_annotations.get(species)
        if new_annotation != old_annotation:
            return False
    
    return True


def init_species_probs_from_dict(
    reaction_participants: Dict[str, List[str]], 
    counters: pd.Series,
    similarity_calc: SimilarityCalculator
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Initialize species match probabilities from reaction participants."""
    species_match_probs = {}

    for rxn_id, query_species_list in reaction_participants.items():
        if rxn_id not in counters:
            continue

        candidate_counter = counters[rxn_id]
        species_probs_for_rxn = {}

        for query_species in query_species_list:
            plausible = {
                cand: count for cand, count in candidate_counter.items()
                if similarity_calc.is_plausible_match(query_species, cand)
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


#------------------------------------------------------------------------------
# Likelihood Calculation Classes
#------------------------------------------------------------------------------

class ParticipantFilter:
    """Filters participants based on cofactor configuration."""
    
    def __init__(self, cofactor_config: CofactorConfig):
        self.cofactor_config = cofactor_config
    
    def filter_cofactors(self, participants: Set[str]) -> Set[str]:
        """Remove cofactors from participant set."""
        return {
            p for p in participants 
            if p and not self.cofactor_config.should_filter(p)
        }


class SpeciesProbabilityCalculator:
    """Calculates species-level match probabilities."""
    
    def __init__(self, config: MatchingConfig):
        self.config = config
        self.normalizer = TextNormalizer()
    
    def compute_species_probability(
        self,
        filtered_candidate_participants: Set[str],
        init_probs: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Tuple[float, Set[str]]:
        """Compute probability product and matched participants."""
        prob_product = 1.0
        query_participants = set()
        kegg_id_to_prob = {}
        
        # Build KEGG ID to probability mapping
        for reaction_name, species_dict in init_probs.items():
            for species_name, kegg_dict in species_dict.items():
                for kegg_id, prob in kegg_dict.items():
                    kegg_id_to_prob[kegg_id] = prob
        
        # Match participants
        for participant in filtered_candidate_participants:
            if "KEGG:" in participant:
                kegg_id = participant.split("KEGG:")[-1].strip()
                if kegg_id in kegg_id_to_prob:
                    prob = kegg_id_to_prob[kegg_id]
                    prob_product *= prob
                    query_participants.add(participant)
                    logger.debug(f"Matched KEGG ID {kegg_id} with probability {prob}")
            else:
                # Try name matching
                std_participant = self.normalizer.standardize_name(participant)
                for reaction_name, species_dict in init_probs.items():
                    for species_name, kegg_dict in species_dict.items():
                        if self.normalizer.standardize_name(species_name) == std_participant:
                            prob = max(kegg_dict.values())
                            prob_product *= prob
                            query_participants.add(participant)
                            logger.debug(f"Matched by name to {species_name} with probability {prob}")
                            break
        
        if not query_participants:
            logger.debug("No matches found in init_probs")
            prob_product = self.config.default_low_probability
        
        return prob_product, query_participants


class LikelihoodCalculator:
    """Calculates reaction likelihoods based on participants."""
    
    def __init__(
        self,
        cofactor_config: CofactorConfig,
        matching_config: MatchingConfig,
        convergence_config: ConvergenceConfig
    ):
        self.cofactor_config = cofactor_config
        self.matching_config = matching_config
        self.convergence_config = convergence_config
        self.participant_filter = ParticipantFilter(cofactor_config)
        self.species_calc = SpeciesProbabilityCalculator(matching_config)
        self.similarity_calc = SimilarityCalculator(matching_config)
    
    def compute_rscores(
        self,
        participant_annotations: Dict[str, Dict[str, str]],
        kegg_recommendations_df: pd.DataFrame,
        prev_rscores: Optional[Dict[Tuple[str, str], float]] = None,
        iteration: int = 0
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], float]]:
        """
        Compute reaction match scores (rScores) for all query-reference reaction pairs.
        
        Args:
            participant_annotations: Dict mapping reaction_id -> {participant_id: kegg_annotation}
            kegg_recommendations_df: DataFrame with candidate reference reactions
            prev_rscores: Previous rScores for tracking convergence
            iteration: Current iteration number
            
        Returns:
            Tuple of (DataFrame with rScores and probabilities, Dict of rScores for convergence tracking)
        """
        logger.info(f"Computing rScores for {len(kegg_recommendations_df)} reaction candidates")
        
        result_df = kegg_recommendations_df.copy()
        result_df['rscore'] = 0.0
        result_df['reaction_probability'] = 0.0
        
        # Filter by match_score_cutoff
        if 'match_score' in result_df.columns:
            initial_count = len(result_df)
            result_df = result_df[
                result_df['match_score'] >= self.convergence_config.match_score_cutoff
            ].copy()
            logger.info(f"Filtered reactions by match_score >= {self.convergence_config.match_score_cutoff}: "
                       f"{len(result_df)}/{initial_count} remaining")
        
        # Store rScores for convergence tracking
        current_rscores = {}
        
        # Compute rScore for each query-reference reaction pair
        for idx, row in result_df.iterrows():
            query_rxn_id = row['id']
            ref_rxn_annotation = row['annotation']
            
            # Get current participant annotations for this query reaction
            query_annotations = participant_annotations.get(query_rxn_id, {})
            
            if not query_annotations:
                logger.debug(f"No participant annotations for query reaction {query_rxn_id}")
                continue
            
            # Compute rScore
            rscore = compute_rscore(
                query_rxn_id,
                row,
                query_annotations,
                self.participant_filter,
                self.similarity_calc
            )
            
            result_df.at[idx, 'rscore'] = rscore
            current_rscores[(query_rxn_id, ref_rxn_annotation)] = rscore
            
            logger.debug(f"rScore for {query_rxn_id} -> {ref_rxn_annotation}: {rscore:.4f}")
        
        # Normalize rScores to probabilities using softmax within each query reaction group
        for query_rxn_id in result_df['id'].unique():
            mask = result_df['id'] == query_rxn_id
            rxn_group = result_df[mask]
            
            if len(rxn_group) == 0:
                continue
            
            # Build score dictionary for softmax
            score_dict = {}
            for idx, row in rxn_group.iterrows():
                score_dict[idx] = row['rscore']
            
            # Apply softmax normalization
            prob_dict = softmax_normalize(score_dict)
            
            # Update probabilities in dataframe
            for idx, prob in prob_dict.items():
                result_df.at[idx, 'reaction_probability'] = prob
        
        logger.info(f"Computed rScores for {len(current_rscores)} query-reference pairs")
        
        return result_df, current_rscores
    
    def compute_reaction_likelihoods(
        self,
        init_probs: Dict[str, Dict[str, Dict[str, float]]],
        kegg_recommendations_df: pd.DataFrame,
        prev_likelihoods: Optional[pd.DataFrame] = None,
        iteration: int = 0
    ) -> pd.DataFrame:
        """
        Compute likelihood scores for reaction recommendations.
        
        DEPRECATED: This method is kept for backward compatibility.
        Use compute_rscores() for EM-style algorithm.
        """
        logger.warning("compute_reaction_likelihoods is deprecated. Use compute_rscores() instead.")
        logger.info(f"Computing likelihoods for {len(kegg_recommendations_df)} reactions")
        logger.debug(f"Reactions in init_probs: {len(init_probs)}")
        
        result_df = kegg_recommendations_df.copy()
        result_df['likelihood'] = 0.0
        
        for idx, row in result_df.iterrows():
            rxn_id = row['id']
            
            if pd.isna(row['participants']) or not row['participants']:
                continue
            
            # Filter candidate participants
            candidate_participants = set(str(row['participants']).split("; "))
            filtered_candidate_participants = self.participant_filter.filter_cofactors(
                candidate_participants
            )
            
            logger.debug(f"Processing reaction {rxn_id}")
            logger.debug(f"Filtered participants: {len(filtered_candidate_participants)}")
            
            # Compute species-level match probability
            prob_product, query_participants = self.species_calc.compute_species_probability(
                filtered_candidate_participants,
                init_probs
            )
            
            # Filter query participants
            filtered_query_participants = self.participant_filter.filter_cofactors(
                query_participants
            )
            
            # Compute Jaccard similarity
            if filtered_query_participants and filtered_candidate_participants:
                jaccard_score = self.similarity_calc.fuzzy_jaccard(
                    filtered_query_participants,
                    filtered_candidate_participants
                )
            else:
                logger.debug("Empty participant sets after filtering")
                jaccard_score = 0.0
            
            # Combine scores
            # new_likelihood = prob_product * jaccard_score
            new_likelihood = (0.7 * prob_product) + (0.3 * jaccard_score)
            
            # Blend with previous likelihood if available
            if prev_likelihoods is not None and not prev_likelihoods.empty:
                prev_row = prev_likelihoods[prev_likelihoods['id'] == row['id']]
                if not prev_row.empty:
                    prev_likelihood = prev_row['likelihood'].iloc[0]
                    alpha = self.convergence_config.get_reaction_alpha(iteration)
                    new_likelihood = (alpha * new_likelihood + (1 - alpha) * prev_likelihood)
            
            logger.debug(f"Scores for {rxn_id}: prob={prob_product:.4f}, jaccard={jaccard_score:.4f}, likelihood={new_likelihood:.4f}")
            result_df.at[idx, 'likelihood'] = new_likelihood
        
        # Normalize likelihoods within each reaction group
        result_df = self._normalize_by_group(result_df)
        
        return result_df
    
    def _normalize_by_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize likelihoods within each reaction group."""
        group_sums = df.groupby('id')['likelihood'].transform('sum')
        mask = group_sums > 0
        df.loc[mask, 'likelihood'] = df.loc[mask, 'likelihood'] / group_sums[mask]
        return df


#------------------------------------------------------------------------------
# Participant Likelihood Update Functions
#------------------------------------------------------------------------------

def participant_likelihoods_to_probs(
    participant_df: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert participant likelihoods DataFrame to probability dictionary."""
    probs = {}
    
    for rxn_id in participant_df['id'].unique():
        numeric_id = ''.join(filter(str.isdigit, rxn_id))
        if numeric_id:
            formatted_rxn_id = f'J{numeric_id}'
            rxn_mask = participant_df['id'] == rxn_id
            rxn_participants = participant_df[rxn_mask]
            
            if formatted_rxn_id not in probs:
                probs[formatted_rxn_id] = {}
            
            for _, row in rxn_participants.iterrows():
                if pd.notna(row['KEGG_ID']) and pd.notna(row['participant_likelihood']):
                    query_species = row['annotation_label']
                    kegg_id = row['KEGG_ID']
                    
                    if query_species not in probs[formatted_rxn_id]:
                        probs[formatted_rxn_id][query_species] = {}
                    
                    probs[formatted_rxn_id][query_species][kegg_id] = row['participant_likelihood']
    
    return probs


def update_participant_likelihoods_singleiter(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame,
    alpha: float = 0.7
) -> pd.DataFrame:
    """
    Perform a single EM iteration of participant likelihood updates.
    
    This implements the E-step and M-step of the EM algorithm:
    - E-step: For each query participant, compute expected annotation distribution
              by aggregating contributions from ALL candidate reference reactions
              weighted by P(r_ref | r_q)
    - M-step: For each query participant, assign annotation that maximizes
              expected likelihood
    
    Args:
        participant_df: DataFrame with participant candidates and KEGG IDs
        reaction_likelihood_df: DataFrame with reaction probabilities (from rScores)
        alpha: Blending parameter (kept for backward compatibility, not used in EM)
        
    Returns:
        Updated participant DataFrame with new likelihoods
    """
    updated_participants_df = participant_df.copy()
    
    if 'participant_likelihood' not in updated_participants_df.columns:
        updated_participants_df['participant_likelihood'] = 0.0
    
    # Use reaction_probability if available (from EM algorithm), otherwise fall back to likelihood
    prob_column = 'reaction_probability' if 'reaction_probability' in reaction_likelihood_df.columns else 'likelihood'
    
    logger.debug(f"Using '{prob_column}' column for participant likelihood updates")
    
    # E-STEP: Compute expected annotation distributions
    # For each query participant, aggregate contributions from all candidate reactions
    
    # Build mapping: participant_id -> {kegg_id -> [indices in dataframe]}
    participant_kegg_indices = {}
    for idx, row in updated_participants_df.iterrows():
        if pd.notna(row.get('KEGG_ID')) and row['KEGG_ID'] != '':
            participant_id = row['id']
            kegg_id = row['KEGG_ID']
            
            if participant_id not in participant_kegg_indices:
                participant_kegg_indices[participant_id] = {}
            if kegg_id not in participant_kegg_indices[participant_id]:
                participant_kegg_indices[participant_id][kegg_id] = []
            
            participant_kegg_indices[participant_id][kegg_id].append(idx)
    
    # Build mapping: query_reaction_id -> {ref_reaction_annotation -> probability}
    reaction_probs = {}
    for _, rxn_row in reaction_likelihood_df.iterrows():
        query_rxn_id = rxn_row['id']
        ref_annotation = rxn_row['annotation']
        prob = rxn_row[prob_column]
        
        if query_rxn_id not in reaction_probs:
            reaction_probs[query_rxn_id] = {}
        reaction_probs[query_rxn_id][ref_annotation] = prob
    
    # For each query participant, compute weighted contributions from all reactions
    for participant_id, kegg_dict in participant_kegg_indices.items():
        # Extract query reaction ID from participant ID (format: reaction_id_participant_name)
        # This assumes participant IDs contain the reaction ID
        query_rxn_id = participant_id.split('_')[0] if '_' in participant_id else participant_id
        
        # Get all candidate reference reactions for this query reaction
        if query_rxn_id not in reaction_probs:
            logger.debug(f"No reaction probabilities found for participant {participant_id}")
            continue
        
        ref_reactions = reaction_probs[query_rxn_id]
        
        # Accumulate weighted contributions for each KEGG ID candidate
        kegg_contributions = {kegg_id: 0.0 for kegg_id in kegg_dict.keys()}
        
        for ref_annotation, ref_prob in ref_reactions.items():
            # Get participant IDs from this reference reaction
            ref_row = reaction_likelihood_df[
                (reaction_likelihood_df['id'] == query_rxn_id) &
                (reaction_likelihood_df['annotation'] == ref_annotation)
            ]
            
            if ref_row.empty:
                continue
            
            ref_participant_ids = ref_row.iloc[0].get('participant_ids', '')
            if pd.isna(ref_participant_ids) or not ref_participant_ids:
                continue
            
            ref_kegg_ids = set(p.strip() for p in ref_participant_ids.split(';') if p.strip())
            
            # For each KEGG ID candidate of this participant, add weighted contribution
            for kegg_id in kegg_dict.keys():
                if kegg_id in ref_kegg_ids:
                    # This KEGG ID appears in this reference reaction
                    # Add contribution weighted by P(r_ref | r_q)
                    kegg_contributions[kegg_id] += ref_prob
        
        # M-STEP: Normalize contributions to get probability distribution
        total_contribution = sum(kegg_contributions.values())
        
        if total_contribution > 0:
            for kegg_id, contribution in kegg_contributions.items():
                normalized_likelihood = contribution / total_contribution
                
                # Update all rows for this participant-KEGG pair
                for idx in kegg_dict[kegg_id]:
                    updated_participants_df.at[idx, 'participant_likelihood'] = normalized_likelihood
        else:
            # No contributions - set uniform distribution
            uniform_prob = 1.0 / len(kegg_dict) if kegg_dict else 0.0
            for kegg_id in kegg_dict.keys():
                for idx in kegg_dict[kegg_id]:
                    updated_participants_df.at[idx, 'participant_likelihood'] = uniform_prob
    
    logger.debug(f"Updated participant likelihoods for {len(participant_kegg_indices)} participants")
    
    return updated_participants_df


class ConvergenceChecker:
    """Checks for convergence in iterative algorithms."""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.stable_iterations = 0
    
    def check_convergence(
        self,
        current_df: pd.DataFrame,
        updated_df: pd.DataFrame,
        iteration: int
    ) -> Tuple[bool, float]:
        """
        Check if the algorithm has converged based on participant likelihood changes.
        
        DEPRECATED: Use check_rscore_convergence() for EM-style algorithm.
        """
        if 'participant_likelihood' not in current_df.columns:
            return False, float('inf')
        
        comparison_df = current_df.merge(
            updated_df[['id', 'KEGG_ID', 'participant_likelihood']],
            on=['id', 'KEGG_ID'],
            suffixes=('_prev', '')
        )
        
        max_diff = (
            comparison_df['participant_likelihood'] -
            comparison_df['participant_likelihood_prev']
        ).abs().max()
        
        max_diff_rounded = round(max_diff, 3)
        
        logger.info(f"Iteration {iteration}: Maximum score change = {max_diff_rounded:.6f}")
        
        if max_diff_rounded <= self.config.threshold:
            self.stable_iterations += 1
            logger.info(f"Stable iteration {self.stable_iterations}/{self.config.stable_count}")
            
            if self.stable_iterations >= self.config.stable_count:
                logger.info(f"Convergence achieved after {iteration} iterations")
                return True, max_diff_rounded
        else:
            self.stable_iterations = 0
        
        return False, max_diff_rounded
    
    def check_rscore_convergence(
        self,
        prev_rscores: Optional[Dict[Tuple[str, str], float]],
        current_rscores: Dict[Tuple[str, str], float],
        iteration: int
    ) -> Tuple[bool, float]:
        """
        Check if the EM algorithm has converged based on rScore changes.
        
        Args:
            prev_rscores: Dictionary of previous rScores {(query_rxn_id, ref_rxn_annotation): rscore}
            current_rscores: Dictionary of current rScores
            iteration: Current iteration number
            
        Returns:
            Tuple of (converged: bool, max_change: float)
        """
        if prev_rscores is None or not prev_rscores:
            logger.info(f"Iteration {iteration}: First iteration, no previous rScores to compare")
            return False, float('inf')
        
        # Compute total change in rScores
        total_change = 0.0
        num_comparisons = 0
        max_change = 0.0
        
        # Compare rScores for common query-reference pairs
        common_keys = set(prev_rscores.keys()).intersection(set(current_rscores.keys()))
        
        for key in common_keys:
            prev_score = prev_rscores[key]
            curr_score = current_rscores[key]
            change = abs(curr_score - prev_score)
            total_change += change
            max_change = max(max_change, change)
            num_comparisons += 1
        
        # Compute average change
        avg_change = total_change / num_comparisons if num_comparisons > 0 else 0.0
        
        logger.info(f"Iteration {iteration}: rScore changes - "
                   f"avg={avg_change:.6f}, max={max_change:.6f}, "
                   f"compared {num_comparisons} reaction pairs")
        
        # Check convergence based on average change
        if avg_change <= self.config.convergence_threshold:
            self.stable_iterations += 1
            logger.info(f"Stable iteration {self.stable_iterations}/{self.config.stable_count}")
            
            if self.stable_iterations >= self.config.stable_count:
                logger.info(f"Convergence achieved after {iteration} iterations "
                           f"(avg rScore change: {avg_change:.6f})")
                return True, avg_change
        else:
            self.stable_iterations = 0
        
        return False, avg_change
    
    def reset(self):
        """Reset the convergence checker state."""
        self.stable_iterations = 0


def discover_new_participants(
    reaction_likelihood_df: pd.DataFrame,
    kegg_features: KEGGReactionFeatures,
    current_participant_ids: Set[str],
    convergence_config: ConvergenceConfig
) -> Set[str]:
    """
    Discover new participant IDs from high-likelihood reactions.
    
    Args:
        reaction_likelihood_df: DataFrame with reaction likelihoods
        kegg_features: KEGG reaction features for extracting participants
        current_participant_ids: Set of currently known participant IDs
        convergence_config: Configuration with discovery thresholds
        
    Returns:
        Set of newly discovered participant IDs
    """
    if not convergence_config.enable_participant_discovery:
        return set()
    
    # Use reaction_probability if available (from EM algorithm), otherwise fall back to likelihood
    prob_column = 'reaction_probability' if 'reaction_probability' in reaction_likelihood_df.columns else 'likelihood'
    
    # Filter for high-likelihood reactions
    high_likelihood_reactions = reaction_likelihood_df[
        reaction_likelihood_df[prob_column] >= convergence_config.min_reaction_likelihood_for_discovery
    ]
    
    logger.info(f"Discovering new participants from {len(high_likelihood_reactions)} high-likelihood reactions")
    
    new_participant_ids = set()
    
    for _, row in high_likelihood_reactions.iterrows():
        if pd.notna(row.get('participant_ids')) and row['participant_ids']:
            # Extract KEGG IDs from the reaction
            participant_ids = set(p.strip() for p in row['participant_ids'].split(';') if p.strip())
            
            # Find IDs that aren't in our current set
            novel_ids = participant_ids - current_participant_ids
            
            if novel_ids:
                prob_value = row.get(prob_column, 0.0)
                logger.debug(f"Reaction {row['id']} ({prob_column}={prob_value:.4f}) suggests new participants: {novel_ids}")
                new_participant_ids.update(novel_ids)
    
    logger.info(f"Discovered {len(new_participant_ids)} new participant IDs")
    return new_participant_ids


def suggest_kegg_candidates_from_reactions(
    reaction_likelihood_df: pd.DataFrame,
    current_participants_df: pd.DataFrame,
    kegg_features: KEGGReactionFeatures,
    convergence_config: ConvergenceConfig
) -> pd.DataFrame:
    """
    Suggest new KEGG ID candidates for existing participants based on
    co-occurrence in high-likelihood reactions.
    
    Args:
        reaction_likelihood_df: DataFrame with reaction likelihoods
        current_participants_df: Current participant DataFrame with KEGG IDs
        kegg_features: KEGG reaction features
        convergence_config: Configuration with thresholds
        
    Returns:
        DataFrame with new candidate rows to add to participants
    """
    new_candidate_rows = []
    
    # Use reaction_probability if available (from EM algorithm), otherwise fall back to likelihood
    prob_column = 'reaction_probability' if 'reaction_probability' in reaction_likelihood_df.columns else 'likelihood'
    
    # Get high-likelihood reactions
    high_likelihood_reactions = reaction_likelihood_df[
        reaction_likelihood_df[prob_column] >= convergence_config.min_reaction_likelihood_for_discovery
    ]
    
    logger.info(f"Analyzing {len(high_likelihood_reactions)} high-likelihood reactions for new KEGG candidates")
    
    for _, reaction in high_likelihood_reactions.iterrows():
        if pd.isna(reaction.get('participant_ids')) or not reaction['participant_ids']:
            continue
        
        # Get all KEGG IDs in this reaction
        reaction_kegg_ids = set(
            p.strip() for p in reaction['participant_ids'].split(';') if p.strip()
        )
        
        # Find which model participants are already in this reaction
        participants_in_reaction = {}
        for _, participant in current_participants_df.iterrows():
            if pd.notna(participant.get('KEGG_ID')) and participant['KEGG_ID'] in reaction_kegg_ids:
                participant_id = participant['id']
                if participant_id not in participants_in_reaction:
                    participants_in_reaction[participant_id] = set()
                participants_in_reaction[participant_id].add(participant['KEGG_ID'])
        
        # For each participant in this reaction, suggest OTHER KEGG IDs as candidates
        for participant_id, existing_kegg_ids in participants_in_reaction.items():
            # Get all existing KEGG IDs for this participant (not just in this reaction)
            all_existing = set(
                current_participants_df[
                    current_participants_df['id'] == participant_id
                ]['KEGG_ID'].dropna()
            )
            
            # Find novel KEGG IDs from this reaction
            novel_kegg_ids = reaction_kegg_ids - all_existing
            
            if novel_kegg_ids:
                # Get a template row for this participant
                template_row = current_participants_df[
                    current_participants_df['id'] == participant_id
                ].iloc[0].copy()
                
                logger.debug(f"Participant {participant_id} in reaction {reaction['annotation']}: "
                           f"suggesting {len(novel_kegg_ids)} new KEGG IDs: {novel_kegg_ids}")
                
                # Create new candidate rows for each novel KEGG ID
                for novel_id in novel_kegg_ids:
                    new_row = template_row.copy()
                    new_row['KEGG_ID'] = novel_id
                    new_row['annotation'] = novel_id  # Update annotation to KEGG ID
                    new_row['participant_likelihood'] = 0.0  # Will be updated in next iteration
                    new_row['match_score'] = 0 # this is an arbitrary default value 
                    new_candidate_rows.append(new_row)
    
    if new_candidate_rows:
        new_candidates_df = pd.DataFrame(new_candidate_rows)
        logger.info(f"Generated {len(new_candidates_df)} new KEGG ID candidates for existing participants")
        return new_candidates_df
    else:
        logger.info("No new KEGG ID candidates found")
        return pd.DataFrame()


def update_participant_likelihoods(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame,
    reaction_participants: Dict,
    model_file: str,
    model_info: Dict,
    kegg_features: KEGGReactionFeatures,
    reactions: List,
    entity_type: str = 'reaction',
    database: str = 'kegg',
    cofactor_config: Optional[CofactorConfig] = None,
    convergence_config: Optional[ConvergenceConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Update participant likelihoods iteratively using EM-style algorithm until convergence.
    
    This implements the full EM algorithm:
    - E₀: Initialize with match scores and compute initial rScores
    - Iterate:
        - E-step: Update participant annotations based on reaction probabilities
        - M-step: Recompute rScores based on updated annotations
        - Check convergence based on rScore changes
    """
    if cofactor_config is None:
        cofactor_config = CofactorConfig()
    if convergence_config is None:
        convergence_config = ConvergenceConfig()
    
    current_participants_df = participant_df.copy()
    convergence_checker = ConvergenceChecker(convergence_config)
    
    # Track all participant IDs we've seen
    all_known_participant_ids = set(current_participants_df['id'].unique())
    
    # Track rScores for convergence checking
    prev_rscores = None
    
    logger.info("Starting EM-style iterative participant likelihood updates")
    
    for iteration in range(1, convergence_config.max_iterations + 1):
        # Log statistics
        if 'participant_likelihood' in current_participants_df.columns:
            likelihood_stats = current_participants_df.groupby('id')['participant_likelihood'].agg(['mean', 'min', 'max'])
            logger.info(f"Iteration {iteration} - Before update:")
            for idx, row in likelihood_stats.head().iterrows():
                logger.debug(f"ID {idx}: mean={row['mean']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")
        
        # Perform single iteration update
        updated_participants_df = update_participant_likelihoods_singleiter(
            current_participants_df,
            reaction_likelihood_df,
            alpha=convergence_config.participant_alpha
        )
        
        # Map ChEBI IDs to KEGG Compound IDs
        logger.info(f"Iteration {iteration}: Mapping ChEBI IDs to KEGG Compound IDs")
        updated_participants_df_with_kegg, high_score_recommendations = map_chebi_to_kegg(
            updated_participants_df
        )
        
        # Suggest new KEGG ID candidates from high-likelihood reactions (if enabled)
        if iteration > 1 and convergence_config.enable_participant_discovery:
            logger.info(f"Iteration {iteration}: Suggesting new KEGG ID candidates from reactions")
            new_kegg_candidates = suggest_kegg_candidates_from_reactions(
                reaction_likelihood_df,
                updated_participants_df_with_kegg,
                kegg_features,
                convergence_config
            )
            
            if not new_kegg_candidates.empty:
                logger.info(f"Adding {len(new_kegg_candidates)} new KEGG ID candidates to participant pool")
                # Merge new candidates with existing participants
                updated_participants_df_with_kegg = pd.concat([
                    updated_participants_df_with_kegg,
                    new_kegg_candidates
                ]).drop_duplicates(subset=['id', 'KEGG_ID']).reset_index(drop=True)
                
                # Only add valid new candidates to high_score_recommendations
                valid_new_candidates = new_kegg_candidates[
                    new_kegg_candidates['KEGG_ID'].notna() &
                    (new_kegg_candidates['KEGG_ID'] != '')
                ].copy()
                
                if not valid_new_candidates.empty:
                    # Ensure new candidates have required columns
                    if 'match_score' not in valid_new_candidates.columns:
                        valid_new_candidates['match_score'] = 0.0
                    
                    high_score_recommendations = pd.concat([
                        high_score_recommendations,
                        valid_new_candidates
                    ]).drop_duplicates(subset=['id', 'KEGG_ID']).reset_index(drop=True)
                    
                    logger.info(f"Added {len(valid_new_candidates)} valid candidates to high_score_recommendations")
        
        # Discover new participant IDs from high-likelihood reactions (if enabled)
        if iteration > 1 and convergence_config.enable_participant_discovery:
            new_participant_ids = discover_new_participants(
                reaction_likelihood_df,
                kegg_features,
                all_known_participant_ids,
                convergence_config
            )
            
            if new_participant_ids:
                logger.info(f"Adding {len(new_participant_ids)} newly discovered participant IDs to search space")
                all_known_participant_ids.update(new_participant_ids)
        
        # Build expanded participant list for reaction extraction
        if convergence_config.enable_participant_discovery:
            # Include high-confidence participants
            high_confidence_mask = (
                updated_participants_df_with_kegg.get('participant_likelihood', 0) >=
                convergence_config.participant_confidence_threshold
            )
            high_confidence_ids = set(
                updated_participants_df_with_kegg[high_confidence_mask]['id'].unique()
            )
            
            # Combine with high-score recommendations
            expanded_participant_ids = list(set(
                list(high_score_recommendations['id'].unique()) +
                list(high_confidence_ids) +
                list(all_known_participant_ids)
            ))
            
            logger.info(f"Expanded participant search space: {len(expanded_participant_ids)} IDs "
                       f"(original: {len(high_score_recommendations['id'].unique())}, "
                       f"high-confidence: {len(high_confidence_ids)}, "
                       f"discovered: {len(all_known_participant_ids)})")
        else:
            expanded_participant_ids = list(high_score_recommendations['id'].unique())

        # Re-extract reactions with expanded participant list
        # logger.info(f"Iteration {iteration}: Re-extracting reactions with expanded participant list")
        reactions, _ = extract_reactions_from_sbml(
            model_file,
            expanded_participant_ids
        )
        
        # Map reactions to KEGG with updated participant set
        # logger.info(f"Iteration {iteration}: Mapping reactions to KEGG")
        normalized_reactions, match_results, _species_relax_levels = map_reactions_to_kegg_with_relaxation(
            reactions,
            high_score_recommendations,
            spectators=False,
            cofactors_to_ignore=cofactor_config.kegg_ids,
            top_k=None,
        )
        
        # Generate recommendation table
        # logger.info(f"Iteration {iteration}: Generating recommendation table")
        updated_kegg_recommendations_df = _generate_recommendation_table(
            model_file,
            match_results,
            {},
            model_info,
            entity_type,
            database,
            {}
        )
        
        # Add participant information
        updated_kegg_recommendations_df['participants'] = (
            updated_kegg_recommendations_df['annotation'].apply(kegg_features.get_participants)
        )
        updated_kegg_recommendations_df['participant_ids'] = (
            updated_kegg_recommendations_df['annotation'].apply(kegg_features.get_participant_ids)
        )
        
        # Build participant annotations dictionary for rScore computation
        # Format: {reaction_id: {participant_id: kegg_annotation}}
        participant_annotations = {}
        for _, row in updated_participants_df_with_kegg.iterrows():
            if pd.notna(row.get('KEGG_ID')) and row['KEGG_ID'] != '':
                # Extract reaction ID from participant ID
                participant_id = row['id']
                # Assume format like "R_HEX1_M_glc__D_c" where reaction is before first underscore after R_
                parts = participant_id.split('_')
                if len(parts) >= 2:
                    reaction_id = '_'.join(parts[:2])  # e.g., "R_HEX1"
                else:
                    reaction_id = participant_id
                
                if reaction_id not in participant_annotations:
                    participant_annotations[reaction_id] = {}
                
                # Use the participant with highest likelihood for this KEGG ID
                if participant_id not in participant_annotations[reaction_id]:
                    participant_annotations[reaction_id][participant_id] = row['KEGG_ID']
                else:
                    # Keep the one with higher likelihood
                    current_likelihood = updated_participants_df_with_kegg[
                        (updated_participants_df_with_kegg['id'] == participant_id) &
                        (updated_participants_df_with_kegg['KEGG_ID'] == participant_annotations[reaction_id][participant_id])
                    ]['participant_likelihood'].iloc[0] if len(updated_participants_df_with_kegg[
                        (updated_participants_df_with_kegg['id'] == participant_id) &
                        (updated_participants_df_with_kegg['KEGG_ID'] == participant_annotations[reaction_id][participant_id])
                    ]) > 0 else 0.0
                    
                    new_likelihood = row.get('participant_likelihood', 0.0)
                    if new_likelihood > current_likelihood:
                        participant_annotations[reaction_id][participant_id] = row['KEGG_ID']
        
        # M-STEP: Compute updated rScores based on current participant annotations
        logger.info(f"Iteration {iteration}: Computing updated rScores (M-step)")
        likelihood_calc = LikelihoodCalculator(
            cofactor_config,
            MatchingConfig(),
            convergence_config
        )
        updated_reaction_likelihood_df, current_rscores = likelihood_calc.compute_rscores(
            participant_annotations,
            updated_kegg_recommendations_df,
            prev_rscores,
            iteration
        )
        
        # Check convergence based on rScore changes
        converged, score_change = convergence_checker.check_rscore_convergence(
            prev_rscores,
            current_rscores,
            iteration
        )
        
        # Update prev_rscores for next iteration
        prev_rscores = current_rscores
        
        # Update state
        current_participants_df = updated_participants_df_with_kegg
        reaction_likelihood_df = updated_reaction_likelihood_df
        
        # Save iteration results
        updated_participants_df_with_kegg.to_csv(
            f'participants_likelihood_iter{iteration}.csv',
            index=False
        )
        
        updated_reaction_likelihood_df.to_csv(
            f'reaction_likelihood_iter{iteration}.csv',
            index=False
        )
        
        if converged:
            break
        
        if iteration == convergence_config.max_iterations:
            logger.warning(
                f"Maximum iterations ({convergence_config.max_iterations}) reached without convergence. "
                f"Last maximum change: {max_diff:.6f}"
            )
    
    return updated_participants_df_with_kegg, updated_reaction_likelihood_df


#------------------------------------------------------------------------------
# Main Workflow Functions
#------------------------------------------------------------------------------

def check_environment(model_file: str) -> bool:
    """Check if the environment is properly configured."""
    available_dbs = get_available_databases()
    logger.info(f"Available databases: {available_dbs}")
    
    all_ok = True
    
    if "chebi" not in available_dbs:
        logger.error("ChEBI chemical database not available!")
        logger.error("Please ensure ChEBI reference files are present in data/chebi/")
        all_ok = False
    
    if "kegg" not in available_dbs:
        logger.error("KEGG reaction database not available!")
        logger.error("Please ensure KEGG reference files are present in data/kegg/")
        all_ok = False
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        logger.error("Please provide a valid SBML model file.")
        all_ok = False
    
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API keys found in environment.")
        logger.warning("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
    
    return all_ok


def map_chebi_to_kegg(recommendations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map ChEBI IDs to KEGG Compound IDs."""
    chebi_to_kegg_map = load_chebi2kegg_dict()
    
    expanded_rows = []
    existing_mappings = {}
    
    if 'KEGG_ID' in recommendations_df.columns and 'participant_likelihood' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            if pd.notna(row['KEGG_ID']) and row['KEGG_ID'] != '':
                key = (row['id'], row['annotation'], row['KEGG_ID'])
                existing_mappings[key] = row['participant_likelihood']
    
    if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            chebi_id = row['annotation']
            kegg_ids = chebi_to_kegg_map.get(chebi_id, [])
            
            if not isinstance(kegg_ids, list):
                kegg_ids = [kegg_ids] if kegg_ids else []
            
            if not kegg_ids:
                row_copy = row.copy()
                row_copy['KEGG_ID'] = ""
                expanded_rows.append(row_copy)
            else:
                for kegg_id in kegg_ids:
                    if kegg_id:
                        row_copy = row.copy()
                        row_copy['KEGG_ID'] = kegg_id
                        key = (row['id'], row['annotation'], kegg_id)
                        if key in existing_mappings:
                            row_copy['participant_likelihood'] = existing_mappings[key]
                        expanded_rows.append(row_copy)
        
        expanded_df = pd.DataFrame(expanded_rows)
        
        if expanded_df.empty:
            recommendations_df['KEGG_ID'] = ""
            return recommendations_df, pd.DataFrame()
        
        combined_df = pd.concat([recommendations_df, expanded_df]).drop_duplicates(
            subset=['id', 'annotation', 'KEGG_ID']
        )
    else:
        recommendations_df['KEGG_ID'] = ""
        return recommendations_df, pd.DataFrame()
    
    filtered_df = combined_df[
        combined_df['KEGG_ID'].notna() &
        (combined_df['KEGG_ID'] != '')
    ]
    
    if not filtered_df.empty:
        high_score_recommendations = filtered_df[
            filtered_df['match_score'] == filtered_df.groupby('id')['match_score'].transform('max')
        ].reset_index(drop=True)
    else:
        high_score_recommendations = pd.DataFrame()
    
    logger.info(f"Expanded {len(recommendations_df)} ChEBI entries to {len(expanded_df)} KEGG mappings")
    logger.info(f"Found {len(filtered_df)} valid KEGG mappings")
    logger.info(f"Selected {len(high_score_recommendations)} high-score recommendations")
    
    return filtered_df, high_score_recommendations


def extract_reaction_participants(
    model_info: Dict,
    recommendations_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Extract participant names for each reaction."""
    reaction_participants = {}
    
    for reaction in model_info['reactions']:
        reaction_id = reaction.split(':')[0].strip()
        participant_str = extract_classifications(reaction.split(':')[1].strip(), 'definition')
        
        participant_names = []
        for participant in participant_str.split('; '):
            matching_rows = recommendations_df[recommendations_df['id'] == participant]
            if not matching_rows.empty and 'annotation_label' in matching_rows.columns:
                participant_names.append(matching_rows['annotation_label'].values[0])
        
        reaction_participants[reaction_id] = participant_names
    
    return reaction_participants


def run_kegg_annotation_workflow(
    model_file: str,
    recommendations_df: pd.DataFrame,
    kegg_features_file: str,
    entity_type: str = 'reaction',
    database: str = 'kegg',
    llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    cofactor_config: Optional[CofactorConfig] = None,
    convergence_config: Optional[ConvergenceConfig] = None,
    matching_config: Optional[MatchingConfig] = None
) -> None:
    """Run the complete KEGG annotation workflow."""
    # Initialize configurations
    if cofactor_config is None:
        cofactor_config = CofactorConfig()
    if convergence_config is None:
        convergence_config = ConvergenceConfig()
    if matching_config is None:
        matching_config = MatchingConfig()
    
    # Check environment
    if not check_environment(model_file):
        logger.error("Environment check failed. Please fix the issues and try again.")
        return
    
    logger.info(f"Model file: {model_file}")
    logger.info(f"LLM model: {llm_model}")
    logger.info(f"Analyzing model: {model_file}")
    
    # Step 1: Extract model information
    all_entity_ids = get_all_reaction_ids(model_file)
    model_info = extract_model_info(model_file, all_entity_ids, entity_type)
    
    # Step 2: Map ChEBI IDs to KEGG Compound IDs
    logger.info("Step 2: Map ChEBI IDs to KEGG Compound IDs")
    _, high_score_recommendations = map_chebi_to_kegg(recommendations_df)
    
    logger.info("\nSample of ChEBI to KEGG mapping:")
    if not high_score_recommendations.empty:
        logger.info(high_score_recommendations[['id', 'display_name', 'annotation', 'KEGG_ID', 'match_score']].head())
    
    # Step 3: Begin rule-based matching to identify reactions
    logger.info("Step 3: Begin rule-based matching to identify reactions")
    reactions, _ = extract_reactions_from_sbml(
        model_file,
        list(high_score_recommendations['id'].unique())
    )
    normalized_reactions, match_results, _species_relax_levels = map_reactions_to_kegg_with_relaxation(
        reactions,
        high_score_recommendations,
        spectators=False,
        cofactors_to_ignore=cofactor_config.kegg_ids,
        top_k=None,
    )
    
    # Build recommendation table
    kegg_recommendations_df = _generate_recommendation_table(
        model_file,
        match_results,
        {},
        model_info,
        entity_type,
        database,
        {}
    )
    
    # Normalize match scores
    kegg_recommendations_df['match_score_norm'] = (
        kegg_recommendations_df['match_score'] /
        kegg_recommendations_df.groupby('id')['match_score'].transform('sum')
    )
    
    # Step 4: Extract reaction participants
    reaction_participants = extract_reaction_participants(model_info, recommendations_df)
    
    # Step 5: Load KEGG reaction features and add participant information
    kegg_features = KEGGReactionFeatures.load_from_file(kegg_features_file)
    
    kegg_recommendations_df['participants'] = (
        kegg_recommendations_df['annotation'].apply(kegg_features.get_participants)
    )
    kegg_recommendations_df['participant_ids'] = (
        kegg_recommendations_df['annotation'].apply(kegg_features.get_participant_ids)
    )
    
    # Step 6: Build participant counters
    merged_participants = (
        kegg_recommendations_df
        .groupby("id")["participants"]
        .agg("; ".join)
    )
    
    counters = merged_participants.apply(
        lambda s: Counter(p.strip() for p in s.split(";") if p.strip())
    )
    
    # Step 7: Initialize probabilities and compute likelihoods
    similarity_calc = SimilarityCalculator(matching_config)
    init_probs = init_species_probs_from_dict(reaction_participants, counters, similarity_calc)
    
    likelihood_calc = LikelihoodCalculator(cofactor_config, matching_config, convergence_config)
    scored_df = likelihood_calc.compute_reaction_likelihoods(init_probs, kegg_recommendations_df)
    
    # Step 8: Update participant KEGG likelihoods iteratively until convergence
    updated_participants_df, updated_reactions_df = update_participant_likelihoods(
        high_score_recommendations,
        scored_df,
        reaction_participants,
        model_file=model_file,
        model_info=model_info,
        kegg_features=kegg_features,
        reactions=reactions,
        entity_type=entity_type,
        database=database,
        cofactor_config=cofactor_config,
        convergence_config=convergence_config
    )
    
    logger.info("\nSample of participants with updated likelihoods after convergence:")
    if not updated_participants_df.empty:
        logger.info(updated_participants_df[['id', 'display_name', 'KEGG_ID', 'participant_likelihood']].head())
    
    updated_participants_df.sort_values(by='participant_likelihood', ascending=False, inplace=True)
    scored_df.sort_values(by='likelihood', ascending=False, inplace=True)
    
    logger.info("KEGG annotation workflow completed successfully.")


#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

def main():
    """Main execution function."""
    # Configuration
    model_file = "tests/glycolysis_part1.xml"
    kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
    llm_model = "meta-llama/llama-3.1-8b-instruct"

    # Print header
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)
    
    # Load recommendations
    logger.info("Step 1: Loading chemical species recommendations")
    recommendations_df = pd.read_csv("recommendations_correctedChEBI.csv")
    # recommendations_df = pd.read_csv("recommendations_LLMChEBI.csv")

    # Run the workflow
    run_kegg_annotation_workflow(
        model_file=model_file,
        recommendations_df=recommendations_df,
        kegg_features_file=kegg_features_file,
        llm_model=llm_model
    )


if __name__ == "__main__":
    main()
