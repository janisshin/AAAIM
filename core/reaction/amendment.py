"""KEGG reaction amendment: likelihoods, convergence, and iterative participant updates."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ..annotation_workflow import _generate_recommendation_table
from .kegg_features import KEGGReactionFeatures
from ..model_info import extract_reactions_from_sbml
from .amendment_config import CofactorConfig, ConvergenceConfig, MatchingConfig
from .matching import map_reactions_to_kegg_with_relaxation
from .scoring import (
    SimilarityCalculator,
    TextNormalizer,
    _species_ids_from_reaction_string,
    compute_rscore,
    softmax_normalize,
)
from .utils import map_chebi_to_kegg

logger = logging.getLogger(__name__)


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
    
    # Build mapping: (reaction_id, participant_id) -> {kegg_id -> [indices in dataframe]}
    # NOTE: `reaction_id` is expected to be present in `participant_df` (added by update_participant_likelihoods()).
    participant_kegg_indices: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for idx, row in updated_participants_df.iterrows():
        if pd.notna(row.get('KEGG_ID')) and row['KEGG_ID'] != '':
            # If reaction_id is missing, we cannot assign this participant to a specific query reaction.
            if 'reaction_id' not in updated_participants_df.columns:
                continue
            if pd.isna(row.get('reaction_id')) or row['reaction_id'] == '':
                continue
            reaction_id = str(row['reaction_id'])
            participant_id = str(row['id'])
            kegg_id = row['KEGG_ID']
            
            key = (reaction_id, participant_id)
            if key not in participant_kegg_indices:
                participant_kegg_indices[key] = {}
            if kegg_id not in participant_kegg_indices[key]:
                participant_kegg_indices[key][kegg_id] = []
            
            participant_kegg_indices[key][kegg_id].append(idx)
    
    # Build mapping: query_reaction_id -> {ref_reaction_annotation -> probability}
    reaction_probs = {}
    for _, rxn_row in reaction_likelihood_df.iterrows():
        query_rxn_id = rxn_row['id']
        ref_annotation = rxn_row['annotation']
        prob = rxn_row[prob_column]
        
        if query_rxn_id not in reaction_probs:
            reaction_probs[query_rxn_id] = {}
        reaction_probs[query_rxn_id][ref_annotation] = prob
    
    # For each query participant (scoped to a reaction), compute weighted contributions from all reactions
    for (query_rxn_id, _participant_id), kegg_dict in participant_kegg_indices.items():
        
        # Get all candidate reference reactions for this query reaction
        if query_rxn_id not in reaction_probs:
            logger.debug(f"No reaction probabilities found for query reaction {query_rxn_id}")
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


def _assign_reaction_id_rows_to_participants(
    participant_df: pd.DataFrame,
    species_to_reactions: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Expand participant candidates into long-form rows with explicit `reaction_id`.

    If a species participates in multiple reactions, the input row is duplicated per reaction.
    """
    if participant_df.empty:
        return participant_df
    if "reaction_id" in participant_df.columns:
        return participant_df

    out_rows = []
    for _, row in participant_df.iterrows():
        participant_id = row.get("id")
        candidate_reactions = species_to_reactions.get(str(participant_id), set())

        if not candidate_reactions:
            row_copy = row.copy()
            row_copy["reaction_id"] = pd.NA
            out_rows.append(row_copy)
            continue

        for rid in candidate_reactions:
            row_copy = row.copy()
            row_copy["reaction_id"] = rid
            out_rows.append(row_copy)

    return pd.DataFrame(out_rows)


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
    current_participant_ids: Set[str],
    convergence_config: ConvergenceConfig
) -> Set[str]:
    """
    Discover new participant IDs from high-likelihood reactions.
    
    Args:
        reaction_likelihood_df: DataFrame with reaction likelihoods
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
    convergence_config: ConvergenceConfig
) -> pd.DataFrame:
    """
    Suggest new KEGG ID candidates for existing participants based on
    co-occurrence in high-likelihood reactions.
    
    Args:
        reaction_likelihood_df: DataFrame with reaction likelihoods
        current_participants_df: Current participant DataFrame with KEGG IDs
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
                    new_row['annotation'] = ""
                    new_row['KEGG_ID'] = novel_id
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
    model_file: str,
    model_info: Dict,
    kegg_features: KEGGReactionFeatures,
    reactions: List,
    reaction_ids: List[str],
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

    # Add an explicit `reaction_id` column so later steps do not rely on brittle string parsing.
    # This makes the EM updates reaction-scoped.
    if "reaction_id" not in current_participants_df.columns:
        # Build reaction_id -> set(species_id) from the extracted reaction strings.
        # IMPORTANT: `reactions` is aligned with `reaction_ids` by construction in callers.
        reaction_to_participants: Dict[str, Set[str]] = {}
        for rid, rxn_str in zip(reaction_ids, reactions):
            reaction_to_participants[str(rid)] = _species_ids_from_reaction_string(rxn_str)
        species_to_reactions: Dict[str, Set[str]] = {}
        for rid, parts in reaction_to_participants.items():
            for pid in parts:
                species_to_reactions.setdefault(pid, set()).add(rid)
        current_participants_df = _assign_reaction_id_rows_to_participants(
            current_participants_df,
            species_to_reactions=species_to_reactions,
        )
    
    # Track rScores for convergence checking
    prev_rscores = None
    
    logger.info("Starting EM-style iterative participant likelihood updates")
    # Reuse calculators/config objects across iterations (they are stateless for a fixed config).
    likelihood_calc = LikelihoodCalculator(
        cofactor_config,
        MatchingConfig(),
        convergence_config,
    )
    save_iteration_csvs: bool = bool(getattr(convergence_config, "save_iteration_csvs", True))
    
    for iteration in range(1, convergence_config.max_iterations + 1):
        # Log statistics
        if 'participant_likelihood' in current_participants_df.columns:
            if "reaction_id" in current_participants_df.columns:
                likelihood_stats = (
                    current_participants_df
                    .groupby(['reaction_id', 'id'])['participant_likelihood']
                    .agg(['mean', 'min', 'max'])
                )
            else:
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
                convergence_config
            )
            
            if not new_kegg_candidates.empty:
                logger.info(f"Adding {len(new_kegg_candidates)} new KEGG ID candidates to participant pool")
                # Merge new candidates with existing participants
                updated_participants_df_with_kegg = pd.concat([
                    updated_participants_df_with_kegg,
                    new_kegg_candidates
                ]).drop_duplicates(
                    subset=['id', 'KEGG_ID'] + (['reaction_id'] if 'reaction_id' in updated_participants_df_with_kegg.columns else [] )
                ).reset_index(drop=True)
                
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
                    ]).drop_duplicates(
                        subset=['id', 'KEGG_ID'] + (['reaction_id'] if 'reaction_id' in high_score_recommendations.columns else [] )
                    ).reset_index(drop=True)
                    
                    logger.info(f"Added {len(valid_new_candidates)} valid candidates to high_score_recommendations")
        
        # Discover new participant IDs from high-likelihood reactions (if enabled)
        if iteration > 1 and convergence_config.enable_participant_discovery:
            new_participant_ids = discover_new_participants(
                reaction_likelihood_df,
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
        logger.info(f"Iteration {iteration}: Re-extracting reactions with expanded participant list")
        reactions, _ = extract_reactions_from_sbml(
            model_file,
            expanded_participant_ids
        )
        
        # Map reactions to KEGG with updated participant set
        logger.info(f"Iteration {iteration}: Mapping reactions to KEGG")
        normalized_reactions, match_results, _species_relax_levels = map_reactions_to_kegg_with_relaxation(
            reactions,
            reaction_ids,
            high_score_recommendations,
            spectators=False,
            cofactors_to_ignore=cofactor_config.kegg_ids,
            top_k=None,
        )

        allowed_reaction_types = {"mappable", "ambiguous_mapping"}
        match_results = [
            rec
            for rec in match_results
            if str((getattr(rec, "metadata", None) or {}).get("reaction_type", "mappable"))
            in allowed_reaction_types
        ]
        
        logger.info(f"Iteration {iteration}: Generating recommendation table")
        updated_kegg_recommendations_df = _generate_recommendation_table(
            model_file,
            match_results,
            {},
            model_info,
            entity_type,
            database,
            {}
        )
    
        updated_kegg_recommendations_df['participants'] = (
            updated_kegg_recommendations_df['annotation'].apply(kegg_features.get_participants)
        )
        updated_kegg_recommendations_df['participant_ids'] = (
            updated_kegg_recommendations_df['annotation'].apply(kegg_features.get_participant_ids)
        )
        
        # Build participant annotations dictionary for rScore computation
        # Format: {reaction_id: {participant_id: kegg_annotation}}
        participant_annotations: Dict[str, Dict[str, str]] = {}
        if "reaction_id" in updated_participants_df_with_kegg.columns:
            # Vectorized: for each (reaction_id, participant_id), keep the KEGG_ID at max participant_likelihood.
            df_ann = updated_participants_df_with_kegg[
                updated_participants_df_with_kegg["KEGG_ID"].notna()
                & (updated_participants_df_with_kegg["KEGG_ID"] != "")
                & updated_participants_df_with_kegg["reaction_id"].notna()
                & (updated_participants_df_with_kegg["reaction_id"] != "")
            ].copy()
            if not df_ann.empty:
                # Ensure consistent string types for keys.
                df_ann["reaction_id"] = df_ann["reaction_id"].astype(str)
                df_ann["id"] = df_ann["id"].astype(str)
                df_ann = df_ann.sort_values("participant_likelihood", ascending=False)
                df_best = df_ann.drop_duplicates(subset=["reaction_id", "id"], keep="first")

                for rid, grp in df_best.groupby("reaction_id", sort=False):
                    participant_annotations[str(rid)] = dict(zip(grp["id"], grp["KEGG_ID"]))
        
        # M-STEP: Compute updated rScores based on current participant annotations
        logger.info(f"Iteration {iteration}: Computing updated rScores (M-step)")
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
        if save_iteration_csvs:
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
                f"Last rScore change metric: {score_change:.6f}"
            )
    
    return updated_participants_df_with_kegg, updated_reaction_likelihood_df
