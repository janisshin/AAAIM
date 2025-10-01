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

import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    annotate_model, 
    curate_model, 
    get_available_databases, 
    database_search,
    normalize_reactions,
    load_chebi2kegg_dict, 
    load_kegg_reaction_features_dict
)
from core.update_model import update_annotation
from core.model_info import (
    extract_reactions_from_sbml, 
    extract_model_info,
    get_all_reaction_ids
)
from core.annotation_workflow import map_reactions_to_kegg, _generate_recommendation_table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

# Common cofactors to ignore in reaction matching
COFACTORS_TO_IGNORE = {
    'C00001',  # H2O
    'C00080',  # H+
    'C00007',  # O2
    'C00027',  # H2O2
    'C00009',  # Phosphate
    'C00013',  # Diphosphate
    'C00008',  # ADP
    'C00002',  # ATP
    'C00003',  # NAD+
    'C00004',  # NADH
    'C00005',  # NADPH
    'C00006',  # NADP+
}


#------------------------------------------------------------------------------
# Text Processing Functions
#------------------------------------------------------------------------------

def extract_classifications(raw_text: str, classification: str) -> str:
    """
    Extract and clean text from different classification formats.
    
    Parameters
    ----------
    raw_text : str
        The raw text to process
    classification : str
        The type of classification to extract ('brite', 'orthology', or 'definition')
        
    Returns
    -------
    str
        Cleaned and formatted text as a semicolon-separated string
    """
    lines = raw_text.splitlines()
    clean_lines = []

    if classification == 'brite':
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # Skip lines with [BR:...] tags
            if "[BR:" in stripped:
                continue
            # Skip EC leaf numbers (pure numbers like 2.2.1.6)
            if re.fullmatch(r"(\d+\.)+\d+", stripped):
                continue
            # Skip lines that start with an R number (reaction ID)
            if re.match(r"R\d{5}", stripped):
                continue
            
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1:
                clean_lines.append(parts[1].strip())
            else:
                clean_lines.append(stripped)
    
    elif classification == 'orthology':
        for line in lines:
            # Split once on spaces to remove the Kxxxxx ID
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                # Remove the EC info if present
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)

    elif classification == 'definition':
        parts = []
        buf = ""
        paren_level = 0  # Track nested parentheses

        i = 0
        while i < len(raw_text):
            c = raw_text[i]

            # Track parentheses
            if c == '(':
                paren_level += 1
            elif c == ')':
                paren_level -= 1

            # Split points: + outside parentheses or <=>
            if c == '+' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
            elif raw_text[i:i+3] == '<=>' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 2  # skip the next two chars of <=>
            elif raw_text[i:i+2] == '->' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 1  
            else:
                buf += c

            i += 1

        # Add remaining buffer
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
    """
    Normalize a probability dictionary so values sum to 1.
    
    Parameters
    ----------
    prob_dict : Dict[str, float]
        Dictionary mapping keys to probability values
        
    Returns
    -------
    Dict[str, float]
        Normalized probability dictionary
    """
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict


def is_plausible_match(query_species: str, cand_species: str, threshold: int = 80) -> bool:
    """
    Determine if a candidate species could match the query species based on name similarity.
    
    Parameters
    ----------
    query_species : str
        The query species name
    cand_species : str
        The candidate species name
    threshold : int, optional
        Minimum similarity score (0-100) to consider a match, by default 80
        
    Returns
    -------
    bool
        True if the match is plausible, False otherwise
    """
    max_score = fuzz.partial_ratio(query_species.lower(), cand_species.lower())
    return max_score >= threshold


def update_species_probs(
    query_species: str, 
    candidate_reactions: List, 
    candidate_probs: Dict
) -> Dict[str, float]:
    """
    Update the species match probabilities for a single query species
    based on the candidate reaction probabilities.
    
    Parameters
    ----------
    query_species : str
        The query species name
    candidate_reactions : List
        List of candidate reactions
    candidate_probs : Dict
        Dictionary mapping candidate reactions to probabilities
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping candidate species to updated probabilities
    """
    updated_probs = {}

    # Loop over each candidate reaction
    for candidate in candidate_reactions:
        prob_candidate = candidate_probs.get(candidate, 0.0)
        
        # Loop over each species in the candidate reaction
        for cand_species in candidate.participants:
            # Check if this candidate species could match the query species
            if is_plausible_match(query_species, cand_species):
                # Accumulate probability weighted by candidate reaction probability
                if cand_species not in updated_probs:
                    updated_probs[cand_species] = 0.0
                updated_probs[cand_species] += prob_candidate

    # Normalize the probabilities so they sum to 1
    total = sum(updated_probs.values())
    if total > 0:
        for species in updated_probs:
            updated_probs[species] /= total

    return updated_probs


def choose_best_annotation(species_probs: Dict[str, float]) -> Optional[str]:
    """
    Choose the best annotation for a query species based on
    the current species match probabilities.
    
    Parameters
    ----------
    species_probs : Dict[str, float]
        Mapping from candidate species to probabilities
        
    Returns
    -------
    Optional[str]
        The candidate species with the highest probability.
        Returns None if no candidates are available.
    """
    if not species_probs:
        return None  # no plausible matches
    
    # Find the candidate with the maximum probability
    best_species = max(species_probs, key=species_probs.get)
    return best_species


def has_converged(
    updated_annotations: Dict[str, str], 
    previous_annotations: Dict[str, str]
) -> bool:
    """
    Check if the EM algorithm has converged for a query reaction.
    
    Convergence occurs when the annotations of all species do not change
    from the previous iteration.
    
    Parameters
    ----------
    updated_annotations : Dict[str, str]
        Mapping from query species to current annotation
    previous_annotations : Dict[str, str]
        Mapping from query species to previous annotation
        
    Returns
    -------
    bool
        True if converged, False otherwise
    """
    # If there were no previous annotations, we haven't converged yet
    if not previous_annotations:
        return False
    
    # Compare updated vs previous for all species
    for species, new_annotation in updated_annotations.items():
        old_annotation = previous_annotations.get(species)
        if new_annotation != old_annotation:
            return False  # At least one species changed, not converged
    
    return True  # All species unchanged, converged


def init_species_probs_from_dict(
    reaction_participants: Dict[str, List[str]], 
    counters: pd.Series
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Initialize species match probabilities for each reaction.

    Parameters
    ----------
    reaction_participants : Dict[str, List[str]]
        Mapping from reaction_id to list of query participants
    counters : pd.Series
        Series mapping reaction_id to Counter of candidate species

    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary: reaction_id -> {query_species: {candidate_species: prob}}
    """
    species_match_probs = {}

    for rxn_id, query_species_list in reaction_participants.items():
        if rxn_id not in counters:   # skip if missing
            continue  

        candidate_counter = counters[rxn_id]
        species_probs_for_rxn = {}

        for query_species in query_species_list:
            # keep only plausible matches
            plausible = {
                cand: count for cand, count in candidate_counter.items()
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


#------------------------------------------------------------------------------
# KEGG Data Processing Functions
#------------------------------------------------------------------------------

def get_participants(annotation: str) -> str:
    """
    Extract participant names from a KEGG reaction annotation.
    
    Parameters
    ----------
    annotation : str
        KEGG reaction annotation
        
    Returns
    -------
    str
        Semicolon-separated list of participant names
    """
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("DEFINITION", "")
    return extract_classifications(definition, 'definition')


def get_participant_ids(annotation: str) -> str:
    """
    Extract participant IDs from a KEGG reaction annotation.
    
    Parameters
    ----------
    annotation : str
        KEGG reaction annotation
        
    Returns
    -------
    str
        Semicolon-separated list of participant IDs
    """
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("EQUATION", "")
    return extract_classifications(definition, 'definition')


def compute_reaction_likelihoods(
    init_probs: Dict[str, Dict[str, Dict[str, float]]], 
    kegg_recommendations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute likelihood scores for each candidate reaction given initial species probabilities.

    Parameters
    ----------
    init_probs : Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary: reaction_id -> {query_species: {candidate_species: prob}}
    kegg_recommendations_df : pd.DataFrame
        DataFrame with columns: ['id', 'annotation', 'participants', 'participant_ids']

    Returns
    -------
    pd.DataFrame
        Same as input, with an extra column 'likelihood' containing the computed likelihood score
    """
    def fuzzy_jaccard(set_a: Set[str], set_b: Set[str], threshold: int = 70) -> float:
        """
        Compute fuzzy Jaccard similarity between two sets of strings.
        
        Parameters
        ----------
        set_a : Set[str]
            First set of strings
        set_b : Set[str]
            Second set of strings
        threshold : int, optional
            Minimum similarity to count as a match (0-100), by default 70
            
        Returns
        -------
        float
            Fuzzy Jaccard similarity score (0-1)
        """
        overlap = 0
        for a in set_a:
            best = max((fuzz.ratio(a, b) / 100 for b in set_b), default=0)
            if best * 100 >= threshold:
                overlap += best  # fractional overlap
        denom = len(set_a) + len(set_b) - overlap
        return overlap / denom if denom > 0 else 0

    likelihoods = []
    
    for _, row in kegg_recommendations_df.iterrows():
        rxn_id = row['id']
        candidate_participants = set(row['participants'].split("; "))
        filtered_candidate_participants = candidate_participants.copy()

        # --- 1. Compute species-level match probability ---
        prob_product = 1.0
        if rxn_id in init_probs:
            query_participants = set([i[0] for i in init_probs[rxn_id].items()])
            if not (query_participants & COFACTORS_TO_IGNORE):
                filtered_candidate_participants -= COFACTORS_TO_IGNORE
            species_probs = init_probs[rxn_id]
            for _, cand_dict in species_probs.items():
                match_probs = [cand_dict[cand] for cand in filtered_candidate_participants if cand in cand_dict]
                prob_product *= max(match_probs) if match_probs else 1e-6
        else:
            prob_product = 1e-6

        # --- 2. Compute Jaccard penalty for extra participants ---
        jaccard_score = fuzzy_jaccard(query_participants, filtered_candidate_participants)

        # --- 3. Combine both scores ---
        likelihood = prob_product * jaccard_score
        likelihoods.append(likelihood)

    # Create a copy to avoid modifying the original
    result_df = kegg_recommendations_df.copy()
    result_df['likelihood'] = likelihoods

    # Rescale so each group of candidate reaction likelihoods sums to 1
    group_sums = result_df.groupby('id')['likelihood'].transform('sum')
    result_df['likelihood'] = result_df['likelihood'] / group_sums
    
    return result_df


def update_participant_kegg_likelihoods_single_iteration(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform a single iteration of updating each candidate participant's KEGG ID
    with their likelihood based on whether that KEGG ID shows up in the reaction likelihood dataframe.
    
    Parameters
    ----------
    participant_df : pd.DataFrame
        DataFrame containing participant information with columns including 'id' and 'KEGG_ID'
    reaction_likelihood_df : pd.DataFrame
        DataFrame containing reaction likelihoods with columns including 'annotation',
        'participant_ids', and 'likelihood'
        
    Returns
    -------
    pd.DataFrame
        Updated participant DataFrame with added 'participant_likelihood' column
    """
    # Create a copy of the input dataframe to avoid modifying the original
    updated_df = participant_df.copy()
    
    # Initialize a new column for participant likelihoods if it doesn't exist
    if 'participant_likelihood' not in updated_df.columns:
        updated_df['participant_likelihood'] = 0.0
    else:
        # Reset likelihoods for new iteration
        updated_df['participant_likelihood'] = 0.0
    
    # Create a mapping from KEGG_ID to participant rows
    kegg_id_to_indices = {}
    for idx, row in updated_df.iterrows():
        if pd.notna(row.get('KEGG_ID')) and row['KEGG_ID'] != '':
            kegg_id = row['KEGG_ID']#[0]
            if kegg_id not in kegg_id_to_indices:
                kegg_id_to_indices[kegg_id] = []
            kegg_id_to_indices[kegg_id].append(idx)
    
    # Iterate through reaction likelihoods
    for _, reaction_row in reaction_likelihood_df.iterrows():
        # Get the reaction likelihood
        reaction_likelihood = reaction_row['likelihood']
        
        # Get participant IDs for this reaction
        if pd.notna(reaction_row.get('participant_ids')) and reaction_row['participant_ids'] != '':
            participant_ids = set(p.strip() for p in reaction_row['participant_ids'].split(';') if p.strip())
            
            # Update likelihood for each participant that appears in this reaction
            for participant_id in participant_ids:
                if participant_id in kegg_id_to_indices:
                    for idx in kegg_id_to_indices[participant_id]:
                        # Accumulate likelihood (we'll take the max later)
                        current_likelihood = updated_df.at[idx, 'participant_likelihood']
                        updated_df.at[idx, 'participant_likelihood'] = max(current_likelihood, reaction_likelihood)
    
    # Normalize likelihoods per participant group (same id)
    for participant_id in updated_df['id'].unique():
        mask = updated_df['id'] == participant_id
        group_sum = updated_df.loc[mask, 'participant_likelihood'].sum()
        if group_sum > 0:
            updated_df.loc[mask, 'participant_likelihood'] = updated_df.loc[mask, 'participant_likelihood'] / group_sum
    
    return updated_df


def update_participant_kegg_likelihoods(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame,
    max_iterations: int = 100,
    convergence_threshold: float = 0.001,
    convergence_count: int = 3
) -> pd.DataFrame:
    """
    Iteratively update each candidate participant's KEGG ID with their likelihood
    based on whether that KEGG ID shows up in the reaction likelihood dataframe.
    Continues until convergence criteria are met.
    
    Parameters
    ----------
    participant_df : pd.DataFrame
        DataFrame containing participant information with columns including 'id' and 'KEGG_ID'
    reaction_likelihood_df : pd.DataFrame
        DataFrame containing reaction likelihoods with columns including 'annotation',
        'participant_ids', and 'likelihood'
    max_iterations : int, optional
        Maximum number of iterations to perform, by default 100
    convergence_threshold : float, optional
        Threshold for considering scores stable (to 3 decimal places), by default 0.001
    convergence_count : int, optional
        Number of consecutive stable iterations required for convergence, by default 3
        
    Returns
    -------
    pd.DataFrame
        Updated participant DataFrame with added 'participant_likelihood' column
    """
    current_df = participant_df.copy()
    
    # Keep track of previous scores for convergence check
    previous_scores = []
    stable_iterations = 0
    
    logger.info("Starting iterative participant likelihood updates")
    
    for iteration in range(1, max_iterations + 1):
        # Perform a single iteration
        updated_df = update_participant_kegg_likelihoods_single_iteration(
            current_df, reaction_likelihood_df
        )
        
        # Calculate the maximum change in likelihood scores
        if 'participant_likelihood' in current_df.columns:
            # Merge dataframes to compare scores
            comparison_df = current_df.merge(
                updated_df[['id', 'KEGG_ID', 'participant_likelihood']],
                on=['id', 'KEGG_ID'],
                suffixes=('_prev', '')
            )
            
            # Calculate maximum absolute difference
            max_diff = (
                comparison_df['participant_likelihood'] -
                comparison_df['participant_likelihood_prev']
            ).abs().max()
            
            # Round to 3 decimal places for comparison
            max_diff_rounded = round(max_diff, 3)
            
            logger.info(f"Iteration {iteration}: Maximum score change = {max_diff_rounded:.6f}")
            
            # Check for convergence
            if max_diff_rounded <= convergence_threshold:
                stable_iterations += 1
                logger.info(f"Stable iteration {stable_iterations}/{convergence_count}")
                
                if stable_iterations >= convergence_count:
                    logger.info(f"Convergence achieved after {iteration} iterations")
                    break
            else:
                # Reset counter if scores changed significantly
                stable_iterations = 0
        
        # Store current scores for next iteration
        previous_scores.append(updated_df['participant_likelihood'].copy())
        current_df = updated_df
        
        # If we've reached max iterations without convergence
        if iteration == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached without convergence. "
                f"Last maximum change: {max_diff_rounded:.6f}"
            )
    
    return current_df


#------------------------------------------------------------------------------
# Main Workflow Functions
#------------------------------------------------------------------------------

def check_environment(model_file: str) -> bool:
    """
    Check if the environment is properly set up for running the workflow.
    
    Parameters
    ----------
    model_file : str
        Path to the SBML model file
        
    Returns
    -------
    bool
        True if environment is properly set up, False otherwise
    """
    # Check if required databases are available
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
    
    # Check if model file exists
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        logger.error("Please provide a valid SBML model file.")
        all_ok = False
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API keys found in environment.")
        logger.warning("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
    
    return all_ok


def load_kegg_reaction_data(data_path: str) -> Dict:
    """
    Load KEGG reaction features from the compressed file.
    
    Parameters
    ----------
    data_path : str
        Path to the KEGG reaction features file
        
    Returns
    -------
    Dict
        Dictionary containing KEGG reaction features
    """
    try:
        with lzma.open(data_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, lzma.LZMAError) as e:
        logger.error(f"Error loading KEGG reaction features: {e}")
        return {}


def map_chebi_to_kegg(recommendations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map ChEBI IDs to KEGG Compound IDs and filter for high-scoring matches.
    If a ChEBI ID maps to multiple KEGG IDs, duplicate the row for each KEGG ID.
    
    Parameters
    ----------
    recommendations_df : pd.DataFrame
        DataFrame containing ChEBI recommendations
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing (filtered_df, high_score_recommendations)
    """
    # Load ChEBI to KEGG mapping
    chebi_to_kegg_map = load_chebi2kegg_dict()
    
    # Create a new DataFrame to store expanded rows
    expanded_rows = []
    
    # Process each row in the recommendations DataFrame
    if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            chebi_id = row['annotation']
            kegg_ids = chebi_to_kegg_map.get(chebi_id, [])
            
            # If kegg_ids is not a list, convert it to a list
            if not isinstance(kegg_ids, list):
                kegg_ids = [kegg_ids] if kegg_ids else []
            
            # If no KEGG IDs found, add the original row with empty KEGG_ID
            if not kegg_ids:
                row_copy = row.copy()
                row_copy['KEGG_ID'] = ""
                expanded_rows.append(row_copy)
            else:
                # Create a duplicate row for each KEGG ID
                for kegg_id in kegg_ids:
                    if kegg_id:  # Only add if KEGG ID is not empty
                        row_copy = row.copy()
                        row_copy['KEGG_ID'] = kegg_id
                        expanded_rows.append(row_copy)
        
        # Create a new DataFrame from the expanded rows
        expanded_df = pd.DataFrame(expanded_rows)
        
        # If expanded_df is empty, return the original DataFrame with empty KEGG_ID column
        if expanded_df.empty:
            recommendations_df['KEGG_ID'] = ""
            return recommendations_df, pd.DataFrame()
    else:
        # If recommendations_df is empty or doesn't have 'annotation' column
        recommendations_df['KEGG_ID'] = ""
        return recommendations_df, pd.DataFrame()
    
    # Filter out rows with empty KEGG_ID
    filtered_df = expanded_df[
        expanded_df['KEGG_ID'].notna() &
        (expanded_df['KEGG_ID'] != '')
    ]
    
    # Keep rows that have the max match_score per id
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


def extract_reaction_participants(model_info: Dict, recommendations_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract reaction participants from model information.
    
    Parameters
    ----------
    model_info : Dict
        Dictionary containing model information
    recommendations_df : pd.DataFrame
        DataFrame containing recommendations
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping reaction IDs to lists of participant names
    """
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
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"
) -> None:
    """
    Run the KEGG reaction annotation workflow.
    
    Parameters
    ----------
    model_file : str
        Path to the SBML model file
    recommendations_df : str
        Path to the recommendations CSV file
    kegg_features_file : str
        Path to the KEGG reaction features file
    entity_type : str, optional
        Type of entity to annotate, by default 'reaction'
    database : str, optional
        Database to use for annotation, by default 'kegg'
    llm_model : str, optional
        LLM model to use, by default "meta-llama/llama-3.1-8b-instruct"
    """
    # Check environment and prerequisites
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
    logger.info(high_score_recommendations[['id', 'display_name', 'annotation', 'KEGG_ID', 'match_score']].head())
    
    # Step 3: Begin rule-based matching to identify reactions
    logger.info("Step 3: Begin rule-based matching to identify reactions")
    reactions, _ = extract_reactions_from_sbml(model_file, list(high_score_recommendations['id'].unique()))
    normalized_reactions = map_reactions_to_kegg(
        reactions, 
        high_score_recommendations[['id', 'KEGG_ID']], 
        spectators=False
    )
    
    # Get KEGG recommendations
    match_results = database_search._get_kegg_recommendations_rulebased(
        normalized_reactions, 
        cofactors_to_ignore=COFACTORS_TO_IGNORE,
        spectators=False
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
    global kegg_reaction_features
    kegg_reaction_features = load_kegg_reaction_data(kegg_features_file)
    
    # Create new columns in the dataframe
    kegg_recommendations_df['participants'] = kegg_recommendations_df['annotation'].apply(get_participants)
    kegg_recommendations_df['participant_ids'] = kegg_recommendations_df['annotation'].apply(get_participant_ids)
    
    # Step 6: Build participant counters
    merged_participants = (
        kegg_recommendations_df
        .groupby("id")["participants"]
        .agg("; ".join)  # concatenate strings with "; "
    )
    
    # Build a set of counters to account for all the species in the list
    counters = merged_participants.apply(
        lambda s: Counter(p.strip() for p in s.split(";") if p.strip())
    )
    
    # Step 7: Initialize probabilities and compute likelihoods
    init_probs = init_species_probs_from_dict(reaction_participants, counters)
    scored_df = compute_reaction_likelihoods(init_probs, kegg_recommendations_df)
    
    # Step 8: Update participant KEGG likelihoods iteratively until convergence
    updated_participants_df = update_participant_kegg_likelihoods(
        high_score_recommendations,
        scored_df,
        max_iterations=50,
        convergence_threshold=0.001,
        convergence_count=3
    )
    
    logger.info("\nSample of participants with updated likelihoods after convergence:")
    logger.info(updated_participants_df[['id', 'display_name', 'KEGG_ID', 'participant_likelihood']].head())
    
    updated_participants_df.sort_values(by='participant_likelihood', ascending=False, inplace=True)
    scored_df.sort_values(by='likelihood', ascending=False, inplace=True)

    # Optional: Save results to file
    updated_participants_df.to_csv(f"{os.path.splitext(model_file)[0]}_participant_likelihoods.csv", index=False)
    scored_df.to_csv(f"{os.path.splitext(model_file)[0]}_reaction_likelihoods.csv", index=False)
    
    logger.info("KEGG annotation workflow completed successfully.")

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

def main():
    """Main function to run the KEGG reaction annotation workflow."""
    # Configuration
    model_file = "tests/glycolysis_part1.xml"
    kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
    llm_model = "meta-llama/llama-3.1-8b-instruct"
    top_k = 10
    # recommendations_file = "recommendations_correctedChEBI.csv"

    # Print header
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)
    
    # first annotate model using ChEBI
    print("Step 1: Identifying the chemical species")
    try:    
        recommendations_df, metrics = annotate_model(
            model_file=model_file,
            llm_model=llm_model,
            entity_type="chemical",
            database="chebi",
            method="rag",
            top_k=top_k,
        )
        # Display annotation results
        if not recommendations_df.empty:
            print("Annotation Results:")
            print(f"Total entities in model: {metrics['total_entities']}")
            print(f"Entities with predictions: {metrics['entities_with_predictions']}")
            print(f"Annotation rate: {metrics['annotation_rate']:.1%}")
            
            if not pd.isna(metrics['accuracy']):
                print(f"Accuracy (where existing annotations available): {metrics['accuracy']:.1%}")
            else:
                print("Accuracy: N/A (no existing annotations to compare against)")
            
            print(f"Total time: {metrics['total_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Annotation Recommendations:")
            sample_df = recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']]# .head(5)
            print(sample_df.to_string(index=False))
            print()
            
            # Save results
            file_name = model_file.split('.')[0]
            output_file = f"{file_name}_initial_chemical_recommendations.csv"
            recommendations_df.to_csv(output_file, index=False)
            print(f"Full annotation results saved to: {output_file}")
            
        else:
            print("No annotation recommendations generated.")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

    # Run the workflow
    run_kegg_annotation_workflow(
        model_file=model_file,
        recommendations_df=recommendations_df,
        kegg_features_file=kegg_features_file,
        llm_model=llm_model
    )


if __name__ == "__main__":
    main()
