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
    get_available_databases, 
    database_search,
    load_chebi2kegg_dict, 
)
# from core.update_model import update_annotation
from core.model_info import (
    extract_reactions_from_sbml, 
    extract_model_info,
    get_all_reaction_ids
)
from core.annotation_workflow import map_reactions_to_kegg, _generate_recommendation_table
# from temp_functions import compute_reaction_likelihoods, participant_likelihoods_to_probs
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
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict


def is_plausible_match(query_species: str, cand_species: str, threshold: int = 80) -> bool:
    max_score = fuzz.partial_ratio(query_species.lower(), cand_species.lower())
    return max_score >= threshold


def update_species_probs(
    query_species: str, 
    candidate_reactions: List, 
    candidate_probs: Dict
) -> Dict[str, float]:
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
    if not species_probs:
        return None  # no plausible matches
    
    # Find the candidate with the maximum probability
    best_species = max(species_probs, key=species_probs.get)
    return best_species


def has_converged(
    updated_annotations: Dict[str, str], 
    previous_annotations: Dict[str, str]
) -> bool:
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
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("DEFINITION", "")
    return extract_classifications(definition, 'definition')


def get_participant_ids(annotation: str) -> str:
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("EQUATION", "")
    return extract_classifications(definition, 'definition')


def participant_likelihoods_to_probs(
    participant_df: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, float]]]:
    probs = {}
    
    # Group by reaction ID to process each reaction's participants
    for rxn_id in participant_df['id'].unique():
        # Convert reaction ID to the format 'J1', 'J2', etc.
        numeric_id = ''.join(filter(str.isdigit, rxn_id))
        if numeric_id:
            formatted_rxn_id = f'J{numeric_id}'
            rxn_mask = participant_df['id'] == rxn_id
            rxn_participants = participant_df[rxn_mask]
            
            # Build probability dictionary for this reaction
            if formatted_rxn_id not in probs:
                probs[formatted_rxn_id] = {}
            
            # Process each participant
            for _, row in rxn_participants.iterrows():
                if pd.notna(row['KEGG_ID']) and pd.notna(row['participant_likelihood']):
                    query_species = row['annotation_label']
                    kegg_id = row['KEGG_ID']
                    
                    # Initialize nested structure if needed
                    if formatted_rxn_id not in probs:
                        probs[formatted_rxn_id] = {}
                    if query_species not in probs[formatted_rxn_id]:
                        probs[formatted_rxn_id][query_species] = {}
                    
                    # Store the likelihood as probability
                    probs[formatted_rxn_id][query_species][kegg_id] = row['participant_likelihood']
    
    return probs

def compute_reaction_likelihoods(
    init_probs: Dict[str, Dict[str, Dict[str, float]]], 
    kegg_recommendations_df: pd.DataFrame,
    prev_likelihoods: pd.DataFrame = None,
    iteration: int = 0
) -> pd.DataFrame:
    print("Starting likelihood computation...")
    print(f"Total reactions to process: {len(kegg_recommendations_df)}")
    print(f"Total reactions in init_probs: {len(init_probs)}")
    print("\nDumping init_probs contents:")
    for rid, data in init_probs.items():
        print(f"\nReaction {rid}:")
        for species, probs in data.items():
            print(f"  {species} -> {probs}")
    def fuzzy_jaccard(set_a: Set[str], set_b: Set[str], threshold: int = 70) -> float:
        """
        Compute fuzzy Jaccard similarity between two sets of strings.
        """
        if not set_a or not set_b:
            return 0.0
            
        overlap = 0
        for a in set_a:
            best = max((fuzz.ratio(a, b) / 100 for b in set_b), default=0)
            if best * 100 >= threshold:
                overlap += best  # fractional overlap
        
        denom = len(set_a) + len(set_b) - overlap
        return overlap / denom if denom > 0 else 0

    # Create a copy to avoid modifying the original
    result_df = kegg_recommendations_df.copy()
    # Initialize likelihood column
    result_df['likelihood'] = 0.0
    
    # Process each reaction
    for idx, row in result_df.iterrows():
        rxn_id = row['id']
        
        # Skip if participants column is missing or empty
        if pd.isna(row['participants']) or not row['participants']:
            continue
            
        # Get and filter candidate participants
        candidate_participants = set(str(row['participants']).split("; "))
        
        # Convert cofactors to lowercase for case-insensitive comparison
        ## JANISTAG THIS IS A PROBLEM
        filtered_candidate_participants = {
            p for p in candidate_participants 
            if p and not any(c in p for c in ['ATP', 'ADP', 'AMP', 'NAD', 'NADP'])  # Direct check for common cofactors
        }
        
        print(f"\nProcessing reaction {rxn_id}")
        print(f"Candidate participants before filtering: {candidate_participants}")
        print(f"Filtered candidate participants: {filtered_candidate_participants}")

        # --- 1. Compute species-level match probability ---
        prob_product = 1.0
        query_participants = set()
        
        def standardize_name(name):
            """Standardize species names for comparison"""
            name = name.lower()
            name = name.replace('α-', 'alpha-').replace('β-', 'beta-')
            name = name.replace('α', 'alpha-').replace('β', 'beta-')  # Handle cases without hyphen
            name = name.replace('-', ' ')
            return name

        # Get all available query species and their KEGG IDs from init_probs
        query_participants = set()
        kegg_id_to_prob = {}
        
        print("\nAvailable query species and their KEGG IDs:")
        for reaction_name, species_dict in init_probs.items():
            for species_name, kegg_dict in species_dict.items():
                for kegg_id, prob in kegg_dict.items():
                    print(f"{species_name} -> {kegg_id}: {prob}")
                    kegg_id_to_prob[kegg_id] = prob

        # Now check if any of the candidate participants' KEGG IDs match our known IDs
        print("\nProcessing candidate participants:")
        for participant in filtered_candidate_participants:
            print(f"Checking participant: {participant}")
            # Parse KEGG ID from participant string if present
            if "KEGG:" in participant:
                kegg_id = participant.split("KEGG:")[-1].strip()
                if kegg_id in kegg_id_to_prob:
                    prob = kegg_id_to_prob[kegg_id]
                    prob_product *= prob
                    query_participants.add(participant)
                    print(f"  Found matching KEGG ID {kegg_id} with probability {prob}")
            else:
                # Try to match by name if no KEGG ID
                std_participant = standardize_name(participant)
                for reaction_name, species_dict in init_probs.items():
                    for species_name, kegg_dict in species_dict.items():
                        if standardize_name(species_name) == std_participant:
                            prob = max(kegg_dict.values())
                            prob_product *= prob
                            query_participants.add(participant)
                            print(f"  Matched by name to {species_name} with probability {prob}")

        if not query_participants:
            print("  No matches found in init_probs")
            prob_product = 1e-6  # Default low probability for no matches

        print(f"Matched participants: {query_participants}")
        print(f"Final prob_product: {prob_product}")

        # --- 2. Compute Jaccard similarity ---
        filtered_query_participants = {
            p for p in query_participants 
            if p and not any(c in p for c in ['ATP', 'ADP', 'AMP', 'NAD', 'NADP'])
        }
        
        print(f"Filtered query participants: {filtered_query_participants}")
        
        if filtered_query_participants and filtered_candidate_participants:
            jaccard_score = fuzzy_jaccard(
                filtered_query_participants, 
                filtered_candidate_participants
            )
        else:
            print("Warning: Empty participant sets after filtering")
            jaccard_score = 0.0

        # --- 3. Combine scores ---
        new_likelihood = prob_product * jaccard_score
        
        # Blend with previous likelihood if available
        if prev_likelihoods is not None and not prev_likelihoods.empty:
            prev_row = prev_likelihoods[prev_likelihoods['id'] == row['id']]
            if not prev_row.empty:
                prev_likelihood = prev_row['likelihood'].iloc[0]
                # Exponential decay factor - gives more weight to new likelihoods in later iterations
                alpha = min(0.1 + (iteration * 0.1), 0.9)  # Increases from 0.1 to 0.9
                blended_likelihood = (alpha * new_likelihood + (1 - alpha) * prev_likelihood)
                new_likelihood = blended_likelihood
        
        print(f"Final scores for {rxn_id}:")
        print(f"prob_product: {prob_product}")
        print(f"jaccard_score: {jaccard_score}")
        print(f"new_likelihood: {new_likelihood}")
        result_df.at[idx, 'likelihood'] = new_likelihood

    # Normalize likelihoods within each reaction group
    group_sums = result_df.groupby('id')['likelihood'].transform('sum')
    print("\nGroup sums before normalization:")
    for rid in result_df['id'].unique():
        print(f"{rid}: {group_sums[result_df['id'] == rid].iloc[0]}")
    
    mask = group_sums > 0  # Avoid division by zero
    result_df.loc[mask, 'likelihood'] = result_df.loc[mask, 'likelihood'] / group_sums[mask]
    
    return result_df

def update_participant_likelihoods_singleiter(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame
) -> pd.DataFrame:
    # Create a copy of the input dataframe to avoid modifying the original
    updated_participants_df = participant_df.copy()
    
    # Initialize or preserve participant likelihoods
    if 'participant_likelihood' not in updated_participants_df.columns:
        updated_participants_df['participant_likelihood'] = 0.0
    # Don't reset existing likelihoods - we'll update them based on new evidence
    
    # Create a mapping from KEGG_ID to participant rows
    kegg_id_to_indices = {}
    for idx, row in updated_participants_df.iterrows():
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
                        # Blend new evidence with existing likelihood
                        # Using exponential moving average with alpha=0.7 to favor newer evidence
                        alpha = 0.7
                        current_likelihood = updated_participants_df.at[idx, 'participant_likelihood']
                        blended_likelihood = (alpha * reaction_likelihood + (1 - alpha) * current_likelihood)
                        updated_participants_df.at[idx, 'participant_likelihood'] = blended_likelihood
    
    # Normalize likelihoods per participant group (same id)
    for participant_id in updated_participants_df['id'].unique():
        mask = updated_participants_df['id'] == participant_id
        group_sum = updated_participants_df.loc[mask, 'participant_likelihood'].sum()
        if group_sum > 0:
            updated_participants_df.loc[mask, 'participant_likelihood'] = updated_participants_df.loc[mask, 'participant_likelihood'] / group_sum
    
    return updated_participants_df


def update_participant_likelihoods(
    participant_df: pd.DataFrame,
    reaction_likelihood_df: pd.DataFrame,
    reaction_participants,
    model_file: str,
    model_info: Dict,
    entity_type: str = 'reaction',
    database: str = 'kegg',
    max_iterations: int = 100,
    convergence_threshold: float = 0.001,
    convergence_count: int = 3,
    cofactors_to_ignore: Set = None
) -> pd.DataFrame:
    current_participants_df = participant_df.copy()
    
    # Keep track of previous scores for convergence check
    previous_scores = []
    stable_iterations = 0
    
    logger.info("Starting iterative participant likelihood updates")
    
    for iteration in range(1, max_iterations + 1):
        # Log likelihood distribution before update
        logger.info(f"\nIteration {iteration} - Before update:")
        if 'participant_likelihood' in current_participants_df.columns:
            likelihood_stats = current_participants_df.groupby('id')['participant_likelihood'].agg(['mean', 'min', 'max'])
            for idx, row in likelihood_stats.iterrows():
                logger.info(f"ID {idx}: mean={row['mean']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")

        # Perform a single iteration
        updated_participants_df = update_participant_likelihoods_singleiter(
            current_participants_df, reaction_likelihood_df
        )
        
        # Log likelihood distribution after update
        logger.info(f"\nIteration {iteration} - After update:")
        likelihood_stats = updated_participants_df.groupby('id')['participant_likelihood'].agg(['mean', 'min', 'max'])
        for idx, row in likelihood_stats.iterrows():
            logger.info(f"ID {idx}: mean={row['mean']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")
        
        # Map ChEBI IDs to KEGG Compound IDs
        logger.info(f"Iteration {iteration}: Mapping ChEBI IDs to KEGG Compound IDs")
        updated_participants_df_with_kegg, high_score_recommendations = map_chebi_to_kegg(updated_participants_df)
        
        # Map reactions to KEGG
        logger.info(f"Iteration {iteration}: Mapping reactions to KEGG")
        ## reactions, _ = extract_reactions_from_sbml(model_file, list(high_score_recommendations['id'].unique()))
        normalized_reactions = map_reactions_to_kegg(
            reactions,
            high_score_recommendations[['id', 'KEGG_ID']],
            spectators=False
        )
        
        # Get KEGG recommendations using rule-based approach
        logger.info(f"Iteration {iteration}: Getting KEGG recommendations using rule-based approach")
        match_results = database_search._get_kegg_recommendations_rulebased(
            normalized_reactions,
            cofactors_to_ignore=cofactors_to_ignore if cofactors_to_ignore else COFACTORS_TO_IGNORE,
            spectators=False
        )
        
        # Generate updated recommendation table
        logger.info(f"Iteration {iteration}: Generating updated recommendation table")
        updated_kegg_recommendations_df = _generate_recommendation_table(
            model_file,
            match_results,
            {},
            model_info,
            entity_type,
            database,
            {}
        )
        # Create new columns in the dataframe
        updated_kegg_recommendations_df['participants'] = updated_kegg_recommendations_df['annotation'].apply(get_participants)
        updated_kegg_recommendations_df['participant_ids'] = updated_kegg_recommendations_df['annotation'].apply(get_participant_ids)        
        
        # Convert updated participant likelihoods to probabilities for next iteration
        logger.info(f"Iteration {iteration}: Converting participant likelihoods to probabilities")
        updated_probs = participant_likelihoods_to_probs(updated_participants_df_with_kegg)
        
        # Compute updated reaction likelihoods using the new probabilities
        logger.info(f"Iteration {iteration}: Computing updated reaction likelihoods")
        updated_reaction_likelihood_df = compute_reaction_likelihoods(
            updated_probs, 
            updated_kegg_recommendations_df,
            reaction_likelihood_df,  # Pass previous likelihoods
            iteration
        )

        # Calculate the maximum change in likelihood scores
        if 'participant_likelihood' in current_participants_df.columns:
            # Merge dataframes to compare scoress
            comparison_df = current_participants_df.merge(
                updated_participants_df[['id', 'KEGG_ID', 'participant_likelihood']],
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
        
        # Update state for next iteration
        previous_scores.append(updated_participants_df['participant_likelihood'].copy())
        current_participants_df = updated_participants_df_with_kegg  # Update participants
        reaction_likelihood_df = updated_reaction_likelihood_df  # Use new reaction likelihoods in next iteration
        
        # Save iteration results
        updated_participants_df_with_kegg.to_csv(f'participants_likelihood_iter{iteration}.csv', index=False)
        updated_reaction_likelihood_df.to_csv(f'reaction_likelihood_iter{iteration}.csv', index=False)

        # If we've reached max iterations without convergence
        if iteration == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached without convergence. "
                
            ) # f"Last maximum change: {max_diff_rounded:.6f}"

    
    return updated_participants_df_with_kegg, updated_reaction_likelihood_df


#------------------------------------------------------------------------------
# Main Workflow Functions
#------------------------------------------------------------------------------

def check_environment(model_file: str) -> bool:
    
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
    try:
        with lzma.open(data_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, lzma.LZMAError) as e:
        logger.error(f"Error loading KEGG reaction features: {e}")
        return {}


def map_chebi_to_kegg(recommendations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load ChEBI to KEGG mapping
    chebi_to_kegg_map = load_chebi2kegg_dict()
    
    # Create a new DataFrame to store expanded rows
    expanded_rows = []
    
    # Keep track of existing KEGG IDs and likelihoods
    existing_mappings = {}
    if 'KEGG_ID' in recommendations_df.columns and 'participant_likelihood' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            if pd.notna(row['KEGG_ID']) and row['KEGG_ID'] != '':
                key = (row['id'], row['annotation'], row['KEGG_ID'])
                existing_mappings[key] = row['participant_likelihood']
    
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
                        # Preserve existing likelihood if available
                        key = (row['id'], row['annotation'], kegg_id)
                        if key in existing_mappings:
                            row_copy['participant_likelihood'] = existing_mappings[key]
                        expanded_rows.append(row_copy)
        
        # Create a new DataFrame from the expanded rows
        expanded_df = pd.DataFrame(expanded_rows)
        
        # If expanded_df is empty, return the original DataFrame with empty KEGG_ID column
        if expanded_df.empty:
            recommendations_df['KEGG_ID'] = ""
            return recommendations_df, pd.DataFrame()
            
        # Combine existing recommendations with new mappings
        combined_df = pd.concat([recommendations_df, expanded_df]).drop_duplicates(subset=['id', 'annotation', 'KEGG_ID'])
    else:
        # If recommendations_df is empty or doesn't have 'annotation' column
        recommendations_df['KEGG_ID'] = ""
        return recommendations_df, pd.DataFrame()
    
    # Filter out rows with empty KEGG_ID
    filtered_df = combined_df[
        combined_df['KEGG_ID'].notna() &
        (combined_df['KEGG_ID'] != '')
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
    global reactions
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
    updated_participants_df, updated_reactions_df = update_participant_likelihoods(
        high_score_recommendations,
        scored_df,
        reaction_participants,
        model_file=model_file,
        model_info=model_info,
        entity_type=entity_type,
        database=database,
        max_iterations=50,
        convergence_threshold=0.001,
        convergence_count=3,
        cofactors_to_ignore=COFACTORS_TO_IGNORE
    )
    
    logger.info("\nSample of participants with updated likelihoods after convergence:")
    logger.info(updated_participants_df[['id', 'display_name', 'KEGG_ID', 'participant_likelihood']].head())
    
    updated_participants_df.sort_values(by='participant_likelihood', ascending=False, inplace=True)
    scored_df.sort_values(by='likelihood', ascending=False, inplace=True)

    # Optional: Save results to file
    # updated_participants_df.to_csv(f"{os.path.splitext(model_file)[0]}_participant_likelihoods.csv", index=False)
    # scored_df.to_csv(f"{os.path.splitext(model_file)[0]}_reaction_likelihoods.csv", index=False)
    
    logger.info("KEGG annotation workflow completed successfully.")

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

def main():
    # Configuration
    model_file = "tests/glycolysis_part1.xml"
    kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
    llm_model = "meta-llama/llama-3.1-8b-instruct"
    top_k = 10

    # Print header
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)
    
    # first annotate model using ChEBI
    print("Step 1: Identifying the chemical species")
    
    file_name = model_file.split('.')[0]
    # recommendations_df = pd.read_csv("glycolysis_part1.xml_recommendations.csv")
    recommendations_df = pd.read_csv(f"recommendations_correctedChEBI.csv")

    # Run the workflow
    run_kegg_annotation_workflow(
        model_file=model_file,
        recommendations_df=recommendations_df,
        kegg_features_file=kegg_features_file,
        llm_model=llm_model
    )


if __name__ == "__main__":
    main()
