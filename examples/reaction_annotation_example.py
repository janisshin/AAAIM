#!/usr/bin/env python3
"""
AAAIM KEGG Reaction Annotation Example

This script demonstrates how to use AAAIM for reaction annotation using the KEGG reaction database.
"""

import os
import sys
import pandas as pd
from itertools import chain
import lzma
import pickle

from dotenv import load_dotenv
load_dotenv()


from pathlib import Path

# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import annotate_model, curate_model, get_available_databases, database_search
from core import normalize_reactions
from core import load_chebi2kegg_dict, load_kegg_reaction_features_dict
from core.update_model import update_annotation
from core.model_info import extract_reactions_from_sbml, extract_model_info
from core.annotation_workflow import map_reactions_to_kegg, _generate_recommendation_table
from core.model_info import get_all_reaction_ids

# Define common cofactors to ignore in reaction matching


def main():
    """
    Main function to demonstrate AAAIM reaction annotation functionality.
    """
    print("AAAIM KEGG Reaction Annotation Example")
    print("=" * 50)
    
    # Configuration
    model_file = "tests/glycolysis_part1.xml"
    # model_file = "tests/test_models/BIOMD0000000190.xml"
    file_name = model_file.split('.')[0]

    # llm_model = "meta-llama/llama-3.3-70b-instruct:free"  # or "gpt-4o-mini"
    llm_model = "meta-llama/llama-3.1-8b-instruct"
    top_k = 10
    cofactors_to_ignore = {
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
    
    entity_type='reaction'
    database='kegg'


    all_entity_ids = get_all_reaction_ids(model_file)
    model_info = extract_model_info(model_file, all_entity_ids, entity_type)


    # Check if KEGG reaction database is available
    available_dbs = get_available_databases()
    print(f"Available databases: {available_dbs}")
    
    if "chebi" not in available_dbs:
        print("ERROR: ChEBI chemical database not available!")
        print("Please ensure ChEBI reference files are present in data/chebi/")
        return
    
    if "kegg" not in available_dbs:
        print("ERROR: KEGG reaction database not available!")
        print("Please ensure KEGG reference files are present in data/kegg/")
        return
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please provide a valid SBML model file.")
        return
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: No API keys found in environment.")
        print("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
        return

    print(f"Model file: {model_file}")
    print(f"LLM model: {llm_model}")
    print()

    print(f"\nAnalyzing model: {model_file}")
    
    # Example 1: Reaction Annotation Workflow
    print("\nEXAMPLE 1. Reaction Annotation Workflow (for models without reaction annotations)")
    print("-" * 60)

    # this line below should be deleted eventually
    recommendations_df = pd.read_csv('recommendations_correctedChEBI.csv')

    print("\nStep 2: Map ChEBI IDs to KEGG Compound IDs")
    print("-" * 60)  
    
    # Load ChEBI to KEGG mapping
    chebi_to_kegg_map = load_chebi2kegg_dict()
    
    # Add KEGG IDs to chemical recommendations if available
    if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
        # Map ChEBI IDs to KEGG IDs
        recommendations_df['KEGG_ID'] = recommendations_df['annotation'].apply(
            lambda x: chebi_to_kegg_map.get(x, "")
        )

    # Filter out rows with empty KEGG_ID
    filtered_df = recommendations_df[recommendations_df['KEGG_ID'].notna() & (recommendations_df['KEGG_ID'] != '')]

    # Keep rows that have the max match_score per id
    high_score_recommendations = filtered_df[
        filtered_df['match_score'] == filtered_df.groupby('id')['match_score'].transform('max')
    ].reset_index(drop=True)
    
    print("\nSample of ChEBI to KEGG mapping:")
    print(high_score_recommendations[['id', 'display_name', 'annotation', 'KEGG_ID', 'match_score']].head().to_string(index=False))

    print("\nStep 3: Begin rule-based matching to identify reactions")
    reactions, _ = extract_reactions_from_sbml(model_file, list(high_score_recommendations['id'].unique()))
    print(f"Reactions: {reactions}")
    normalized_reactions = map_reactions_to_kegg(reactions, high_score_recommendations[['id', 'KEGG_ID']], spectators=False)
    print(f"Normalized reactions: {normalized_reactions}")

    # Get KEGG recommendations
    match_results = database_search._get_kegg_recommendations_rulebased(
        normalized_reactions, cofactors_to_ignore = cofactors_to_ignore,
        spectators=False)

    # Build recommendation table
    kegg_recommendations_df = _generate_recommendation_table(model_file, 
                                                        match_results, 
                                                        {}, 
                                                        model_info, 
                                                        entity_type, 
                                                        database, 
                                                        {})

    kegg_output_file = f"{file_name}_kegg_reaction_recommendations.csv"
    if not kegg_recommendations_df.empty:
        print("\nSample KEGG reaction recommendations:")
        print(kegg_recommendations_df.head(10).to_string(index=False))
        
        # Save results
        kegg_recommendations_df.to_csv(kegg_output_file, index=False)
        print(f"\nKEGG reaction recommendations saved to: {kegg_output_file}")
    else:
        print("\nNo KEGG reaction recommendations generated.")


def init_species_probs(query_reaction, candidate_reactions):
    """
    Initialize species match probabilities for a query reaction
    given a set of candidate reference reactions.

    Returns a dictionary mapping each query species to a dictionary of
    candidate species and their initial probabilities.
    """
    species_match_probs = {}

    for query_species in query_reaction.participants:

        # Collect all candidate species that could plausibly match this query species
        possible_matches = set()
        for candidate in candidate_reactions:
            for cand_species in candidate.participants:
                if is_plausible_match(query_species, cand_species):
                    possible_matches.add(cand_species)

        # Initialize probabilities
        if possible_matches:
            prob = 1.0 / len(possible_matches)
            species_match_probs[query_species] = {s: prob for s in possible_matches}
        else:
            # No plausible matches found; start with empty dict
            species_match_probs[query_species] = {}

    return species_match_probs


# Example helper function (very simple)
def is_plausible_match(query_species, cand_species):
    """
    Decide if a candidate species could match the query species.
    For now, match if names are identical or similar.
    """
    return query_species.lower() == cand_species.lower()

def normalize(prob_dict):
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict

def update_species_probs(query_species, candidate_reactions, candidate_probs):
    """
    Update the species match probabilities for a single query species
    based on the candidate reaction probabilities.
    
    Returns a dictionary mapping candidate species to updated probabilities.
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

def choose_best_annotation(species_probs):
    """
    Choose the best annotation for a query species based on
    the current species match probabilities.
    
    Parameters:
        species_probs (dict): Mapping from candidate species to probabilities.
        
    Returns:
        str: The candidate species with the highest probability.
             Returns None if no candidates are available.
    """
    if not species_probs:
        return None  # no plausible matches
    
    # Find the candidate with the maximum probability
    best_species = max(species_probs, key=species_probs.get)
    return best_species

def has_converged(updated_annotations, previous_annotations):
    """
    Check if the EM algorithm has converged for a query reaction.
    
    Convergence occurs when the annotations of all species do not change
    from the previous iteration.
    
    Parameters:
        updated_annotations (dict): Mapping from query species to current annotation.
        previous_annotations (dict): Mapping from query species to previous annotation.
        
    Returns:
        bool: True if converged, False otherwise.
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




def amend_reaction_and_species_annotations(model_info, recommendations_df, top_k=10, max_iter=10):

    # get reaction recommendations


    # 2. Initialize probabilities for each candidate reaction
    #    (e.g., uniform distribution to start)
    recommendations_df['match_score_norm'] = (
        recommendations_df['match_score'] /
        recommendations_df.groupby('id')['match_score'].transform('sum')
        )

    # 3. Initialize species match probabilities
    #    For each query participant, assign probabilities over candidate participants
    species_match_probs = init_species_probs(query_reaction, candidate_reactions)
    
    for iteration in range(max_iter):

        # ---- Expectation Step ----
        # Compute how likely each candidate mapping explains the query reaction
        for candidate in candidate_reactions:
            candidate_probs[candidate] = compute_match_likelihood(
                query_reaction, candidate, species_match_probs
            )
        normalize(candidate_probs)

        # Update species match probabilities using weighted candidates
        for query_species in query_reaction.participants:
            species_match_probs[query_species] = update_species_probs(
                query_species, candidate_reactions, candidate_probs
            )

        # ---- Maximization Step ----
        # Assign best annotation for each query species based on match probabilities
        updated_annotations = {}
        for query_species in query_reaction.participants:
            updated_annotations[query_species] = choose_best_annotation(
                species_match_probs[query_species]
            )

        # ---- Check convergence ----
        if has_converged(updated_annotations, query_reaction.annotations):
            break

        # Update query reaction with new annotations
        query_reaction.annotations = updated_annotations

    return query_reaction.annotations
























    print("Time to update the file with annotations!")
    
    



if __name__ == "__main__":
    main()
    print("\nKEGG reaction annotation example completed!")
