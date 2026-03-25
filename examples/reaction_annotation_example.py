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
from core.annotation_workflow import map_reactions_to_kegg_with_relaxation, _generate_recommendation_table
from core.model_info import get_all_reaction_ids

# Define common cofactors to ignore in reaction matching


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


def main():
    """
    Main function to demonstrate AAAIM reaction annotation functionality.
    """
    print("AAAIM KEGG Reaction Annotation Example")
    print("=" * 50)
    
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
    normalized_reactions, match_results, species_relax_levels = map_reactions_to_kegg_with_relaxation(
        reactions,
        high_score_recommendations,
        spectators=False,
        cofactors_to_ignore=cofactors_to_ignore,
        top_k=None,
    )
    print(f"Normalized reactions: {normalized_reactions}")
    print(f"Species relaxation levels (0=strict ChEBI→KEGG): {species_relax_levels}")

    # KEGG recommendations are produced inside map_reactions_to_kegg_with_relaxation

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


if __name__ == "__main__":
    main()
    print("\nKEGG reaction annotation example completed!")
