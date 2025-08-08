#!/usr/bin/env python3
"""
AAAIM KEGG Reaction Annotation Example

This script demonstrates how to use AAAIM for reaction annotation using the KEGG reaction database.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import annotate_model, curate_model, get_available_databases, database_search
from core.update_model import update_annotation

def main():
    """
    Main function to demonstrate AAAIM reaction annotation functionality.
    """
    print("AAAIM KEGG Reaction Annotation Example")
    print("=" * 55)
    
    # Configuration
    llm_model = "meta-llama/llama-3.3-70b-instruct:free"  # or "gpt-4o-mini"

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
    
    # Example model file (you can replace this with your own model)
    model_file = "tests/test_models/BIOMD0000000190.xml"
    
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
    print("\n1. Reaction Annotation Workflow (for models without reaction annotations)")
    print("-" * 65)
    

    # first annotate model using ChEBI
    try:
        # Annotate genes in the model
        recommendations_df, metrics = annotate_model(
            model_file=model_file,
            llm_model=llm_model,  # You can change this to your preferred model
            entity_type="chemical",
            database="chebi",
            method="direct",
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
            sample_df = recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(5)
            print(sample_df.to_string(index=False))
            print()
            
            # Save results
            output_file = "recommendations.csv"
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

    # Example 2: Custom KEGG Reaction Mapping
    print("\n2. Custom KEGG Reaction Mapping")
    print("-" * 65)
    
    # Import necessary modules
    import lzma
    import pickle
    import json
    from collections import Counter
    
    # Load ChEBI to KEGG mapping
    try:
        chebi_to_kegg_map_file = "../data/kegg/chebi_to_kegg_map.lzma"
        if os.path.exists(chebi_to_kegg_map_file):
            with lzma.open(chebi_to_kegg_map_file, 'rb') as f:
                chebi_to_kegg_map = pickle.load(f)
            print(f"Loaded ChEBI to KEGG mapping with {len(chebi_to_kegg_map)} entries")
        else:
            print(f"ChEBI to KEGG mapping file not found: {chebi_to_kegg_map_file}")
            chebi_to_kegg_map = {}
            
        # Add KEGG IDs to chemical recommendations if available
        if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
            # Map ChEBI IDs to KEGG IDs
            recommendations_df['KEGG_ID'] = recommendations_df['annotation'].apply(
                lambda x: chebi_to_kegg_map.get(x, "Not mapped")
            )
            print("\nSample of ChEBI to KEGG mapping:")
            print(recommendations_df[['id', 'display_name', 'annotation', 'KEGG_ID']].head(5).to_string(index=False))
            
        # Load KEGG reaction data
        kegg_reactions_file = "../data/kegg/parsed_kegg_reactions.lzma"
        if os.path.exists(kegg_reactions_file):
            with lzma.open(kegg_reactions_file, 'rb') as f:
                kegg_reactions = pickle.load(f)
            print(f"\nLoaded {len(kegg_reactions)} KEGG reactions")
        else:
            print(f"KEGG reactions file not found: {kegg_reactions_file}")
            kegg_reactions = {}
            
        # Define common cofactors to ignore in reaction matching
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
        
        def normalize_reactions(model_reactions, cofactors_to_ignore):
            """
            Normalize reaction data for comparison by filtering out common cofactors
            and tracking stoichiometry.
            
            Args:
                model_reactions: List of reaction dictionaries
                cofactors_to_ignore: Set of cofactor IDs to ignore
                
            Returns:
                List of normalized reaction dictionaries
            """
            normalized_reactions = []
            
            for rxn in model_reactions:
                subs = filter_and_count(rxn.get('substrates', []), cofactors_to_ignore)
                prods = filter_and_count(rxn.get('products', []), cofactors_to_ignore)
                
                normalized_reactions.append({
                    'original_reaction': rxn.get('id', 'Unknown'),
                    'substrate_counter': subs,
                    'product_counter': prods,
                    'direction': rxn.get('direction', 'forward')
                })
                
            return normalized_reactions
            
        def filter_and_count(kegg_list, cofactors_to_ignore):
            """
            Filter out cofactors and count occurrences of each metabolite.
            
            Args:
                kegg_list: List of KEGG IDs
                cofactors_to_ignore: Set of cofactor IDs to ignore
                
            Returns:
                Counter object with metabolite counts
            """
            counter = Counter()
            for kegg_id in kegg_list:
                if kegg_id is None:
                    continue  # skip unmapped
                if kegg_id not in cofactors_to_ignore:
                    counter[kegg_id] += 1  # track stoichiometry
            return counter
            
        def compute_similarity(counter1, counter2, cofactors_to_ignore):
            """
            Compute Jaccard-like similarity between two reaction sides.
            
            Args:
                counter1: Counter for first reaction side
                counter2: Counter for second reaction side
                cofactors_to_ignore: Set of cofactor IDs to ignore
                
            Returns:
                Similarity score between 0 and 1
            """
            # Filter out cofactors
            c1 = {k: v for k, v in counter1.items() if k not in cofactors_to_ignore}
            c2 = {k: v for k, v in counter2.items() if k not in cofactors_to_ignore}
            
            if not c1 and not c2:
                return 1.0  # perfect match if both empty
                
            intersection = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
            union = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
            
            if union == 0:
                return 0.0
            return intersection / union
            
        def get_kegg_recommendations(normalized_reactions, kegg_reactions, cofactors_to_ignore, top_n=5):
            """
            Match model reactions to KEGG reactions based on metabolite similarity.
            
            Args:
                normalized_reactions: List of normalized model reactions
                kegg_reactions: Dictionary of KEGG reactions
                cofactors_to_ignore: Set of cofactor IDs to ignore
                top_n: Number of top matches to return per reaction
                
            Returns:
                List of dictionaries with match results
            """
            match_results = []
            
            for model_rxn in normalized_reactions:
                model_subs = model_rxn['substrate_counter']
                model_prods = model_rxn['product_counter']
                
                matches = []
                
                for kegg_id, kegg_rxn in kegg_reactions.items():
                    kegg_subs = kegg_rxn.get('substrate_counter', Counter())
                    kegg_prods = kegg_rxn.get('product_counter', Counter())
                    
                    # Score both orientations
                    score_forward = compute_similarity(model_subs, kegg_subs, cofactors_to_ignore) + \
                                    compute_similarity(model_prods, kegg_prods, cofactors_to_ignore)
                    
                    score_reverse = compute_similarity(model_subs, kegg_prods, cofactors_to_ignore) + \
                                    compute_similarity(model_prods, kegg_subs, cofactors_to_ignore)
                    
                    max_score = max(score_forward, score_reverse) / 2  # average of two comparisons
                    
                    matches.append({
                        'kegg_id': kegg_id,
                        'score_forward': score_forward / 2,
                        'score_reverse': score_reverse / 2,
                        'final_score': max_score
                    })
                
                # Sort matches by final score (descending)
                matches.sort(key=lambda x: x['final_score'], reverse=True)
                
                # Keep top N matches
                top_matches = matches[:top_n]
                
                match_results.append({
                    'model_reaction': model_rxn['original_reaction'],
                    'top_kegg_matches': top_matches
                })
                
            return match_results
            
        def build_recommendation_table(match_results, top_k=5):
            """
            Build a recommendation table from match results.
            
            Args:
                match_results: List of dictionaries with match results
                top_k: Number of best matches to retain per reaction
                
            Returns:
                List of dictionaries for DataFrame conversion
            """
            rows = []
            
            for entry in match_results:
                model_rxn_str = entry['model_reaction']
                kegg_matches = entry['top_kegg_matches'][:top_k]
                
                for match in kegg_matches:
                    rows.append({
                        'Model_Reaction': model_rxn_str,
                        'KEGG_Reaction_ID': match['kegg_id'],
                        'Score_Forward': round(match['score_forward'], 3),
                        'Score_Reverse': round(match['score_reverse'], 3),
                        'Final_Score': round(match['final_score'], 3)
                    })
                    
            return rows
        
        # Extract model reactions from the annotated model
        # For this example, we'll create a simple test set
        print("\nCreating test reaction set...")
        test_reactions = [
            {
                'id': 'R1',
                'substrates': ['C00031', 'C00103'],  # Glucose + ATP
                'products': ['C00668', 'C00008'],    # Glucose-6-P + ADP
                'direction': 'forward'
            },
            {
                'id': 'R2',
                'substrates': ['C00668', 'C00003'],  # Glucose-6-P + NAD+
                'products': ['C00198', 'C00004'],    # Glucono-1,5-lactone 6-P + NADH
                'direction': 'forward'
            }
        ]
        
        # Normalize reactions
        normalized_reactions = normalize_reactions(test_reactions, cofactors_to_ignore)
        print(f"Normalized {len(normalized_reactions)} test reactions")
        
        # Get KEGG recommendations
        match_results = get_kegg_recommendations(normalized_reactions, kegg_reactions, cofactors_to_ignore)
        
        # Build recommendation table
        recommendation_rows = build_recommendation_table(match_results)
        kegg_recommendations_df = pd.DataFrame(recommendation_rows)
        
        if not kegg_recommendations_df.empty:
            print("\nSample KEGG reaction recommendations:")
            print(kegg_recommendations_df.head(10).to_string(index=False))
            
            # Save results
            kegg_output_file = "kegg_reaction_recommendations.csv"
            kegg_recommendations_df.to_csv(kegg_output_file, index=False)
            print(f"\nKEGG reaction recommendations saved to: {kegg_output_file}")
        else:
            print("\nNo KEGG reaction recommendations generated.")
            
    except Exception as e:
        print(f"\nError in custom KEGG reaction mapping: {e}")
        import traceback
        traceback.print_exc()

    chebi_annotated_model = model_file.split('.')[0]+'_annotated.xml'

    if os.path.exists(output_file):
        update_annotation(
            original_model_path=model_file,
            recommendation_table="recommendations.csv",  # or a pandas DataFrame
            new_model_path=chebi_annotated_model,
            qualifier="is"  # (optional) bqbiol qualifier, default is 'is'
            )
    else: 
        return
    
    if os.path.exists(chebi_annotated_model):
        try:
            # Annotate reactions in the model
            reaction_recommendations_df, reaction_metrics = annotate_model(
                model_file=model_file,
                llm_model="gpt-4o-mini",
                entity_type="reaction",
                database="kegg",
                method="direct",
                max_entities=5
            )
            # Display annotation results
            if not reaction_recommendations_df.empty:
                print("Annotation Results:")
                print(f"Total entities in model: {reaction_metrics['total_entities']}")
                print(f"Entities with predictions: {reaction_metrics['entities_with_predictions']}")
                print(f"Annotation rate: {reaction_metrics['annotation_rate']:.1%}")
                
                if not pd.isna(reaction_metrics['accuracy']):
                    print(f"Accuracy (where existing annotations available): {reaction_metrics['accuracy']:.1%}")
                else:
                    print("Accuracy: N/A (no existing annotations to compare against)")
                
                print(f"Total time: {reaction_metrics['total_time']:.2f}s")
                print()
                
                # Show sample recommendations
                print("Sample Annotation Recommendations:")
                reaction_samples_df = reaction_recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(5)
                print(reaction_samples_df.to_string(index=False))
                print()
                
                # Save results
                output_file = "reaction_recommendations.csv"
                reaction_recommendations_df.to_csv(output_file, index=False)
                print(f"Full reaction annotation results saved to: {output_file}")
                
            else:
                print("No reaction annotation recommendations generated.")
                if 'error' in reaction_metrics:
                    print(f"Error: {reaction_metrics['error']}")

        except Exception as e:
            print(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()

        if os.path.exists(output_file):
            try: 
                update_annotation(
                    original_model_path=model_file,
                    recommendation_table="recommendations.csv",  # or a pandas DataFrame
                    new_model_path=model_file.split('.')[0]+'_annotated.xml',
                    qualifier="is"  # (optional) bqbiol qualifier, default is 'is'
                    )
            except Exception as e:
                print(f"Reaction annotation failed: {e}")
        else: 
            return
         
    
if __name__ == "__main__":
    main()
