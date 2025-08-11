#!/usr/bin/env python3
"""
AAAIM KEGG Reaction Annotation Example

This script demonstrates how to use AAAIM for reaction annotation using the KEGG reaction database.
"""

import os
import sys
import pandas as pd
import lzma
import pickle

from dotenv import load_dotenv
load_dotenv()


from pathlib import Path

# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import annotate_model, curate_model, get_available_databases, database_search
from core import normalize_reactions, build_recommendation_table
from core import load_chebi2kegg_dict, load_kegg_reaction_features_dict
from core.update_model import update_annotation


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
    
    # Load ChEBI to KEGG mapping
    chebi_to_kegg_map = load_chebi2kegg_dict()
    
    try: # JANISTAG why are we trying (+ excepting) this? 
        # Add KEGG IDs to chemical recommendations if available
        if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
            # Map ChEBI IDs to KEGG IDs
            recommendations_df['KEGG_ID'] = recommendations_df['annotation'].apply(
                lambda x: chebi_to_kegg_map.get(x, "Not mapped")
            )
            print("\nSample of ChEBI to KEGG mapping:")
            print(recommendations_df[['id', 'display_name', 'annotation', 'KEGG_ID']].head(5).to_string(index=False))
        
        ##############################################

        # Load KEGG reaction data
        kegg_reaction_features = load_kegg_reaction_features_dict()
        
        # Extract model reactions from the annotated model
        # For this example, we'll create a simple test set
        print("\nCreating test reaction set...")
        test_reactions = [
            {
                'id': 'R1',
                'substrates': ['C00031', 'C00103'],  # Glucose + ATP
                'products': ['C00668', 'C00008'],    # Glucose-6-P + ADP
            },
            {
                'id': 'R2',
                'substrates': ['C00668', 'C00003'],  # Glucose-6-P + NAD+
                'products': ['C00198', 'C00004'],    # Glucono-1,5-lactone 6-P + NADH
            }
        ]
        
        # Normalize reactions
        normalized_reactions = normalize_reactions(test_reactions, cofactors_to_ignore)
        print(f"Normalized {len(normalized_reactions)} test reactions")
        
        # Get KEGG recommendations
        match_results = database_search._get_kegg_recommendations(
            normalized_reactions, kegg_reaction_features, 
            cofactors_to_ignore)
        
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
    print("Time to update the file with annotations!")

    """if os.path.exists(output_file):
        update_annotation(
            original_model_path=model_file,
            recommendation_table="recommendations.csv",  # or a pandas DataFrame
            new_model_path=chebi_annotated_model,
            qualifier="is"  # (optional) bqbiol qualifier, default is 'is'
            )
    else: 
        return"""
    
    """if os.path.exists(chebi_annotated_model):
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
            return"""
        


"""
# Example 2: Reaction Curation Workflow (for models with existing reaction annotations)
print("\n2. Reaction Curation Workflow (for models with existing reaction annotations)")
print("-" * 65)

try:
    # Try to curate existing reaction annotations
    curation_df, curation_metrics = curate_model(
        model_file=model_file,
        llm_model="gpt-4o-mini",
        entity_type="reaction", 
        database="kegg",
        method="direct"
    )
    
    if not curation_df.empty:
        print(f"Generated {len(curation_df)} reaction curation recommendations")
        print("\nSample curations:")
        print(curation_df[['id', 'display_name', 'annotation', 'annotation_label', 'existing']].head())
        
        # Save results
        output_file = "reaction_curation_results.csv"
        curation_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        print(f"\nCuration metrics: {curation_metrics}")
    else:
        print("No existing reaction annotations found in model - curation not applicable")
    
except Exception as e:
    print(f"Reaction curation failed: {e}")"""




if __name__ == "__main__":
    main()
    print("\nKEGG reaction annotation example completed!")
