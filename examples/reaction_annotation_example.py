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

from core import annotate_model, curate_model, get_available_databases

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
            sample_df2 = recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score', 'existing']].head(5)
            print(sample_df2.to_string(index=False))
            print()
            
            # Save results
            output_file = "simple_annotation_results.csv"
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

    try:
        
        # parse reactions from model_file and extract the substrates and reactions
            # first find reactions that already have KEGG annotations
            # should you modify extract_model_info or add a follow-up function? 

        # use model_info.parse_reaction_equation.
        
        
        
        
        # Annotate reactions in the model
        recommendations_df, metrics = annotate_model(
            model_file=model_file,
            llm_model="gpt-4o-mini",
            entity_type="reaction",
            database="kegg",
            method="direct",
            max_entities=5
        )
        
        if not recommendations_df.empty:
            print(f"Generated {len(recommendations_df)} reaction annotation recommendations")
            print("\nSample recommendations:")
            print(recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score']].head())
            
            # Save results
            output_file = "reaction_annotation_results.csv"
            recommendations_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("No reaction annotation recommendations generated")
        
        print(f"\nMetrics: {metrics}")
        
    except Exception as e:
        print(f"Reaction annotation failed: {e}")
    
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
        print(f"Reaction curation failed: {e}")
    
    print("\nKEGG reaction annotation example completed!")

if __name__ == "__main__":
    main()
