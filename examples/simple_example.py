#!/usr/bin/env python3
"""
Simple AAAIM Example

Shows annotation, curation, user feedback, and model update workflows.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main AAAIM interfaces
from core import annotate_model, curate_model, print_results, update_annotation

def main():
    """
    Simple example of using AAAIM to annotate, curate, and refine models.
    """
    print("AAAIM Simple Annotation & Curation Example")
    print("=" * 50)
    
    # Configuration
    model_curation_file = "tests/test_models/BIOMD0000000190.xml"
    model_file = "tests/test_models/190_few_anno.xml"
    # llm_model = "meta-llama/llama-3.3-70b-instruct:free"
    llm_model = "Llama-3.3-70B-Instruct"
    max_entities = 5  # None will evaluate all species
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please ensure the test model is available.")
        return
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: No API keys found in environment.")
        print("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
        return
    
    print(f"Model file: {model_file}")
    print(f"LLM model: {llm_model}")
    print()
    
    try:
        # ── Example 1: Annotation workflow ──────────────────────────────
        print("EXAMPLE 1: Annotation workflow (all species)")
        print("-" * 80)
        
        result = annotate_model(
            model_file=model_file,
            llm_model=llm_model,
            max_entities=max_entities,
            entity_type="chemical",
            database="chebi"
        )
        
        # Display annotation results
        if not result.recommendations_df.empty:
            print("Annotation Results:")
            print(f"Total entities in model: {result.metrics['total_entities']}")
            print(f"Entities with predictions: {result.metrics['entities_with_predictions']}")
            print(f"Annotation rate: {result.metrics['annotation_rate']:.1%}")
            
            if not pd.isna(result.metrics['accuracy']):
                print(f"Accuracy (where existing annotations available): {result.metrics['accuracy']:.1%}")
            else:
                print("Accuracy: N/A (no existing annotations to compare against)")
            
            print(f"Total time: {result.metrics['total_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Annotation Recommendations:")
            sample_df = result.recommendations_df[
                ['id', 'display_name', 'curated_name', 'annotation', 'annotation_label', 'match_score', 'status']
            ].head(5)
            print(sample_df.to_string(index=False))
            print()
            
        else:
            print("No annotation recommendations generated.")
            if 'error' in result.metrics:
                print(f"Error: {result.metrics['error']}")
            sys.exit()

        # ── Example 2: Curation workflow ────────────────────────────────
        print("EXAMPLE 2: Curation workflow (models with existing annotations)")
        print("-" * 60)
        
        result_curate = curate_model(
            model_file=model_curation_file,
            llm_model=llm_model,
            max_entities=max_entities,
            entity_type="chemical",
            database="chebi"
        )
        
        # Display curation results
        if not result_curate.recommendations_df.empty:
            print("Curation Results:")
            print(f"Total entities with existing annotations: {result_curate.metrics['total_entities']}")
            print(f"Entities with predictions: {result_curate.metrics['entities_with_predictions']}")
            print(f"Accuracy: {result_curate.metrics['accuracy']:.1%}")
            print(f"Total time: {result_curate.metrics['total_time']:.2f}s")
            print()
            
            # Show sample recommendations
            print("Sample Curation Recommendations:")
            sample_df = result_curate.recommendations_df[
                ['id', 'display_name', 'curated_name', 'annotation', 'annotation_label', 'match_score', 'status']
            ].head(5)
            print(sample_df.to_string(index=False))
            print()
        else:
            if 'error' in result_curate.metrics:
                print(f"Curation failed: {result_curate.metrics['error']}")
        
        print("\n" + "="*60 + "\n")

        # ── Example 3: User Feedback ────────────────────────────────────
        print("EXAMPLE 3: User feedback to revise recommendations")
        print("-" * 80)
        print("Providing feedback: 'Species CoA should be coenzyme A.'")
        result_curate = result_curate.revise("Species CoA should be coenzyme A.")

        print("Revised Curation Results:")
        print(f"Entities with predictions: {result_curate.metrics['entities_with_predictions']}")
        print(f"Accuracy: {result_curate.metrics['accuracy']:.1%}")
        print()

        print("\nRevised recommendations:")
        sample_df = result_curate.recommendations_df[
            ['id', 'display_name', 'annotation', 'annotation_label', 'match_score']
        ].head(5)
        print(sample_df.to_string(index=False))
        print()
        # For interactive feedback in a terminal, use:
        #   result_curate.feedback_loop()

        print("\n" + "="*60 + "\n")

        # ── Example 4: Update model ─────────────────────────────────────
        print("EXAMPLE 4: Updating model with new annotations")
        print("-" * 80)
        print("A recommendation table was generated for the user to inspect:", 
              model_file+'_recommendations.csv')
        print("User can edit the 'update_annotation' column of the CSV.")
        print("-" * 80)
        print("Assuming the user wants to delete the first annotation and add the second...")
        result_curate.recommendations_df.loc[1, 'update_annotation'] = 'delete'
        result_curate.recommendations_df.loc[2, 'update_annotation'] = 'add'
        # the rest of the annotations are ignored
        result_curate.recommendations_df.loc[3:, 'update_annotation'] = 'ignore'
        print("-" * 80)
        update_annotation(
            original_model_path=model_file,
            recommendation_table=result_curate.recommendations_df,
            new_model_path=model_file+'_updated.xml'
        )
        print("Model updated successfully")
        print("Saved to: ", model_file+'_updated.xml')
        print("-" * 80)

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
