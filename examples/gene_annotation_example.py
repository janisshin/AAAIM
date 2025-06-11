#!/usr/bin/env python3
"""
AAAIM Gene Annotation Example

This script demonstrates how to use AAAIM for gene annotation using the NCBI gene database.
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
    Main function to demonstrate AAAIM gene annotation functionality.
    """
    print("AAAIM Gene Annotation Example")
    print("=" * 50)
    
    # Check if NCBI gene database is available
    available_dbs = get_available_databases()
    print(f"Available databases: {available_dbs}")
    
    if "ncbigene" not in available_dbs:
        print("ERROR: NCBI gene database not available!")
        print("Please ensure NCBI gene reference files are present in data/ncbigene/")
        return
    
    # Example model file (you can replace this with your own model)
    model_file = "tests/test_models/BIOMD0000000190.xml"  # This is a chemical model, but we'll use it for demo
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please provide a valid SBML model file.")
        return
    
    print(f"\nAnalyzing model: {model_file}")
    
    # Example 1: Gene Annotation Workflow
    print("\n1. Gene Annotation Workflow (for models without gene annotations)")
    print("-" * 60)
    
    try:
        # Annotate genes in the model
        recommendations_df, metrics = annotate_model(
            model_file=model_file,
            llm_model="gpt-4o-mini",  # You can change this to your preferred model
            entity_type="gene",
            database="ncbigene",
            method="direct",
            max_entities=5  # Limit to 5 entities for demo
        )
        
        if not recommendations_df.empty:
            print(f"Generated {len(recommendations_df)} gene annotation recommendations")
            print("\nSample recommendations:")
            print(recommendations_df[['id', 'display_name', 'annotation', 'annotation_label', 'match_score']].head())
            
            # Save results
            output_file = "gene_annotation_results.csv"
            recommendations_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("No gene annotation recommendations generated")
        
        print(f"\nMetrics: {metrics}")
        
    except Exception as e:
        print(f"Gene annotation failed: {e}")
    
    # Example 2: Gene Curation Workflow (only if the model has existing gene annotations)
    print("\n2. Gene Curation Workflow (for models with existing gene annotations)")
    print("-" * 60)
    
    try:
        # Try to curate existing gene annotations
        curation_df, curation_metrics = curate_model(
            model_file=model_file,
            llm_model="gpt-4o-mini",
            entity_type="gene", 
            database="ncbigene",
            method="direct"
        )
        
        if not curation_df.empty:
            print(f"Generated {len(curation_df)} gene curation recommendations")
            print("\nSample curations:")
            print(curation_df[['id', 'display_name', 'annotation', 'annotation_label', 'existing']].head())
            
            # Save results
            output_file = "gene_curation_results.csv"
            curation_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            print(f"\nCuration metrics: {curation_metrics}")
        else:
            print("No existing gene annotations found in model - curation not applicable")
        
    except Exception as e:
        print(f"Gene curation failed: {e}")
    
    # Example 3: Compare chemical vs gene annotation
    print("\n3. Comparison: Chemical vs Gene Annotation")
    print("-" * 60)
    
    try:
        # Annotate same model as chemicals
        chemical_df, chemical_metrics = annotate_model(
            model_file=model_file,
            llm_model="gpt-4o-mini",
            entity_type="chemical",
            database="chebi", 
            method="direct",
            max_entities=5
        )
        
        print("Chemical annotation results:")
        if not chemical_df.empty:
            print(f"  - Generated {len(chemical_df)} chemical recommendations")
            print(f"  - Accuracy: {chemical_metrics.get('accuracy', 'N/A')}")
        else:
            print("  - No chemical recommendations generated")
        
        print("Gene annotation results:")
        if 'recommendations_df' in locals() and not recommendations_df.empty:
            print(f"  - Generated {len(recommendations_df)} gene recommendations")
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
        else:
            print("  - No gene recommendations generated")
            
    except Exception as e:
        print(f"Comparison failed: {e}")
    
    print("\nGene annotation example completed!")

if __name__ == "__main__":
    main() 