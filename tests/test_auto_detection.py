"""
Test script for automatic entity type detection.
"""

from utils.evaluation import evaluate_single_model
import pandas as pd

# Test model file
test_model_file = "/Users/luna/Desktop/CRBM/AMAS_proj/Models/BioModels/BIOMD0000000039.xml"

print("="*80)
print("Testing Automatic Entity Type Detection")
print("="*80)
print(f"\nModel: {test_model_file}")
print(f"\nConfiguration:")
print(f"  - entity_type: auto")
print(f"  - database: ['chebi', 'uniprot']")
print(f"  - method: direct")
print(f"  - llm_model: meta-llama/llama-3.3-70b-instruct:free")
print(f"  - top_k: 3")
print(f"  - max_entities: 10")
print("\n" + "="*80)

# Run evaluation with auto entity type detection
result_df = evaluate_single_model(
    model_file=test_model_file,
    llm_model="meta-llama/llama-3.3-70b-instruct:free",
    method="direct",
    top_k=3,
    max_entities=10,  # Limit to 10 entities for testing
    entity_type="auto",  # Enable auto detection
    database=["chebi", "uniprot"],  # Support both chemicals and proteins
    verbose=True
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if result_df is not None and not result_df.empty:
    print(f"\nTotal species evaluated: {len(result_df)}")
    
    # Show detected entity types
    print("\nDetected Entity Types:")
    entity_type_counts = result_df['detected_entity_type'].value_counts()
    for entity_type, count in entity_type_counts.items():
        print(f"  - {entity_type}: {count}")
    
    # Show sample results
    print("\nSample Results (first 5 species):")
    print("\n" + "-"*80)
    for idx, row in result_df.head(5).iterrows():
        print(f"\nSpecies ID: {row['species_id']}")
        print(f"  Display Name: {row['display_name']}")
        print(f"  Detected Type: {row['detected_entity_type']}")
        print(f"  LLM Synonyms: {row['synonyms_LLM']}")
        print(f"  Predictions: {row['predictions']}")
        print(f"  Prediction Names: {row['predictions_names']}")
        print(f"  Accuracy: {row['accuracy']}")
    
    # Save results
    output_file = "test_auto_detection_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")
    
else:
    print("\nNo results generated. Check the logs above for errors.")

print("\n" + "="*80)

