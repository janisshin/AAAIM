"""
Unit tests for automatic entity type detection components.
"""

from core.llm_interface import parse_llm_response
from utils.evaluation import _get_database_for_entity_type
from utils.constants import EntityType, DatabaseID

print("="*80)
print("Unit Tests for Auto Entity Type Detection")
print("="*80)

# Test 1: Parse LLM response with entity types
print("\n" + "="*80)
print("TEST 1: Parsing LLM Response with Entity Types")
print("="*80)

test_response = """
A (chemical): "acetyl-CoA", "acetyl coenzyme A"
B (chemical): "citric acid", "sodium citrate"
C (protein): "citrate synthase", "CS"
D (gene): "RELA", "p65", "NFKB3"
E (complex): "IKK complex", "IkappaB kinase complex"
F (unknown): "UNK"
Reason: This is a test model with mixed entity types.
"""

synonyms_dict, entity_type_dict, reason = parse_llm_response(test_response)

print("\nSynonyms Dictionary:")
for species_id, synonyms in synonyms_dict.items():
    print(f"  {species_id}: {synonyms}")

print("\nEntity Type Dictionary:")
for species_id, entity_type in entity_type_dict.items():
    print(f"  {species_id}: {entity_type}")

print(f"\nReason: {reason}")

# Validate results
assert len(synonyms_dict) == 6, "Should have 6 species"
assert len(entity_type_dict) == 6, "Should have 6 entity types"
assert entity_type_dict['A'] == 'chemical', "A should be chemical"
assert entity_type_dict['C'] == 'protein', "C should be protein"
assert entity_type_dict['D'] == 'gene', "D should be gene"
assert entity_type_dict['E'] == 'complex', "E should be complex"
assert entity_type_dict['F'] == 'unknown', "F should be unknown"

print("\n✓ TEST 1 PASSED")

# Test 2: Database mapping for entity types
print("\n" + "="*80)
print("TEST 2: Database Mapping for Entity Types")
print("="*80)

test_cases = [
    ("chemical", ["chebi", "uniprot"], "chebi"),
    ("gene", ["chebi", "ncbigene"], "ncbigene"),
    ("protein", ["uniprot", "chebi"], "uniprot"),
    ("complex", ["chebi", "uniprot"], "chebi"),  # Should return first available
    ("unknown", ["chebi", "uniprot"], None),
]

print("\nTest cases:")
for entity_type, allowed_dbs, expected in test_cases:
    result = _get_database_for_entity_type(entity_type, allowed_dbs)
    status = "✓" if result == expected else "✗"
    print(f"  {status} Entity: {entity_type:10} | Allowed: {str(allowed_dbs):25} | Expected: {str(expected):10} | Got: {str(result):10}")
    assert result == expected, f"Failed for {entity_type}"

print("\n✓ TEST 2 PASSED")

# Test 3: Constants verification
print("\n" + "="*80)
print("TEST 3: Constants Verification")
print("="*80)

print("\nEntityType enum values:")
for entity in EntityType:
    print(f"  - {entity.value}")

assert EntityType.CHEMICAL in EntityType, "CHEMICAL should exist"
assert EntityType.GENE in EntityType, "GENE should exist"
assert EntityType.PROTEIN in EntityType, "PROTEIN should exist"
assert EntityType.COMPLEX in EntityType, "COMPLEX should exist"
assert EntityType.UNKNOWN in EntityType, "UNKNOWN should exist"

print("\nDatabaseID enum values:")
for db in DatabaseID:
    print(f"  - {db.value}")

assert DatabaseID.CHEBI in DatabaseID, "CHEBI should exist"
assert DatabaseID.NCBIGENE in DatabaseID, "NCBIGENE should exist"
assert DatabaseID.UNIPROT in DatabaseID, "UNIPROT should exist"

print("\n✓ TEST 3 PASSED")

# Test 4: Test parsing without entity types (backward compatibility)
print("\n" + "="*80)
print("TEST 4: Backward Compatibility (no entity types)")
print("="*80)

test_response_old = """
A: "acetyl-CoA", "acetyl coenzyme A"
B: "citric acid", "sodium citrate"
Reason: Old format without entity types.
"""

synonyms_dict_old, entity_type_dict_old, reason_old = parse_llm_response(test_response_old)

print("\nSynonyms Dictionary:")
for species_id, synonyms in synonyms_dict_old.items():
    print(f"  {species_id}: {synonyms}")

print("\nEntity Type Dictionary (should default to 'unknown'):")
for species_id, entity_type in entity_type_dict_old.items():
    print(f"  {species_id}: {entity_type}")

assert len(synonyms_dict_old) == 2, "Should have 2 species"
assert entity_type_dict_old['A'] == 'unknown', "A should default to unknown"
assert entity_type_dict_old['B'] == 'unknown', "B should default to unknown"

print("\n✓ TEST 4 PASSED")

print("\n" + "="*80)
print("ALL UNIT TESTS PASSED ✓")
print("="*80)
print("\nThe auto entity type detection implementation is working correctly!")
print("To test with a real model, set your OPENROUTER_API_KEY and run:")
print("  python test_auto_detection.py")
print("="*80)

