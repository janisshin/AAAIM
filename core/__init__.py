"""
AAAIM Core Module
"""

# Main annotation interface - for models with no or limited annotations
from .annotation_workflow import annotate_model, annotate_single_model, map_reactions_to_kegg_with_relaxation

# Main curation interface - for models with existing annotations
from .curation_workflow import curate_model, curate_single_model

# Shared functionality
from .annotation_workflow import print_results, normalize_reactions
from .update_model import update_annotation
from .database_search import (
    load_chebi2kegg_dict,
    load_kegg_reaction_features_dict,
    score_model_against_kegg_reaction,
)
from .hierarchy_relaxation import (
    compute_relaxation_penalty,
    expand_chebi_with_metadata,
    iter_chebi_for_species,
    detect_problematic_metabolites,
    detect_unmapped_metabolites,
    detect_unmapped_species_ids,
    get_ancestors,
    load_chebi_child_map,
    load_chebi_parent_map,
    merge_chebi_to_kegg_mapping,
    normalize_chebi,
    normalize_reaction,
    parse_chebi_obo,
    progressive_normalization,
    select_metabolites_to_relax,
    should_continue_iteration,
    unified_reaction_objective,
    unified_reaction_objective_weighted,
)

# Individual components
from .model_info import find_species_with_chebi_annotations, find_species_with_annotations_and_qualifiers, find_species_with_ncbigene_annotations, extract_model_info, format_prompt, get_species_display_names, get_all_species_ids, detect_model_format
from .llm_interface import SYSTEM_PROMPT, SYSTEM_PROMPT_CHEMICAL, SYSTEM_PROMPT_GENE, get_system_prompt, query_llm, parse_llm_response
from .database_search import get_species_recommendations_direct, search_database, get_available_databases, Recommendation
from .reaction_deduplication import Reaction, deduplicate_reactions

__all__ = [
    # Main interfaces
    'annotate_model',  
    'annotate_single_model', 
    'curate_model', 
    'curate_single_model',
    'print_results',
    'update_annotation',
    
    # Individual components
    'get_all_species_ids',
    'find_species_with_chebi_annotations',
    'find_species_with_annotations_and_qualifiers',
    'find_species_with_ncbigene_annotations',
    'get_species_display_names',
    'extract_model_info',
    'format_prompt',
    'detect_model_format',
    'SYSTEM_PROMPT',
    'SYSTEM_PROMPT_CHEMICAL',
    'SYSTEM_PROMPT_GENE',
    'get_system_prompt',
    'query_llm',
    'parse_llm_response',
    'get_species_recommendations_direct',
    'search_database',
    'get_available_databases',
    'Recommendation',
    'map_reactions_to_kegg_with_relaxation',
    'load_chebi_parent_map',
    'load_chebi_child_map',
    'expand_chebi_with_metadata',
    'iter_chebi_for_species',
    'compute_relaxation_penalty',
    'merge_chebi_to_kegg_mapping',
    'normalize_chebi',
    'normalize_reaction',
    'parse_chebi_obo',
    'progressive_normalization',
    'get_ancestors',
    'detect_unmapped_metabolites',
    'detect_unmapped_species_ids',
    'detect_problematic_metabolites',
    'select_metabolites_to_relax',
    'score_model_against_kegg_reaction',
    'unified_reaction_objective',
    'unified_reaction_objective_weighted',
    'should_continue_iteration',
    'Reaction',
    'deduplicate_reactions',
] 