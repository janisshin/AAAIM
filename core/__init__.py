"""
AAAIM Core Module
"""

# Main annotation interface - for models with no or limited annotations
from .annotation_workflow import annotate_model, annotate_single_model

# Main curation interface - for models with existing annotations
from .curation_workflow import curate_model, curate_single_model

# Shared functionality
from .annotation_workflow import print_results

# Individual components for advanced users
from .model_info import find_species_with_chebi_annotations, find_species_with_ncbigene_annotations, extract_model_info, format_prompt, get_species_display_names, get_all_species_ids, detect_model_format
from .llm_interface import SYSTEM_PROMPT, SYSTEM_PROMPT_CHEMICAL, SYSTEM_PROMPT_GENE, get_system_prompt, query_llm, parse_llm_response
from .database_search import get_species_recommendations_direct, search_database, get_available_databases, Recommendation

__all__ = [
    # Main interfaces - what most users will use
    'annotate_model',  
    'annotate_single_model', 
    'curate_model', 
    'curate_single_model',
    'print_results',
    
    # Individual components
    'get_all_species_ids',
    'find_species_with_chebi_annotations',
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
    'Recommendation'
] 