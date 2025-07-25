"""
Curation Workflow for AAAIM

Main interface for curating a single model.
Provides the primary function that users will call to get recommendation tables
for models that already have existing annotations.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from core.model_info import find_species_with_chebi_annotations, find_species_with_ncbigene_annotations, extract_model_info, format_prompt
from core.llm_interface import get_system_prompt, query_llm, parse_llm_response
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag

logger = logging.getLogger(__name__)

def curate_single_model(model_file: str, 
                  llm_model: str = "gpt-4o-mini",
                  method: str = "direct",
                  top_k: int = 3,
                  max_entities: int = None,
                  entity_type: str = "chemical",
                  database: str = "chebi",
                  tax_id: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    This is the main function users will call to get curation recommendations
    for a model that already has existing annotations.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use ("gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct:free")
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to return per species
        max_entities: Maximum number of entities to annotate (None for all)
        entity_type: Type of entities to annotate ("chemical", "gene", "protein")
        database: Target database ("chebi", "ncbigene", "uniprot")
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
        - recommendations_df: AMAS-compatible DataFrame with annotation recommendations
        - metrics_dict: Dictionary with evaluation metrics and timing information
    """
    start_time = time.time()
    
    logger.info(f"Starting curation for model: {model_file}")
    logger.info(f"Using LLM model: {llm_model}")
    logger.info(f"Using method: {method} for database search")
    logger.info(f"Entity type: {entity_type}, Database: {database}")
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")
    
    # Step 1: Find existing annotations
    logger.info(">>>Step 1: Finding existing annotations...<<<")
    if entity_type == "chemical" and database == "chebi":
        existing_annotations = find_species_with_chebi_annotations(model_file)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "gene" and database == "ncbigene":
        existing_annotations = find_species_with_ncbigene_annotations(model_file)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    else:
        # Future: support other entity types and databases
        logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
        existing_annotations = {}
    
    if not existing_annotations:
        logger.warning("No existing annotations found in model")
        return pd.DataFrame(), {"error": "No existing annotations found"}
    
    # Select entities to evaluate
    if max_entities:
        specs_to_evaluate = list(existing_annotations.keys())[:max_entities]
        logger.info(f"Selected {len(specs_to_evaluate)} entities for curation")
    else:
        specs_to_evaluate = list(existing_annotations.keys())
        logger.info(f"Curation all {len(specs_to_evaluate)} entities")
    
    # Extract model context
    logger.info(">>>Step 2: Extracting model context...<<<")
    model_info = extract_model_info(model_file, specs_to_evaluate, entity_type)
    
    if not model_info:
        logger.error("Failed to extract model context")
        return pd.DataFrame(), {"error": "Failed to extract model context"}
    
    logger.info(f"Extracted context for model: {model_info['model_name']}")
    
    # Format prompt for LLM
    logger.info(">>>Step 3: Querying LLM ({llm_model})...<<<")
    prompt = format_prompt(model_file, specs_to_evaluate, entity_type)
    
    if not prompt:
        logger.error("Failed to format prompt")
        return pd.DataFrame(), {"error": "Failed to format prompt"}
    
    # Query LLM
    llm_start = time.time()
    
    try:
        # Get appropriate system prompt for entity type
        system_prompt = get_system_prompt(entity_type)
        result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
        llm_time = time.time() - llm_start
        
        if not result:
            logger.error("No response from LLM")
            return pd.DataFrame(), {"error": "No response from LLM"}
        
        logger.info(f"LLM response received in {llm_time:.2f}s")
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return pd.DataFrame(), {"error": f"LLM query failed: {e}"}
    
    # Parse LLM response
    synonyms_dict, reason = parse_llm_response(result)
    
    if not synonyms_dict:
        logger.error("Failed to parse LLM response")
        return pd.DataFrame(), {"error": "Failed to parse LLM response"}
    
    logger.info(f"Parsed synonyms for {len(synonyms_dict)} entities")
    
    # Search database
    logger.info(f">>>Step 4: Searching {database} database...<<<")
    search_start = time.time()
    
    if database == "chebi":
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="chebi")
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == "ncbigene":
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    else:
        # Future: support other databases
        logger.error(f"Database {database} not yet supported")
        return pd.DataFrame(), {"error": f"Database {database} not yet supported"}
    
    search_time = time.time() - search_start
    logger.info(f"Database search completed in {search_time:.2f}s")
    
    # Generate recommendation table
    logger.info(">>>Step 5: Generating recommendation table...<<<")
    recommendations_df = _generate_recommendation_table(
        model_file, recommendations, existing_annotations, model_info, entity_type
    )
    
    # Step 9: Calculate metrics
    total_time = time.time() - start_time
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, total_time, llm_time, search_time
    )
    
    logger.info(f"Curation completed in {total_time:.2f}s")
    logger.info(f"Generated {len(recommendations_df)} recommendations")
    
    return recommendations_df, metrics

def _generate_recommendation_table(model_file: str, 
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any],
                                 entity_type: str = "chemical",
                                 database: str = "chebi") -> pd.DataFrame:
    """
    Generate AMAS-compatible recommendation table.
    
    Args:
        model_file: Path to model file
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_info: Model information dictionary
        entity_type: Type of entity being annotated
        
    Returns:
        DataFrame in AMAS format
    """
    rows = []
    filename = Path(model_file).name
    
    # Track which (species_id, annotation) pairs are already in the table
    seen_pairs = set()
    for rec in recommendations:
        if not rec.candidates:
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': '',
                'annotation_label': '',
                'match_score': 0.0,
                'existing': 0,
                'update_annotation': 'ignore'
            }
            rows.append(row)
            continue
        for i, candidate in enumerate(rec.candidates):
            if database == "chebi":
                candidate_display = f"CHEBI:{candidate}"
            elif database == "ncbigene":
                candidate_display = f"NCBIGENE:{candidate}"
            existing = 1 if candidate in existing_annotations.get(rec.id, []) else 0
            match_score = rec.match_score[i]
            if existing:
                update_action = 'keep'
            else:
                update_action = 'ignore'
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': candidate_display,
                'annotation_label': rec.candidate_names[i],
                'match_score': match_score,
                'existing': existing,
                'update_annotation': update_action
            }
            rows.append(row)
            seen_pairs.add((rec.id, candidate))
    # Add rows for existing annotations not predicted
    for species_id, ann_list in existing_annotations.items():
        for ann in ann_list:
            if (species_id, ann) not in seen_pairs:
                if database == "chebi":
                    candidate_display = f"CHEBI:{ann}"
                elif database == "ncbigene":
                    candidate_display = f"NCBIGENE:{ann}"
                else:
                    candidate_display = ann
                row = {
                    'file': filename,
                    'type': entity_type,
                    'id': species_id,
                    'display_name': model_info["display_names"].get(species_id, species_id),
                    'annotation': candidate_display,
                    'annotation_label': '',
                    'match_score': None,
                    'existing': 1,
                    'update_annotation': 'keep'
                }
                rows.append(row)
    return pd.DataFrame(rows)

def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    
    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        Dictionary with metrics
    """
    if recommendations_df.empty:
        return {
            'total_entities': 0,
            'entities_with_predictions': 0,
            'annotation_rate': 0.0,
            'total_predictions': 0,
            'matches': 0,
            'accuracy': 0.0,
            'total_time': total_time,
            'llm_time': llm_time,
            'search_time': search_time
        }
    
    total_entities = len(existing_annotations)
    entities_with_predictions = recommendations_df[recommendations_df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / total_entities if total_entities > 0 else 0
    
    # Calculate accuracy based on existing annotations
    total_predictions = len(recommendations_df[recommendations_df['annotation'] != ''])
    matches = len(recommendations_df[recommendations_df['existing'] == 1])
    accuracy = matches / entities_with_predictions if entities_with_predictions > 0 else 0
    
    return {
        'total_entities': total_entities,
        'entities_with_predictions': entities_with_predictions,
        'annotation_rate': annotation_rate,
        'total_predictions': total_predictions,
        'matches': matches,
        'accuracy': accuracy,
        'total_time': total_time,
        'llm_time': llm_time,
        'search_time': search_time
    }

def print_results(results_df: pd.DataFrame):
    """
    Print evaluation results summary.
    
    Args:
        results_df: DataFrame with evaluation results
    """
    if results_df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % results_df['model'].nunique())
    print("Number of models with predictions: %d" % results_df[results_df['annotation'] != '']['model'].nunique())
    
    # Calculate per-model averages
    model_accuracy = results_df.groupby('model')['existing'].mean().mean()
    print("Average accuracy (per model): %.02f" % model_accuracy)
    
    mean_processing_time = results_df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)
    
    num_elements = results_df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements
    print("Ave. total time (per element, per model): %.02f" % mean_processing_time_per_element)
    
    # LLM time
    mean_llm_time = results_df.groupby('model')['llm_time'].first().mean()
    print("Ave. LLM time (per model): %.02f" % mean_llm_time)
    
    mean_llm_time_per_element = mean_llm_time / num_elements
    print("Ave. LLM time (per element, per model): %.02f" % mean_llm_time_per_element)
    
    # Average number of predictions per species
    average_predictions = results_df[results_df['annotation'] != ''].groupby('model').size().mean()
    print(f"Average number of predictions per model: {average_predictions}")

# Main interface function for users
def curate_model(model_file: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main interface function for curating a single model.
    
    This is the primary function users should call for models with existing annotations.
    
    Args:
        model_file: Path to SBML model file
        **kwargs: Additional arguments passed to curate_single_model
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
    """
    return curate_single_model(model_file, **kwargs) 