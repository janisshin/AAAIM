"""
Curation Workflow for AAAIM

Main interface for curating a single model.
Provides the primary function that users will call to get recommendation tables
for models that already have existing annotations.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from core.model_info import find_species_with_chebi_annotations, find_species_with_annotations_and_qualifiers, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, extract_model_info, format_prompt
from core.llm_interface import get_system_prompt, query_llm, parse_llm_response
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag, load_uniprot_label_dict, load_ncbigene_label_dict, load_chebi_label_dict

logger = logging.getLogger(__name__)

def curate_single_model(model_file: str, 
                  llm_model: str = "Llama-3.3-70B-Instruct",
                  method: str = "direct",
                  top_k: int = 3,
                  max_entities: int = None,
                  entity_type: str = "chemical",
                  database: str = "chebi",
                  tax_id: str = None,
                  chunk_size: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    This is the main function users will call to get curation recommendations
    for a model that already has existing annotations.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use ("gpt-4o-mini", "Llama-3.3-70B-Instruct")
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to return per species
        max_entities: Maximum number of entities to annotate (None for all)
        entity_type: Type of entities to annotate ("chemical", "gene", "protein", "auto")
        database: Target database ("chebi", "ncbigene", "uniprot")
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        chunk_size: Size of chunks to split large models into (default: 50, None for no chunking)
        
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
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "chebi")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "gene" and database == "ncbigene":
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "ncbigene")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "protein" and database == "uniprot":
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "uniprot")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    else:
        # Future: support other entity types and databases
        logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
        existing_annotations = {}
        qualifier_annotations = {}
    
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
    
    if chunk_size and len(specs_to_evaluate) > chunk_size:
        logger.info(f"Breaking {len(specs_to_evaluate)} entities into chunks of {chunk_size}")
        
        # Break down large models into chunks
        species_chunks = []
        for i in range(0, len(specs_to_evaluate), chunk_size):
            chunk = specs_to_evaluate[i:i + chunk_size]
            species_chunks.append(chunk)
        
        # Process each chunk and accumulate results
        all_synonyms_dict = {}
        all_reasons = []
        total_llm_time = 0
        
        for chunk_idx, chunk in enumerate(species_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(species_chunks)} ({len(chunk)} entities)")
            
            # Format prompt for this chunk
            prompt = format_prompt(model_file, chunk, entity_type, top_k)
            
            if not prompt:
                logger.error(f"Failed to format prompt for chunk {chunk_idx + 1}")
                continue
            
            llm_start = time.time()
            try:
                # Get appropriate system prompt for entity type
                system_prompt = get_system_prompt(entity_type)
                result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
                chunk_llm_time = time.time() - llm_start
                total_llm_time += chunk_llm_time
                
                if not result:
                    logger.error(f"No response from LLM for chunk {chunk_idx + 1}")
                    continue
                
                logger.info(f"Chunk {chunk_idx + 1} LLM response received in {chunk_llm_time:.2f}s")
                
            except Exception as e:
                logger.error(f"LLM query failed for chunk {chunk_idx + 1}: {e}")
                continue
            
            # Parse LLM response
            chunk_synonyms_dict, chunk_entity_type_dict, chunk_reason = parse_llm_response(result, entity_type)
            
            # Accumulate synonyms
            all_synonyms_dict.update(chunk_synonyms_dict)
            
            # Accumulate reasons
            if chunk_reason:
                all_reasons.append(f"Chunk {chunk_idx + 1}: {chunk_reason}")
        
        # Combine all reasons
        if all_reasons:
            reason = ' '.join(all_reasons)
        else:
            reason = ""
        
        # Use accumulated synonyms
        synonyms_dict = all_synonyms_dict
        llm_time = total_llm_time
        
    else:
        # Single prompt for all entities
        prompt = format_prompt(model_file, specs_to_evaluate, entity_type, top_k)
        
        if not prompt:
            logger.error("Failed to format prompt")
            return pd.DataFrame(), {"error": "Failed to format prompt"}
        
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
        synonyms_dict, entity_type_dict, reason = parse_llm_response(result, entity_type)
    
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
    elif database == "uniprot":
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id)
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
        model_file, recommendations, existing_annotations, model_info, entity_type, database, qualifier_annotations
    )
    
    # Step 9: Calculate metrics
    total_time = time.time() - start_time
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, max_entities, total_time, llm_time, search_time
    )
    
    logger.info(f"Curation completed in {total_time:.2f}s")
    logger.info(f"Generated {len(recommendations_df)} recommendations")
    
    recommendations_df.to_csv(f"{Path(model_file).name}_recommendations.csv", index=False)
    logger.info(f"Recommendations saved to {Path(model_file).name}_recommendations.csv")
    
    return recommendations_df, metrics

def _generate_recommendation_table(model_file: str, 
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any],
                                 entity_type: str = "chemical",
                                 database: str = "chebi",
                                 qualifier_annotations: Dict[str, List[str]] = None) -> pd.DataFrame:
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
            # Get qualifier information for this species
            # For existing annotations, show the qualifier used
            # For new predictions, show 'is' as default
            if qualifier_annotations and rec.id in qualifier_annotations and qualifier_annotations[rec.id]:
                # Get all qualifiers for this species
                all_qualifiers = list(qualifier_annotations[rec.id].values())
                specific_qualifier = ', '.join(all_qualifiers) if all_qualifiers else 'is'
            else:
                specific_qualifier = 'is'  # Default for new predictions
            
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': '',
                'annotation_label': '',
                'match_score': 0.0,
                'existing': 0,
                'update_annotation': 'ignore',
                'qualifier': specific_qualifier
            }
            rows.append(row)
            continue
        for i, candidate in enumerate(rec.candidates):
            candidate_display = f"{database.upper()}:{candidate}"
            existing = 1 if candidate in existing_annotations.get(rec.id, []) else 0
            match_score = rec.match_score[i]
            if existing:
                update_action = 'keep'
            else:
                update_action = 'ignore'
            
            # Get qualifier information for this species
            # One annotation per row: use the specific qualifier for this candidate if it exists; otherwise 'is'
            if existing == 1 and qualifier_annotations:
                specific_qualifier = qualifier_annotations.get(rec.id, {}).get(candidate, 'is')
            else:
                specific_qualifier = 'is'
            
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': candidate_display,
                'annotation_label': rec.candidate_names[i],
                'match_score': match_score,
                'existing': existing,
                'update_annotation': update_action,
                'qualifier': specific_qualifier
            }
            rows.append(row)
            seen_pairs.add((rec.id, candidate))
    # Add rows for existing annotations not predicted
    for species_id, ann_list in existing_annotations.items():
        for ann in ann_list:
            if (species_id, ann) not in seen_pairs:
                candidate_display = f"{database.upper()}:{ann}"
                
                # Get qualifier information for this species
                # For existing annotations, show the qualifier used for this specific annotation
                if qualifier_annotations:
                    specific_qualifier = qualifier_annotations.get(species_id, {}).get(ann, 'is')
                else:
                    specific_qualifier = 'is'
                
                # get the label for existing annotation
                if database == "chebi":
                    dict = load_chebi_label_dict()
                elif database == "ncbigene":
                    dict = load_ncbigene_label_dict()
                elif database == "uniprot":
                    dict = load_uniprot_label_dict()
                if ann in dict:
                    label = dict[ann]
                else:
                    logger.warning(f"Annotation {ann} not found in {database} label dictionary")
                    label = ann
                
                row = {
                    'file': filename,
                    'type': entity_type,
                    'id': species_id,
                    'display_name': model_info["display_names"].get(species_id, species_id),
                    'annotation': candidate_display,
                    'annotation_label': label,
                    'match_score': None,
                    'existing': 1,
                    'update_annotation': 'keep',
                    'qualifier': specific_qualifier
                }
                rows.append(row)
    # Order rows by id
    df = pd.DataFrame(rows)
    if not df.empty and 'id' in df.columns:
        df = df.sort_values(by=['id']).reset_index(drop=True)
    return df

def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      max_entities: int,
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    
    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations
        max_entities: Maximum number of entities to annotate (None for all)
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
    
    if max_entities is None:
        max_entities = len(existing_annotations)
    
    entities_with_predictions = recommendations_df[recommendations_df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / max_entities if max_entities > 0 else np.nan
    
    # Calculate accuracy based on existing annotations
    total_predictions = len(recommendations_df[recommendations_df['annotation'] != ''])
    matches = len(recommendations_df[recommendations_df['existing'] == 1])
    accuracy = matches / entities_with_predictions if entities_with_predictions > 0 else 0
    
    return {
        'total_entities': max_entities,
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