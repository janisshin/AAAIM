"""
Annotation Workflow for AAAIM

Main interface for annotating a single model that has no or limited existing annotations.
Provides the primary function that users will call to get recommendation tables
for all species in a model.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import numpy as np
import re
from collections import Counter
import itertools
import warnings
from core.model_info import find_species_with_chebi_annotations, find_species_with_annotations_and_qualifiers, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, find_reactions_with_kegg_annotations, extract_model_info, format_prompt, get_all_species_ids
from core.model_info import get_all_reaction_ids
from core.llm_interface import get_system_prompt, query_llm, parse_llm_response
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag, load_chebi_label_dict, load_ncbigene_label_dict, load_uniprot_label_dict, load_kegg_label_dict
from core.database_search import cancel_spectators

logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning noise (e.g., concat dtype changes)
warnings.filterwarnings("ignore", category=FutureWarning)



def annotate_single_model(model_file: str, 
                  llm_model: str = "Llama-3.3-70B-Instruct",
                  method: str = "direct",
                  top_k: int = 3,
                  max_entities: int = None,
                  entity_type: str = "chemical",
                  database: str = "chebi",
                  tax_id: str = None,
                  chunk_size: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Annotate a single model that has no or limited existing annotations.
    
    This is the main function users will call to get annotation recommendations
    for all species in a model, regardless of existing annotations.
    
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
    
    logger.info(f"Starting annotation for model: {model_file}")
    logger.info(f"Using LLM model: {llm_model}")
    logger.info(f"Using method: {method} for database search")
    logger.info(f"Entity type: {entity_type}, Database: {database}")
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")
    
    if entity_type=='reaction':
        # Step 1: Get reactions from model
        logger.info(">>>Step 1: Getting reactions from model...<<<")
        all_entity_ids = get_all_reaction_ids(model_file)

        if not all_entity_ids:
            logger.warning("No reactions found in model")
            return pd.DataFrame(), {"error": "No reactions found in model"}
        
        logger.info(f"Found {len(all_entity_ids)} reactions in model")
    else:
        # Step 1: Get species from model
        logger.info(">>>Step 1: Getting species from model...<<<")
        all_entity_ids = get_all_species_ids(model_file, entity_type)
        
        if not all_entity_ids:
            logger.warning("No species found in model")
            return pd.DataFrame(), {"error": "No species found in model"}
        
        logger.info(f"Found {len(all_entity_ids)} species in model")
    
    # Check for existing annotations (for metrics calculation)
    existing_annotations = {}
    qualifier_annotations = {}
    if entity_type == "chemical" and database == "chebi":
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "chebi")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "gene" and database == "ncbigene":
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "ncbigene")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "protein" and database == "uniprot":
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "uniprot")
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == "reaction" and database == "kegg":
        existing_annotations, qualifier_annotations = find_reactions_with_kegg_annotations(model_file)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    else:
        # Future: support other entity types and databases
        logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
    
    if max_entities:
        entities_to_evaluate = all_entity_ids[:max_entities]
        logger.info(f"Selected {max_entities} entities for annotation")
    else:
        entities_to_evaluate = all_entity_ids
        logger.info(f"Annotate all {len(entities_to_evaluate)} entities")
        
    # Step 2: Extract model context
    logger.info(">>>Step 2: Extracting model context...<<<")

    model_info = extract_model_info(model_file, entities_to_evaluate, entity_type)
    
    if not model_info:
        logger.error("Failed to extract model context")
        return pd.DataFrame(), {"error": "Failed to extract model context"}
    
    logger.info(f"Extracted context for model: {model_info['model_name']}")
    
    # Format prompt for LLM
    logger.info(f">>>Step 3: Querying LLM ({llm_model})...<<<")
    
    # Track conversation context for potential feedback rounds
    all_prompts = []
    all_responses = []
    system_prompt = get_system_prompt(entity_type)

    if chunk_size and len(entities_to_evaluate) > chunk_size:
        logger.info(f"Breaking {len(entities_to_evaluate)} entities into chunks of {chunk_size}")
        
        # Break down large models into chunks
        species_chunks = []
        for i in range(0, len(entities_to_evaluate), chunk_size):
            chunk = entities_to_evaluate[i:i + chunk_size]
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
            
            all_prompts.append(prompt)
            
            llm_start = time.time()
            try:
                result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
                chunk_llm_time = time.time() - llm_start
                total_llm_time += chunk_llm_time
                
                if not result:
                    logger.error(f"No response from LLM for chunk {chunk_idx + 1}")
                    continue
                
                all_responses.append(result)
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
        prompt = format_prompt(model_file, entities_to_evaluate, entity_type, top_k)
        
        if not prompt:
            logger.error("Failed to format prompt")
            return pd.DataFrame(), {"error": "Failed to format prompt"}
        
        all_prompts.append(prompt)
        
        llm_start = time.time()
        try:
            result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
            llm_time = time.time() - llm_start
            
            if not result:
                logger.error("No response from LLM")
                return pd.DataFrame(), {"error": "No response from LLM"}
            
            all_responses.append(result)
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

    if reason:
        print(f"LLM Reason: {reason}")

    # Step 4: Search database
    logger.info(f">>>Step 4: Searching {database} database...<<<")
    search_start = time.time()
    
    if database == "chebi":
        if method == "direct":
            recommendations = get_species_recommendations_direct(entities_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(entities_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == "ncbigene":
        if method == "direct":
            recommendations = get_species_recommendations_direct(entities_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(entities_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == "uniprot":
        if method == "direct":
            recommendations = get_species_recommendations_direct(entities_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(entities_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == "kegg":
        if method == "direct":
            recommendations = get_species_recommendations_direct(entities_to_evaluate, synonyms_dict, database="kegg", top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(entities_to_evaluate, synonyms_dict, database="kegg")
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    else:
        logger.error(f"Database {database} not yet supported")
        return pd.DataFrame(), {"error": f"Database {database} not yet supported"}

    search_time = time.time() - search_start
    logger.info(f"Database search completed in {search_time:.2f}s")
    
    # Generate recommendation table
    logger.info(">>>Step 5: Generating recommendation table...<<<")
    recommendations_df = _generate_recommendation_table(
        model_file, recommendations, existing_annotations, model_info, entity_type, database, qualifier_annotations,
        synonyms_dict=synonyms_dict, reason=reason
    )
    
    # Step 10: Calculate metrics
    total_time = time.time() - start_time
    
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, max_entities, len(all_entity_ids), total_time, llm_time, search_time
    )

    csv_path = f"{Path(model_file).name}_recommendations.csv"
    recommendations_df.to_csv(csv_path, index=False)
    print(f"Recommendations saved to {csv_path}")
    logger.info(f"Annotation completed in {total_time:.2f}s – {len(recommendations_df)} recommendations")

    from core.feedback import AnnotationResult, build_initial_conversation
    combined_prompt = "\n\n".join(all_prompts)
    combined_response = "\n\n".join(all_responses)

    return AnnotationResult(
        recommendations_df, metrics,
        model_file=model_file,
        conversation_history=build_initial_conversation(system_prompt, combined_prompt, combined_response),
        entities_to_evaluate=entities_to_evaluate,
        entity_type=entity_type,
        database=database,
        method=method,
        llm_model=llm_model,
        top_k=top_k,
        tax_id=tax_id,
        existing_annotations=existing_annotations,
        qualifier_annotations=qualifier_annotations,
        model_info=model_info,
        csv_path=csv_path,
    )

def _generate_recommendation_table(model_file: str, 
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any],
                                 entity_type: str = "chemical",
                                 database: str = "chebi",
                                 qualifier_annotations: Dict[str, List[str]] = None,
                                 synonyms_dict: Dict[str, List[str]] = None,
                                 reason: str = "") -> pd.DataFrame:
    """
    Generate AMAS-compatible recommendation table.
    
    Args:
        model_file: Path to model file
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations (may be empty)
        model_info: Model information dictionary
        entity_type: Type of entity being annotated
        database: Database being used for search
        qualifier_annotations: Dictionary of qualifier annotations
        synonyms_dict: Dictionary mapping species IDs to LLM-suggested synonyms
        reason: LLM reasoning text

    Returns:
        DataFrame in AMAS format
    """
    rows = []
    filename = Path(model_file).name
    if synonyms_dict is None:
        synonyms_dict = {}
    if qualifier_annotations is None:
        qualifier_annotations = {}

    seen_pairs = set()

    for rec in recommendations:
        curated_name = synonyms_dict.get(rec.id, [""])[0]

        if not rec.candidates:
            if rec.id in qualifier_annotations and qualifier_annotations[rec.id]:
                all_qualifiers = list(qualifier_annotations[rec.id].values())
                specific_qualifier = ', '.join(all_qualifiers) if all_qualifiers else 'is'
            else:
                specific_qualifier = 'is'

            if database == "chebi":
                lbl_dict = load_chebi_label_dict()
            elif database == "ncbigene":
                lbl_dict = load_ncbigene_label_dict()
            elif database == "uniprot":
                lbl_dict = load_uniprot_label_dict()
            elif database == "kegg":
                lbl_dict = load_kegg_label_dict()
            else:
                lbl_dict = {}
            label = lbl_dict.get(rec.id, rec.id)

            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': '',
                'annotation_label': label,
                'match_score': 0.0,
                'status': '',
                'update_annotation': 'ignore',
                'qualifier': specific_qualifier
            }
            rows.append(row)
            continue
        
        for i, candidate in enumerate(rec.candidates):
            candidate_display = f"{database.upper()}:{candidate}"
            is_existing = candidate in existing_annotations.get(rec.id, [])
            match_score = rec.match_score[i]

            if is_existing:
                status = 'original and predicted'
                update_action = 'keep'
            else:
                status = 'predicted only'
                if i == 0 and match_score > 0.5:
                    update_action = 'add'
                else:
                    update_action = 'ignore'

            if is_existing and qualifier_annotations:
                specific_qualifier = qualifier_annotations.get(rec.id, {}).get(candidate, 'is')
            else:
                specific_qualifier = 'is'
            
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': candidate_display,
                'annotation_label': rec.candidate_names[i],
                'match_score': match_score,
                'status': status,
                'update_annotation': update_action,
                'qualifier': specific_qualifier
            }
            rows.append(row)
            seen_pairs.add((rec.id, candidate))

    # Add rows for existing annotations not predicted
    if existing_annotations:
        if database == "chebi":
            lbl_dict = load_chebi_label_dict()
        elif database == "ncbigene":
            lbl_dict = load_ncbigene_label_dict()
        elif database == "uniprot":
            lbl_dict = load_uniprot_label_dict()
        elif database == "kegg":
            lbl_dict = load_kegg_label_dict()
        else:
            lbl_dict = {}

        for species_id, ann_list in existing_annotations.items():
            for ann in ann_list:
                if (species_id, ann) not in seen_pairs:
                    candidate_display = f"{database.upper()}:{ann}"
                    curated_name = synonyms_dict.get(species_id, [""])[0]

                    if qualifier_annotations:
                        specific_qualifier = qualifier_annotations.get(species_id, {}).get(ann, 'is')
                    else:
                        specific_qualifier = 'is'

                    label = lbl_dict.get(ann, ann)

                    row = {
                        'file': filename,
                        'type': entity_type,
                        'id': species_id,
                        'display_name': model_info["display_names"].get(species_id, species_id),
                        'curated_name': curated_name,
                        'annotation': candidate_display,
                        'annotation_label': label,
                        'match_score': None,
                        'status': 'original only',
                        'update_annotation': 'keep',
                        'qualifier': specific_qualifier
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty and 'id' in df.columns:
        status_order = {'original and predicted': 0, 'original only': 1, 'predicted only': 2, '': 3}
        df['_status_order'] = df['status'].map(status_order).fillna(3)
        df = df.sort_values(by=['id', '_status_order']).reset_index(drop=True)
        df = df.drop(columns=['_status_order'])

    if reason:
        reason_row = pd.DataFrame([{
            'file': filename, 'type': '', 'id': 'Reason:',
            'display_name': reason, 'curated_name': '',
            'annotation': '', 'annotation_label': '',
            'match_score': None, 'status': '',
            'update_annotation': '', 'qualifier': ''
        }])
        if df.empty:
            df = reason_row
        else:
            reason_row = reason_row.reindex(columns=df.columns)
            df = pd.concat([reason_row, df], ignore_index=True)

    return df

def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      max_entities: int,
                      total_species: int,
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for annotation workflow.
    
    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations (may be empty)
        max_entities: Maximum number of entities to annotate (None for all)
        total_species: Total number of species in the model
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        Dictionary with metrics
    """
    if recommendations_df.empty:
        return {
            'total_entities': max_entities,
            'entities_with_predictions': 0,
            'annotation_rate': 0.0,
            'total_predictions': 0,
            'matches': 0,
            'accuracy': np.nan,
            'total_time': total_time,
            'llm_time': llm_time,
            'search_time': search_time
        }
    
    if max_entities is None:
        max_entities = total_species

    # Filter out Reason row for metrics calculation
    df = recommendations_df[recommendations_df['id'] != 'Reason:'] if not recommendations_df.empty else recommendations_df

    entities_with_predictions = df[df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / max_entities if max_entities > 0 else np.nan
    
    # Calculate accuracy based on existing annotations
    total_predictions = len(df[df['annotation'] != ''])
    matches = len(df[df['status'] == 'original and predicted'])
    
    # Accuracy = matches / entities with existing annotations
    entities_with_existing = len(existing_annotations)
    if not existing_annotations:
        accuracy = np.nan
    else:
        accuracy = matches / entities_with_existing if entities_with_existing > 0 else np.nan
    
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
    Adapted from AMAS test_LLM_synonyms_plain.ipynb for annotation workflow
    
    Args:
        results_df: DataFrame with evaluation results
    """
    if results_df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % results_df['model'].nunique())
    print("Number of models with predictions: %d" % results_df[results_df['annotation'] != '']['model'].nunique())
    
    # Calculate per-model averages - handle NaN accuracy values
    results_df = results_df[results_df['id'] != 'Reason:'].copy()
    results_df['_is_match'] = (results_df['status'] == 'original and predicted').astype(int)
    model_accuracies = results_df.groupby('model')['_is_match'].mean()
    valid_accuracies = model_accuracies[~pd.isna(model_accuracies)]
    
    if len(valid_accuracies) > 0:
        print("Average accuracy (per model, where existing annotations available): %.02f" % valid_accuracies.mean())
    else:
        print("Average accuracy: N/A (no existing annotations)")
    
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

def normalize_reactions(model_reactions, cofactors_to_ignore):
    """
    Normalize reaction data for comparison by filtering out common cofactors
    and tracking stoichiometry.
    
    Args:
        model_reactions: List of reaction dictionaries
        cofactors_to_ignore: Set of cofactor IDs to ignore
        
    Returns:
        List of normalized reaction dictionaries
    """
    normalized_reactions = []
    
    for rxn in model_reactions:
        subs = filter_and_count(rxn.get('substrates', []), cofactors_to_ignore)
        prods = filter_and_count(rxn.get('products', []), cofactors_to_ignore)
        
        normalized_reactions.append({
            'reaction_name_in_model': rxn.get('id', 'Unknown'),
            'substrate_counter': subs,
            'product_counter': prods,
        })
                
    return normalized_reactions
            
def filter_and_count(kegg_list, cofactors_to_ignore):
    """
    Filter out cofactors and count occurrences of each metabolite.
    
    Args:
        kegg_list: List of KEGG IDs
        cofactors_to_ignore: Set of cofactor IDs to ignore
        
    Returns:
        Counter object with metabolite counts
    """
    counter = Counter()

    for kegg_id in kegg_list:
        if kegg_id is None:
            continue  # skip unmapped
        if kegg_id: # not in cofactors_to_ignore:
            counter[kegg_id] += 1  # track stoichiometry
    return counter      
    

def map_reactions_to_kegg(rxn_list: List[str], id_df: pd.DataFrame, spectators=False) -> List[Dict[str, Any]]:
    """
    Map reaction strings to KEGG reaction identifiers.
    
    This function processes a list of reaction strings and maps the metabolites
    in each reaction to their corresponding KEGG IDs using the provided mapping DataFrame.
    
    Args:
        rxn_list: List of reaction strings in the format "id: reactants -> products"
        id_df: DataFrame with columns 'id' and 'KEGG_ID' mapping metabolite IDs to KEGG IDs
        
    Returns:
        List of dictionaries containing mapped reaction information:
        - id: Reaction identifier
        - reaction_string: Original reaction string
        - substrates: List of Counter objects with mapped substrate KEGG IDs and stoichiometry
        - products: List of Counter objects with mapped product KEGG IDs and stoichiometry
    """
    # Create lookup table from DataFrame for faster mapping
    id_lookup = id_df.set_index('id')['KEGG_ID']
    
    def parse_reaction_equation(rxn_str: str) -> Tuple[Counter, Counter]:
        """
        Parse a reaction equation string into reactants and products.
        
        Args:
            rxn_str: Reaction equation string (e.g., "A + 2 B -> C + D")
            
        Returns:
            Tuple of (reactants_counter, products_counter) where each counter
            maps metabolite IDs to their stoichiometric coefficients
        """
        # Split reaction string into left-hand side (reactants) and right-hand side (products)
        if "=>" in rxn_str or "->" in rxn_str:
            lhs, rhs = re.split(r"=>|->", rxn_str)
        else:
            return Counter(), Counter()
        
        def parse_metabolites(side: str) -> Counter:
            """
            Parse one side of a reaction equation into a Counter of metabolites.
            
            Args:
                side: String representing one side of a reaction equation
                
            Returns:
                Counter mapping metabolite IDs to their stoichiometric coefficients
            """
            side = side.strip()
            if not side:  # Empty or all whitespace
                return Counter()
            
            result = Counter()
            # Process each metabolite term (separated by +)
            for term in side.split("+"):
                parts = term.strip().split()
                
                if len(parts) == 1:
                    # No explicit coefficient (assumed to be 1)
                    coeff = 1
                    met = parts[0]
                else:
                    # First part is coefficient, last part is metabolite ID
                    try:
                        coeff = float(parts[0])
                    except ValueError:
                        # If conversion fails, assume coefficient is 1
                        coeff = 1
                        met = term.strip()
                    else:
                        met = parts[-1]
                
                # Remove $ prefix if present (sometimes used in model IDs)
                met = met.lstrip('$')
                result[met] += coeff
                
            return result
        
        reactants = parse_metabolites(lhs)
        products = parse_metabolites(rhs)
        return reactants, products
    
    def map_metabolites_to_kegg(counter: Counter, mapping_df: pd.Series) -> List[Counter]:
        """
        Map metabolite IDs to KEGG IDs while preserving stoichiometry.
        
        For each metabolite in the counter, finds all possible KEGG IDs and
        generates all possible combinations of mappings.
        
        Args:
            counter: Counter mapping metabolite IDs to stoichiometric coefficients
            mapping_df: Series mapping metabolite IDs to KEGG IDs
            
        Returns:
            List of Counter objects representing all possible KEGG ID mappings
        """
        # For each metabolite in the counter, get possible KEGG IDs
        id_choices = []
        
        for met, coeff in counter.items():
            # Try to find KEGG IDs for this metabolite
            try:
                kegg_ids = mapping_df.loc[met]
                
                if isinstance(kegg_ids, pd.Series) or len(kegg_ids) > 1:
                    # Multiple KEGG IDs for this metabolite
                    choices = [(kid[0], coeff) for kid in kegg_ids.tolist()]
                else:
                    # Single KEGG ID
                    choices = [(kegg_ids[0], coeff)]
                    
                id_choices.append(choices)
            except (KeyError, IndexError):
                # Metabolite not found in mapping, skip it
                logger.debug(f"No KEGG mapping found for metabolite: {met}")
                continue
        
        if not id_choices:
            return []
            
        # Generate all possible combinations of KEGG IDs
        counters = []
        for combo in itertools.product(*id_choices):
            counters.append(Counter(dict(combo)))
            
        return counters

    # Process each reaction
    output = []
    
    # Extract reaction IDs from reaction strings
    rxn_ids = [rxn.split(":", 1)[0] if ":" in rxn else rxn for rxn in rxn_list]
    
    for idx, rxn in enumerate(rxn_list):
        # Extract reaction string (remove ID prefix if present)
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn

        # Parse reaction equation into reactants and products
        reactants, products = parse_reaction_equation(rxn_str)

        if not spectators: 
            # Stoichiometric cancellation -- eliminate specatators
            reactants, products = cancel_spectators(reactants, products)

        # Map metabolite IDs to KEGG IDs
        substrates_mapped = map_metabolites_to_kegg(reactants, id_lookup)
        products_mapped = map_metabolites_to_kegg(products, id_lookup)

        # Store mapped reaction
        output.append({
            "id": rxn_ids[idx],
            "reaction_string": rxn_str,
            "substrates": substrates_mapped,
            "products": products_mapped
        })
    
    return output

# Main interface function for users
def annotate_model(model_file: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main interface function for annotating a single model.
    
    This is the primary function users should call for models without existing annotations.
    
    Args:
        model_file: Path to SBML model file
        **kwargs: Additional arguments passed to annotate_single_model
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
    """
    return annotate_single_model(model_file, **kwargs) 