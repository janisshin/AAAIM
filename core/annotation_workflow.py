"""
Annotation Workflow for AAAIM

Main interface for annotating a single model that has no or limited existing annotations.
Provides the primary function that users will call to get recommendation tables
for all species in a model.
"""

import time
import pandas as pd
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from pathlib import Path
import logging
import numpy as np
import re
from collections import Counter
import itertools
from core.model_info import find_species_with_chebi_annotations, find_species_with_annotations_and_qualifiers, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, find_reactions_with_kegg_annotations, extract_model_info, format_prompt, get_all_species_ids
from core.model_info import get_all_reaction_ids
from core.llm_interface import get_system_prompt, query_llm, parse_llm_response
from core.data_types import Recommendation
from core.database_search import (
    cancel_spectators,
    extract_classifications,
    get_species_recommendations_direct,
    get_species_recommendations_rag,
    load_chebi2kegg_dict,
    load_chebi_label_dict,
    load_kegg_label_dict,
    load_ncbigene_label_dict,
    load_uniprot_label_dict,
    _get_kegg_recommendations_rulebased,
    score_model_against_kegg_reaction,
)
from core.hierarchy_relaxation import (
    build_kegg_mapping_dataframe,
    iter_chebi_for_species,
    load_chebi_child_map,
    load_chebi_parent_map,
    merge_chebi_to_kegg_mapping,
    normalize_chebi,
    select_relaxations_by_global_improvement,
    select_metabolites_to_relax,
    should_continue_iteration,
    unified_reaction_objective,
)


logger = logging.getLogger(__name__)


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
        prompt = format_prompt(model_file, entities_to_evaluate, entity_type, top_k)
        
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
            reaction_definitions = [i.split(':')[1] for i in model_info['reactions']]
            reaction_participants = [extract_classifications(i, 'definition') for i in reaction_definitions]
            recommendations = get_species_recommendations_rag(entities_to_evaluate, synonyms_dict, database="kegg", reaction_participants=reaction_participants)
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
        model_file, recommendations, existing_annotations, model_info, entity_type, database, qualifier_annotations
    )
    
    # Step 10: Calculate metrics
    total_time = time.time() - start_time
    
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, max_entities, len(all_entity_ids), total_time, llm_time, search_time
    )
        
    logger.info(f"Annotation completed in {total_time:.2f}s")
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
        recommendations: List of Recommendation or ReactionRecommendation objects
        existing_annotations: Dictionary of existing annotations (may be empty)
        model_info: Model information dictionary
        entity_type: Type of entity being annotated
        database: Database being used for search
        qualifier_annotations: Dictionary of qualifier annotations

    Returns:
        DataFrame in AMAS format
    """
    from core.data_types import ReactionRecommendation
    
    rows = []
    filename = Path(model_file).name
    
    for rec in recommendations:
        if not rec.candidates:
            # No candidates found
            # Get qualifier information for this species
            # For existing annotations, show the qualifier used
            # For new predictions, show 'is' as default
            if rec.id in qualifier_annotations and qualifier_annotations[rec.id]:
                # Get all qualifiers for this species
                all_qualifiers = list(qualifier_annotations[rec.id].values())
                specific_qualifier = ', '.join(all_qualifiers) if all_qualifiers else 'is'
            else:
                specific_qualifier = 'is'  # Default for new predictions

            # get the label for existing annotation
            if database == "chebi":
                dict = load_chebi_label_dict()
            elif database == "ncbigene":
                dict = load_ncbigene_label_dict()
            elif database == "uniprot":
                dict = load_uniprot_label_dict()
            elif database == "kegg":
                dict = load_kegg_label_dict()
            if rec.id in dict:
                label = dict[rec.id]
            else:
                logger.warning(f"Annotation {rec.id} not found in {database} label dictionary")
                label = rec.id

            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'annotation': '',
                'annotation_label': label,
                'match_score': 0.0,
                'existing': 0,
                'update_annotation': 'ignore',
                'qualifier': specific_qualifier
            }
            rows.append(row)
            continue
        
        # Add row for each candidate
        for i, candidate in enumerate(rec.candidates):
            candidate_display = f"{database.upper()}:{candidate}"

            # Determine if this is an existing annotation
            existing = 1 if candidate in existing_annotations.get(rec.id, []) else 0
            
            # match score - handle potential index out of range
            match_score = rec.match_score[i] if i < len(rec.match_score) else 0.0
            
            # Determine update action - for new annotations, suggest adding top candidates
            if existing:
                update_action = 'keep'
            elif i == 0 and match_score > 0.5:  # Top candidate with good score
                update_action = 'add'
            else:
                update_action = 'ignore'
            
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
                'annotation_label': rec.candidate_names[i] if i < len(rec.candidate_names) else candidate,
                'match_score': match_score,
                'existing': existing,
                'update_annotation': update_action,
                'qualifier': specific_qualifier
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)

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
    
    entities_with_predictions = recommendations_df[recommendations_df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / max_entities if max_entities > 0 else np.nan
    
    # Calculate accuracy based on existing annotations
    total_predictions = len(recommendations_df[recommendations_df['annotation'] != ''])
    matches = len(recommendations_df[recommendations_df['existing'] == 1])
    
    # If no existing annotations, accuracy is NA
    if not existing_annotations:
        accuracy = np.nan
    else:
        accuracy = matches / max_entities if max_entities > 0 else np.nan
    
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
    model_accuracies = results_df.groupby('model')['existing'].mean()
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
    Prepare KEGG-mapped reaction dicts for comparison: drop cofactors and keep
    substrate/product multisets (Counters).

    **Not** ``hierarchy_relaxation.normalize_reaction``: that function maps
    ChEBI terms to (relaxed) KEGG compound ID sets using the ontology. This
    function assumes ``model_reactions`` already carry KEGG IDs in substrate /
    product lists.

    Args:
        model_reactions: List of reaction dictionaries
        cofactors_to_ignore: Set of cofactor IDs to ignore
        
    Returns:
        List of normalized reaction dictionaries

    See Also:
        ``hierarchy_relaxation.normalize_reaction`` — ChEBI → KEGG compound sets.
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


def parse_reaction_equation(rxn_str: str) -> Tuple[Counter, Counter]:
    """
    Parse a reaction equation string into reactant and product metabolite Counters.

    Same rules as ``map_reactions_to_kegg`` (``+`` terms, optional stoichiometry).
    """
    def parse_metabolites(side: str) -> Counter:
        side = side.strip()
        if not side:
            return Counter()
        result = Counter()
        for term in side.split("+"):
            parts = term.strip().split()
            if len(parts) == 1:
                coeff = 1
                met = parts[0]
            else:
                try:
                    coeff = float(parts[0])
                except ValueError:
                    coeff = 1
                    met = term.strip()
                else:
                    met = parts[-1]
            met = met.lstrip("$")
            result[met] += coeff
        return result

    if "=>" in rxn_str or "->" in rxn_str:
        lhs, rhs = re.split(r"=>|->", rxn_str)
    else:
        return Counter(), Counter()

    reactants = parse_metabolites(lhs)
    products = parse_metabolites(rhs)
    return reactants, products


def collect_species_ids_from_rxn_list(rxn_list: List[str], spectators: bool = False) -> Set[str]:
    """All metabolite species IDs appearing in ``rxn_list`` after optional spectator cancellation."""
    out: Set[str] = set()
    for rxn in rxn_list:
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn
        reactants, products = parse_reaction_equation(rxn_str)
        if not spectators:
            reactants, products = cancel_spectators(reactants, products)
        out.update(reactants.keys())
        out.update(products.keys())
    return out


def map_reactions_to_kegg(rxn_list: List[str], reaction_ids: List[str], id_df: pd.DataFrame, spectators=False) -> List[Dict[str, Any]]:
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
    def map_metabolites_to_kegg(counter: Counter, mapping_df: pd.DataFrame) -> List[Counter]:
        """
        Map metabolite IDs to KEGG IDs while preserving stoichiometry.
        
        For each metabolite in the counter, finds all possible KEGG IDs and
        generates all possible combinations of mappings.
        
        Args:
            counter: Counter mapping metabolite IDs to stoichiometric coefficients
            mapping_df: DataFrame mapping metabolite IDs to KEGG IDs (+ optional metadata)
            
        Returns:
            List of Counter objects representing all possible KEGG ID mappings
        """
        # For each metabolite in the counter, get possible KEGG IDs
        id_choices = dict()
        for met, coeff in counter.items():
            # Try to find KEGG IDs for this metabolite
            try:
                met_rows = mapping_df[mapping_df["id"] == met]
                if met_rows.empty:
                    raise KeyError(met)

                if "direction" in met_rows.columns and "distance" in met_rows.columns:
                    choices = []
                    for _, row in met_rows.iterrows():
                        kid = row.get("KEGG_ID")
                        if pd.isna(kid) or str(kid).strip() == "":
                            continue
                        choices.append(
                            {
                                "kegg_id": str(kid).strip(),
                                "canonical_id": str(row.get("canonical_id", "")).strip(),
                                "direction": str(row.get("direction", "exact")).strip(),
                                "distance": int(row.get("distance", 0) or 0),
                            }
                        )
                else:
                    # Backward-compatible path when only id/KEGG_ID are present.
                    choices = [str(kid).strip() for kid in met_rows["KEGG_ID"].tolist() if str(kid).strip()]
                
                id_choices[met] = {
                    'species_id': met,
                    'coeff': coeff,
                    'candidates': choices
                }
                # id_choices.append(choices)
            except (KeyError, IndexError):
                # Metabolite not found in mapping, skip it
                logger.debug(f"No KEGG mapping found for metabolite: {met}")
                continue
        
        if not id_choices:
            return []
                       
        return id_choices

    # Keep full rows so we can propagate relaxation metadata when available.
    id_lookup = id_df.copy()

    # Process each reaction
    output = []
       
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

        # Map metabolite CHEBI IDs to KEGG IDs
        substrates_mapped = map_metabolites_to_kegg(reactants, id_lookup)
        products_mapped = map_metabolites_to_kegg(products, id_lookup)

        # Store mapped reaction
        output.append({
            "id": reaction_ids[idx],
            "reaction_string": rxn_str,
            "substrates": substrates_mapped,
            "products": products_mapped
        })
    
    return output


def _participant_species_from_normalized_reaction(nr: Dict[str, Any]) -> Set[str]:
    s: Set[str] = set()
    for side in ("substrates", "products"):
        block = nr.get(side, {})
        if isinstance(block, dict):
            s.update(block.keys())
    return s


def _aggregate_best_penalized_scores(match_results: List[Any]) -> float:
    """
    Mean penalized score over score-eligible reactions:
    - include mappable reactions by best penalized match
    - include ambiguous_mapping reactions with default low score
    - exclude non_mappable reactions
    """
    best_by_rxn: Dict[str, float] = {}
    classification_by_rxn: Dict[str, str] = {}
    ambiguous_default_by_rxn: Dict[str, float] = {}
    for rec in match_results:
        rid = rec.id
        meta = getattr(rec, "metadata", None) or {}
        rtype = str(meta.get("reaction_type", "mappable"))
        classification_by_rxn[rid] = rtype
        if rtype == "ambiguous_mapping":
            ambiguous_default_by_rxn[rid] = float(meta.get("ambiguous_default_score", 0.0))
        if rtype != "mappable":
            continue
        if not rec.match_score:
            continue
        sc = float(rec.match_score[0])
        prev = best_by_rxn.get(rid)
        if prev is None or sc > prev:
            best_by_rxn[rid] = sc
    scored: List[float] = []
    for rid, rtype in classification_by_rxn.items():
        if rtype == "non_mappable":
            continue
        if rtype == "ambiguous_mapping":
            scored.append(float(ambiguous_default_by_rxn.get(rid, 0.0)))
            continue
        if rid in best_by_rxn:
            scored.append(float(best_by_rxn[rid]))
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def _reaction_coverage_stats(match_results: List[Any]) -> Dict[str, Any]:
    """Coverage counts and percentages by reaction classification."""
    reaction_type_by_id: Dict[str, str] = {}
    for rec in match_results:
        rid = rec.id
        meta = getattr(rec, "metadata", None) or {}
        reaction_type_by_id[rid] = str(meta.get("reaction_type", "mappable"))
    total = len(reaction_type_by_id)
    counts = {
        "mappable": 0,
        "ambiguous_mapping": 0,
        "non_mappable": 0,
    }
    for rtype in reaction_type_by_id.values():
        if rtype in counts:
            counts[rtype] += 1
    successful_mapped = counts["mappable"]
    denom = float(total) if total else 1.0
    return {
        "counts": counts,
        "percent_mappable": round(100.0 * counts["mappable"] / denom, 2),
        "percent_successfully_mapped": round(100.0 * max(successful_mapped, 0) / denom, 2),
        "percent_ambiguous_mapping": round(100.0 * counts["ambiguous_mapping"] / denom, 2),
        "percent_non_mappable": round(100.0 * counts["non_mappable"] / denom, 2),
    }


def _top_kegg_reference_from_matches(match_results: List[Any]) -> Dict[str, str]:
    """Best-scoring KEGG reaction id per model reaction id from split recommendations."""
    best: Dict[str, Tuple[str, float]] = {}
    for rec in match_results:
        if not rec.candidates or not rec.match_score:
            continue
        rid = rec.id
        kid = rec.candidates[0]
        sc = float(rec.match_score[0])
        prev = best.get(rid)
        if prev is None or sc > prev[1]:
            best[rid] = (kid, sc)
    return {rid: t[0] for rid, t in best.items()}


def _species_ids_for_chebi_relax_targets(
    chebi_hit: Set[str],
    species_ids: Iterable[str],
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    max_relax_level: int,
) -> Set[str]:
    """Map ChEBI ids from ``select_metabolites_to_relax`` to relaxable model species ids."""
    ch_norm = {str(c).strip() for c in chebi_hit if str(c).strip()}
    out: Set[str] = set()
    for sid in species_ids:
        if int(relax_level.get(sid, 0)) >= max_relax_level:
            continue
        for ch in iter_chebi_for_species(species_to_chebi, str(sid)):
            if str(ch).strip() in ch_norm:
                out.add(sid)
                break
    return out


def _leave_one_out_penalized_matcher_factory(
    nr: Dict[str, Any],
    ref_kegg: str,
    part: Set[str],
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    merged_kegg: Mapping[str, Set[str]],
    parent_map: Mapping[str, Set[str]],
    max_ancestor_depth: int,
    cofactors: Set[str],
    spectators: bool,
    penalty_lam: float,
    max_relax_level: int,
):
    """
    Leave-one-ChEBI-out matcher returning **penalized** objective only (no raw scores
    in control flow that uses this closure).
    """
    reaction_relax_levels = {sid: int(relax_level.get(sid, 0) or 0) for sid in part}

    def reaction_matcher(exclude_chebi: Optional[str]) -> float:
        sub_c = _kegg_counters_from_normalized_block(
            nr.get("substrates"),
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth,
            exclude_chebi,
        )
        prod_c = _kegg_counters_from_normalized_block(
            nr.get("products"),
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth,
            exclude_chebi,
        )
        base = score_model_against_kegg_reaction(
            sub_c,
            prod_c,
            ref_kegg,
            cofactors_to_ignore=cofactors,
            spectators=spectators,
        )[0]
        return unified_reaction_objective(
            base,
            reaction_relax_levels if reaction_relax_levels else None,
            lam=penalty_lam,
            max_relax_level=max_relax_level,
        )

    return reaction_matcher


def _kegg_counters_from_normalized_block(
    block: Any,
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    merged_kegg: Mapping[str, Set[str]],
    parent_map: Mapping[str, Set[str]],
    max_ancestor_depth: int,
    exclude_chebi: Optional[str],
) -> Counter:
    """
    Rebuild KEGG compound counters for one side of a normalized reaction using
    ``normalize_chebi`` at each species' current relaxation level.

    When ``exclude_chebi`` is set, species annotated with that ChEBI are omitted
    (leave-one-ChEBI-out for problematic-metabolite detection).
    """
    ctr: Counter = Counter()
    if not isinstance(block, dict):
        return ctr
    ex = (exclude_chebi or "").strip()
    for met_id, v in block.items():
        if not isinstance(v, dict):
            continue
        coeff = float(v.get("coeff", 1))
        lvl = int(relax_level.get(met_id, 0))
        keggs_union: Set[str] = set()
        for ch in iter_chebi_for_species(species_to_chebi, str(met_id)):
            c = str(ch).strip()
            if not c:
                continue
            if ex and c == ex:
                continue
            keggs_union.update(
                normalize_chebi(
                    c, merged_kegg, parent_map, level=lvl, max_depth=max_ancestor_depth
                )
            )
        for kid in keggs_union:
            if kid:
                ctr[kid] += coeff
    return ctr


def _species_to_chebi_from_recommendations(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build species id → list of distinct ChEBI annotation strings.

    All recommendation rows are retained (no single-row-per-id collapse). Order
    follows ``match_score`` descending when that column exists, so higher-ranked
    candidates appear first in each list.
    """
    if "match_score" in df.columns:
        sorted_df = df.sort_values("match_score", ascending=False)
    else:
        sorted_df = df
    out: Dict[str, List[str]] = {}
    for _, row in sorted_df.iterrows():
        sid = str(row["id"]).strip()
        ann = str(row["annotation"]).strip()
        if not sid or not ann:
            continue
        out.setdefault(sid, [])
        if ann not in out[sid]:
            out[sid].append(ann)
    return out


def map_reactions_to_kegg_with_relaxation(
    rxn_list: List[str],
    reaction_ids: List[str],
    species_recommendations_df: pd.DataFrame,
    *,
    parent_map: Optional[Mapping[str, Set[str]]] = None,
    chebi_to_kegg: Optional[Mapping[str, Any]] = None,
    obo_path: Optional[str] = None,
    parent_map_gz: Optional[str] = None,
    spectators: bool = False,
    max_relax_level: int = 2,
    max_ancestor_depth: int = 2,
    max_descendant_depth: Optional[int] = None,
    score_gain_threshold: float = 0.0,
    score_tolerance: float = 1e-3,
    max_relaxation_rounds: int = 8,
    cofactors_to_ignore: Optional[Set[str]] = None,
    top_k: Optional[int] = None,
    penalty_lam: float = 0.1,
    run_matching: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, int]]:
    """
    Single iterative loop: normalize → penalized KEGG matching → relax targets → converge.

    **Per iteration**

    1. Build ``normalized_reactions`` via ``map_reactions_to_kegg`` (ChEBI→KEGG at
       current ``relax_level`` per species).
    2. ``_get_kegg_recommendations_rulebased`` computes raw similarity internally and
       ranks only on ``unified_reaction_objective``; aggregate ``score`` for control
       flow is the mean best penalized score per model reaction.
    3. ``select_metabolites_to_relax`` (unmapped ∪ score-sensitive) yields ChEBI terms;
       these map back to **species ids** in ``to_relax`` only (no global relaxation).
    4. ``should_continue_iteration`` stops when ``to_relax`` is empty and the penalized
       aggregate is stable vs ``previous_best_score`` (initialized to ``-inf``).
    5. Increment ``relax_level`` only for species in ``to_relax`` (capped at
       ``max_relax_level``).

    Args:
        max_descendant_depth: Maximum downward traversal depth used for
            relaxation-aware ChEBI expansion. If None, defaults to
            ``max_ancestor_depth`` for backward-compatible behavior.
        rxn_list: Reaction strings ``"RID: lhs -> rhs"`` as for map_reactions_to_kegg.
        species_recommendations_df: Must include ``id`` (species) and ``annotation``
            (ChEBI ID). Optional ``match_score`` ranks rows; **all** distinct
            ``annotation`` values per ``id`` are kept (not only the top row).
        parent_map: Optional precomputed child→parents map; otherwise loaded via
            ``load_chebi_parent_map`` (gz or OBO under data/chebi/).
        chebi_to_kegg: Optional raw ChEBI→KEGG dict; defaults to
            ``load_chebi2kegg_dict()``.
        score_gain_threshold: Minimum score increase (leave-one-out minus baseline)
            required to flag a ChEBI term as score-sensitive.
        score_tolerance: Stop when there is nothing to relax and the mean best penalized
            score per reaction changes by less than this vs the previous iteration.
        run_matching: If False, performs a single mapping pass and returns an empty
            match list (no refinement).

    Returns:
        (normalized_reactions, kegg_match_results, species_relax_level_by_id)
    """

    def compute_global_score(levels: Mapping[str, int]) -> float:
        """Evaluate the full-model global objective at the provided relaxation levels."""
        trial_id_kegg_df = build_kegg_mapping_dataframe(
            species_to_chebi,
            levels,
            merged_kegg,
            parent_map,
            max_ancestor_depth=max_ancestor_depth,
            child_map=child_map,
            max_descendant_depth=down_depth,
        )
        trial_normalized_reactions = map_reactions_to_kegg(
            rxn_list, reaction_ids, trial_id_kegg_df, spectators=spectators
        )
        trial_match_results = _get_kegg_recommendations_rulebased(
            trial_normalized_reactions,
            cofactors_to_ignore=cofactors,
            top_k=top_k,
            spectators=spectators,
            relaxation_levels_by_entity=levels,
            penalty_lam=penalty_lam,
            max_relax_level=max_relax_level,
            species_to_chebi=species_to_chebi,
            parent_map=parent_map,
            child_map=child_map,
            chebi_to_kegg=merged_kegg,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=down_depth,
        )
        return float(_aggregate_best_penalized_scores(trial_match_results))


    if species_recommendations_df is None or species_recommendations_df.empty:
        return [], [], {}

    required_cols = {"id", "annotation"}
    if not required_cols.issubset(species_recommendations_df.columns):
        raise ValueError(
            f"species_recommendations_df must contain columns {required_cols}, "
            f"got {set(species_recommendations_df.columns)}"
        )

    df = species_recommendations_df.dropna(subset=["annotation"])
    df = df[df["annotation"].astype(str).str.strip() != ""]
    if df.empty:
        return [], [], {}

    species_to_chebi = _species_to_chebi_from_recommendations(df)

    if parent_map is None:
        parent_map = load_chebi_parent_map(obo_path=obo_path, gz_path=parent_map_gz)
    child_map = load_chebi_child_map(parent_map=parent_map)
    if chebi_to_kegg is None:
        chebi_to_kegg = load_chebi2kegg_dict()

    merged_kegg = merge_chebi_to_kegg_mapping(chebi_to_kegg)
    down_depth = int(max_ancestor_depth) if max_descendant_depth is None else int(max_descendant_depth)

    relax_level: Dict[str, int] = {sid: 0 for sid in species_to_chebi}

    cofactors = cofactors_to_ignore if cofactors_to_ignore is not None else set()
    normalized_reactions: List[Dict[str, Any]] = []
    match_results: List[Any] = []

    max_iterations = 1 if not run_matching else max(1, int(max_relaxation_rounds))
    previous_best_score: float = float("-inf")

    for _iteration in range(max_iterations):
        # --- Step 1: build normalized reactions ---
        id_kegg_df = build_kegg_mapping_dataframe(
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth=max_ancestor_depth,
            child_map=child_map,
            max_descendant_depth=down_depth,
        )
        normalized_reactions = map_reactions_to_kegg(
            rxn_list, reaction_ids, id_kegg_df, spectators=spectators
        )

        if not run_matching:
            match_results = []
            break

        # --- Step 2: KEGG matching (raw similarity inside matcher; ranking = penalized only) ---
        match_results = _get_kegg_recommendations_rulebased(
            normalized_reactions,
            cofactors_to_ignore=cofactors,
            top_k=top_k,
            spectators=spectators,
            relaxation_levels_by_entity=relax_level,
            penalty_lam=penalty_lam,
            max_relax_level=max_relax_level,
            species_to_chebi=species_to_chebi,
            parent_map=parent_map,
            child_map=child_map,
            chebi_to_kegg=merged_kegg,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=down_depth,
        )
        score = _aggregate_best_penalized_scores(match_results)
        coverage = _reaction_coverage_stats(match_results)
        logger.info(f"Reaction coverage: {coverage}")

        # --- Step 3: build candidate species (unmapped + optionally problematic) ---
        participants_union: Set[str] = set()
        for nr in normalized_reactions:
            participants_union |= _participant_species_from_normalized_reaction(nr)
        if not participants_union:
            participants_union = collect_species_ids_from_rxn_list(rxn_list, spectators=spectators)
        participants_union &= set(species_to_chebi.keys())

        candidate_species: Set[str] = set()
        top_ref = _top_kegg_reference_from_matches(match_results)
        species_in_any_part: Set[str] = set()

        for nr in normalized_reactions:
            part = _participant_species_from_normalized_reaction(nr) & set(species_to_chebi.keys())
            if not part:
                continue
            species_in_any_part |= part
            chebi_union = sorted(
                {
                    c
                    for s in part
                    for c in iter_chebi_for_species(species_to_chebi, str(s))
                    if c
                }
            )
            if not chebi_union:
                continue

            ref_kegg = top_ref.get(nr.get("id"))
            if ref_kegg:
                matcher = _leave_one_out_penalized_matcher_factory(
                    nr,
                    ref_kegg,
                    part,
                    species_to_chebi,
                    relax_level,
                    merged_kegg,
                    parent_map,
                    max_ancestor_depth,
                    cofactors,
                    spectators,
                    penalty_lam,
                    max_relax_level,
                )
            else:
                matcher = None

            chebi_to_relax = select_metabolites_to_relax(
                chebi_union,
                merged_kegg,
                parent_map,
                matcher,
                score_threshold=score_gain_threshold,
                participant_species=part,
                species_to_chebi=species_to_chebi,
                relax_level=relax_level,
                max_depth=max_ancestor_depth,
            )
            candidate_species |= _species_ids_for_chebi_relax_targets(
                chebi_to_relax,
                part,
                species_to_chebi,
                relax_level,
                max_relax_level,
            )

        orphan = participants_union - species_in_any_part
        if orphan:
            orch_chebi = sorted(
                {
                    c
                    for s in orphan
                    for c in iter_chebi_for_species(species_to_chebi, str(s))
                    if c
                }
            )
            if orch_chebi:
                chebi_orphan = select_metabolites_to_relax(
                    orch_chebi,
                    merged_kegg,
                    parent_map,
                    None,
                    participant_species=orphan,
                    species_to_chebi=species_to_chebi,
                    relax_level=relax_level,
                    max_depth=max_ancestor_depth,
                )
                candidate_species |= _species_ids_for_chebi_relax_targets(
                    chebi_orphan,
                    orphan,
                    species_to_chebi,
                    relax_level,
                    max_relax_level,
                )

        # Global objective gate: relax only species that improve full-model score.
        to_relax: Set[str] = set(
            select_relaxations_by_global_improvement(
                sorted(candidate_species),
                relax_level,
                compute_global_score,
                max_relax_level=max_relax_level,
                delta_threshold=score_gain_threshold,
            )
        )

        # --- Step 4: convergence (penalized score + relaxation state) ---
        if not should_continue_iteration(
            score,
            previous_best_score,
            relax_level,
            to_relax,
            score_tolerance=score_tolerance,
        ):
            previous_best_score = score
            break

        previous_best_score = score

        # --- Step 5: apply relaxation (only entities in to_relax) ---
        for entity in to_relax:
            relax_level[entity] = min(relax_level.get(entity, 0) + 1, max_relax_level)

    return normalized_reactions, match_results, relax_level


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