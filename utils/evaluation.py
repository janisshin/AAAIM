"""
Evaluation Utilities for AAAIM

Internal evaluation functions for testing and validation.
Replicates the evaluation workflow from AMAS test_LLM_synonyms_plain.ipynb.
"""

import os
import time
import pandas as pd
import numpy as np
import lzma
import pickle
import re
import warnings
import contextlib
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from core.curation_workflow import curate_model
from core.model_info import find_species_with_chebi_annotations, extract_model_info, format_prompt
from core.llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response
from core.database_search import Recommendation, get_species_recommendations_direct, get_species_recommendations_rag, clear_chromadb_cache

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_outputs(verbose: bool = True):
    """
    Context manager to suppress various outputs when verbose=False.
    """
    if not verbose:
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to devnull for sentence transformers progress bars
        with open(os.devnull, 'w') as devnull:
            try:
                # Set environment variables to suppress progress bars
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
                
                # Suppress tqdm
                import tqdm
                original_tqdm_disable = getattr(tqdm.tqdm, '__init__', None)
                tqdm.tqdm.__init__ = lambda self, *args, **kwargs: original_tqdm_disable(self, *args, **{**kwargs, 'disable': True})
                
                yield
                
            finally:
                # Restore tqdm
                if original_tqdm_disable:
                    tqdm.tqdm.__init__ = original_tqdm_disable
                
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        yield

def _configure_verbosity(verbose: bool = True):
    """
    Configure logging and warning verbosity levels.
    
    Args:
        verbose: If True, show all logging. If False, minimize output.
    """
    if not verbose:
        # Set logging to WARNING level for AAAIM modules
        logging.getLogger('core').setLevel(logging.WARNING)
        logging.getLogger('utils').setLevel(logging.WARNING)
        
        # Suppress HTTP request logs
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        
        # Suppress ChromaDB logs
        logging.getLogger('chromadb').setLevel(logging.WARNING)
        logging.getLogger('sqlite3').setLevel(logging.WARNING)
        
        # Suppress sentence transformers logs
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Suppress ChromaDB telemetry messages
        os.environ.setdefault('ANONYMIZED_TELEMETRY', 'false')
        
        # Set transformers logging
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except ImportError:
            pass
    else:
        # Reset to normal logging
        logging.getLogger('core').setLevel(logging.INFO)
        logging.getLogger('utils').setLevel(logging.INFO)
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('httpcore').setLevel(logging.INFO)
        logging.getLogger('openai').setLevel(logging.INFO)
        logging.getLogger('chromadb').setLevel(logging.INFO)
        logging.getLogger('sentence_transformers').setLevel(logging.INFO)
        
        try:
            import transformers
            transformers.logging.set_verbosity_warning()
        except ImportError:
            pass

# Cache for loaded dictionaries
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_CHEBI_FORMULA_DICT: Optional[Dict[str, str]] = None

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / "chebi2label.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def load_chebi_formula_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to formula dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their formulas
    """
    global _CHEBI_FORMULA_DICT
    
    if _CHEBI_FORMULA_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / "chebi_shortened_formula_comp.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI formula data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_FORMULA_DICT = pickle.load(f)
    
    return _CHEBI_FORMULA_DICT

def get_recall(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate recall metric.
    Replicates tools.getRecall from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Recall value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    species_to_test = ref_keys.intersection(pred_keys)
    recall = {}
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        recall[one_k] = num_intersection / len(set(ref[one_k])) if ref[one_k] else 0
    
    if mean:
        return np.round(np.mean([recall[val] for val in recall.keys()]), 3) if recall else 0.0
    else:
        return {val: np.round(recall[val], 3) for val in recall.keys()}

def get_precision(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate precision metric.
    Replicates tools.getPrecision from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Precision value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    precision = {}
    species_to_test = ref_keys.intersection(pred_keys)
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        num_predicted = len(set(pred[one_k]))
        if num_predicted == 0:
            precision[one_k] = 0.0
        else:
            precision[one_k] = num_intersection / num_predicted
    
    if mean:
        if precision:
            return np.round(np.mean([precision[val] for val in precision.keys()]), 3)
        else:
            return 0.0
    else:
        return {val: np.round(precision[val], 3) for val in precision.keys()}

def get_species_statistics(recommendations: List[Recommendation], 
                          refs_formula: Dict[str, List[str]], 
                          refs_chebi: Dict[str, List[str]], 
                          model_mean: bool = False) -> Dict[str, Any]:
    """
    Calculate species statistics including formula and ChEBI-based metrics.
    Replicates getSpeciesStatistics from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        recommendations: List of Recommendation objects
        refs_formula: Reference formulas {id: [formulas]}
        refs_chebi: Reference ChEBI IDs {id: [chebi_ids]}
        model_mean: If True, return model-level averages
        
    Returns:
        Dictionary with recall and precision statistics
    """
    # Convert recommendations to prediction dictionaries
    preds_chebi = {val.id: [k for k in val.candidates] for val in recommendations}
    
    # Convert ChEBI predictions to formulas
    formula_dict = load_chebi_formula_dict()
    preds_formula = {}
    for k in preds_chebi.keys():
        formulas = []
        for chebi_id in preds_chebi[k]:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        preds_formula[k] = formulas
    
    # Calculate metrics
    recall_formula = get_recall(ref=refs_formula, pred=preds_formula, mean=model_mean)
    precision_formula = get_precision(ref=refs_formula, pred=preds_formula, mean=model_mean)
    recall_chebi = get_recall(ref=refs_chebi, pred=preds_chebi, mean=model_mean)
    precision_chebi = get_precision(ref=refs_chebi, pred=preds_chebi, mean=model_mean)
    
    return {
        'recall_formula': recall_formula, 
        'recall_chebi': recall_chebi, 
        'precision_formula': precision_formula, 
        'precision_chebi': precision_chebi
    }

def find_species_with_formulas(model_file: str) -> Dict[str, List[str]]:
    """
    Find species with existing ChEBI annotations that have chemical formulas.
    Replicates the logic from AMAS species_annotation.py exist_annotation_formula.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary mapping species IDs to their ChEBI annotation IDs (only for species with formulas)
    """
    # Get all species with ChEBI annotations
    existing_annotations = find_species_with_chebi_annotations(model_file)
    
    if not existing_annotations:
        return {}
    
    # Load ChEBI to formula dictionary
    formula_dict = load_chebi_formula_dict()
    
    # Filter to only species that have at least one ChEBI with a formula
    species_with_formulas = {}
    for species_id, chebi_ids in existing_annotations.items():
        formulas = []
        for chebi_id in chebi_ids:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        
        # Only include species that have at least one formula
        if formulas:
            species_with_formulas[species_id] = chebi_ids
    
    return species_with_formulas

def evaluate_single_model(model_file: str, 
                         llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                         method: str = "direct",
                         max_entities: int = None,
                         entity_type: str = "chemical",
                         database: str = "chebi",
                         save_llm_results: bool = True,
                         output_dir: str = './results/',
                         verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Generate species evaluation statistics for one model.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use
        method: Method to use for database search ("direct", "rag")
        max_entities: Maximum number of entities to evaluate (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results to files
        output_dir: Directory to save results
        verbose: If True, show detailed logging. If False, minimize output.
        
    Returns:
        DataFrame with evaluation results or None if failed
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    try:
        model_name = Path(model_file).name
        if verbose:
            logger.info(f"Evaluating model: {model_name}")
        
        # Get existing annotations to determine entities to evaluate
        if entity_type == "chemical" and database == "chebi":
            existing_annotations = find_species_with_formulas(model_file)
        else:
            if verbose:
                logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
            return None
        
        if not existing_annotations:
            if verbose:
                logger.warning(f"No existing annotations with formulas found in {model_name}")
            return None
        
        # Limit entities if specified
        specs_to_evaluate = list(existing_annotations.keys())
        if max_entities:
            specs_to_evaluate = specs_to_evaluate[:max_entities]
        
        if verbose:
            logger.info(f"Evaluating {len(specs_to_evaluate)} entities in {model_name}")
        
        # Run annotation with access to LLM results
    
        # Extract model context and query LLM
        model_info = extract_model_info(model_file, specs_to_evaluate)
        prompt = format_prompt(model_file, specs_to_evaluate)
        
        # Query LLM and get response
        llm_start = time.time()
        llm_response = query_llm(prompt, SYSTEM_PROMPT, model=llm_model)
        llm_time = time.time() - llm_start
        
        # Parse LLM response
        synonyms_dict, reason = parse_llm_response(llm_response)
        
        # Search database
        search_start = time.time()
        with suppress_outputs(verbose):
            if method == "direct":
                recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict)
            elif method == "rag":
                recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict)
            else:
                if verbose:
                    logger.error(f"Invalid method: {method}")
                return None
        search_time = time.time() - search_start
        
        total_time = llm_time + search_time
        
        if not recommendations:
            if verbose:
                logger.warning(f"No recommendations generated for {model_name}")
            return None
        
        # Convert to AMAS-compatible format with LLM results
        result_df = _convert_format(
            recommendations, existing_annotations, model_name, 
            synonyms_dict, reason, total_time, llm_time, search_time
        )
        
        # Save LLM results if requested
        if save_llm_results:
            _save_llm_results(model_file, llm_model, output_dir, synonyms_dict, reason)
        
        return result_df
        
    except Exception as e:
        if verbose:
            logger.error(f"Failed to evaluate model {model_file}: {e}")
        return None

def evaluate_models_in_folder(model_dir: str,
                             num_models: str = 'all',
                             llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                             method: str = "direct",
                             max_entities: int = None,
                             entity_type: str = "chemical",
                             database: str = "chebi",
                             save_llm_results: bool = True,
                             output_dir: str = './results/',
                             output_file: str = 'evaluation_results.csv',
                             start_at: int = 1,
                             verbose: bool = False) -> pd.DataFrame:
    """
    Generate species evaluation statistics for multiple models in a directory.
    Replicates evaluate_models from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        model_dir: Directory containing SBML model files
        num_models: Number of models to evaluate ('all' or integer)
        llm_model: LLM model to use
        method: Method to use for database search ("direct", "rag")
        max_entities: Maximum entities per model (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results
        output_dir: Directory to save results
        output_file: Name of output CSV file
        start_at: Model index to start at (1-based)
        verbose: If True, show detailed logging. If False, minimize output.
        
    Returns:
        Combined DataFrame with all evaluation results
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    # Clear any existing ChromaDB clients to avoid conflicts
    clear_chromadb_cache()
    
    # Get model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml')]
    model_files.sort()  # Ensure consistent ordering
    
    # Determine which models to evaluate
    if num_models == 'all':
        num_models = len(model_files)
        model_files = model_files[start_at-1:]
    else:
        num_models = int(min(num_models, len(model_files) - start_at + 1))
        model_files = model_files[start_at-1:start_at+num_models-1]
    
    logger.info(f"Evaluating {len(model_files)} models starting from index {start_at}")
    
    # Initialize result storage
    all_results = []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    for idx, model_file in enumerate(model_files):
        actual_idx = idx + start_at
        print(f"Evaluating {actual_idx}/{start_at + len(model_files) - 1}: {model_file}")
        
        model_path = os.path.join(model_dir, model_file)
        
        # Evaluate single model
        result_df = evaluate_single_model(
            model_file=model_path,
            llm_model=llm_model,
            method=method,
            max_entities=max_entities,
            entity_type=entity_type,
            database=database,
            save_llm_results=save_llm_results,
            output_dir=output_dir,
            verbose=verbose
        )
        
        if result_df is not None:
            all_results.append(result_df)
            
            # Save intermediate results in a subfolder
            intermediate_dir = output_path / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            intermediate_file = intermediate_dir / f"{output_file}_{actual_idx}.csv"
            result_df.to_csv(intermediate_file, index=False)
            logger.info(f"Saved intermediate results to: {intermediate_file}")
        else:
            logger.warning(f"Skipping {model_file} - no results generated")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save final results
        final_file = output_path / output_file
        combined_df.to_csv(final_file, index=False)
        logger.info(f"Saved final results to: {final_file}")
        
        return combined_df
    else:
        logger.warning("No results generated for any models")
        return pd.DataFrame()

def _convert_format(recommendations: List[Recommendation],
                                   existing_annotations: Dict[str, List[str]],
                                   model_name: str,
                                   synonyms_dict: Dict[str, List[str]],
                                   reason: str,
                                   total_time: float,
                                   llm_time: float,
                                   search_time: float) -> pd.DataFrame:
    """
    Convert AAAIM recommendations to AMAS evaluation format with LLM results.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_name: Name of the model file
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        DataFrame in AMAS evaluation format
    """
    # Load required dictionaries
    label_dict = load_chebi_label_dict()
    formula_dict = load_chebi_formula_dict()
    
    # Prepare reference data for statistics calculation
    refs_chebi = existing_annotations
    refs_formula = {}
    for species_id, chebi_ids in existing_annotations.items():
        formulas = []
        for chebi_id in chebi_ids:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:
                    formulas.append(formula)
        refs_formula[species_id] = formulas
    
    # Calculate statistics
    stats = get_species_statistics(recommendations, refs_formula, refs_chebi, model_mean=False)
    
    # Convert to AMAS format
    result_rows = []
    for rec in recommendations:
        species_id = rec.id
        
        # Get existing annotation names
        existing_ids = existing_annotations.get(species_id, [])
        existing_names = [label_dict.get(chebi_id, chebi_id) for chebi_id in existing_ids]
        exist_annotation_name = ', '.join(existing_names) if existing_names else 'NA'
        
        # Get LLM synonyms
        llm_synonyms = synonyms_dict.get(species_id, [])
        
        # Get predictions and their names
        predictions = rec.candidates
        prediction_names = [label_dict.get(chebi_id, chebi_id) for chebi_id in predictions]
        
        # Calculate match scores
        match_scores = []
        if rec.match_score and llm_synonyms:
            match_scores = [match_score for match_score in rec.match_score]
        else:
            match_scores = [0.0] * len(predictions)
                
        # Get statistics for this species
        recall_formula = stats['recall_formula'].get(species_id, 0) if isinstance(stats['recall_formula'], dict) else 0
        precision_formula = stats['precision_formula'].get(species_id, 0) if isinstance(stats['precision_formula'], dict) else 0
        recall_chebi = stats['recall_chebi'].get(species_id, 0) if isinstance(stats['recall_chebi'], dict) else 0
        precision_chebi = stats['precision_chebi'].get(species_id, 0) if isinstance(stats['precision_chebi'], dict) else 0

        # Calculate accuracy (1 if recall_formula > 0, 0 otherwise)
        accuracy = 1 if recall_formula > 0 else 0
        
        # Create row in AMAS format
        row = {
            'model': model_name,
            'species_id': species_id,
            'display_name': rec.synonyms[0] if rec.synonyms else species_id,  # Use first synonym as display name
            'synonyms_LLM': llm_synonyms,
            'reason': reason,
            'exist_annotation_chebi': existing_ids,
            'exist_annotation_name': exist_annotation_name,
            'predictions': predictions,
            'predictions_names': prediction_names,
            'match_score': match_scores,
            'recall_formula': recall_formula,
            'precision_formula': precision_formula,
            'recall_chebi': recall_chebi,
            'precision_chebi': precision_chebi,
            'accuracy': accuracy,
            'total_time': total_time,
            'llm_time': llm_time,
            'query_time': search_time
        }
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)

def _save_llm_results(model_file: str, llm_model: str, output_dir: str, 
                     synonyms_dict: Dict[str, List[str]], reason: str):
    """
    Save LLM results to file.
    
    Args:
        model_file: Path to model file
        llm_model: LLM model used
        output_dir: Output directory
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
    """
    model_name = Path(model_file).stem
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "meta-llama/llama-3.3-70b-instruct":
        llm_name = "llama-3.3-70b-instruct"
    else:
        llm_name = llm_model

    output_dir = output_dir+llm_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{model_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"LLM: {llm_model}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Synonyms:\n")
        for species_id, synonyms in synonyms_dict.items():
            f.write(f"{species_id}: {synonyms}\n")
        f.write(f"\nReason: {reason}\n")
    print(f"LLM results saved to: {output_file}")

def print_evaluation_results(results_csv: str):
    """
    Print evaluation results summary.
    Replicates print_results from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        results_csv: Path to results CSV file
    """
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % df['model'].nunique())
    print("Number of models with predictions: %d" % df[df['predictions'] != '[]']['model'].nunique())
    
    # Calculate per-model averages
    model_accuracy = df.groupby('model')['accuracy'].mean().mean()
    print("Average accuracy (per model): %.02f" % model_accuracy)
    
    mean_processing_time = df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)
    
    num_elements = df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements if num_elements > 0 else 0
    print("Ave. total time (per element, per model): %.02f" % mean_processing_time_per_element)
    
    # LLM time
    mean_llm_time = df.groupby('model')['llm_time'].first().mean()
    print("Ave. LLM time (per model): %.02f" % mean_llm_time)
    
    mean_llm_time_per_element = mean_llm_time / num_elements if num_elements > 0 else 0
    print("Ave. LLM time (per element, per model): %.02f" % mean_llm_time_per_element)
    
    # Average number of predictions per species
    def safe_eval_predictions(x):
        """Safely evaluate predictions string."""
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except Exception:
            return []
    
    df['parsed_predictions'] = df['predictions'].apply(safe_eval_predictions)
    df['num_predictions'] = df['parsed_predictions'].apply(len)
    average_predictions = df['num_predictions'].mean()
    print(f"Average number of predictions per species: {average_predictions:.2f}")

def calculate_species_statistics(recommendations: List[Recommendation],
                                existing_annotations: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate evaluation statistics for species recommendations.
    Simplified version of getSpeciesStatistics from AMAS.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        
    Returns:
        Dictionary with recall and precision statistics
    """
    stats = {}
    
    for rec in recommendations:
        species_id = rec.id
        predicted_ids = rec.candidates
        existing_ids = existing_annotations.get(species_id, [])
        
        # Calculate simple recall and precision
        if existing_ids:
            # Recall: fraction of existing annotations that were predicted
            matches = set(predicted_ids) & set(existing_ids)
            recall = len(matches) / len(existing_ids) if existing_ids else 0
        else:
            recall = 0
        
        if predicted_ids:
            # Precision: fraction of predictions that match existing annotations
            matches = set(predicted_ids) & set(existing_ids)
            precision = len(matches) / len(predicted_ids) if predicted_ids else 0
        else:
            precision = 0
        
        stats[species_id] = {
            'recall_chebi': recall,
            'precision_chebi': precision,
            'recall_formula': 0,  # Not implemented
            'precision_formula': 0  # Not implemented
        }
    
    return stats

def process_saved_llm_responses(response_folder: str, 
                               model_dir: str, 
                               prev_results_csv: str, 
                               method: str = "direct",
                               output_dir: str = './results/', 
                               output_file: str = 'reprocessed_results.csv',
                               verbose: bool = False) -> pd.DataFrame:
    """
    Process saved LLM response files to generate species evaluation statistics,
    keeping the same species information as in the previous results.
    
    Args:
        response_folder: Path to folder containing saved LLM response files
        model_dir: Path to directory containing the original model files
        prev_results_csv: Path to previous results CSV from evaluate_models
        method: Method to use for database search ("direct", "rag")
        output_dir: Path to directory where results should be saved
        output_file: Name of the output CSV file
        verbose: If True, show detailed logging. If False, minimize output.
        
    Returns:
        DataFrame with evaluation results
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    # Clear any existing ChromaDB clients to avoid conflicts
    clear_chromadb_cache()
    
    # Load previous results
    if not os.path.exists(prev_results_csv):
        raise FileNotFoundError(f"Previous results file not found: {prev_results_csv}")
    
    prev_df = pd.read_csv(prev_results_csv)
    
    # Group previous results by model
    model_data = {}
    for model_name in prev_df['model'].unique():
        model_df = prev_df[prev_df['model'] == model_name]
        model_data[model_name] = {
            'species_info': {},
            'llm_time': model_df['llm_time'].iloc[0] if 'llm_time' in model_df.columns else 0
        }
        
        # Store species-specific information
        for _, row in model_df.iterrows():
            species_id = row['species_id']
            model_data[model_name]['species_info'][species_id] = {
                'display_name': row['display_name'],
                'exist_annotation_chebi': row['exist_annotation_chebi'],
                'exist_annotation_name': row['exist_annotation_name']
            }
    
    # List to track models with parsing errors
    parse_errors = []
    
    # Process each LLM response file
    response_files = [f for f in os.listdir(response_folder) if f.endswith('.txt')]
    
    all_results = []
    
    for idx, response_file in enumerate(response_files):
        print(f"Processing {idx+1}/{len(response_files)}: {response_file}")
        
        # Extract model name from filename (remove .txt extension but keep .xml)
        model_name = response_file.replace('.txt', '.xml')
        # The model file path in model_dir  
        model_file = os.path.join(model_dir, model_name)
        
        # Read response file
        with open(os.path.join(response_folder, response_file), 'r') as f:
            content = f.read()
        
        # Extract response part
        # First try format with "RESULT:"
        result_match = re.search(r'RESULT:\s*([\s\S]*)', content)
        if result_match:
            result = result_match.group(1).strip()
        else:
            # Try format with "Synonyms:" section
            synonyms_match = re.search(r'Synonyms:\s*([\s\S]*?)(?=Reason:|$)', content)
            reason_match = re.search(r'Reason:\s*([\s\S]*)', content)
            
            if synonyms_match:
                synonyms_text = synonyms_match.group(1).strip()
                reason_text = reason_match.group(1).strip() if reason_match else ""
                # Reconstruct the result in the format expected by parse_llm_response
                result = synonyms_text + '\nReason: ' + reason_text
            else:
                print(f"Could not find parseable content in {response_file}, skipping")
                parse_errors.append(f"{response_file}: Could not find parseable content")
                continue
        
        # Find the model name as it appears in the previous results
        if model_name not in model_data:
            print(f"Model {model_name} not found in previous results, skipping")
            parse_errors.append(f"{response_file}: Model not found in previous results")
            continue
            
        # Parse the LLM response
        try:
            synonyms_dict, reason = parse_llm_response(result)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {response_file}: {e}")
            parse_errors.append(f"{response_file}: Error parsing LLM response - {str(e)}")
            continue
        
        try:
            # Get species from previous results
            species_info = model_data[model_name]['species_info']
            specs_to_evaluate = list(species_info.keys())
            
            # Time the ChEBI dictionary search
            query_start_time = time.time()
            with suppress_outputs(verbose):
                if method == "direct":
                    recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict)
                elif method == "rag":
                    recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict)
                else:
                    print(f"Invalid method: {method}")
                    return None
            query_end_time = time.time()
            dict_search_time = query_end_time - query_start_time
            
            # Get existing annotations for statistics calculation
            existing_annotations = find_species_with_formulas(model_file)
            
            # Previous LLM time from original run
            previous_llm_time = model_data[model_name]['llm_time']
            
            # Convert to AMAS-compatible format
            result_df = _convert_format(
                recommendations, existing_annotations, model_name, 
                synonyms_dict, reason, previous_llm_time + dict_search_time, 
                previous_llm_time, dict_search_time
            )
            
            if not result_df.empty:
                all_results.append(result_df)
        
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            parse_errors.append(f"{response_file}: Error during processing - {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        output_file_path = output_path / output_file
        combined_df.to_csv(output_file_path, index=False)
        print(f"Saved results to {output_file_path}")
        
        # Save parse errors to file
        error_file_path = output_path / 'parse_errors.txt'
        with open(error_file_path, 'w') as f:
            f.write(f"Total errors: {len(parse_errors)}\n\n")
            for error in parse_errors:
                f.write(f"{error}\n")
        print(f"Saved {len(parse_errors)} parse errors to {error_file_path}")
        
        return combined_df
    else:
        print("No results generated for any models")
        return pd.DataFrame()



def compare_results(*csv_paths: str) -> dict:
    """
    Compare results from multiple CSVs by filtering to only include common models and species.
    Prints detailed statistics for each CSV and a summary comparison table.
    
    Args:
        *csv_paths: Paths to result CSVs (must be at least 2)
        
    Returns:
        Dictionary mapping CSV path to filtered DataFrame (only common models/species)
    """
    if len(csv_paths) < 2:
        raise ValueError("At least two CSV paths must be provided.")

    # Helper to safely parse predictions
    def safe_eval_predictions(x):
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except Exception:
            return []

    # Load all DataFrames and check existence
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        dfs.append(pd.read_csv(path))

    # Find common models and species_ids across all DataFrames
    model_sets = [set(df['model'].unique()) for df in dfs]
    common_models = set.intersection(*model_sets)

    # For each model, find common species_ids across all DataFrames
    common_model_species = {}
    for model in common_models:
        species_sets = []
        for df in dfs:
            species_sets.append(set(df[df['model'] == model]['species_id'].unique()))
        common_species = set.intersection(*species_sets)
        if common_species:
            common_model_species[model] = common_species

    # Flatten to set of (model, species_id) pairs
    common_pairs = set()
    for model, species_set in common_model_species.items():
        for species_id in species_set:
            common_pairs.add((model, species_id))

    if not common_pairs:
        print("No common models and species found across all results.")
        return {}

    # Filter each DataFrame to only include common (model, species_id) pairs
    filtered_dfs = []
    for df in dfs:
        filtered = df[df.apply(lambda row: (row['model'], row['species_id']) in common_pairs, axis=1)].copy()
        filtered_dfs.append(filtered)

    # Print stats for each filtered DataFrame
    print("="*70)
    print("COMPARISON OF RESULTS (filtered to common models/species)")
    print("="*70)
    stats = []
    for csv_path, filtered_df in zip(csv_paths, filtered_dfs):
        print(f"\n{'='*60}\nRESULTS FOR: {csv_path}\n{'='*60}")
        n_models = filtered_df['model'].nunique()
        n_models_with_preds = filtered_df[filtered_df['predictions'] != '[]']['model'].nunique()
        print(f"Number of models assessed: {n_models}")
        print(f"Number of models with predictions: {n_models_with_preds}")

        # Per-model averages
        if 'accuracy' in filtered_df.columns:
            model_accuracy = filtered_df.groupby('model')['accuracy'].mean().mean()
            print(f"Average accuracy (per model): {model_accuracy:.2f}")
        else:
            model_accuracy = None

        if 'total_time' in filtered_df.columns:
            mean_processing_time = filtered_df.groupby('model')['total_time'].first().mean()
            print(f"Ave. total time (per model): {mean_processing_time:.2f}")
            num_elements = filtered_df.groupby('model').size().mean()
            mean_processing_time_per_element = mean_processing_time / num_elements if num_elements > 0 else 0
            print(f"Ave. total time (per element, per model): {mean_processing_time_per_element:.2f}")
        else:
            mean_processing_time = None
            mean_processing_time_per_element = None
            num_elements = None

        if 'llm_time' in filtered_df.columns:
            mean_llm_time = filtered_df.groupby('model')['llm_time'].first().mean()
            print(f"Ave. LLM time (per model): {mean_llm_time:.2f}")
            mean_llm_time_per_element = mean_llm_time / num_elements if num_elements and num_elements > 0 else 0
            print(f"Ave. LLM time (per element, per model): {mean_llm_time_per_element:.2f}")
        else:
            mean_llm_time = None
            mean_llm_time_per_element = None

        filtered_df['parsed_predictions'] = filtered_df['predictions'].apply(safe_eval_predictions)
        filtered_df['num_predictions'] = filtered_df['parsed_predictions'].apply(len)
        avg_preds_per_species = filtered_df['num_predictions'].mean()
        print(f"Average number of predictions per species: {avg_preds_per_species:.2f}")

        stats.append({
            'CSV': csv_path,
            'Models Assessed': n_models,
            'Average Accuracy': model_accuracy,
            'Average Total Time': mean_processing_time,
            'Average LLM Time': mean_llm_time,
            'Avg Predictions per Species': avg_preds_per_species
        })

    # Print summary table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(stats)
    summary_df = summary_df.set_index('CSV')
    print(summary_df.to_string(float_format="%.2f"))

    # Return dictionary of filtered DataFrames
    # return {csv_path: filtered_df for csv_path, filtered_df in zip(csv_paths, filtered_dfs)}
