"""
Evaluation Utilities for AAAIM

Internal evaluation functions for testing and validation.
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from core.curation_workflow import curate_model
from core.model_info import find_species_with_chebi_annotations, extract_model_info, format_prompt, find_species_with_ncbigene_annotations, get_species_display_names, detect_model_format
from core.llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response, get_system_prompt
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag, clear_chromadb_cache
from utils.constants import REF_CHEBI2LABEL, REF_NCBIGENE2LABEL, REF_CHEBI2FORMULA, CHEBI_URI_PATTERNS, NCBIGENE_URI_PATTERNS, UNIPROT_URI_PATTERNS, ModelType

REF_RESULTS = "/Users/luna/Desktop/CRBM/AMAS_proj/Results/biomd_species_accuracy_AMAS.csv"

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
_NCBIGENE_LABEL_DICT: Optional[Dict[str, str]] = None

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / REF_CHEBI2LABEL
        
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
        data_file = Path(__file__).parent.parent / "data" / "chebi" / REF_CHEBI2FORMULA
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI formula data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_FORMULA_DICT = pickle.load(f)
    
    return _CHEBI_FORMULA_DICT

def load_ncbigene_label_dict() -> Dict[str, str]:
    """
    Load the NCBI gene ID to label dictionary.
    
    Returns:
        Dictionary mapping NCBI gene IDs to their labels
    """
    global _NCBIGENE_LABEL_DICT
    
    if _NCBIGENE_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "ncbigene" / REF_NCBIGENE2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"NCBI gene label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _NCBIGENE_LABEL_DICT = pickle.load(f)
    
    return _NCBIGENE_LABEL_DICT

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
                          refs_exact: Dict[str, List[str]], 
                          model_mean: bool = False) -> Dict[str, Any]:
    """
    Calculate species statistics including formula and exact-based metrics.
    Replicates getSpeciesStatistics from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        recommendations: List of Recommendation objects
        refs_formula: Reference formulas {id: [formulas]}
        refs_exact: Reference ChEBI IDs {id: [chebi_ids]}
        model_mean: If True, return model-level averages
        
    Returns:
        Dictionary with recall and precision statistics
    """
    # Convert recommendations to prediction dictionaries
    preds_exact = {val.id: [k for k in val.candidates] for val in recommendations}
    
    # Convert ChEBI predictions to formulas
    formula_dict = load_chebi_formula_dict()
    preds_formula = {}
    for k in preds_exact.keys():
        formulas = []
        for chebi_id in preds_exact[k]:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        preds_formula[k] = formulas
    
    # Calculate metrics
    recall_formula = get_recall(ref=refs_formula, pred=preds_formula, mean=model_mean)
    precision_formula = get_precision(ref=refs_formula, pred=preds_formula, mean=model_mean)
    recall_exact = get_recall(ref=refs_exact, pred=preds_exact, mean=model_mean)
    precision_exact = get_precision(ref=refs_exact, pred=preds_exact, mean=model_mean)
    
    return {
        'recall_formula': recall_formula, 
        'precision_formula': precision_formula, 
        'recall_exact': recall_exact, 
        'precision_exact': precision_exact
    }

def find_species_with_formulas(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find species with existing ChEBI annotations that have chemical formulas.
    Replicates the logic from AMAS species_annotation.py exist_annotation_formula.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        Dictionary mapping species IDs to their ChEBI annotation IDs (only for species with formulas)
    """
    # Get all species with ChEBI annotations
    existing_annotations = find_species_with_chebi_annotations(model_file, bqbiol_qualifiers)
    
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

def find_species_with_gene_annotations(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find species with existing NCBI gene annotations.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
    Returns:
        Dictionary mapping species IDs to their NCBI gene annotation IDs
    """
    # Get all species with NCBI gene annotations
    existing_annotations = find_species_with_ncbigene_annotations(model_file, bqbiol_qualifiers)
    
    if not existing_annotations:
        return {}
    
    # Return all species that have NCBI gene annotations
    return existing_annotations

def evaluate_single_model(model_file: str, 
                         llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                         method: str = "direct",
                         top_k: int = 3,
                         max_entities: int = None,
                         entity_type: str = "chemical",
                         database: str = "chebi",
                         save_llm_results: bool = True,
                         save_llm_results_folder: str = None,
                         output_dir: str = './results/',
                         verbose: bool = True,
                         tax_id: str = None,
                         tax_name: str = None,
                         bqbiol_qualifiers: list = None) -> Optional[pd.DataFrame]:
    """
    Generate species evaluation statistics for one model.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to return per species
        max_entities: Maximum number of entities to evaluate (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results to files
        save_llm_results_folder: Custom folder name for LLM results. If None, uses timestamp.
        output_dir: Directory to save results
        verbose: If True, show detailed logging. If False, minimize output.
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        tax_name: For gene/protein annotations, the organism's tax_name for species-specific lookup
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        DataFrame with evaluation results or None if failed
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    try:
        model_name = Path(model_file).name
        if verbose:
            logger.info(f"Evaluating model: {model_name}")
            if tax_id:
                logger.info(f"Using organism-specific search for tax_id: {tax_id}")
        
        # Get existing annotations to determine entities to evaluate
        if entity_type == "chemical" and database == "chebi":
            existing_annotations = find_species_with_formulas(model_file, bqbiol_qualifiers)
        elif entity_type == "gene" and database == "ncbigene":
            existing_annotations = find_species_with_gene_annotations(model_file, bqbiol_qualifiers)
        else:
            if verbose:
                logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
            return None
        if not existing_annotations:
            if verbose:
                logger.warning(f"No existing annotations found in {model_name}")
            return None
        
        # Limit entities if specified
        specs_to_evaluate = list(existing_annotations.keys())
        if max_entities:
            specs_to_evaluate = specs_to_evaluate[:max_entities]
        
        if verbose:
            logger.info(f"Evaluating {len(specs_to_evaluate)} entities in {model_name}")
        
        # Run annotation with access to LLM results
    
        # Extract model context and query LLM
        model_info = extract_model_info(model_file, specs_to_evaluate, entity_type)
        prompt = format_prompt(model_file, specs_to_evaluate, entity_type)
        
        # Query LLM and get response
        llm_start = time.time()
        # Get appropriate system prompt for entity type
        system_prompt = get_system_prompt(entity_type)
        llm_response = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
        llm_time = time.time() - llm_start
        
        # Parse LLM response
        synonyms_dict, reason = parse_llm_response(llm_response)
        
        # Search database
        search_start = time.time()
        with suppress_outputs(verbose):
            if method == "direct":
                if database == "chebi":
                    recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
                elif database == "ncbigene":
                    recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                else:
                    if verbose:
                        logger.error(f"Database {database} not supported")
                    return None
            elif method == "rag":
                if database == "chebi":
                    recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
                elif database == "ncbigene":
                    recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                else:
                    if verbose:
                        logger.error(f"Database {database} not supported")
                    return None
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
        
        # Convert to evaluation format with LLM results
        result_df = _convert_format(
            recommendations, existing_annotations, model_name, 
            synonyms_dict, reason, total_time, llm_time, search_time, entity_type, database, tax_id, tax_name, model_file
        )
        
        # Save LLM results if requested
        if save_llm_results:
            _save_llm_results(model_file, llm_model, output_dir, synonyms_dict, reason, entity_type, save_llm_results_folder)
        
        return result_df
        
    except Exception as e:
        if verbose:
            logger.error(f"Failed to evaluate model {model_file}: {e}")
        return None

def get_model_taxonomy(model_file, tax_dict_file):
    tax_dict_df = pd.read_csv(tax_dict_file)
    model_file_id = model_file.replace('.xml', '')
    if model_file_id in tax_dict_df['id'].values:
        tax_id = tax_dict_df[tax_dict_df['id'] == model_file_id]['tax_id'].values[0]
        tax_name = tax_dict_df[tax_dict_df['id'] == model_file_id]['organism'].values[0]
    else:
        tax_id = None
        tax_name = None
    return tax_id, tax_name

def evaluate_models_in_folder(model_dir: str,
                             num_models: str = 'all',
                             llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                             method: str = "direct",
                             top_k: int = 3,
                             max_entities: int = None,
                             entity_type: str = "chemical",
                             database: str = "chebi",
                             save_llm_results: bool = True,
                             save_llm_results_folder: str = None,
                             output_dir: str = './results/',
                             output_file: str = 'evaluation_results.csv',
                             start_at: int = 1,
                             verbose: bool = False,
                             tax_id: str = None,
                             tax_dict_file: str = None,
                             bqbiol_qualifiers: list = None) -> pd.DataFrame:
    """
    Generate species evaluation statistics for multiple models in a directory.
    Replicates evaluate_models from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        model_dir: Directory containing SBML model files
        num_models: Number of models to evaluate ('all' or integer)
        llm_model: LLM model to use
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to return per species
        max_entities: Maximum entities per model (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        save_llm_results: Whether to save LLM results
        save_llm_results_folder: Custom folder name for LLM results. If None, uses timestamp.
        output_dir: Directory to save results
        output_file: Name of output CSV file
        start_at: Model index to start at (1-based)
        verbose: If True, show detailed logging. If False, minimize output.
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        tax_dict_file: File containing taxonomy information for model files
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        
    Returns:
        Combined DataFrame with all evaluation results
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")
    
    # Clear any existing ChromaDB clients to avoid conflicts
    clear_chromadb_cache()
    
    # Get model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml') or f.endswith('.sbml')]
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
    
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "meta-llama/llama-3.3-70b-instruct":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "Llama-3.3-70B-Instruct":
        llm_name = "Llama-3.3-70B-instruct-Meta"
    elif llm_model == "Llama-4-Maverick-17B-128E-Instruct-FP8":
        llm_name = "llama-4-maverick-17b-128e-instruct-fp8"
    else:
        llm_name = llm_model

    # Use custom folder name or timestamp-based folder name
    if not save_llm_results_folder:
        timestamp = time.strftime('%Y%m%d_%H%M')
        save_llm_results_folder = f"{llm_name}/{entity_type}/{timestamp}"

    print(f"LLM results will be saved to: {output_dir + save_llm_results_folder}")

    # Evaluate each model
    for idx, model_file in enumerate(model_files):
        actual_idx = idx + start_at
        print(f"Evaluating {actual_idx}/{start_at + len(model_files) - 1}: {model_file}")
        
        model_path = os.path.join(model_dir, model_file)
        tax_name = None
        if tax_id == 9606:
            tax_name = "Homo sapiens"
        elif tax_id == 511145:
            tax_name = "Escherichia coli"
        elif tax_id == 10090:
            tax_name = "Mus musculus"
        if tax_dict_file:
            tax_id, tax_name = get_model_taxonomy(model_file, tax_dict_file)
            if not tax_id:
                logger.warning(f"No tax_id found for {model_file}")
                continue
        
        # Evaluate single model
        result_df = evaluate_single_model(
            model_file=model_path,
            llm_model=llm_model,
            method=method,
            top_k=top_k,
            max_entities=max_entities,
            entity_type=entity_type,
            database=database,
            save_llm_results=save_llm_results,
            save_llm_results_folder=save_llm_results_folder,
            output_dir=output_dir,
            verbose=verbose,
            tax_id=tax_id,
            tax_name=tax_name,
            bqbiol_qualifiers=bqbiol_qualifiers
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
                                   search_time: float,
                                   entity_type: str = "chemical",
                                   database: str = "chebi",
                                   tax_id: str = None,
                                   tax_name: str = None,
                                   model_file: str = None) -> pd.DataFrame:
    """
    Convert AAAIM recommendations to evaluation format with LLM results.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_name: Name of the model file
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        entity_type: Type of entity being annotated
        database: Database being used
        tax_id: For gene/protein annotations, the organism's tax_id
        tax_name: For gene/protein annotations, the organism's tax_name
        model_file: Path to the model file (optional)

    Returns:
        DataFrame in evaluation format
    """
    # Load required dictionaries based on database
    if database == "chebi":
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
    
    elif database == "ncbigene":
        label_dict = load_ncbigene_label_dict()
        
        # For NCBI gene, we don't have formulas, so we use gene IDs directly
        refs_gene = existing_annotations
        refs_formula = {}  # Empty for gene annotations
        
        # Calculate statistics (simplified for genes)
        stats = {
            'recall_formula': {species_id: 0 for species_id in existing_annotations.keys()},
            'precision_formula': {species_id: 0 for species_id in existing_annotations.keys()},
            'recall_exact': {},  # Will be calculated below
            'precision_exact': {}  # Will be calculated below
        }
        
        # Calculate gene-based recall and precision
        for species_id in existing_annotations.keys():
            existing_ids = existing_annotations.get(species_id, [])
            predicted_ids = []
            for rec in recommendations:
                if rec.id == species_id:
                    predicted_ids = rec.candidates
                    break
            
            if existing_ids:
                matches = set(predicted_ids) & set(existing_ids)
                recall = len(matches) / len(existing_ids) if existing_ids else 0
            else:
                recall = 0
            
            if predicted_ids:
                matches = set(predicted_ids) & set(existing_ids)
                precision = len(matches) / len(predicted_ids) if predicted_ids else 0
            else:
                precision = 0
            
            stats['recall_exact'][species_id] = recall
            stats['precision_exact'][species_id] = precision
    
    else:
        # Default/unknown database
        label_dict = {}
        stats = {
            'recall_formula': {species_id: 0 for species_id in existing_annotations.keys()},
            'precision_formula': {species_id: 0 for species_id in existing_annotations.keys()},
            'recall_exact': {species_id: 0 for species_id in existing_annotations.keys()},
            'precision_exact': {species_id: 0 for species_id in existing_annotations.keys()}
        }
    
    # Get display names from the model file if available
    display_names = {}
    if model_file is not None:
        display_names = get_species_display_names(model_file, entity_type)
    
    # Convert to AMAS format
    result_rows = []
    for rec in recommendations:
        species_id = rec.id
        
        # Get existing annotation names
        existing_ids = existing_annotations.get(species_id, [])
        existing_names = [label_dict.get(db_id, db_id) for db_id in existing_ids]
        exist_annotation_name = ', '.join(existing_names) if existing_names else 'NA'
        
        # Get LLM synonyms
        llm_synonyms = synonyms_dict.get(species_id, [])
        
        # Get predictions and their names
        predictions = rec.candidates
        prediction_names = [label_dict.get(db_id, db_id) for db_id in predictions]
        
        # Calculate match scores
        match_scores = []
        if rec.match_score and llm_synonyms:
            match_scores = [match_score for match_score in rec.match_score]
        else:
            match_scores = [0.0] * len(predictions)
                
        # Get statistics for this species
        recall_formula = stats['recall_formula'].get(species_id, 0) if isinstance(stats['recall_formula'], dict) else 0
        precision_formula = stats['precision_formula'].get(species_id, 0) if isinstance(stats['precision_formula'], dict) else 0
        recall_exact = stats['recall_exact'].get(species_id, 0) if isinstance(stats['recall_exact'], dict) else 0
        precision_exact = stats['precision_exact'].get(species_id, 0) if isinstance(stats['precision_exact'], dict) else 0

        # Calculate accuracy (1 if recall > 0, 0 otherwise)
        # For chemical entities, use recall_formula; for gene entities, use recall_exact
        if entity_type == "chemical":
            accuracy = 1 if recall_formula > 0 else 0
        else:  # gene or other entity types
            accuracy = 1 if recall_exact > 0 else 0
        
        # Use display name from SBML if available
        display_name = display_names.get(species_id, '')
        
        # Create row in AMAS format
        row = {
            'model': model_name,
            'species_id': species_id,
            'display_name': display_name,
            'synonyms_LLM': llm_synonyms,
            'reason': reason,
            'exist_annotation_id': existing_ids,
            'exist_annotation_name': exist_annotation_name,
            'predictions': predictions,
            'predictions_names': prediction_names,
            'match_score': match_scores,
            'recall_formula': recall_formula,
            'precision_formula': precision_formula,
            'recall_exact': recall_exact,
            'precision_exact': precision_exact,
            'accuracy': accuracy,
            'total_time': total_time,
            'llm_time': llm_time,
            'query_time': search_time,
            'tax_id': tax_id,
            'tax_name': tax_name
        }
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)

def _save_llm_results(model_file: str, llm_model: str, output_dir: str, 
                     synonyms_dict: Dict[str, List[str]], reason: str, entity_type: str, 
                     save_llm_results_folder: str = None):
    """
    Save LLM results to file.
    
    Args:
        model_file: Path to model file
        llm_model: LLM model used
        output_dir: Output directory
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        entity_type: Type of entity being annotated
        save_llm_results_folder: Custom folder name for LLM results. If None, uses timestamp.
    """
    model_name = Path(model_file).stem
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "meta-llama/llama-3.3-70b-instruct":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "Llama-3.3-70B-Instruct":
        llm_name = "Llama-3.3-70B-instruct-Meta"
    elif llm_model == "Llama-4-Maverick-17B-128E-Instruct-FP8":
        llm_name = "llama-4-maverick-17b-128e-instruct-fp8"
    else:
        llm_name = llm_model

    # Use custom folder name or timestamp-based folder name
    if save_llm_results_folder:
        output_dir = output_dir + save_llm_results_folder
    else:
        # Generate timestamp-based folder name
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir + f"{llm_name}/{entity_type}/{timestamp}"

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

def print_evaluation_results(results_csv: str, ref_results_csv = REF_RESULTS):
    """
    Print evaluation results summary.
    Replicates print_results from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        results_csv: Path to results CSV file
        ref_results_csv: Path to reference results CSV file to filter against
    """
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("No results to display")
        return
    
    # Filter by reference results if provided
    if ref_results_csv:
        ref_df = pd.read_csv(ref_results_csv)
        if not ref_df.empty:
            # Create a set of (model, species_id) pairs from reference
            ref_pairs = set(zip(ref_df['model'], ref_df['species_id']))
            # Filter current results to only include pairs that exist in reference
            mask = df.apply(lambda row: (row['model'], row['species_id']) in ref_pairs, axis=1)
            df = df[mask]
            
            if df.empty:
                print("No overlapping results found between current results and reference")
                return
            
            print(f"Filtered results to {len(df)} entries that exist in reference: {ref_results_csv}")
        else:
            print(f"Reference file is empty: {ref_results_csv}")
    else:
        print(f"Showing all results")
    
    print("Number of models assessed: %d" % df['model'].nunique())
    print("Number of models with predictions: %d" % df[df['predictions'] != '[]']['model'].nunique())
    print("Number of annotations evaluated: %d" % len(df))    
    # Calculate per-model averages
    model_accuracy = df.groupby('model')['accuracy'].mean().mean()
    print("Average accuracy (per model): %.02f" % model_accuracy)
    
    recall_formula = df.groupby('model')['recall_formula'].mean().mean()
    print("Ave. recall (formula): %.02f" % recall_formula)
    
    precision_formula = df.groupby('model')['precision_formula'].mean().mean()
    print("Ave. precision (formula): %.02f" % precision_formula)
    
    recall_exact = df.groupby('model')['recall_exact'].mean().mean()
    print("Ave. recall (exact): %.02f" % recall_exact)
    
    precision_exact = df.groupby('model')['precision_exact'].mean().mean()
    print("Ave. precision (exact): %.02f" % precision_exact)
    
    # Calculate per-species averages
    species_accuracy = df['accuracy'].mean()
    print("Average accuracy (per species): %.02f" % species_accuracy)
    
    species_recall_formula = df['recall_formula'].mean()
    print("Ave. recall (formula, per species): %.02f" % species_recall_formula)
    
    species_precision_formula = df['precision_formula'].mean()
    print("Ave. precision (formula, per species): %.02f" % species_precision_formula)
    
    species_recall_exact = df['recall_exact'].mean()
    print("Ave. recall (exact, per species): %.02f" % species_recall_exact)
    
    species_precision_exact = df['precision_exact'].mean()
    print("Ave. precision (exact, per species): %.02f" % species_precision_exact)

    # Total time
    mean_processing_time = df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)
    
    # Total time per element
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
            'recall_exact': recall,
            'precision_exact': precision,
            'recall_formula': 0,  # Not implemented
            'precision_formula': 0  # Not implemented
        }
    
    return stats

def process_saved_llm_responses(response_folder: str, 
                               model_dir: str, 
                               prev_results_csv: str, 
                               method: str = "direct",
                               top_k: int = 3,
                               entity_type: str = "chemical",
                               database: str = "chebi",
                               tax_id: str = None,
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
        top_k: Number of top candidates to retrieve per species
        entity_type: Type of entity being annotated
        database: Database being used
        tax_id: Taxonomy ID for NCBI gene search, list or string
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
                'exist_annotation_id': row['exist_annotation_id'],
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

        # Read response file
        with open(os.path.join(response_folder, response_file), 'r') as f:
            content = f.read()
        
        # # Extract response part
        # # First try format with "RESULT:"
        # result_match = re.search(r'RESULT:\s*([\s\S]*)', content)
        # if result_match:
        #     result = result_match.group(1).strip()
        # else:
        #     # Try format with "Synonyms:" section
        #     synonyms_match = re.search(r'Synonyms:\s*([\s\S]*?)(?=Reason:|$)', content)
        #     reason_match = re.search(r'Reason:\s*([\s\S]*)', content)
             
        #     if synonyms_match:
        #         synonyms_text = synonyms_match.group(1).strip()
        #         reason_text = reason_match.group(1).strip() if reason_match else ""
        #         # Reconstruct the result in the format expected by parse_llm_response
        #         result = synonyms_text + '\nReason: ' + reason_text
        #     else:
        #         print(f"Could not find parseable content in {response_file}, skipping")
        #         parse_errors.append(f"{response_file}: Could not find parseable content")
        #         continue
        
        # Find the model name as it appears in the previous results
        if model_name not in model_data:
            # try .sbml
            model_name = model_name.replace('.xml', '.sbml')
            if model_name not in model_data:
                print(f"Model {model_name} not found in previous results, skipping")
                parse_errors.append(f"{response_file}: Model not found in previous results")
                continue
        model_file = os.path.join(model_dir, model_name)

        # Parse the LLM response
        try:
            # synonyms_dict, reason = parse_llm_response(result)
            synonyms_dict, reason = parse_llm_response(content)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {response_file}: {e}")
            parse_errors.append(f"{response_file}: Error parsing LLM response - {str(e)}")
            continue
        
        try:
            # Only evaluate species that exist in BOTH the previous results AND the LLM response
            species_from_prev_results = set(model_data[model_name]['species_info'].keys())
            species_with_llm_synonyms = set(synonyms_dict.keys())
            
            # Find intersection - only species that have both previous results AND LLM synonyms
            specs_to_evaluate = list(species_from_prev_results & species_with_llm_synonyms)
            
            if not specs_to_evaluate:
                print(f"No overlapping species between previous results and LLM response for {model_name}, skipping")
                parse_errors.append(f"{response_file}: No overlapping species found")
                continue
            
            # print(f"Evaluating {len(specs_to_evaluate)} species for {model_name} (intersection of {len(species_from_prev_results)} prev and {len(species_with_llm_synonyms)} LLM)")
            
            # Time the ChEBI dictionary search
            query_start_time = time.time()
            with suppress_outputs(verbose):
                if method == "direct":
                    if database == "chebi":
                        recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
                    elif database == "ncbigene":
                        recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                    else:
                        print(f"Database {database} not supported")
                        return None
                elif method == "rag":
                    if database == "chebi":
                        recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
                    elif database == "ncbigene":
                        recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                    else:
                        print(f"Database {database} not supported")
                        return None
                else:
                    print(f"Invalid method: {method}")
                    return None
            query_end_time = time.time()
            dict_search_time = query_end_time - query_start_time
            
            # Get existing annotations for statistics calculation
            if entity_type == "chemical" and database == "chebi":
                existing_annotations = find_species_with_formulas(model_file)
            elif entity_type == "gene" and database == "ncbigene":
                existing_annotations = find_species_with_gene_annotations(model_file)
            else:
                existing_annotations = {}
            
            # Filter existing_annotations to match the species we're actually evaluating
            existing_annotations = {species_id: existing_annotations[species_id] 
                                    for species_id in specs_to_evaluate 
                                    if species_id in existing_annotations}
            
            # Previous LLM time from original run
            previous_llm_time = model_data[model_name]['llm_time']
            
            # Convert to AMAS-compatible format
            result_df = _convert_format(
                recommendations, existing_annotations, model_name, 
                synonyms_dict, reason, previous_llm_time + dict_search_time, 
                previous_llm_time, dict_search_time, entity_type, database, tax_id, model_file=model_file
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

def compare_results(*csv_paths: str, ref_results_csv: str = REF_RESULTS) -> dict:
    """
    Compare results from multiple CSVs by filtering to only include common models and species.
    Prints detailed statistics for each CSV and a summary comparison table.
    
    Args:
        *csv_paths: Paths to result CSVs (must be at least 2)
        ref_results_csv: Path to reference results CSV file to filter against
        
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

    # Filter by reference results if provided
    if ref_results_csv:
        ref_df = pd.read_csv(ref_results_csv)
        if not ref_df.empty:
            # Create a set of (model, species_id) pairs from reference
            ref_pairs = set(zip(ref_df['model'], ref_df['species_id']))
            
            # Filter each DataFrame to only include pairs that exist in reference
            filtered_dfs = []
            for df in dfs:
                mask = df.apply(lambda row: (row['model'], row['species_id']) in ref_pairs, axis=1)
                filtered_df = df[mask]
                filtered_dfs.append(filtered_df)
            print(f"Filtered all results to only include entries that exist in reference: {ref_results_csv}")
        else:
            print(f"Reference file is empty: {ref_results_csv}")
    else:
        print(f"Showing all results, filtering to common models/species in models")

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
    print("COMPARISON OF RESULTS")
    print("="*70)
    stats = []
    for csv_path, filtered_df in zip(csv_paths, filtered_dfs):
        print(f"\n{'='*60}\nRESULTS FOR: {csv_path}\n{'='*60}")
        n_models = filtered_df['model'].nunique()
        n_models_with_preds = filtered_df[filtered_df['predictions'] != '[]']['model'].nunique()
        print(f"Number of models assessed: {n_models}")
        print(f"Number of models with predictions: {n_models_with_preds}")
        print(f"Number of species tested: {filtered_df['species_id'].nunique()}")

        # Per-model averages
        if 'accuracy' in filtered_df.columns:
            model_accuracy = filtered_df.groupby('model')['accuracy'].mean().mean()
            print(f"Average accuracy (per model): {model_accuracy:.2f}")
            print(f"Average accuracy (per species): {filtered_df['accuracy'].mean():.2f}")
        else:
            model_accuracy = None
        if 'recall_formula' in filtered_df.columns:
            recall_formula = filtered_df.groupby('model')['recall_formula'].mean().mean()
            print(f"Average recall (formula) (per model): {recall_formula:.2f}")
        else:
            recall_formula = None
        if 'precision_formula' in filtered_df.columns:
            precision_formula = filtered_df.groupby('model')['precision_formula'].mean().mean()
            print(f"Average precision (formula) (per model): {precision_formula:.2f}")
        else:
            precision_formula = None
        if 'recall_exact' in filtered_df.columns:
            recall_exact = filtered_df.groupby('model')['recall_exact'].mean().mean()
            print(f"Average recall (exact) (per model): {recall_exact:.2f}")
            print(f"Average recall (exact) (per species): {filtered_df['recall_exact'].mean():.2f}")
        else:
            recall_exact = None
        if 'precision_exact' in filtered_df.columns:
            precision_exact = filtered_df.groupby('model')['precision_exact'].mean().mean()
            print(f"Average precision (exact) (per model): {precision_exact:.2f}")
            print(f"Average precision (exact) (per species): {filtered_df['precision_exact'].mean():.2f}")
        else:
            precision_exact = None
        # Total time
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
            'Average Recall (Formula)': recall_formula,
            'Average Precision (Formula)': precision_formula,
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

def add_distance_columns_to_results(results_csv: str, output_csv: str = None, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Add 'distance_string' and 'distance_embedding' columns to a results CSV.
    - distance_string: normalized character difference between exist_annotation_name and predictions_names
    - distance_embedding: cosine distance between the embedding of the existing annotation name and the prediction name
    Args:
        results_csv: Path to the input results CSV
        output_csv: Path to save the new CSV (optional)
        model_name: SentenceTransformer model to use for embeddings
    Returns:
        DataFrame with new columns
    """
    import ast
    df = pd.read_csv(results_csv)
    model = SentenceTransformer(model_name)
    def norm_char_dist(a, b):
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            return 1.0
        return abs(len(a) - len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0
    def get_first(lst):
        if isinstance(lst, list) and lst:
            return lst[0]
        if isinstance(lst, str):
            try:
                val = ast.literal_eval(lst)
                if isinstance(val, list) and val:
                    return val[0]
            except Exception:
                return lst
        return ''
    distance_strings = []
    distance_embeddings = []
    for idx, row in df.iterrows():
        exist_name = get_first(row.get('exist_annotation_name', ''))
        pred_name = get_first(row.get('predictions_names', ''))
        # String distance
        dist_str = norm_char_dist(str(exist_name), str(pred_name))
        distance_strings.append(dist_str)
        # Embedding distance
        if exist_name and pred_name:
            try:
                emb_exist = model.encode(str(exist_name), convert_to_numpy=True)
                emb_pred = model.encode(str(pred_name), convert_to_numpy=True)
                dist_emb = float(cosine_distances([emb_exist], [emb_pred])[0][0])
            except Exception:
                dist_emb = 1.0
        else:
            dist_emb = 1.0
        distance_embeddings.append(dist_emb)
    df['distance_string'] = distance_strings
    df['distance_embedding'] = distance_embeddings
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

def debug_evaluation_differences(original_csv: str, processed_csv: str, 
                                response_folder: str, model_dir: str,
                                specific_models: List[str] = None,
                                specific_species: List[str] = None) -> pd.DataFrame:
    """
    Debug function to identify exact differences between evaluate_models_in_folder 
    and process_saved_llm_responses results.
    
    Args:
        original_csv: Path to original results
        processed_csv: Path to processed results  
        response_folder: Path to LLM response files
        model_dir: Path to model directory
        specific_models: List of specific models to debug (optional)
        specific_species: List of specific species to debug (optional)
        
    Returns:
        DataFrame with detailed comparison
    """
    orig_df = pd.read_csv(original_csv)
    proc_df = pd.read_csv(processed_csv)
    
    print("=== DEBUGGING EVALUATION DIFFERENCES ===")
    
    # Find models/species with different accuracy
    merged = orig_df.merge(proc_df, on=['model', 'species_id'], suffixes=('_orig', '_proc'))
    differences = merged[merged['accuracy_orig'] != merged['accuracy_proc']]
    
    if specific_models:
        differences = differences[differences['model'].isin(specific_models)]
    if specific_species:
        differences = differences[differences['species_id'].isin(specific_species)]
    
    print(f"Found {len(differences)} species with different accuracy scores")
    
    debug_results = []
    
    for _, row in differences.head(10).iterrows():  # Debug first 10 differences
        model_name = row['model']
        species_id = row['species_id']
        
        print(f"\n=== DEBUGGING: {model_name} - {species_id} ===")
        print(f"Original accuracy: {row['accuracy_orig']}, Processed accuracy: {row['accuracy_proc']}")
        
        # Load and parse LLM response
        response_file = model_name.replace('.xml', '.txt').replace('.sbml', '.txt')
        response_path = os.path.join(response_folder, response_file)
        
        if os.path.exists(response_path):
            with open(response_path, 'r') as f:
                content = f.read()
            
            try:
                synonyms_dict, reason = parse_llm_response(content)
                synonyms_for_species = synonyms_dict.get(species_id, [])
                print(f"LLM synonyms for {species_id}: {synonyms_for_species}")
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                synonyms_for_species = []
        else:
            print(f"Response file not found: {response_path}")
            synonyms_for_species = []
        
        # Get existing annotations
        model_file = os.path.join(model_dir, model_name)
        if os.path.exists(model_file):
            existing_annotations = find_species_with_formulas(model_file)
            existing_for_species = existing_annotations.get(species_id, [])
            print(f"Existing annotations for {species_id}: {existing_for_species}")
        else:
            print(f"Model file not found: {model_file}")
            existing_for_species = []
        
        # Compare predictions
        orig_preds = row['predictions_orig']
        proc_preds = row['predictions_proc']
        print(f"Original predictions: {orig_preds}")
        print(f"Processed predictions: {proc_preds}")
        
        # Compare recall/precision
        print(f"Original recall_formula: {row['recall_formula_orig']}, precision: {row['precision_formula_orig']}")
        print(f"Processed recall_formula: {row['recall_formula_proc']}, precision: {row['precision_formula_proc']}")
        
        debug_results.append({
            'model': model_name,
            'species_id': species_id,
            'accuracy_orig': row['accuracy_orig'],
            'accuracy_proc': row['accuracy_proc'],
            'synonyms_llm': synonyms_for_species,
            'existing_annotations': existing_for_species,
            'predictions_orig': orig_preds,
            'predictions_proc': proc_preds,
            'recall_formula_orig': row['recall_formula_orig'],
            'recall_formula_proc': row['recall_formula_proc']
        })
    
    return pd.DataFrame(debug_results)

def evaluate_llm_synergy(*csv_results: str, 
                        output_file: str = 'llm_synergy_analysis.csv',
                        analysis_level: str = 'model') -> pd.DataFrame:
    """
    Evaluate LLM synergy by analyzing how different LLMs perform on different models or species.
    This function identifies where different LLMs excel and quantifies potential gains from combining them.
    
    Args:
        *csv_results: Paths to CSV files containing results from different LLMs
        output_file: Path to save the synergy analysis results
        analysis_level: 'model' for per-model analysis, 'species' for per-species analysis
        
    Returns:
        DataFrame with synergy analysis results (per-model or per-species statistics)
    """
    if len(csv_results) < 2:
        raise ValueError("Need at least 2 LLM result files to evaluate synergy")
    
    if analysis_level not in ['model', 'species']:
        raise ValueError("analysis_level must be 'model' or 'species'")
    
    # Load all result files
    llm_dfs = []
    llm_names = []
    
    for csv_path in csv_results:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        llm_dfs.append(df)
        
        # Extract LLM name from filename or model column
        llm_name = Path(csv_path).stem
        if 'llm_model' in df.columns:
            llm_name = df['llm_model'].iloc[0] if not df['llm_model'].isnull().all() else llm_name
        llm_names.append(llm_name)
    
    print(f"Analyzing synergy between {len(llm_names)} LLMs: {', '.join(llm_names)}")
    print(f"Analysis level: {analysis_level}")
    
    # Create a combined analysis
    synergy_results = []
    
    if analysis_level == 'model':
        # Get all unique models
        all_models = set()
        for df in llm_dfs:
            all_models.update(df['model'].unique())
        
        # Analyze each model
        for model_name in all_models:
            model_results = {}
            
            # Get results for this model from each LLM (aggregate across species)
            llm_model_performances = {}
            for i, (llm_name, df) in enumerate(zip(llm_names, llm_dfs)):
                model_data = df[df['model'] == model_name]
                
                if not model_data.empty:
                    # Calculate model-level aggregated metrics
                    llm_model_performances[llm_name] = {
                        'avg_recall_exact': model_data['recall_exact'].mean(),
                        'avg_precision_exact': model_data['precision_exact'].mean(),
                        'avg_recall_formula': model_data['recall_formula'].mean(),
                        'avg_precision_formula': model_data['precision_formula'].mean(),
                        'avg_accuracy': model_data['accuracy'].mean(),
                        'num_species': len(model_data),
                        'total_predictions': len(model_data[model_data['predictions'].notna()]),
                        'llm_time': model_data['llm_time'].iloc[0] if 'llm_time' in model_data.columns else 0,
                        'all_predictions': set(),  # Will collect all unique predictions
                        'all_existing': set()  # Will collect all unique existing annotations
                    }
                    
                    # Collect all predictions and existing annotations for oracle calculation
                    for _, row in model_data.iterrows():
                        # Handle predictions
                        preds = row.get('predictions', [])
                        if isinstance(preds, str):
                            try:
                                import ast
                                preds = ast.literal_eval(preds)
                            except:
                                preds = []
                        if isinstance(preds, list):
                            llm_model_performances[llm_name]['all_predictions'].update(preds)
                        
                        # Handle existing annotations
                        existing = row.get('exist_annotation_id', [])
                        if isinstance(existing, str):
                            try:
                                import ast
                                existing = ast.literal_eval(existing)
                            except:
                                existing = []
                        if isinstance(existing, list):
                            llm_model_performances[llm_name]['all_existing'].update(existing)
            
            if len(llm_model_performances) < 2:
                continue  # Skip if not enough LLMs have data for this model
            
            # Calculate synergy metrics for this model
            model_results['model'] = model_name
            model_results['num_species'] = list(llm_model_performances.values())[0]['num_species']
            
            # Individual LLM performances
            for llm_name in llm_names:
                if llm_name in llm_model_performances:
                    model_results[f'{llm_name}_avg_recall_formula'] = llm_model_performances[llm_name]['avg_recall_formula']
                    model_results[f'{llm_name}_avg_precision_formula'] = llm_model_performances[llm_name]['avg_precision_formula']
                    model_results[f'{llm_name}_avg_accuracy'] = llm_model_performances[llm_name]['avg_accuracy']
                    model_results[f'{llm_name}_llm_time'] = llm_model_performances[llm_name]['llm_time']
                else:
                    model_results[f'{llm_name}_avg_recall_formula'] = 0
                    model_results[f'{llm_name}_avg_precision_formula'] = 0
                    model_results[f'{llm_name}_avg_accuracy'] = 0
                    model_results[f'{llm_name}_llm_time'] = 0
            
            # Best individual performance across LLMs
            best_recall = max([llm_model_performances[llm]['avg_recall_formula'] for llm in llm_model_performances])
            best_precision = max([llm_model_performances[llm]['avg_precision_formula'] for llm in llm_model_performances])
            best_accuracy = max([llm_model_performances[llm]['avg_accuracy'] for llm in llm_model_performances])
            
            model_results['best_individual_recall'] = best_recall
            model_results['best_individual_precision'] = best_precision
            model_results['best_individual_accuracy'] = best_accuracy
            
            # Oracle combination (union of all predictions)
            all_predictions = set()
            all_existing = set()
            for llm_name in llm_model_performances:
                all_predictions.update(llm_model_performances[llm_name]['all_predictions'])
                all_existing.update(llm_model_performances[llm_name]['all_existing'])
            
            # Calculate oracle performance
            if all_existing:
                oracle_matches = all_predictions & all_existing
                oracle_recall = len(oracle_matches) / len(all_existing) if all_existing else 0
                oracle_precision = len(oracle_matches) / len(all_predictions) if all_predictions else 0
                oracle_accuracy = 1 if oracle_recall > 0 else 0
            else:
                oracle_recall = 0
                oracle_precision = 0
                oracle_accuracy = 0
            
            model_results['oracle_recall'] = oracle_recall
            model_results['oracle_precision'] = oracle_precision
            model_results['oracle_accuracy'] = oracle_accuracy
            
            # Synergy potential (improvement over best individual)
            model_results['synergy_recall_gain'] = oracle_recall - best_recall
            model_results['synergy_precision_gain'] = oracle_precision - best_precision
            model_results['synergy_accuracy_gain'] = oracle_accuracy - best_accuracy
            
            # Performance complementarity (how much LLMs differ on this model)
            recall_values = [llm_model_performances[llm]['avg_recall_formula'] for llm in llm_model_performances]
            precision_values = [llm_model_performances[llm]['avg_precision_formula'] for llm in llm_model_performances]
            accuracy_values = [llm_model_performances[llm]['avg_accuracy'] for llm in llm_model_performances]
            
            model_results['recall_std'] = np.std(recall_values)
            model_results['precision_std'] = np.std(precision_values)
            model_results['accuracy_std'] = np.std(accuracy_values)
            
            # Which LLM performed best for this model
            best_llm_recall = max(llm_model_performances.keys(), key=lambda x: llm_model_performances[x]['avg_recall_formula'])
            best_llm_precision = max(llm_model_performances.keys(), key=lambda x: llm_model_performances[x]['avg_precision_formula'])
            best_llm_accuracy = max(llm_model_performances.keys(), key=lambda x: llm_model_performances[x]['avg_accuracy'])
            
            model_results['best_llm_recall'] = best_llm_recall
            model_results['best_llm_precision'] = best_llm_precision
            model_results['best_llm_accuracy'] = best_llm_accuracy
            
            synergy_results.append(model_results)
    
    else:  # species level
        # Get all unique (model, species_id) pairs
        all_species_pairs = set()
        for df in llm_dfs:
            pairs = list(zip(df['model'], df['species_id']))
            all_species_pairs.update(pairs)
        
        # Analyze each species
        for model_name, species_id in all_species_pairs:
            species_results = {}
            
            # Get results for this species from each LLM
            llm_species_performances = {}
            for llm_name, df in zip(llm_names, llm_dfs):
                species_data = df[(df['model'] == model_name) & (df['species_id'] == species_id)]
                
                if not species_data.empty:
                    row = species_data.iloc[0]
                    
                    # Get predictions and existing annotations
                    preds = row.get('predictions', [])
                    if isinstance(preds, str):
                        try:
                            import ast
                            preds = ast.literal_eval(preds)
                        except:
                            preds = []
                    
                    existing = row.get('exist_annotation_id', [])
                    if isinstance(existing, str):
                        try:
                            import ast
                            existing = ast.literal_eval(existing)
                        except:
                            existing = []
                    
                    llm_species_performances[llm_name] = {
                        'recall_exact': row['recall_exact'],
                        'precision_exact': row['precision_exact'],
                        'recall_formula': row['recall_formula'],
                        'precision_formula': row['precision_formula'],
                        'accuracy': row['accuracy'],
                        'predictions': set(preds) if isinstance(preds, list) else set(),
                        'existing': set(existing) if isinstance(existing, list) else set()
                    }
            
            if len(llm_species_performances) < 2:
                continue  # Skip if not enough LLMs have data for this species
            
            # Calculate synergy metrics for this species
            species_results['model'] = model_name
            species_results['species_id'] = species_id
            
            # Individual LLM performances
            for llm_name in llm_names:
                if llm_name in llm_species_performances:
                    species_results[f'{llm_name}_recall_formula'] = llm_species_performances[llm_name]['recall_formula']
                    species_results[f'{llm_name}_precision_formula'] = llm_species_performances[llm_name]['precision_formula']
                    species_results[f'{llm_name}_accuracy'] = llm_species_performances[llm_name]['accuracy']
                else:
                    species_results[f'{llm_name}_recall_formula'] = 0
                    species_results[f'{llm_name}_precision_formula'] = 0
                    species_results[f'{llm_name}_accuracy'] = 0
            
            # Best individual performance across LLMs
            best_recall = max([llm_species_performances[llm]['recall_formula'] for llm in llm_species_performances])
            best_precision = max([llm_species_performances[llm]['precision_formula'] for llm in llm_species_performances])
            best_accuracy = max([llm_species_performances[llm]['accuracy'] for llm in llm_species_performances])
            
            species_results['best_individual_recall'] = best_recall
            species_results['best_individual_precision'] = best_precision
            species_results['best_individual_accuracy'] = best_accuracy
            
            # Oracle combination (union of all predictions)
            all_predictions = set()
            all_existing = set()
            for llm_name in llm_species_performances:
                all_predictions.update(llm_species_performances[llm_name]['predictions'])
                all_existing.update(llm_species_performances[llm_name]['existing'])
            
            # Calculate oracle performance
            if all_existing:
                oracle_matches = all_predictions & all_existing
                oracle_recall = len(oracle_matches) / len(all_existing) if all_existing else 0
                oracle_precision = len(oracle_matches) / len(all_predictions) if all_predictions else 0
                oracle_accuracy = 1 if oracle_recall > 0 else 0
            else:
                oracle_recall = 0
                oracle_precision = 0
                oracle_accuracy = 0
            
            species_results['oracle_recall'] = oracle_recall
            species_results['oracle_precision'] = oracle_precision
            species_results['oracle_accuracy'] = oracle_accuracy
            
            # Synergy potential (improvement over best individual)
            species_results['synergy_recall_gain'] = oracle_recall - best_recall
            species_results['synergy_precision_gain'] = oracle_precision - best_precision
            species_results['synergy_accuracy_gain'] = oracle_accuracy - best_accuracy
            
            # Which LLM performed best for this species
            best_llm_recall = max(llm_species_performances.keys(), key=lambda x: llm_species_performances[x]['recall_exact'])
            best_llm_precision = max(llm_species_performances.keys(), key=lambda x: llm_species_performances[x]['precision_exact'])
            best_llm_accuracy = max(llm_species_performances.keys(), key=lambda x: llm_species_performances[x]['accuracy'])
            
            species_results['best_llm_recall'] = best_llm_recall
            species_results['best_llm_precision'] = best_llm_precision
            species_results['best_llm_accuracy'] = best_llm_accuracy
            
            synergy_results.append(species_results)
    
    # Convert to DataFrame
    synergy_df = pd.DataFrame(synergy_results)
    
    if synergy_df.empty:
        print("No overlapping data found between LLM results")
        return pd.DataFrame()
    
    # Save results
    synergy_df.to_csv(output_file, index=False)
    print(f"Saved synergy analysis to {output_file}")
    
    # Print summary statistics
    if analysis_level == 'model':
        print("\n=== LLM SYNERGY ANALYSIS SUMMARY (PER-MODEL) ===")
        print(f"Total models analyzed: {len(synergy_df)}")
        
        # Overall synergy potential
        avg_synergy_recall = synergy_df['synergy_recall_gain'].mean()
        avg_synergy_precision = synergy_df['synergy_precision_gain'].mean()
        avg_synergy_accuracy = synergy_df['synergy_accuracy_gain'].mean()
        
        print(f"Average synergy gains:")
        print(f"  Recall: {avg_synergy_recall:.3f}")
        print(f"  Precision: {avg_synergy_precision:.3f}")
        print(f"  Accuracy: {avg_synergy_accuracy:.3f}")
        
        # Models with highest synergy potential
        high_synergy = synergy_df.nlargest(100, 'synergy_recall_gain')
        print(f"\nTop 100 models with highest synergy potential (recall):")
        for _, row in high_synergy.iterrows():
            print(f"  {row['model']} (gain: {row['synergy_recall_gain']:.3f})")
        
        # LLM performance distribution
        print(f"\nLLM performance distribution:")
        for llm_name in llm_names:
            if f'{llm_name}_avg_recall_exact' in synergy_df.columns:
                avg_recall = synergy_df[f'{llm_name}_avg_recall_exact'].mean()
                best_count = sum(synergy_df['best_llm_recall'] == llm_name)
                print(f"  {llm_name}: avg recall {avg_recall:.3f}, best on {best_count} models")
        
        # Complementarity analysis
        avg_recall_std = synergy_df['recall_std'].mean()
        avg_precision_std = synergy_df['precision_std'].mean()
        print(f"\nLLM Complementarity (higher = more diverse):")
        print(f"  Average recall std dev: {avg_recall_std:.3f}")
        print(f"  Average precision std dev: {avg_precision_std:.3f}")
        
        # Distribution of which LLM is best
        print(f"\nBest LLM distribution (by recall):")
        best_llm_counts = synergy_df['best_llm_recall'].value_counts()
        for llm_name, count in best_llm_counts.items():
            print(f"  {llm_name}: best on {count} models ({100*count/len(synergy_df):.1f}%)")
    
    else:  # species level
        print("\n=== LLM SYNERGY ANALYSIS SUMMARY (PER-SPECIES) ===")
        print(f"Total species analyzed: {len(synergy_df)}")
        
        # Overall synergy potential
        avg_synergy_recall = synergy_df['synergy_recall_gain'].mean()
        avg_synergy_precision = synergy_df['synergy_precision_gain'].mean()
        avg_synergy_accuracy = synergy_df['synergy_accuracy_gain'].mean()
        
        print(f"Average synergy gains:")
        print(f"  Recall: {avg_synergy_recall:.3f}")
        print(f"  Precision: {avg_synergy_precision:.3f}")
        print(f"  Accuracy: {avg_synergy_accuracy:.3f}")
        
        # Species with highest synergy potential
        high_synergy = synergy_df.nlargest(5, 'synergy_recall_gain')
        print(f"\nTop 5 species with highest synergy potential (recall):")
        for _, row in high_synergy.iterrows():
            print(f"  {row['model']}:{row['species_id']} (gain: {row['synergy_recall_gain']:.3f})")
        
        # LLM performance distribution
        print(f"\nLLM performance distribution:")
        for llm_name in llm_names:
            if f'{llm_name}_recall_exact' in synergy_df.columns:
                avg_recall = synergy_df[f'{llm_name}_recall_exact'].mean()
                best_count = sum(synergy_df['best_llm_recall'] == llm_name)
                print(f"  {llm_name}: avg recall {avg_recall:.3f}, best on {best_count} species")
        
        # Distribution of which LLM is best
        print(f"\nBest LLM distribution (by recall):")
        best_llm_counts = synergy_df['best_llm_recall'].value_counts()
        for llm_name, count in best_llm_counts.items():
            print(f"  {llm_name}: best on {count} species ({100*count/len(synergy_df):.1f}%)")
    
    return synergy_df

def filter_qualifiers_in_results(results_csv: str, 
                                 bqbiol_qualifiers: List[str],
                                 model_dir: str,
                                 output_csv: str = None,
                                 entity_type: str = "chemical",
                                 database: str = "chebi") -> pd.DataFrame:
    """
    Filter previously saved results to only include species that use the specified bqbiol qualifiers.
    
    Args:
        results_csv: Path to CSV file containing previously saved evaluation results
        bqbiol_qualifiers: List of bqbiol qualifiers to filter for (e.g. ['is', 'isVersionOf', 'hasPart'])
        model_dir: Directory containing the original SBML model files
        output_csv: Path to save filtered results (optional)
        entity_type: Type of entity ("chemical" or "gene")
        database: Database being used ("chebi" or "ncbigene")
        
    Returns:
        Filtered DataFrame containing only results for species with the specified qualifiers
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    # Load the results
    df = pd.read_csv(results_csv)
    
    if df.empty:
        return df
    
    # Get unique models from the results
    models_in_results = df['model'].unique()
    
    filtered_rows = []
    
    for model_name in models_in_results:
        # Find the corresponding model file
        model_file = None
        for ext in ['.xml', '.sbml']:
            potential_path = os.path.join(model_dir, model_name.replace('.xml', '').replace('.sbml', '') + ext)
            if os.path.exists(potential_path):
                model_file = potential_path
                break
        
        if not model_file:
            logger.warning(f"Model file not found for {model_name}, skipping")
            continue
        
        # Get annotations for this model using the specified qualifiers
        if entity_type == "chemical" and database == "chebi":
            qualified_annotations = find_species_with_chebi_annotations(model_file, bqbiol_qualifiers)
        elif entity_type == "gene" and database == "ncbigene":
            qualified_annotations = find_species_with_ncbigene_annotations(model_file, bqbiol_qualifiers)
        else:
            logger.warning(f"Entity type {entity_type} with database {database} not supported")
            continue
        
        # Filter rows for this model to only include species with qualified annotations
        model_rows = df[df['model'] == model_name]
        for _, row in model_rows.iterrows():
            species_id = row['species_id']
            if species_id in qualified_annotations:
                filtered_rows.append(row)
    
    # Create filtered DataFrame
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.reset_index(drop=True, inplace=True)
    else:
        filtered_df = pd.DataFrame()
    
    # Save if output path provided
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        logger.info(f"Filtered results saved to: {output_csv}")
    
    return filtered_df

def analyze_bqbiol_qualifier_statistics(model_dir: str, 
                                       output_file: str = 'bqbiol_qualifier_statistics.csv',
                                       verbose: bool = True) -> pd.DataFrame:
    """
    Analyze how many species have annotations using each bqbiol qualifier across all models,
    broken down by ontology (chebi, uniprot, ncbigene, etc.).
    
    Args:
        model_dir: Directory containing SBML model files
        output_file: Path to save the statistics table
        verbose: If True, show detailed logging
        
    Returns:
        DataFrame with statistics table where rows are ontologies, columns are qualifiers,
        and cells show the number of species that contain that ontology term for that qualifier
    """
    from core.model_info import find_species_with_chebi_annotations, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, detect_model_format
    from utils.constants import CHEBI_URI_PATTERNS, NCBIGENE_URI_PATTERNS, UNIPROT_URI_PATTERNS, ModelType
    
    # Common bqbiol qualifiers to check
    common_qualifiers = [
        'is', 'isVersionOf', 'hasVersion', 'isDescribedBy', 'hasPart','isPartOf',
        'hasProperty', 'isPropertyOf', 'isEncodedBy', 'encodes', 'isHomologTo',  
        'occursIn', 'hasTaxon', 'isRelatedTo'
    ]
    
    # Initialize statistics storage - start with main ontologies, others will be added dynamically
    ontology_stats = {
        'chebi': {qualifier: 0 for qualifier in common_qualifiers},
        'ncbigene': {qualifier: 0 for qualifier in common_qualifiers},
        'uniprot': {qualifier: 0 for qualifier in common_qualifiers},
    }
    
    models_with_qualifiers = {qualifier: set() for qualifier in common_qualifiers}
    model_type_counts = {'SBML': 0, 'SBML-qual': 0, 'SBML-fbc': 0}
    discovered_ontologies = set()
    
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml') or f.endswith('.sbml')]
    
    if verbose:
        print(f"Analyzing {len(model_files)} models for bqbiol qualifier statistics...")
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        if verbose:
            print(f"Processing {model_file}...")
        
        # Detect model type
        try:
            model_type, format_info = detect_model_format(model_path)
            model_type_counts[model_type.value] += 1
        except Exception as e:
            if verbose:
                logger.warning(f"Error detecting model type for {model_file}: {e}")
            continue
        
        for qualifier in common_qualifiers:
            # Check for each ontology using specific qualifiers
            try:
                # ChEBI annotations
                chebi_annotations = find_species_with_chebi_annotations(model_path, [qualifier])
                if chebi_annotations:
                    ontology_stats['chebi'][qualifier] += len(chebi_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # NCBI Gene annotations  
                ncbigene_annotations = find_species_with_ncbigene_annotations(model_path, [qualifier])
                if ncbigene_annotations:
                    ontology_stats['ncbigene'][qualifier] += len(ncbigene_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # UniProt annotations
                uniprot_annotations = find_species_with_uniprot_annotations(model_path, [qualifier])
                if uniprot_annotations:
                    ontology_stats['uniprot'][qualifier] += len(uniprot_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # Detect and count other ontologies
                other_ontologies = _detect_and_count_other_ontologies(model_path, qualifier, model_type)
                for ontology, count in other_ontologies.items():
                    if count > 0:
                        # Add new ontology if not seen before
                        if ontology not in ontology_stats:
                            ontology_stats[ontology] = {q: 0 for q in common_qualifiers}
                        
                        ontology_stats[ontology][qualifier] += count
                        models_with_qualifiers[qualifier].add(model_file)
                        discovered_ontologies.add(ontology)
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Error processing {model_file} for qualifier {qualifier}: {e}")
                continue
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(ontology_stats).T  # Transpose so ontologies are rows
    stats_df.index.name = 'Ontology'
    
    # Add a summary row showing number of models that contain each qualifier
    model_counts = {qualifier: len(models_with_qualifiers[qualifier]) for qualifier in common_qualifiers}
    summary_row = pd.DataFrame([model_counts], index=['Models_with_qualifier'])
    stats_df = pd.concat([stats_df, summary_row])
    
    # Save results
    stats_df.to_csv(output_file)
    if verbose:
        print(f"Statistics saved to: {output_file}")
        print(f"\nModel type distribution:")
        for model_type, count in model_type_counts.items():
            print(f"  {model_type}: {count} models")
        
        print(f"\nDiscovered ontologies: {sorted(discovered_ontologies)}")
        print("\nSummary:")
        print(stats_df)
        
        print(f"\nNumber of models that contain at least one annotation for each qualifier:")
        for qualifier in common_qualifiers:
            count = len(models_with_qualifiers[qualifier])
            print(f"  {qualifier}: {count} models")
    
    return stats_df

def _detect_and_count_other_ontologies(model_file: str, qualifier: str, model_type: 'ModelType') -> Dict[str, int]:
    """
    Enhanced helper function to detect and count annotations for various ontologies for a specific qualifier.
    Handles different model types (SBML, SBML_FBC, SBML_QUAL).
    Extracts ontology names from identifiers.org URLs automatically.
    """
    import libsbml
    from utils.constants import ModelType
    
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    ontology_counts = {}
    
    def extract_ontologies_from_qualifier_content(qualifier_content: str):
        """Extract ontology names from identifiers.org URLs in qualifier content."""
        # Pattern to match the term after 'identifiers.org/' and before any following / or :
        identifiers_pattern = r'http[s]?://identifiers\.org/([^/:]+)'
        miriam_pattern = r'urn:miriam:([^:\s<>"]+)'
        
        # Find all identifiers.org URLs
        identifiers_matches = re.findall(identifiers_pattern, qualifier_content)
        miriam_matches = re.findall(miriam_pattern, qualifier_content)
        
        # Combine and count ontologies
        all_ontologies = identifiers_matches + miriam_matches
        
        for ontology in all_ontologies:
            # Clean up ontology name (remove any trailing characters)
            ontology = ontology.strip()
            if ontology and ontology not in ['chebi', 'ncbigene', 'uniprot']:  # Skip main ones we handle separately
                if ontology not in ontology_counts:
                    ontology_counts[ontology] = 0
                ontology_counts[ontology] += 1
    
    def check_annotations_for_species(species_list):
        """Helper to check annotations for a list of species objects."""
        for species in species_list:
            if species.isSetAnnotation():
                annotation_str = species.getAnnotation().toXMLString()
                
                # Check if this species has the qualifier
                qualifier_match = re.search(
                    r'<bqbiol:{}[^>]*?>.*?</bqbiol:{}>'.format(
                        re.escape(qualifier), re.escape(qualifier)
                    ), 
                    annotation_str, 
                    flags=re.DOTALL
                )
                
                if qualifier_match:
                    qualifier_content = qualifier_match.group(0)
                    extract_ontologies_from_qualifier_content(qualifier_content)
    
    # Handle different model types
    if model_type == ModelType.SBML:
        # Regular SBML models - check species
        check_annotations_for_species(model.getListOfSpecies())
        
    elif model_type == ModelType.SBML_FBC:
        # SBML-FBC models - check species and gene products
        check_annotations_for_species(model.getListOfSpecies())
        
        fbc_plugin = model.getPlugin("fbc")
        if fbc_plugin:
            gene_products = []
            for gene_product in fbc_plugin.getListOfGeneProducts():
                gene_products.append(gene_product)
            check_annotations_for_species(gene_products)
    
    elif model_type == ModelType.SBML_QUAL:
        # SBML-qual models - check qualitative species
        qual_plugin = model.getPlugin("qual")
        if qual_plugin:
            qual_species = []
            for qual_spec in qual_plugin.getListOfQualitativeSpecies():
                qual_species.append(qual_spec)
            check_annotations_for_species(qual_species)
    
    return ontology_counts
