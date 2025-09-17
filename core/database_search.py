"""
Database Search for AAAIM

Handles database searches for annotation candidates.
Currently supports ChEBI, extensible to other databases.
"""

import os
import re
import lzma
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from collections import Counter
from itertools import product
import sys
import chromadb
from chromadb.utils import embedding_functions
from utils.constants import REF_CHEBI2LABEL, REF_NAMES2CHEBI, REF_NCBIGENE2LABEL, REF_NAMES2NCBIGENE, REF_UNIPROT2LABEL, REF_NAMES2UNIPROT
from utils.constants import REF_CHEBI2KEGG_COMPOUND, REF_KEGG_REACTION2NAME, REF_KEGG2EC, REF_KEGG_REACTION_FEATURES, REF_KEGG_PARSED_REACTIONS
from core.data_types import Recommendation, ReactionRecommendation


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Global ChromaDB client cache to avoid conflicts
_CHROMADB_CLIENTS = {}

# Cache for loaded dictionaries
_CHEBI_CLEANNAMES_DICT: Optional[Dict[str, List[str]]] = None
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_NCBIGENE_NAMES_DICT: Optional[Dict[str, List[str]]] = None
_NCBIGENE_LABEL_DICT: Optional[Dict[str, str]] = None
_UNIPROT_NAMES_DICT: Optional[Dict[str, List[str]]] = None
_UNIPROT_LABEL_DICT: Optional[Dict[str, str]] = None
_CHEBI2KEGG_DICT: Optional[Dict[str, str]] = None
_KEGG_REACTION2NAME_DICT: Optional[Dict[str, str]] = None
_KEGG2EC_DICT: Optional[Dict[str, Dict[str, List[str]]]] = None
_KEGG_REACTION_FEATURES_DICT: Optional[Dict[str, Dict[str, Any]]] = None
_KEGG_PARSED_REACTIONS_DICT: Optional[Dict[str, Dict[str, Any]]] = None

def get_data_dir() -> Path:
    """Get the path to the AAAIM data directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data" 

def load_chebi_cleannames_dict() -> Dict[str, List[str]]:
    """
    Load the ChEBI clean names to ChEBI ID dictionary.
    
    Returns:
        Dictionary mapping clean names to lists of ChEBI IDs
    """
    global _CHEBI_CLEANNAMES_DICT
    
    if _CHEBI_CLEANNAMES_DICT is None:
        data_file = get_data_dir() / "chebi" / REF_NAMES2CHEBI
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI cleannames data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_CLEANNAMES_DICT = pickle.load(f)
    
    return _CHEBI_CLEANNAMES_DICT

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = get_data_dir() / "chebi" / REF_CHEBI2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def load_ncbigene_names_dict(tax_id: str = None) -> Dict[str, List[str]]:
    """
    Load the NCBI gene names to NCBI gene ID dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the old combined file for backwards compatibility.
    
    Returns:
        Dictionary mapping clean names to lists of NCBI gene IDs
    """
    global _NCBIGENE_NAMES_DICT
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"ncbigene_names_{tax_id or 'combined'}"
    
    # Check if we have this specific version cached
    if not hasattr(load_ncbigene_names_dict, '_cache'):
        load_ncbigene_names_dict._cache = {}
    
    if cache_key in load_ncbigene_names_dict._cache:
        return load_ncbigene_names_dict._cache[cache_key]
    
    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "ncbigene" / f"names2ncbigene_tax{tax_id}_protein-coding.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "ncbigene" / REF_NAMES2NCBIGENE
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"NCBI gene names data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"NCBI gene names data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        names_dict = pickle.load(f)
    
    # Cache the result
    load_ncbigene_names_dict._cache[cache_key] = names_dict
    
    return names_dict

def load_ncbigene_label_dict() -> Dict[str, str]:
    """
    Load the NCBI gene ID to label dictionary.
    
    Returns:
        Dictionary mapping NCBI gene IDs to their labels
    """
    global _NCBIGENE_LABEL_DICT
    
    if _NCBIGENE_LABEL_DICT is None:
        data_file = get_data_dir() / "ncbigene" / REF_NCBIGENE2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"NCBI gene label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _NCBIGENE_LABEL_DICT = pickle.load(f)
    
    return _NCBIGENE_LABEL_DICT

def load_uniprot_names_dict(tax_id: str = None) -> Dict[str, List[str]]:
    """
    Load the UniProt clean names to UniProt ID dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the old combined file for backwards compatibility.
    
    Returns:
        Dictionary mapping clean names to lists of UniProt IDs
    """
    global _UNIPROT_NAMES_DICT
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"uniprot_names_{tax_id or 'combined'}"
    
    # Check if we have this specific version cached
    if not hasattr(load_uniprot_names_dict, '_cache'):
        load_uniprot_names_dict._cache = {}
    
    if cache_key in load_uniprot_names_dict._cache:
        return load_uniprot_names_dict._cache[cache_key]
    
    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "uniprot" / f"names2uniprot_tax{tax_id}.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "uniprot" / REF_NAMES2UNIPROT
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"UniProt names data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"UniProt names data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        names_dict = pickle.load(f)
    
    # Cache the result
    load_uniprot_names_dict._cache[cache_key] = names_dict
    
    return names_dict

def load_uniprot_label_dict(tax_id: str = None) -> Dict[str, str]:
    """
    Load the UniProt ID to label dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the combined file
    
    Returns:
        Dictionary mapping UniProt IDs to their labels
    """
    global _UNIPROT_LABEL_DICT
    
    if _UNIPROT_LABEL_DICT is not None:
        return _UNIPROT_LABEL_DICT

    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "uniprot" / f"uniprot2label_tax{tax_id}.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "uniprot" / REF_UNIPROT2LABEL
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"UniProt label data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"UniProt label data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        label_dict = pickle.load(f)
    
    return label_dict

def load_chebi2kegg_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to KEGG compound ID mapping dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to KEGG compound IDs
    """
    global _CHEBI2KEGG_DICT
    
    if _CHEBI2KEGG_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_CHEBI2KEGG_COMPOUND
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI to KEGG compound mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI2KEGG_DICT = pickle.load(f)
    
    return _CHEBI2KEGG_DICT

def load_kegg_label_dict(): 
    return [] # JANISTAG

def load_kegg_reaction2name_dict() -> Dict[str, str]:
    """
    Load the KEGG reaction ID to name dictionary.
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their names
    """
    global _KEGG_REACTION2NAME_DICT
    
    if _KEGG_REACTION2NAME_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_REACTION2NAME
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG reaction to name mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_REACTION2NAME_DICT = pickle.load(f)
    
    return _KEGG_REACTION2NAME_DICT

def load_kegg2ec_dict() -> Dict[str, Dict[str, List[str]]]:
    """
    Load the KEGG ID to EC number mapping dictionary.
    
    Returns:
        Dictionary mapping KEGG IDs to EC numbers with additional metadata
    """
    global _KEGG2EC_DICT
    
    if _KEGG2EC_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG2EC
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG to EC mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG2EC_DICT = pickle.load(f)
    
    return _KEGG2EC_DICT

def load_kegg_reaction_features_dict() -> Dict[str, Dict[str, Any]]:
    """
    Load the parsed KEGG reactions dictionary containing detailed reaction features.
    
    The dictionary contains information about KEGG reactions including:
        {'R01600': {
            'ENTRY': 'R01600                      Reaction',
            'NAME': 'ATP:beta-D-glucose 6-phosphotransferase',
            'DEFINITION': 'ATP + beta-D-Glucose <=> ADP + beta-D-Glucose 6
            'EQUATION': 'C00002 + C00221 <=> C00008 + C01172',
            'RCLASS': 'RC00002  C00002_C00008\nRC00017  C00221_C01172',
            'ENZYME': '2.7.1.1         2.7.1.2',
            'PATHWAY': 'rn00010  Glycolysis / Gluconeogenesis\nrn01100 
            'BRITE': 'Enzymatic reactions [BR:br08201]\n2. Transferase 
            'ORTHOLOGY': 'K00844  hexokinase [EC:2.7.1.1]\nK00845  glucoki
            }
        }
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their feature dictionaries
    """
    global _KEGG_REACTION_FEATURES_DICT
    
    if _KEGG_REACTION_FEATURES_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_REACTION_FEATURES
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG reaction features data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_REACTION_FEATURES_DICT = pickle.load(f)
    
    return _KEGG_REACTION_FEATURES_DICT


def load_kegg_parsed_reactions_dict() -> Dict[str, Dict[str, Any]]:
    """
    Load the list of dicts containing detailed reaction features.
    
    Each dictionary contains information about KEGG reactions including:
    - 'reaction_id': 'R00002',
    - 'name': 'reduced ferredoxin:dinitrogen oxidoreductase (ATP-hydrolysing)',
    - 'ec_numbers': ['1.18.6.1'],
    - 'direction': 'reversible',
    - 'substrates': ['C00002', 'C00001', 'C00138'],
    - 'products': ['C05359', 'C00009', 'C00008', 'C00139'],
    - 'pathways': [],
    - 'raw_equation': '16 C00002 + 16 C00001 + 8 C00138 <=> 8 C05359 + 16 C00009 + 16 C00008 + 8 C00139'}
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their feature dictionaries
    """
    global _KEGG_PARSED_REACTIONS_DICT
    
    if _KEGG_PARSED_REACTIONS_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_PARSED_REACTIONS
        
        if not data_file.exists():
            raise FileNotFoundError(f"Parsed KEGG reactions data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_PARSED_REACTIONS_DICT = pickle.load(f)
    
    return _KEGG_PARSED_REACTIONS_DICT



def remove_symbols(text: str) -> str:
    """
    Remove all characters except numbers and letters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with only alphanumeric characters
    """
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def extract_classifications(raw_text, classification):
    """
    classification (str): either 'brite' or 'orthology'
    Extracts only the BRITE hierarchy (excluding [BR:...] tags, EC leaf nodes, 
    and reaction entries).
    """
    lines = raw_text.splitlines()
    clean_lines = []

    if classification == 'brite':
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # Skip lines with [BR:...] tags
            if "[BR:" in stripped:
                continue
            # Skip EC leaf numbers (pure numbers like 2.2.1.6)
            if re.fullmatch(r"(\d+\.)+\d+", stripped):
                continue
            # Skip lines that start with an R number (reaction ID)
            if re.match(r"R\d{5}", stripped):
                continue
            
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1:
                clean_lines.append(parts[1].strip())
            else:
                clean_lines.append(stripped)
        return "; ".join(clean_lines)
    
    elif classification == 'orthology':
        for line in lines:
        # Split once on spaces to remove the Kxxxxx ID
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                # Remove the EC info if present
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)
        return "; ".join(clean_lines)
    

def get_species_recommendations_direct(species_ids: List[str], synonyms_dict, database: str = "chebi", tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find recommendations by directly matching against database synonyms.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - database (str): Database to search ("chebi", "ncbigene", "uniprot")
    - tax_id (str/list): For ncbigene/uniprot database, the organism's tax_id for organism-specific lookup. If list, search all tax_ids for each species.
    - top_k (int): Number of top candidates to return per species based on hit_count.
    
    Returns:
    - list: List of Recommendation objects with candidates and names.
    """
    if database == "chebi":
        return _get_chebi_recommendations_direct(species_ids, synonyms_dict, top_k=top_k)
    elif database == "ncbigene":
        return _get_ncbigene_recommendations_direct(species_ids, synonyms_dict, tax_id=tax_id, top_k=top_k)
    elif database == "uniprot":
        return _get_uniprot_recommendations_direct(species_ids, synonyms_dict, tax_id=tax_id, top_k=top_k)
    elif database == "kegg":
        # return _get_kegg_recommendations_rulebased(species_ids, synonyms_dict, top_k=top_k)
        return _get_kegg_recommendations_direct(species_ids, synonyms_dict, top_k=top_k)
    else:
        logger.error(f"Database {database} not supported for direct search")
        return []

def _get_chebi_recommendations_direct(species_ids: List[str], synonyms_dict, top_k: int = 3) -> List[Recommendation]:
    """
    Find ChEBI recommendations by directly matching against ChEBI synonyms.
    """
    cleannames_dict = load_chebi_cleannames_dict()
    label_dict = load_chebi_label_dict()
    
    recommendations = []
    
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}  # Dictionary to track how many times each candidate appears
        
        # Query for each synonym
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            # Check all entries in cleannames dict for matches
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_synonym == ref_name.lower():
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        
                        if chebi_id not in all_candidates:
                            all_candidates.append(chebi_id)
                            all_candidate_names.append(chebi_name)
                            hit_count[chebi_id] = 1
                        else:
                            hit_count[chebi_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        # Calculate normalized match scores (hit_count / number_of_synonyms)
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    
    return recommendations

def _get_ncbigene_recommendations_direct(species_ids: List[str], synonyms_dict, tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find NCBI gene recommendations by directly matching against NCBI gene synonyms.
    Args:
        species_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        tax_id: Organism's tax_id for each species (str, list). If list, search all tax_ids for each species.
        top_k: Number of top candidates to return per species based on hit_count.
    """
    label_dict = load_ncbigene_label_dict()
    recommendations = []
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        # Determine which tax_ids to search
        if isinstance(tax_id, list):
            tax_ids_to_search = tax_id
        else:
            tax_ids_to_search = [tax_id]
        # Query for each synonym and each tax_id
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            for tid in tax_ids_to_search:
                try:
                    names_dict = load_ncbigene_names_dict(tax_id=tid)
                except Exception:
                    logger.warning(f"Error loading NCBI gene names for tax_id {tid}: {e}")
                    continue
                for ref_name, gene_ids in names_dict.items():
                    if norm_synonym == ref_name.lower():
                        for gene_id in gene_ids:
                            gene_name = label_dict.get(gene_id, gene_id)
                            if gene_id not in all_candidates:
                                all_candidates.append(gene_id)
                                all_candidate_names.append(gene_name)
                                hit_count[gene_id] = 1
                            else:
                                hit_count[gene_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations

def _get_uniprot_recommendations_direct(species_ids: List[str], synonyms_dict, tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find UniProt recommendations by directly matching against UniProt synonyms.
    Args:
        species_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        tax_id: Organism's tax_id for each species (str, list). If list, search all tax_ids for each species.
        top_k: Number of top candidates to return per species based on hit_count.
    """
    label_dict = load_uniprot_label_dict(tax_id=tax_id)
    recommendations = []
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        # Determine which tax_ids to search
        if isinstance(tax_id, list):
            tax_ids_to_search = tax_id
        else:
            tax_ids_to_search = [tax_id]
        # Query for each synonym and each tax_id
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            for tid in tax_ids_to_search:
                try:
                    names_dict = load_uniprot_names_dict(tax_id=tid)
                except Exception as e:
                    logger.warning(f"Error loading UniProt names for tax_id {tid}: {e}")
                    continue
                for ref_name, uniprot_ids in names_dict.items():
                    if norm_synonym == ref_name.lower():
                        for uniprot_id in uniprot_ids:
                            uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                            if uniprot_id not in all_candidates:
                                all_candidates.append(uniprot_id)
                                all_candidate_names.append(uniprot_name)
                                hit_count[uniprot_id] = 1
                            else:
                                hit_count[uniprot_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations


def _get_kegg_recommendations_rulebased(normalized_reactions, cofactors_to_ignore: set = {}, top_k: int = None, spectators=False) -> List[Recommendation]:
    """
    Find KEGG reaction recommendations by matching model reactions to KEGG reactions.
    
    Args:
        species_ids: List of reaction IDs to evaluate
        cofactors_to_ignore: Set of KEGG IDs of cofactors to ignore
        top_k: Number of top candidates to return per reaction
        
    Returns:
        List of Recommendation objects with candidates and match scores
    """
    try:
        logger.info(f"Loading KEGG reaction data...")
        # Load KEGG reaction data
        kegg_parsed_reactions_dict = load_kegg_parsed_reactions_dict()
        kegg_reaction_features_dict = load_kegg_reaction_features_dict()
        logger.info(f"Loaded {len(kegg_parsed_reactions_dict)} parsed KEGG reactions")
        logger.info(f"Loaded {len(kegg_reaction_features_dict)} KEGG reaction features")
        
        recommendations = []
        
        for reaction_id in normalized_reactions:
            reaction_label = reaction_id.get('id')
            
            reaction_str = reaction_id.get('reaction_string')
            # Extract substrate and product counters
            model_subs = reaction_id.get('substrates', Counter())
            model_prods = reaction_id.get('products', Counter())
            
            # Check if reactions only contain cofactors
            model_sub_keys = {key for counter in model_subs for key in counter.keys()}
            model_prod_keys = {key for counter in model_prods for key in counter.keys()}
            
            only_cofactors_subs = all(key in cofactors_to_ignore for key in model_sub_keys)
            only_cofactors_prods = all(key in cofactors_to_ignore for key in model_prod_keys)
            
            # If either substrates or products only contain cofactors, don't ignore cofactors in filtering
            if only_cofactors_subs or only_cofactors_prods:
                filtered_reaction_list = filter_kegg_reactions(model_subs, model_prods)
            else:
                # Otherwise, try both with and without ignoring cofactors
                filtered_reaction_list = filter_kegg_reactions(model_subs, model_prods) + \
                    filter_kegg_reactions(model_subs, model_prods, cofactors_to_ignore=cofactors_to_ignore)
                filtered_reaction_list = set(filtered_reaction_list)
            matches = []

            # in case there are multiple candidates for a substrate or product group,
            # create a list of cartesian products of substrate and product groups
            cartesian_products = list(product(model_subs, model_prods))
            # Compare with each KEGG reaction
            for kegg_id in filtered_reaction_list:
                kegg_subs = Counter(set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('substrates', [])))
                kegg_prods = Counter(set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('products', [])))

                if not spectators:
                    kegg_subs, kegg_prods = cancel_spectators(kegg_subs, kegg_prods)

                for i in cartesian_products: 
                    # Score both orientations (forward and reverse)

                    if only_cofactors_subs or only_cofactors_prods:
                        score_forward = compute_similarity(i[0], kegg_subs) + \
                                compute_similarity(i[1], kegg_prods)                
                        score_reverse = compute_similarity(i[0], kegg_prods) + \
                                compute_similarity(i[1], kegg_subs)
                        
                    else: 
                        score_forward = compute_similarity(i[0], kegg_subs, cofactors_to_ignore) + \
                                compute_similarity(i[1], kegg_prods, cofactors_to_ignore)                
                        score_reverse = compute_similarity(i[0], kegg_prods, cofactors_to_ignore) + \
                                compute_similarity(i[1], kegg_subs, cofactors_to_ignore)

                    score_forward /= 2
                    score_reverse /= 2
                    
                    max_score = max(score_forward, score_reverse)
                    
                    matches.append({
                        'model_reaction_id': reaction_label,
                        'kegg_reaction_id': kegg_id,
                        'score_forward': score_forward,
                        'score_reverse': score_reverse,
                        'match_score': max_score
                    })
            
            # Sort matches by final score (descending)
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Keep top_k matches
            if top_k:
                top_matches = matches[:top_k]
            else:
                top_matches = matches
            
            # Extract candidates and scores for recommendation
            candidates = [match['kegg_reaction_id'] for match in top_matches]
            match_scores = [match['match_score'] for match in top_matches]
            
            # Get reaction names from KEGG
            candidate_names = []
            for kegg_id in candidates:
                orthology = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("ORTHOLOGY", "")
                candidate_names.append(extract_classifications(orthology, 'orthology'))

            # Create recommendation object
            recommendation = ReactionRecommendation(
                id=reaction_label,
                synonyms=[],
                equation=reaction_str, 
                substrates=model_subs,
                products=model_prods,
                candidates=candidates,
                candidate_names=candidate_names,
                match_score=match_scores
            )
            recommendations.append(recommendation)
            
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in KEGG recommendation: {e}")
        import traceback
        traceback.print_exc()
        return []


def _get_kegg_recommendations_direct(reaction_ids: List[str], synonyms_dict, top_k: int = 3, species_recs=None) -> List[Recommendation]:
    """
    Find KEGG recommendations by directly matching against KEGG compound synonyms.
    Args:
        reaction_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        top_k: Number of top candidates to return per species based on hit_count.
    """
    # Load necessary KEGG dictionaries
    logger.info(f"Loading KEGG reaction data...")
    kegg_reaction_features_dict = load_kegg_reaction_features_dict()
    logger.info(f"Loaded {len(kegg_reaction_features_dict)} KEGG reactions")

    recommendations = []
    
    for reaction_id in reaction_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(reaction_id, [reaction_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(reaction_id, [reaction_id])
        else:
            synonyms = [reaction_id]
        
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=reaction_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        
        # Query for each synonym
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            
            if norm_synonym.startswith('R') and len(norm_synonym)==5 and norm_synonym[-5:].isdigit():
                kegg_reaction_id = norm_synonym.upper()
                if kegg_reaction_id in kegg_reaction_features_dict:
                    kegg_name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        hit_count[kegg_id] = 1
                    else:
                        hit_count[kegg_id] += 1
            
            # Then try direct name matching with KEGG reaction names
            for kegg_id in kegg_reaction_features_dict:
                name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                if norm_synonym == remove_symbols(name.lower()): # this could benefit from fuzzy matching
                    kegg_name = name
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        hit_count[kegg_id] = 1
                    else:
                        hit_count[kegg_id] += 1
            
            # Also check for partial matches in reaction orthology/names if no direct matches found
            # if not all_candidates:
            for kegg_id in kegg_reaction_features_dict:
                orthology = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("ORTHOLOGY", "")
                clean_orthology = remove_symbols(extract_classifications(orthology, 'orthology').lower())
                name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                clean_name = remove_symbols(name.lower())

                if (norm_synonym in clean_orthology or clean_orthology in norm_synonym) and clean_orthology:
                    kegg_orthology = orthology
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_orthology)
                        # Lower confidence for partial matches
                        hit_count[kegg_id] = 0.5
                    else:
                        hit_count[kegg_id] += 0.5

                elif (norm_synonym in clean_name or clean_name in norm_synonym) and clean_name:
                    kegg_name = name
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        # Lower confidence for partial matches
                        hit_count[kegg_id] = 0.5
                    else:
                        hit_count[kegg_id] += 0.5

        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate])
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            if top_k:
                top_candidates = candidate_tuples[:top_k]
            else:
                top_candidates = candidate_tuples

            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=reaction_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    
    return recommendations


def _get_kegg_recommendations_RAG(reaction_ids: List[str], top_k: int = None, spectators=False)-> List[Recommendation]:
    pass


def filter_kegg_reactions(model_subs: List[Counter], model_prods: List[Counter], cofactors_to_ignore={}) -> List[Dict[str, Any]]:
    """
    Filter KEGG reactions based on substrate and product matching.
    
    Args:
        model_subs: List of Counter objects representing model substrates
        model_prods: List of Counter objects representing model products
        kegg_parsed_reactions_dict: Dictionary of KEGG reaction data
        cofactors_to_ignore: set of KEGG IDs of cofactors
        
    Returns:
        List of KEGG reactions that contain all model substrates and products
    """
    kegg_parsed_reactions_dict = load_kegg_parsed_reactions_dict() # load_kegg_reaction_features_dict
    # Get unique keys from the model substrates and products
    model_sub_keys = {key for counter in model_subs for key in counter.keys() if key not in cofactors_to_ignore}
    model_prod_keys = {key for counter in model_prods for key in counter.keys() if key not in cofactors_to_ignore}
    
    filtered_reactions = []
    partial_matches = []
    
    for kegg_id in kegg_parsed_reactions_dict:
        # Get sets of KEGG substrates and products
        kegg_subs_set = set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('substrates', []))
        kegg_prods_set = set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('products', []))
        
        # Check if all model metabolites are in KEGG reaction (ignore counts)
        subs_match = model_sub_keys.issubset(kegg_subs_set)
        prods_match = model_prod_keys.issubset(kegg_prods_set)
        
        if subs_match and prods_match:
            filtered_reactions.append(kegg_id)
        elif subs_match:
            partial_matches.append(kegg_id)
        elif prods_match:
            partial_matches.append(kegg_id)

    #if not filtered_reactions: 
    #    filtered_reactions=partial_matches
    
    return filtered_reactions

def cancel_spectators(model_subs: Counter, model_prods: Counter):
    """
    Cancel spectator metabolites (same metabolite and same stoichiometry)
    from substrates and products. Works directly with Counters.
    
    Parameters
    ----------
    model_subs : Counter
        Substrate stoichiometry, e.g., Counter({"ATP": 1, "Glucose": 1})
    model_prods : Counter
        Product stoichiometry, e.g., Counter({"ADP": 1, "Glucose": 1})
    
    Returns
    -------
    (Counter, Counter)
        New Counters after cancellation
    """
    # Find the intersection (min stoichiometry of each metabolite)
    common = model_subs & model_prods   # stoichiometry-aware AND

    # Subtract the common terms from both sides
    new_subs = model_subs - common
    new_prods = model_prods - common

    return new_subs, new_prods

def compute_similarity(counter1: Counter, counter2: Counter, cofactors_to_ignore: set = {}) -> float:
    """
    Compute Jaccard-like similarity between two reaction sides with stoichiometry awareness.
    
    This function calculates a similarity score between two sets of metabolites,
    taking into account their stoichiometric coefficients and filtering out common cofactors.
    
    Args:
        counter1: Counter object for first reaction side (substrates or products)
        counter2: Counter object for second reaction side (substrates or products)
        cofactors_to_ignore: Set of cofactor IDs to ignore in the comparison
        
    Returns:
        Similarity score between 0.0 (no similarity) and 1.0 (identical)
    """
    # Filter out cofactors
    c1 = {k: v for k, v in counter1.items() if k not in cofactors_to_ignore}
    c2 = {k: v for k, v in counter2.items() if k not in cofactors_to_ignore}

    # Perfect match if both are empty after filtering cofactors
    if not c1 and not c2:
        return 1.0
    
    # Calculate stoichiometry-aware Jaccard similarity
    # Sum of minimum values (intersection) divided by sum of maximum values (union)
    intersection = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
    union = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
    
    if union == 0:
        return 0.0
        
    return intersection / union


def get_embedding_function(model_type: str = "default"):
    """
    Get the appropriate embedding function based on model type.
    
    Args:
        model_type: Type of embedding model ("default", "openai")
        
    Returns:
        ChromaDB embedding function
    """
    if model_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        logger.info("Using OpenAI text-embedding-ada-002 model")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002",
        )
    else:  # default
        logger.info("Using sentence transformer all-MiniLM-L6-v2 model")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

def get_chromadb_client(persist_directory: str, collection_name: str, model_type: str = "default"):
    """
    Get or create a ChromaDB client and collection, handling conflicts properly.
    
    Args:
        persist_directory: Directory for ChromaDB storage
        collection_name: Name of the collection
        model_type: Type of embedding model
        
    Returns:
        Tuple of (client, collection)
    """
    client_key = f"{persist_directory}_{collection_name}_{model_type}"
    
    if client_key in _CHROMADB_CLIENTS:
        return _CHROMADB_CLIENTS[client_key]
    
    try:
        # Try to initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Get embedding function
        embedding_function = get_embedding_function(model_type)
        
        # Get the collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        # Cache the client and collection
        _CHROMADB_CLIENTS[client_key] = (client, collection)
        
        logger.info(f"Using RAG embeddings from collection '{collection_name}' with {model_type} model")
        
        return client, collection
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle the specific "already exists" error
        if "already exists" in error_msg and "different settings" in error_msg:
            logger.warning(f"ChromaDB client conflict detected. Attempting to use in-memory client as fallback.")
            
            try:
                # Try using an in-memory client as fallback (this won't persist but will work for queries)
                client = chromadb.Client()
                
                # Try to load the collection from persistent storage manually
                # This is a workaround - the collection might not be available in memory
                raise ValueError(f"ChromaDB client conflict. Please restart Python session or check for other running processes using {persist_directory}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback client also failed: {fallback_error}")
                raise ValueError(f"ChromaDB unavailable due to client conflict. Error: {e}")
        else:
            logger.error(f"Could not access ChromaDB collection '{collection_name}': {e}")
            raise ValueError(f"ChromaDB collection not available. Make sure embeddings have been created first. Error: {e}")

def force_clear_chromadb():
    """
    Force clear ChromaDB cache and try to cleanup any hanging clients.
    """
    global _CHROMADB_CLIENTS
    
    # Clear our cache
    _CHROMADB_CLIENTS.clear()
    
    # Try to garbage collect
    import gc
    gc.collect()
    
    logger.info("Forced ChromaDB cleanup completed")

def get_species_recommendations_rag(
    species_ids: List[str], 
    synonyms_dict, 
    model_type: str = "default",
    persist_directory: str = "chroma_storage",
    collection_name: str = None,
    top_k: int = 3,
    database: str = "chebi",
    tax_id: str = None
) -> List[Recommendation]:
    """
    Find recommendations using RAG embeddings.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - collection_name (str): ChromaDB collection name. If None, will be set to default collection name.
    - model_type (str): Type of embedding model ("default", "openai").
    - persist_directory (str): ChromaDB storage directory.
    - top_k (int): Number of top candidates to retrieve per species.
    - database (str): Database to search ("chebi", "ncbigene").
    - tax_id (str/list): For ncbigene database, the organism's tax_id for organism-specific lookup. Use 9606 by default. If list, search all tax_ids for each species.
    
    Returns:
    - list: List of Recommendation objects with candidates and similarity scores.
    """
    persist_directory = os.path.join(get_data_dir(), persist_directory)
    recommendations = []
    # Helper to get collection for a given tax_id
    def get_collection_for_taxid(tid):
        if database == "ncbigene":
            cname = f"ncbigene_default_tax{tid}"
        elif database == "uniprot":
            cname = f"uniprot_default_tax{tid}"
        else:
            cname = f"{database}_default_tax{tid}"
        client, collection = get_chromadb_client(persist_directory, cname, model_type)
        return collection
    # If database is ncbigene/uniprot and tax_id is a list, aggregate results
    if database in ["ncbigene", "uniprot"] and isinstance(tax_id, list):
        for spec_id in species_ids:
            if isinstance(synonyms_dict, dict):
                synonyms = synonyms_dict.get(spec_id, [spec_id])
            elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
                synonyms = synonyms_dict[0].get(spec_id, [spec_id])
            else:
                synonyms = [spec_id]
            if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
                recommendation = Recommendation(
                    id=spec_id,
                    synonyms=synonyms,
                    candidates=[],
                    candidate_names=[],
                    match_score=[]
                )
                recommendations.append(recommendation)
                continue
            agg_candidates = {}
            agg_names = {}
            for tid in tax_id:
                try:
                    collection = get_collection_for_taxid(tid)
                except Exception as e:
                    logger.warning(f"Could not access {database.upper()} RAG collection for tax_id {tid}: {e}")
                    continue
                for synonym in synonyms:
                    try:
                        results = collection.query(
                            query_texts=[synonym],
                            n_results=top_k,
                            include=["metadatas", "distances"]
                        )
                        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                            db_id = metadata.get('ncbigene_id', 'Unknown')
                            db_name = metadata.get('name', 'Unknown')
                            similarity_score = round(1 - distance, 3)
                            if db_id not in agg_candidates or similarity_score > agg_candidates[db_id]:
                                agg_candidates[db_id] = similarity_score
                                agg_names[db_id] = db_name
                    except Exception as e:
                        logger.warning(f"Error querying synonym '{synonym}' for species '{spec_id}' in tax_id {tid}: {e}")
                        continue
            # Sort and select top_k
            sorted_candidates = sorted(agg_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
            all_candidates = [db_id for db_id, _ in sorted_candidates]
            all_candidate_names = [agg_names[db_id] for db_id, _ in sorted_candidates]
            match_score_list = [agg_candidates[db_id] for db_id, _ in sorted_candidates]
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=all_candidates,
                candidate_names=all_candidate_names,
                match_score=match_score_list
            )
            recommendations.append(recommendation)
        return recommendations
    # If database is ncbigene/uniprot and tax_id is a str or None (single organism)
    if database in ["ncbigene", "uniprot"]:
        if not tax_id:
            default_tax_id = 9606
            logger.warning(f"No tax_id provided for {database} RAG search. Using default tax_id {default_tax_id}.")
            tax_id = default_tax_id
        if collection_name is None and model_type == "default":
            collection_name = f"{database}_default_tax{tax_id}"
        elif collection_name is None and model_type == "openai":
            collection_name = f"{database}_openai_tax{tax_id}"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access {database.upper()} RAG collection '{collection_name}': {e}")
            raise
    elif database == "chebi":
        if collection_name is None and model_type == "default":
            collection_name = "chebi_default_numonly"
        elif collection_name is None and model_type == "openai":
            collection_name = "chebi_openai_numonly"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access ChEBI RAG collection '{collection_name}': {e}")
            raise
    else:
        logger.error(f"Database {database} not supported for RAG search")
        return []
    # Standard single-collection logic
    for spec_id in species_ids:
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        all_candidates = []
        all_candidate_names = []
        candidate_scores = {}
        candidate_names = {}  # Keep track of candidate names separately
        for synonym in synonyms:
            try:
                results = collection.query(
                    query_texts=[synonym],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    if database == "chebi":
                        db_id = metadata.get('chebi_id', 'Unknown')
                    elif database == "ncbigene":
                        db_id = metadata.get('ncbigene_id', 'Unknown')
                    elif database == "uniprot":
                        db_id = metadata.get('uniprot_id', 'Unknown')
                    else:
                        db_id = metadata.get('id', 'Unknown')
                    db_name = metadata.get('name', 'Unknown')
                    similarity_score = round(1 - distance, 3)
                    if db_id not in candidate_scores:
                        all_candidates.append(db_id)
                        all_candidate_names.append(db_name)
                        candidate_scores[db_id] = similarity_score
                        candidate_names[db_id] = db_name  # Store name mapping
                    else:
                        candidate_scores[db_id] = max(candidate_scores[db_id], similarity_score)
                        # Keep the name from first occurrence or update if needed
                        if db_id not in candidate_names:
                            candidate_names[db_id] = db_name
                # Only keep the top_k candidates
                if len(candidate_scores) > top_k:
                    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    all_candidates = [db_id for db_id, _ in sorted_candidates]
                    all_candidate_names = [candidate_names[db_id] for db_id, _ in sorted_candidates]
                    candidate_scores = dict(sorted_candidates)
            except Exception as e:
                logger.warning(f"Error querying synonym '{synonym}' for species '{spec_id}': {e}")
                continue
        match_score_list = [candidate_scores.get(candidate, 0.0) for candidate in all_candidates]
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations

def search_database(entity_name: str, 
                   entity_type: str, 
                   database: str = "chebi",
                   max_candidates: int = 10,
                   tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search for annotation candidates in specified database.
    Currently supports ChEBI, NCBI gene, and UniProt, extensible to other databases.
    
    Args:
        entity_name: Name of entity to search for
        entity_type: Type of entity (chemical, gene, protein)
        database: Database to search in ("chebi", "ncbigene", "uniprot")
        max_candidates: Maximum number of candidates to return
        tax_id: For ncbigene/uniprot database, the organism's tax_id for organism-specific lookup
        
    Returns:
        List of tuples (database_id, confidence, description)
    """
    if database.lower() == "chebi":
        return _search_chebi(entity_name, max_candidates)
    elif database.lower() == "ncbigene":
        return _search_ncbigene(entity_name, max_candidates, tax_id=tax_id)
    elif database.lower() == "uniprot":
        return _search_uniprot(entity_name, max_candidates, tax_id=tax_id)
    else:
        logger.warning(f"Database {database} not yet supported")
        return []

def _search_chebi(entity_name: str, max_candidates: int = 10) -> List[Tuple[str, float, str]]:
    """
    Search ChEBI database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        
    Returns:
        List of tuples (chebi_id, confidence, description)
    """
    try:
        cleannames_dict = load_chebi_cleannames_dict()
        label_dict = load_chebi_label_dict()
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, chebi_ids in cleannames_dict.items():
            if norm_name == ref_name.lower():
                for chebi_id in chebi_ids:
                    chebi_name = label_dict.get(chebi_id, chebi_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((chebi_id, confidence, chebi_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((chebi_id, confidence, chebi_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"ChEBI search failed for {entity_name}: {e}")
        return []

def _search_ncbigene(entity_name: str, max_candidates: int = 10, tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search NCBI gene database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        tax_id: Organism's tax_id for organism-specific gene lookup
        
    Returns:
        List of tuples (ncbigene_id, confidence, description)
    """
    try:
        names_dict = load_ncbigene_names_dict(tax_id=tax_id)
        label_dict = load_ncbigene_label_dict()
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, gene_ids in names_dict.items():
            if norm_name == ref_name.lower():
                for gene_id in gene_ids:
                    gene_name = label_dict.get(gene_id, gene_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((gene_id, confidence, gene_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, gene_ids in names_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for gene_id in gene_ids:
                        gene_name = label_dict.get(gene_id, gene_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((gene_id, confidence, gene_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"NCBI gene search failed for {entity_name}: {e}")
        return []

def _search_uniprot(entity_name: str, max_candidates: int = 10, tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search UniProt database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        tax_id: Organism's tax_id for organism-specific UniProt lookup
        
    Returns:
        List of tuples (uniprot_id, confidence, description)
    """
    try:
        names_dict = load_uniprot_names_dict(tax_id=tax_id)
        label_dict = load_uniprot_label_dict(tax_id=tax_id)
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, uniprot_ids in names_dict.items():
            if norm_name == ref_name.lower():
                for uniprot_id in uniprot_ids:
                    uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((uniprot_id, confidence, uniprot_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, uniprot_ids in names_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for uniprot_id in uniprot_ids:
                        uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((uniprot_id, confidence, uniprot_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"UniProt search failed for {entity_name}: {e}")
        return []

def is_database_available(database: str) -> bool:
    """
    Check if a database is available for searching.
    
    Args:
        database: Database name to check
        
    Returns:
        True if database is available
    """
    if database.lower() == "chebi":
        try:
            data_dir = get_data_dir()
            cleannames_file = data_dir / "chebi" / REF_NAMES2CHEBI
            labels_file = data_dir / "chebi" / REF_CHEBI2LABEL
            return cleannames_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "ncbigene":
        try:
            data_dir = get_data_dir()
            names_file = data_dir / "ncbigene" / REF_NAMES2NCBIGENE
            labels_file = data_dir / "ncbigene" / REF_NCBIGENE2LABEL
            return names_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "uniprot":
        try:
            data_dir = get_data_dir()
            # Check for organism-specific files (common tax_ids: 9606 for human, 10090 for mouse)
            common_tax_ids = ["9606", "10090"]
            for tax_id in common_tax_ids:
                names_file = data_dir / "uniprot" / f"names2uniprot_tax{tax_id}.lzma"
                labels_file = data_dir / "uniprot" / f"uniprot2label_tax{tax_id}.lzma"
                if names_file.exists() and labels_file.exists():
                    return True
            # Also check for old combined files for backwards compatibility
            names_file = data_dir / "uniprot" / REF_NAMES2UNIPROT
            labels_file = data_dir / "uniprot" / REF_UNIPROT2LABEL
            return names_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "kegg":
        try:
            data_dir = get_data_dir()
            chebi_to_kegg_map_file = data_dir / "kegg" / REF_CHEBI2KEGG_COMPOUND
            names_file = data_dir / "kegg" / REF_KEGG_REACTION2NAME
            ec_file = data_dir / "kegg" / REF_KEGG2EC
            reactions_file = data_dir / "kegg" / "parsed_kegg_reactions.lzma"
            return (chebi_to_kegg_map_file.exists() and 
                   names_file.exists() and 
                   ec_file.exists() and 
                   reactions_file.exists())
        except Exception:
            return False
    
    return False

def get_available_databases() -> List[str]:
    """
    Get list of available databases.
    
    Returns:
        List of available database names
    """
    available = []
    
    if is_database_available("chebi"):
        available.append("chebi")
    
    if is_database_available("ncbigene"):
        available.append("ncbigene")
    
    if is_database_available("uniprot"):
        available.append("uniprot")

    if is_database_available("kegg"):
        available.append("kegg")
    
    # Future databases can be added here
    # if is_database_available("go"):
    #     available.append("go")
    
    return available

def clear_chromadb_cache():
    """Clear the ChromaDB client cache."""
    global _CHROMADB_CLIENTS
    for client in _CHROMADB_CLIENTS.values():
        try:
            client.reset()
        except Exception:
            pass
    _CHROMADB_CLIENTS.clear()
    logger.info("ChromaDB cache cleared")

def list_available_organisms(data_dir=None):
    """
    List available organism-specific NCBI gene reference files.
    
    Args:
        data_dir: Directory containing reference files (default: auto-detect)
        
    Returns:
        list: List of available tax_ids
    """
    if data_dir is None:
        data_dir = get_data_dir() / "ncbigene"
    else:
        data_dir = Path(data_dir)
    
    # Look for organism-specific files
    pattern = "names2ncbigene_tax*_protein-coding.lzma"
    files = list(data_dir.glob(pattern))
    
    tax_ids = []
    for f in files:
        # Extract tax_id from filename: names2ncbigene_tax{tax_id}_protein_coding.lzma
        parts = f.stem.split('_')
        if len(parts) >= 2 and parts[1].startswith('tax'):
            tax_id = parts[1][3:]  # Remove 'tax' prefix
            tax_ids.append(tax_id)
    
    tax_ids.sort()
    return tax_ids

def get_organism_files_info(data_dir=None):
    """
    Get information about available organism-specific files.
    
    Args:
        data_dir: Directory containing reference files (default: auto-detect)
        
    Returns:
        dict: Information about available files per organism
    """
    if data_dir is None:
        data_dir = get_data_dir() / "ncbigene"
    else:
        data_dir = Path(data_dir)
    
    tax_ids = list_available_organisms(data_dir)
    
    organism_info = {}
    for tax_id in tax_ids:
        names2gene_file = data_dir / f"names2ncbigene_tax{tax_id}_protein-coding.lzma"
        gene2names_file = data_dir / f"ncbigene2names_tax{tax_id}_protein-coding.lzma"
        
        organism_info[tax_id] = {
            'has_names2gene': names2gene_file.exists(),
            'has_gene2names': gene2names_file.exists(),
            'names2gene_file': str(names2gene_file) if names2gene_file.exists() else None,
            'gene2names_file': str(gene2names_file) if gene2names_file.exists() else None,
            'complete': names2gene_file.exists() and gene2names_file.exists()
        }
    
    return organism_info 