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
import sys
import chromadb
from chromadb.utils import embedding_functions
from utils.constants import REF_CHEBI2LABEL, REF_NAMES2CHEBI, REF_NCBIGENE2LABEL, REF_NAMES2NCBIGENE
from core.data_types import Recommendation

logger = logging.getLogger(__name__)

# Global ChromaDB client cache to avoid conflicts
_CHROMADB_CLIENTS = {}

# Cache for loaded dictionaries
_CHEBI_CLEANNAMES_DICT: Optional[Dict[str, List[str]]] = None
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_NCBIGENE_NAMES_DICT: Optional[Dict[str, List[str]]] = None
_NCBIGENE_LABEL_DICT: Optional[Dict[str, str]] = None

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
        # Try to load old combined file for backwards compatibility
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

def load_ncbigene_gene2names_dict(tax_id: str) -> Dict[str, List[str]]:
    """
    Load the NCBI gene ID to names dictionary for RAG embeddings.
    
    Args:
        tax_id: The organism's tax_id for organism-specific lookup.
    
    Returns:
        Dictionary mapping NCBI gene IDs to lists of names/synonyms
    """
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"ncbigene_gene2names_{tax_id}"
    
    # Check if we have this specific version cached
    if not hasattr(load_ncbigene_gene2names_dict, '_cache'):
        load_ncbigene_gene2names_dict._cache = {}
    
    if cache_key in load_ncbigene_gene2names_dict._cache:
        return load_ncbigene_gene2names_dict._cache[cache_key]
    
    # Load organism-specific gene2names file
    data_file = get_data_dir() / "ncbigene" / f"ncbigene2names_tax{tax_id}_protein-coding.lzma"
    
    if not data_file.exists():
        raise FileNotFoundError(f"NCBI gene2names data file not found for tax_id {tax_id}: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        gene2names_dict = pickle.load(f)
    
    # Cache the result
    load_ncbigene_gene2names_dict._cache[cache_key] = gene2names_dict
    
    return gene2names_dict

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

def remove_symbols(text: str) -> str:
    """
    Remove all characters except numbers and letters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with only alphanumeric characters
    """
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def get_species_recommendations_direct(species_ids: List[str], synonyms_dict, database: str = "chebi", tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find recommendations by directly matching against database synonyms.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - database (str): Database to search ("chebi", "ncbigene")
    - tax_id (str/list): For ncbigene database, the organism's tax_id for organism-specific lookup. If list, search all tax_ids for each species.
    - top_k (int): Number of top candidates to return per species based on hit_count.
    
    Returns:
    - list: List of Recommendation objects with candidates and names.
    """
    if database == "chebi":
        return _get_chebi_recommendations_direct(species_ids, synonyms_dict, top_k=top_k)
    elif database == "ncbigene":
        return _get_ncbigene_recommendations_direct(species_ids, synonyms_dict, tax_id=tax_id, top_k=top_k)
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
        cname = f"ncbigene_default_tax{tid}"
        client, collection = get_chromadb_client(persist_directory, cname, model_type)
        return collection
    # If database is ncbigene and tax_id is a list, aggregate results
    if database == "ncbigene" and isinstance(tax_id, list):
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
                    logger.warning(f"Could not access NCBI gene RAG collection for tax_id {tid}: {e}")
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
    # If database is ncbigene and tax_id is a str or None (single organism)
    if database == "ncbigene":
        if not tax_id:
            logger.warning("No tax_id provided for ncbigene RAG search. Using default tax_id 9606.")
            tax_id = 9606
        if collection_name is None:
            collection_name = f"ncbigene_default_tax{tax_id}"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access NCBI gene RAG collection '{collection_name}': {e}")
            raise
    elif database == "chebi":
        if collection_name is None:
            collection_name = "chebi_default_numonly"
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
    Currently supports ChEBI and NCBI gene, extensible to other databases.
    
    Args:
        entity_name: Name of entity to search for
        entity_type: Type of entity (chemical, gene, protein)
        database: Database to search in ("chebi", "ncbigene")
        max_candidates: Maximum number of candidates to return
        tax_id: For ncbigene database, the organism's tax_id for organism-specific lookup
        
    Returns:
        List of tuples (database_id, confidence, description)
    """
    if database.lower() == "chebi":
        return _search_chebi(entity_name, max_candidates)
    elif database.lower() == "ncbigene":
        return _search_ncbigene(entity_name, max_candidates, tax_id=tax_id)
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
    
    # Future databases can be added here
    # if is_database_available("uniprot"):
    #     available.append("uniprot")
    
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