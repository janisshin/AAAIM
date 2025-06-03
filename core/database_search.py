"""
Database Search for AAAIM

Handles database searches for annotation candidates.
Currently supports ChEBI, extensible to other databases.
"""

import os
import re
import lzma
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import sys
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Global ChromaDB client cache to avoid conflicts
_CHROMADB_CLIENTS = {}

# Cache for loaded dictionaries
_CHEBI_CLEANNAMES_DICT: Optional[Dict[str, List[str]]] = None
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None

@dataclass
class Recommendation:
    """
    Recommendation dataclass
    """
    id: str  # ID for the species
    synonyms: list  # List of synonyms predicted by LLM
    candidates: list  # List of ChEBI IDs
    candidate_names: list  # List of names of the predicted candidates
    match_score: list  # Match scores (normalized hit count for direct search, cosine similarity for RAG)

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
        data_file = get_data_dir() / "chebi" / "cleannames2chebi.lzma"
        
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
        data_file = get_data_dir() / "chebi" / "chebi2label.lzma"
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def remove_symbols(text: str) -> str:
    """
    Remove all characters except numbers and letters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with only alphanumeric characters
    """
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def get_species_recommendations_direct(species_ids: List[str], synonyms_dict) -> List[Recommendation]:
    """
    Find ChEBI recommendations by directly matching against ChEBI synonyms.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    
    Returns:
    - list: List of Recommendation objects with candidates and names.
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
    collection_name: str = "chebi_default",
    model_type: str = "default",
    persist_directory: str = "chroma_storage",
    top_k: int = 5
) -> List[Recommendation]:
    """
    Find ChEBI recommendations using RAG embeddings.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - collection_name (str): ChromaDB collection name.
    - model_type (str): Type of embedding model ("default", "openai").
    - persist_directory (str): ChromaDB storage directory.
    - top_k (int): Number of top candidates to retrieve per species.
    
    Returns:
    - list: List of Recommendation objects with candidates and similarity scores.
    """
    persist_directory = os.path.join(get_data_dir(), persist_directory)
    
    # Use the client manager to get or create client and collection
    client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
    
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
        candidate_scores = {}  # Dictionary to track best score for each candidate
        
        # Query embeddings for each synonym
        for synonym in synonyms:
            try:
                # Query the collection
                results = collection.query(
                    query_texts=[synonym],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
                
                # Process results
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    chebi_id = metadata.get('chebi_id', 'Unknown')
                    chebi_name = metadata.get('name', 'Unknown')
                    similarity_score = round(1 - distance, 3)  # Convert distance to similarity and only keep 3 decimal places
                    
                    if chebi_id not in candidate_scores:
                        all_candidates.append(chebi_id)
                        all_candidate_names.append(chebi_name)
                        candidate_scores[chebi_id] = similarity_score
                    else:
                        # Keep the best (highest) similarity score for this candidate
                        candidate_scores[chebi_id] = max(candidate_scores[chebi_id], similarity_score)
            
                # Only keep the top_k candidate for each species
                if len(candidate_scores) > top_k:
                    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    all_candidates = [chebi_id for chebi_id, _ in sorted_candidates]
                    all_candidate_names = [all_candidate_names[all_candidates.index(chebi_id)] for chebi_id, _ in sorted_candidates if chebi_id in all_candidates]
                    candidate_scores = dict(sorted_candidates)

            except Exception as e:
                logger.warning(f"Error querying synonym '{synonym}' for species '{spec_id}': {e}")
                continue
        
        # Convert candidate_scores dict to list in the same order as candidates
        match_score_list = [candidate_scores.get(candidate, 0.0) for candidate in all_candidates]
        
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

def search_database(entity_name: str, 
                   entity_type: str, 
                   database: str = "chebi",
                   max_candidates: int = 10) -> List[Tuple[str, float, str]]:
    """
    Search for annotation candidates in specified database.
    Currently supports ChEBI, extensible to other databases.
    
    Args:
        entity_name: Name of entity to search for
        entity_type: Type of entity (chemical, gene, protein)
        database: Database to search in (currently only "chebi")
        max_candidates: Maximum number of candidates to return
        
    Returns:
        List of tuples (database_id, confidence, description)
    """
    if database.lower() == "chebi":
        return _search_chebi(entity_name, max_candidates)
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
            cleannames_file = data_dir / "chebi" / "cleannames2chebi.lzma"
            labels_file = data_dir / "chebi" / "chebi2label.lzma"
            return cleannames_file.exists() and labels_file.exists()
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
    
    # Future databases can be added here
    # if is_database_available("ncbigene"):
    #     available.append("ncbigene")
    
    return available 

def clear_chromadb_cache():
    """
    Clear the ChromaDB client cache. Useful for cleaning up between batch operations.
    """
    force_clear_chromadb() 