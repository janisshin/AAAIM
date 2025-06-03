#!/usr/bin/env python3
"""
ChEBI Reference Data Loader for RAG-based Entity Linking

This script loads ChEBI reference data from a compressed pickle file and creates
embeddings using ChromaDB for RAG-based chemical entity linking. It supports
both default sentence transformer models and OpenAI embedding models.

Usage:
    python load_data.py --help
    python load_data.py --model default --collection chebi_default
    python load_data.py --model openai --collection chebi_openai
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import compress_pickle
from tqdm import tqdm
import sys
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chunk_list(lst: List, size: int):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def load_chebi_data(ref_data_path: str) -> Dict[str, List[str]]:
    """
    Load ChEBI reference data from compressed pickle file.
    
    Args:
        ref_data_path: Path to the reference file
        
    Returns:
        Dictionary mapping IDs to lists of names/synonyms
    """
    logger.info(f"Loading data from {ref_data_path}")
    
    if not os.path.exists(ref_data_path):
        raise FileNotFoundError(f"Data file not found: {ref_data_path}")
    
    try:
        with open(ref_data_path, 'rb') as handle:
            chebi_data = compress_pickle.load(handle, compression="lzma")
        
        logger.info(f"Loaded {len(chebi_data)} entries")
        return chebi_data
    
    except Exception as e:
        logger.error(f"Error loading ChEBI data: {e}")
        raise


def prepare_documents_for_indexing(chebi_data: Dict[str, List[str]]) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Convert ChEBI data into documents suitable for ChromaDB indexing.
    
    Each ChEBI entry becomes multiple documents - one for each name/synonym.
    This allows for better retrieval when searching for chemical names.
    
    Args:
        chebi_data: Dictionary mapping ChEBI IDs to lists of names/synonyms
        
    Returns:
        Tuple of (ids, documents, metadatas)
    """
    logger.info("Preparing documents for indexing...")
    
    ids = []
    documents = []
    metadatas = []
    doc_id = 0
    
    for chebi_id, names in tqdm(chebi_data.items(), desc="Processing ChEBI entries"):
        if not names:  # Skip entries with no names
            continue
            
        # Create a document for each name/synonym
        for name in names:
            if not name or name.strip() == "":  # Skip empty names
                continue
                
            # Clean the name
            cleaned_name = name.strip()
            
            # Simple metadata - just essential information
            metadata = {
                "chebi_id": chebi_id,
                "name": cleaned_name
            }
            
            ids.append(f"{chebi_id}_{doc_id}")
            documents.append(cleaned_name)  # Use the name directly as document
            metadatas.append(metadata)
            doc_id += 1
    
    logger.info(f"Prepared {len(documents)} documents for indexing")
    return ids, documents, metadatas


def get_embedding_function(model_type: str):
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


def create_embeddings(
    ids: List[str],
    documents: List[str], 
    metadatas: List[Dict[str, Any]],
    collection_name: str,
    model_type: str = "default",
    persist_directory: str = "chroma_storage",
    batch_size: int = 500
) -> None:
    """
    Create embeddings and index documents using ChromaDB.
    
    Args:
        ids: List of document IDs
        documents: List of document texts
        metadatas: List of document metadata
        collection_name: Name for the ChromaDB collection
        model_type: Type of embedding model ("default", "openai")
        persist_directory: Directory to store the ChromaDB database
        batch_size: Number of documents to process in each batch
    """
    logger.info(f"Creating embeddings with {model_type} model...")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get embedding function
    embedding_function = get_embedding_function(model_type)
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"model": model_type, "purpose": "entity_linking"}
    )
    
    # Index documents in batches
    total_docs = len(documents)
    logger.info(f"Indexing {total_docs} documents in batches of {batch_size}")
    
    try:
        for i, (id_batch, doc_batch, meta_batch) in enumerate(
            zip(
                chunk_list(ids, batch_size), 
                chunk_list(documents, batch_size), 
                chunk_list(metadatas, batch_size)
            )
        ):
            start_doc = i * batch_size
            end_doc = min((i + 1) * batch_size, total_docs)
            percent_done = (end_doc / total_docs) * 100 if total_docs > 0 else 0
            logger.info(
                f"Processing batch {i+1}, documents {start_doc} to {end_doc} "
                f"({percent_done:.2f}% complete)"
            )
            
            collection.add(
                ids=id_batch,
                documents=doc_batch,
                metadatas=meta_batch
            )
        
        final_count = collection.count()
        logger.info(f"Successfully indexed {final_count} documents in collection '{collection_name}'")
        logger.info(f"Collection saved to {persist_directory}")
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise


def test_search(
    collection_name: str,
    model_type: str = "default",
    persist_directory: str = "chroma_storage",
    test_queries: List[str] = None
) -> None:
    """
    Test the created embeddings with sample queries.
    
    Args:
        collection_name: Name of the ChromaDB collection
        model_type: Type of embedding model used
        persist_directory: Directory where ChromaDB database is stored
        test_queries: List of test chemical names to search for
    """
    if test_queries is None:
        test_queries = [
            "glucose",
            "caffeine", 
            "aspirin",
            "water",
            "ethanol"
        ]
    
    logger.info("Testing the embeddings with sample queries...")
    
    # Initialize client and get collection
    client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = get_embedding_function(model_type)
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        # Test each query
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=5,
                    include=["metadatas", "distances", "documents"]
                )
                
                print(f"\nTop 5 results for '{query}':")
                for i, (metadata, distance, document) in enumerate(zip(
                    results['metadatas'][0], 
                    results['distances'][0], 
                    results['documents'][0]
                )):
                    chebi_id = metadata.get('chebi_id', 'Unknown')
                    name = metadata.get('name', 'Unknown')
                    similarity = 1 - distance
                    print(f"  {i+1}. {chebi_id}: {name} (similarity: {similarity:.3f})")
                    
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
                
    except Exception as e:
        logger.error(f"Error accessing collection '{collection_name}': {e}")
        logger.error("Make sure the collection exists and was created with the same model type")


def main():
    parser = argparse.ArgumentParser(
        description="Load ChEBI reference data and create embeddings for RAG-based entity linking"
    )
    
    parser.add_argument(
        "--ref_data_path",
        type=str,
        default="chebi/chebi2names.lzma",
        help="Path to the reference file (default: chebi/chebi2names.lzma)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="chebi_default",
        help="Name for the ChromaDB collection (default: chebi_default)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["default", "openai"],
        default="default",
        help="Embedding model type (default: default)"
    )
    
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="Directory to store ChromaDB database (default: chroma_storage)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for indexing (default: 500)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after creating embeddings"
    )
    
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run test queries (skip embedding creation)"
    )
    
    args = parser.parse_args()
    
    # Ensure the persist directory exists
    os.makedirs(args.persist_directory, exist_ok=True)
    
    try:
        if not args.test_only:
            # Load ChEBI data
            chebi_data = load_chebi_data(args.ref_data_path)
            
            # Prepare documents for indexing
            ids, documents, metadatas = prepare_documents_for_indexing(chebi_data)
            
            # Create embeddings
            create_embeddings(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                collection_name=args.collection,
                model_type=args.model,
                persist_directory=args.persist_directory,
                batch_size=args.batch_size
            )
        
        # Run tests if requested
        if args.test or args.test_only:
            test_search(
                collection_name=args.collection,
                model_type=args.model,
                persist_directory=args.persist_directory
            )
            
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
