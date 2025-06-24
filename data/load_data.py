#!/usr/bin/env python3
"""
ChEBI Reference Data Loader for RAG-based Entity Linking

This script loads ChEBI reference data from a compressed pickle file and creates
embeddings using ChromaDB for RAG-based chemical entity linking. It supports
both default sentence transformer models and OpenAI embedding models.

Usage:
    python load_data.py --help
    python load_data.py --database chebi --model default --collection chebi_default
    python load_data.py --database ncbigene --model default --tax_id 9606
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
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


def load_reference_data(ref_data_path: str) -> Dict[str, List[str]]:
    """
    Load reference data (ChEBI or gene) from compressed pickle file.
    """
    logger.info(f"Loading data from {ref_data_path}")
    if not os.path.exists(ref_data_path):
        raise FileNotFoundError(f"Data file not found: {ref_data_path}")
    try:
        with open(ref_data_path, 'rb') as handle:
            data = compress_pickle.load(handle, compression="lzma")
        logger.info(f"Loaded {len(data)} entries")
        return data
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        raise


def prepare_documents_for_indexing(ref_data: Dict[str, List[str]], database: str) -> tuple[list, list, list]:
    """
    Convert reference data into documents for ChromaDB indexing.
    For ChEBI: chebi_id, for gene: ncbigene_id.
    """
    logger.info("Preparing documents for indexing...")
    ids = []
    documents = []
    metadatas = []
    doc_id = 0
    for entry_id, names in tqdm(ref_data.items(), desc="Processing entries"):
        if not names:
            continue
        for name in names:
            if not name or name.strip() == "":
                continue
            cleaned_name = name.strip()
            if database == "chebi":
                metadata = {"chebi_id": entry_id, "name": cleaned_name}
                ids.append(f"{entry_id}_{doc_id}")
            elif database == "ncbigene":
                metadata = {"ncbigene_id": entry_id, "name": cleaned_name}
                ids.append(f"{entry_id}_{doc_id}")
            else:
                raise ValueError(f"Unsupported database: {database}")
            documents.append(cleaned_name)
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
    test_queries: Optional[list] = None,
    database: str = "chebi"
) -> None:
    """
    Test the created embeddings with sample queries.
    """
    if test_queries is None:
        if database == "chebi":
            test_queries = ["glucose", "D-glucose", "blood sugar"]
        elif database == "ncbigene":
            test_queries = ["TP53", "BRCA1", "RAS"]
        else:
            test_queries = ["test"]
    logger.info("Testing the embeddings with sample queries...")
    client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = get_embedding_function(model_type)
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["embeddings","metadatas", "distances", "documents"]
                )
                print(f"\nTop 3 results for '{query}':")
                for i, (embedding, metadata, distance, document) in enumerate(zip(
                    results['embeddings'][0],
                    results['metadatas'][0], 
                    results['distances'][0], 
                    results['documents'][0]
                )):
                    if database == "chebi":
                        entry_id = metadata.get('chebi_id', 'Unknown')
                    elif database == "ncbigene":
                        entry_id = metadata.get('ncbigene_id', 'Unknown')
                    else:
                        entry_id = metadata.get('id', 'Unknown')
                    name = metadata.get('name', 'Unknown')
                    similarity = 1 - distance
                    print(f"  {i+1}. {entry_id}: {name} (similarity: {similarity:.3f})")
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
    except Exception as e:
        logger.error(f"Error accessing collection '{collection_name}': {e}")
        logger.error("Make sure the collection exists and was created with the same model type")


def main():
    parser = argparse.ArgumentParser(
        description="Load reference data and create embeddings for RAG-based entity linking (ChEBI or NCBI gene)"
    )
    parser.add_argument(
        "--database",
        type=str,
        choices=["chebi", "ncbigene"],
        default="chebi",
        help="Database to use: 'chebi' or 'ncbigene' (default: chebi)"
    )
    parser.add_argument(
        "--tax_id",
        type=str,
        default=None,
        help="Taxonomy ID for gene database (required for ncbigene)"
    )
    parser.add_argument(
        "--ref_data_path",
        type=str,
        default=None,
        help="Path to the reference file (default: auto for selected database)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Name for the ChromaDB collection (default: auto for selected database)"
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
    # Determine defaults for ref_data_path and collection
    if args.ref_data_path is None:
        if args.database == "chebi":
            args.ref_data_path = str(Path("chebi/chebi2names.lzma"))
        elif args.database == "ncbigene":
            if not args.tax_id:
                raise ValueError("--tax_id is required for ncbigene database")
            args.ref_data_path = str(Path(f"ncbigene/ncbigene2names_tax{args.tax_id}_protein-coding.lzma"))
    if args.collection is None:
        if args.database == "chebi":
            args.collection = "chebi_default_numonly"
        elif args.database == "ncbigene":
            if not args.tax_id:
                raise ValueError("--tax_id is required for ncbigene database")
            args.collection = f"ncbigene_default_tax{args.tax_id}"
    os.makedirs(args.persist_directory, exist_ok=True)
    try:
        if not args.test_only:
            ref_data = load_reference_data(args.ref_data_path)
            ids, documents, metadatas = prepare_documents_for_indexing(ref_data, args.database)
            create_embeddings(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                collection_name=args.collection,
                model_type=args.model,
                persist_directory=args.persist_directory,
                batch_size=args.batch_size
            )
        if args.test or args.test_only:
            test_search(
                collection_name=args.collection,
                model_type=args.model,
                persist_directory=args.persist_directory,
                database=args.database
            )
        logger.info("Process completed successfully!")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
