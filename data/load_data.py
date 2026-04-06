#!/usr/bin/env python3
"""
ChEBI Reference Data Loader for RAG-based Entity Linking

This script loads ChEBI reference data from a compressed pickle file and creates
embeddings using ChromaDB for RAG-based ontology entity linking. It supports
both default sentence transformer models and OpenAI embedding models.

Usage:
    python load_data.py --help
    python load_data.py --database chebi --model default --collection chebi_default
    python load_data.py --database ncbigene --model default --tax_id 9606
    python load_data.py --database uniprot --model default --tax_id 9606
    python load_data.py --database kegg --model default
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import compress_pickle
from tqdm import tqdm
import sys
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


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
    
    elif classification == 'orthology':
        for line in lines:
        # Split once on spaces to remove the Kxxxxx ID
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                # Remove the EC info if present
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)

    elif classification == 'definition':
        parts = []
        buf = ""
        paren_level = 0  # Track nested parentheses

        i = 0
        while i < len(raw_text):
            c = raw_text[i]

            # Track parentheses
            if c == '(':
                paren_level += 1
            elif c == ')':
                paren_level -= 1

            # Split points: + outside parentheses or <=>
            if c == '+' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
            elif raw_text[i:i+3] == '<=>' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 2  # skip the next two chars of <=>
            elif raw_text[i:i+2] == '->' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 1  
            else:
                buf += c

            i += 1

        # Add remaining buffer
        if buf:
            parts.append(buf.strip())
        # parts = [p for p in parts if p]
        strip_dollars = [p.lstrip("$") for p in parts if p]
        clean_lines = [re.sub(r'^[\d\w\(\)\+\-]+?\s+', '', p.strip()) for p in strip_dollars]
    
    return "; ".join(set(clean_lines))


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


def build_chunks_for_embedding(kegg_reactions):
    """Convert parsed KEGG reactions into text + metadata chunks for Chroma."""
    def flatten_list(lst):
        """
        Flattens a list into a ';'-separated string.
        Empty lists return an empty string.
        """
        return ";".join(map(str, lst)) if lst else ""

    chunks = {}

    for reaction in kegg_reactions:
        reaction_id = reaction
        name = kegg_reactions[reaction].get("NAME", "")
        ec_number = kegg_reactions[reaction].get("ENZYME", "").split()
        ec_number = flatten_list([line.strip() for line in ec_number if line.strip()])
        definition = kegg_reactions[reaction].get("DEFINITION", "")
        participants = extract_classifications(definition, 'definition')
        equation = kegg_reactions[reaction].get("EQUATION", "")
        brite = kegg_reactions[reaction].get("BRITE","")
        if brite: 
            brite = extract_classifications(brite, 'brite')
        pathways = kegg_reactions[reaction].get("PATHWAY", "").splitlines()
        pathways = flatten_list([line.strip() for line in pathways if line.strip()])
        orthology = kegg_reactions[reaction].get("ORTHOLOGY","")
        if orthology: 
            orthology = extract_classifications(orthology, 'orthology')

        # Construct the text to embed
        text = f"Reaction type: {orthology};{name};{brite};{pathways}\nReaction: {participants}"

        # Store in dictionary keyed by compound_id
        chunks[reaction_id] = {
            "text": text,
            "metadata": {
                "kegg_id": reaction_id,
                "name": name,
                "ec_number": ec_number,
                "definition": definition,
                "participants": participants,
                "equation": equation,
                "brite": brite,
                "orthology": orthology,
                "pathways": pathways,
            }
        }

    return chunks


def prepare_documents_for_indexing(ref_data: Dict[str, List[str]] | List[Dict], database: str) -> tuple[list, list, list]:
    """
    Convert reference data into documents for ChromaDB indexing.
    """
    logger.info("Preparing documents for indexing...")
    ids = []
    documents = []
    metadatas = []
    doc_id = 0
    
    # Handle KEGG data which is a list of dictionaries
    if database == "kegg":
        # Use the specialized KEGG chunking function for richer text representation
        chunks = build_chunks_for_embedding(ref_data)
        for chunk in tqdm(chunks, desc="Processing KEGG reactions"):
            ids.append(chunk)
            documents.append(chunks[chunk].get("text"))
            metadatas.append(chunks[chunk].get("metadata"))
    else:
        # Handle ChEBI and NCBI gene data which are dictionaries
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
                elif database == "uniprot":
                    metadata = {"uniprot_id": entry_id, "name": cleaned_name}
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


def batch_add_to_chroma(collection, ids, documents, metadatas, model=None, batch_size=5000):
    """
    Add documents to Chroma in batches, with optional pre-embedding.
    
    Args:
        collection: ChromaDB collection
        ids: List of document IDs
        documents: List of document texts
        metadatas: List of document metadata
        model: Optional SentenceTransformer model for pre-embedding
        batch_size: Number of documents to process in each batch
    """
    if model:
        logger.info("🔍 Pre-embedding all documents...")
        embeddings = model.encode(documents, show_progress_bar=True)
        
        logger.info("🧠 Inserting into Chroma in batches with pre-computed embeddings...")
        for i in tqdm(range(0, len(ids), batch_size), desc="Storing in Chroma"):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size]
            )
    else:
        logger.info("🧠 Inserting into Chroma in batches...")
        for i in tqdm(range(0, len(ids), batch_size), desc="Storing in Chroma"):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
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
        # For default model (SentenceTransformer), we can optimize by pre-computing embeddings
        if model_type == "default":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            batch_add_to_chroma(collection, ids, documents, metadatas, model, batch_size)
        else:
            # For other models (like OpenAI), use the standard approach
            batch_add_to_chroma(collection, ids, documents, metadatas, None, batch_size)
        
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
        elif database == "uniprot":
            test_queries = ["TP53", "RAS", "EGFR"]
        elif database == "kegg":
            test_queries = ["hydrolase", "dehydrogenase", "kinase"]
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
                    elif database == "uniprot":
                        entry_id = metadata.get('uniprot_id', 'Unknown')
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
        choices=["chebi", "ncbigene", "uniprot", "kegg"],
        default="chebi",
        help="Database to use: 'chebi' or 'ncbigene' or 'uniprot' or 'kegg' (default: chebi)"
    )
    parser.add_argument(
        "--tax_id",
        type=str,
        default=None,
        help="Taxonomy ID for gene database (required for ncbigene and uniprot)"
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
        elif args.database == "uniprot":
            if not args.tax_id:
                raise ValueError("--tax_id is required for uniprot database")
            args.ref_data_path = str(Path(f"uniprot/uniprot2names_tax{args.tax_id}.lzma"))
        elif args.database == "kegg":
            args.ref_data_path = str(Path("kegg/kegg_reaction_features.lzma"))
            # Check if JSON format is available (for backward compatibility)
            json_path = str(Path("kegg/merged_kegg_reactions.json"))
            if os.path.exists(json_path) and not os.path.exists(args.ref_data_path):
                logger.info(f"Using JSON format for KEGG reactions: {json_path}")
                args.ref_data_path = json_path
    if args.collection is None:
        if args.database == "chebi":
            args.collection = "chebi_default_numonly"
        elif args.database == "ncbigene":
            if not args.tax_id:
                raise ValueError("--tax_id is required for ncbigene database")
            args.collection = f"ncbigene_default_tax{args.tax_id}"
        elif args.database == "uniprot":
            if not args.tax_id:
                raise ValueError("--tax_id is required for uniprot database")
            args.collection = f"uniprot_default_tax{args.tax_id}"
        elif args.database == "kegg":
            args.collection = "kegg_reactions_default"
    os.makedirs(args.persist_directory, exist_ok=True)
    try:
        if not args.test_only:
            # Load reference data
            if args.database == "kegg" and args.ref_data_path.endswith(".json"):
                logger.info(f"Loading KEGG data from JSON: {args.ref_data_path}")
                with open(args.ref_data_path, "r") as f:
                    ref_data = json.load(f)
            else:
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
