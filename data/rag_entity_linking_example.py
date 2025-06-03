#!/usr/bin/env python3
"""
RAG-based Entity Linking Example for ChEBI

This script demonstrates how to use the created ChEBI embeddings for
RAG-based entity linking, where chemical mentions in text are linked
to their corresponding ChEBI IDs with the help of an LLM.

Usage:
    python rag_entity_linking_example.py --help
    python rag_entity_linking_example.py --query "aspirin"
    python rag_entity_linking_example.py --text "The patient was treated with glucose and caffeine"
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_embedding_function(model_type: str):
    """
    Get the appropriate embedding function based on model type.
    
    Args:
        model_type: Type of embedding model ("default", "openai", or "llama")
        
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
    elif model_type == "llama":
        logger.info("Using Ollama nomic-embed-text model")
        return embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
    else:  # default
        logger.info("Using sentence transformer all-MiniLM-L6-v2 model")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )


class ChEBIEntityLinker:
    """
    RAG-based entity linker for ChEBI chemical entities.
    """
    
    def __init__(
        self, 
        collection_name: str = "chebi_entities",
        model_type: str = "default",
        persist_directory: str = "chroma_storage",
        top_k: int = 5
    ):
        """
        Initialize the entity linker.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_type: Type of embedding model used
            persist_directory: Directory where ChromaDB database is stored
            top_k: Number of top candidates to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.model_type = model_type
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get embedding function
        self.embedding_function = get_embedding_function(model_type)
        
        # Get the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Initialized ChEBI entity linker with {model_type} model, collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Could not access collection '{collection_name}': {e}. Make sure embeddings have been created first.")
    
    def retrieve_candidates(self, chemical_mention: str) -> List[Dict[str, Any]]:
        """
        Retrieve candidate ChEBI entities for a chemical mention.
        
        Args:
            chemical_mention: Chemical name or mention to search for
            
        Returns:
            List of candidate entities with metadata and similarity scores
        """
        try:
            results = self.collection.query(
                query_texts=[chemical_mention],
                n_results=self.top_k,
                include=["metadatas", "distances", "documents"]
            )
            
            candidates = []
            for metadata, distance, document in zip(
                results['metadatas'][0], 
                results['distances'][0], 
                results['documents'][0]
            ):
                candidate = {
                    'chebi_id': metadata.get('chebi_id', 'Unknown'),
                    'name': metadata.get('name', 'Unknown'),
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'distance': distance,
                    'document': document
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error retrieving candidates for '{chemical_mention}': {e}")
            return []
    
    def format_candidates_for_llm(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Format candidates for LLM processing.
        
        Args:
            candidates: List of candidate entities
            
        Returns:
            Formatted string for LLM input
        """
        if not candidates:
            return "No candidates found."
        
        formatted = "Candidate ChEBI entities:\n"
        for i, candidate in enumerate(candidates, 1):
            formatted += f"{i}. {candidate['chebi_id']}: {candidate['name']} "
            formatted += f"(similarity: {candidate['similarity_score']:.3f})\n"
        
        return formatted
    
    def link_entity(self, chemical_mention: str, context: str = "") -> Dict[str, Any]:
        """
        Link a chemical mention to its ChEBI ID using RAG.
        
        Args:
            chemical_mention: Chemical name to link
            context: Optional context text
            
        Returns:
            Dictionary with linking results
        """
        logger.info(f"Linking entity: '{chemical_mention}'")
        
        # Retrieve candidates
        candidates = self.retrieve_candidates(chemical_mention)
        
        if not candidates:
            return {
                'query': chemical_mention,
                'context': context,
                'candidates': [],
                'best_match': None,
                'confidence': 0.0
            }
        
        # For now, return the best candidate (highest similarity)
        # In a full RAG system, this would be passed to an LLM for final decision
        best_match = candidates[0]
        
        return {
            'query': chemical_mention,
            'context': context,
            'candidates': candidates,
            'best_match': best_match,
            'confidence': best_match['similarity_score']
        }


def create_llm_prompt(chemical_mention: str, candidates: List[Dict[str, Any]], context: str = "") -> str:
    """
    Create a prompt for LLM-based entity linking decision.
    
    Args:
        chemical_mention: Chemical name to link
        candidates: List of candidate ChEBI entities
        context: Optional context text
        
    Returns:
        Formatted prompt for LLM
    """
    prompt = f"""You are an expert in chemical entity linking. Your task is to determine which ChEBI ID best corresponds to the chemical mention "{chemical_mention}"."""
    
    if context:
        prompt += f""" The mention appears in this context: "{context}"."""
    
    prompt += f"""

Here are the top candidate ChEBI entities retrieved from the database:

"""
    
    for i, candidate in enumerate(candidates, 1):
        prompt += f"""{i}. ChEBI ID: {candidate['chebi_id']}
   Name: {candidate['name']}
   Similarity Score: {candidate['similarity_score']:.3f}
   
"""
    
    prompt += """Please analyze these candidates and determine:
1. Which ChEBI ID is the best match for the mention (or "NONE" if no good match)
2. Your confidence level (0.0 to 1.0)
3. Brief explanation for your choice

Respond in JSON format:
{
    "best_chebi_id": "CHEBI:XXXXX",
    "confidence": 0.95,
    "explanation": "Your reasoning here"
}"""
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based entity linking example for ChEBI"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Chemical name to search for"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        help="Text containing chemical mentions"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="chebi_entities",
        help="ChromaDB collection name"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["default", "openai", "llama"],
        default="default",
        help="Embedding model type"
    )
    
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="ChromaDB storage directory"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidates to retrieve"
    )
    
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Show the LLM prompt that would be used"
    )
    
    args = parser.parse_args()
    
    if not args.query and not args.text:
        parser.error("Either --query or --text must be provided")
    
    try:
        # Initialize the entity linker
        linker = ChEBIEntityLinker(
            collection_name=args.collection,
            model_type=args.model,
            persist_directory=args.persist_directory,
            top_k=args.top_k
        )
        
        if args.query:
            # Single query mode
            print(f"\nSearching for: '{args.query}'")
            result = linker.link_entity(args.query)
            
            print(f"\nTop {len(result['candidates'])} candidates:")
            for i, candidate in enumerate(result['candidates'], 1):
                print(f"  {i}. {candidate['chebi_id']}: {candidate['name']}")
                print(f"     Similarity: {candidate['similarity_score']:.3f}")
                print()
            
            if result['best_match']:
                print(f"Best match: {result['best_match']['chebi_id']} - {result['best_match']['name']}")
                print(f"Confidence: {result['confidence']:.3f}")
            
            # Show LLM prompt if requested
            if args.show_prompt and result['candidates']:
                print("\n" + "="*50)
                print("LLM PROMPT:")
                print("="*50)
                prompt = create_llm_prompt(args.query, result['candidates'])
                print(prompt)
        
        elif args.text:
            # Text processing mode (simple example)
            print(f"\nProcessing text: '{args.text}'")
            
            # Simple chemical name extraction (in practice, you'd use NER)
            common_chemicals = ["glucose", "caffeine", "aspirin", "water", "ethanol", "insulin", "dopamine"]
            found_chemicals = [chem for chem in common_chemicals if chem.lower() in args.text.lower()]
            
            if not found_chemicals:
                print("No known chemical names found in text.")
                return
            
            print(f"Found chemical mentions: {found_chemicals}")
            
            for chemical in found_chemicals:
                print(f"\n--- Linking '{chemical}' ---")
                result = linker.link_entity(chemical, context=args.text)
                
                if result['best_match']:
                    print(f"Best match: {result['best_match']['chebi_id']} - {result['best_match']['name']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                else:
                    print("No suitable candidates found")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 