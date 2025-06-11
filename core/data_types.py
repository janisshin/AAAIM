"""
Shared data types
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Recommendation:
    """
    Recommendation dataclass for database search results.
    """
    id: str  # ID for the species
    synonyms: list  # List of synonyms predicted by LLM
    candidates: list  # List of database IDs (ChEBI IDs, NCBI gene IDs, etc.)
    candidate_names: list  # List of names of the predicted candidates
    match_score: list  # Match scores (normalized hit count for direct search, cosine similarity for RAG) 