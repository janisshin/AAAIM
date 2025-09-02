"""
Shared data types
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import Counter

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

@dataclass
class ReactionRecommendation(Recommendation):
    """
    Extended Recommendation class specifically for reaction annotations.
    Includes additional fields for reaction-specific information.
    """
    substrates: List[Counter]  # List of substrate Counter objects
    products: List[Counter]  # List of product Counter objects
    equation: str  # Original reaction equation string
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata about the reaction 