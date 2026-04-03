"""Configuration dataclasses for KEGG reaction amendment / likelihood workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class CofactorConfig:
    """Configuration for cofactors to ignore in reaction matching."""

    cofactors_dict: Dict[str, str] = field(
        default_factory=lambda: {
            "C00001": "H2O",
            "C00080": "H+",
            "C00007": "O2",
            "C00027": "H2O2",
            "C00009": "Phosphate",
            "C00013": "Diphosphate",
            "C00008": "ADP",
            "C00002": "ATP",
            "C00003": "NAD+",
            "C00004": "NADH",
            "C00005": "NADPH",
            "C00006": "NADP+",
        }
    )

    @property
    def kegg_ids(self) -> Set[str]:
        return set(self.cofactors_dict.keys())

    @property
    def name_patterns(self) -> List[str]:
        return list(self.cofactors_dict.values())

    def should_filter(self, participant: str) -> bool:
        return any(pattern in participant for pattern in self.name_patterns)


@dataclass
class ConvergenceConfig:
    """Configuration for iterative convergence / EM-style updates."""

    max_iterations: int = 5
    threshold: float = 0.001
    stable_count: int = 3

    match_score_cutoff: float = 0.1
    convergence_threshold: float = 0.001

    reaction_alpha_start: float = 0.1
    reaction_alpha_increment: float = 0.1
    reaction_alpha_max: float = 0.9
    participant_alpha: float = 0.7

    participant_confidence_threshold: float = 0.3
    enable_participant_discovery: bool = True
    min_reaction_likelihood_for_discovery: float = 0.1

    def get_reaction_alpha(self, iteration: int) -> float:
        return min(
            self.reaction_alpha_start + (iteration * self.reaction_alpha_increment),
            self.reaction_alpha_max,
        )


@dataclass
class MatchingConfig:
    """Configuration for fuzzy string matching (rapidfuzz)."""

    similarity_threshold: int = 80
    jaccard_threshold: int = 70
    default_low_probability: float = 1e-6
