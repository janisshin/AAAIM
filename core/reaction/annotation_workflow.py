from typing import NamedTuple, Optional

import pandas as pd
from collections import Counter

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from .kegg_features import KEGGReactionFeatures
from .species_probability import init_species_probs_from_dict
from core.model_info import extract_model_info, extract_reactions_from_sbml, get_all_reaction_ids
from core.annotation_workflow import _generate_recommendation_table
from .matching import map_reactions_to_kegg_with_relaxation
from .amendment_config import CofactorConfig, ConvergenceConfig, MatchingConfig
from .amendment import LikelihoodCalculator, SimilarityCalculator, update_participant_likelihoods
from .utils import check_environment, extract_reaction_participants, map_chebi_to_kegg


class KeggAnnotationWorkflowResult(NamedTuple):
    """DataFrames from :func:`run_kegg_annotation_workflow` (ChEBI→KEGG map, KEGG reaction
    candidates, scored candidates, updated participant likelihoods)."""

    high_score_recommendations: pd.DataFrame
    kegg_recommendations: pd.DataFrame
    scored_reactions: pd.DataFrame
    updated_participants: pd.DataFrame


def run_kegg_annotation_workflow(
    model_file: str,
    recommendations_df: pd.DataFrame,
    kegg_features_file: str,
    entity_type: str = "reaction",
    database: str = "kegg",
    llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    cofactor_config: Optional[CofactorConfig] = None,
    convergence_config: Optional[ConvergenceConfig] = None,
    matching_config: Optional[MatchingConfig] = None,
) -> Optional[KeggAnnotationWorkflowResult]:
    """Run the complete KEGG annotation workflow.

    Returns:
        KeggAnnotationWorkflowResult with four DataFrames, or ``None`` if the
        environment check fails (see :func:`~core.reaction.utils.check_environment`).
    """
    if cofactor_config is None:
        cofactor_config = CofactorConfig()
    if convergence_config is None:
        convergence_config = ConvergenceConfig()
    if matching_config is None:
        matching_config = MatchingConfig()

    if not check_environment(model_file):
        logger.error("Environment check failed. Please fix the issues and try again.")
        return None

    logger.info("Model file: %s", model_file)
    logger.info("LLM model: %s", llm_model)
    logger.info("Analyzing model: %s", model_file)

    reaction_ids = get_all_reaction_ids(model_file)
    model_info = extract_model_info(model_file, reaction_ids, entity_type)

    logger.info("Step 2: Map ChEBI IDs to KEGG Compound IDs")
    _, high_score_recommendations = map_chebi_to_kegg(recommendations_df)

    logger.info("\nSample of ChEBI to KEGG mapping:")
    if not high_score_recommendations.empty:
        logger.info(
            high_score_recommendations[
                ["id", "display_name", "annotation", "KEGG_ID", "match_score"]
            ].head()
        )

    logger.info("Step 3: Begin rule-based matching to identify reactions")
    reactions, _ = extract_reactions_from_sbml(
        model_file,
        list(high_score_recommendations["id"].unique()),
    )
    _, match_results, _species_relax_levels = map_reactions_to_kegg_with_relaxation(
        reactions,
        reaction_ids, 
        high_score_recommendations,
        spectators=False,
        cofactors_to_ignore=cofactor_config.kegg_ids,
        top_k=None,
    )

    # Only keep reaction candidates that are eligible for updating.
    allowed_reaction_types = {"mappable", "ambiguous_mapping"}
    match_results = [
        rec
        for rec in match_results
        if str((getattr(rec, "metadata", None) or {}).get("reaction_type", "mappable")) in allowed_reaction_types
    ]

    kegg_recommendations_df = _generate_recommendation_table(
        model_file,
        match_results,
        {},
        model_info,
        entity_type,
        database,
        {},
    )

    kegg_recommendations_df["match_score_norm"] = (
        kegg_recommendations_df["match_score"]
        / kegg_recommendations_df.groupby("id")["match_score"].transform("sum")
    )

    reaction_participants = extract_reaction_participants(model_info, recommendations_df)

    kegg_features = KEGGReactionFeatures.load_from_file(kegg_features_file)

    kegg_recommendations_df["participants"] = kegg_recommendations_df["annotation"].apply(
        kegg_features.get_participants
    )
    kegg_recommendations_df["participant_ids"] = kegg_recommendations_df["annotation"].apply(
        kegg_features.get_participant_ids
    )

    merged_participants = kegg_recommendations_df.groupby("id")["participants"].agg("; ".join)
    counters = merged_participants.apply(
        lambda s: Counter(p.strip() for p in s.split(";") if p.strip())
    )

    similarity_calc = SimilarityCalculator(matching_config)
    init_probs = init_species_probs_from_dict(
        reaction_participants, counters, similarity_calc.is_plausible_match
    )

    likelihood_calc = LikelihoodCalculator(cofactor_config, matching_config, convergence_config)
    scored_df = likelihood_calc.compute_reaction_likelihoods(init_probs, kegg_recommendations_df)

    updated_participants_df, _ = update_participant_likelihoods(
        high_score_recommendations,
        scored_df,
        model_file,
        model_info=model_info,
        kegg_features=kegg_features,
        reactions=reactions,
        reaction_ids=reaction_ids,
        entity_type=entity_type,
        database=database,
        cofactor_config=cofactor_config,
        convergence_config=convergence_config,
    )

    logger.info("\nSample of participants with updated likelihoods after convergence:")
    if not updated_participants_df.empty:
        logger.info(
            updated_participants_df[["id", "display_name", "KEGG_ID", "participant_likelihood"]].head()
        )

    updated_participants_df.sort_values(by="participant_likelihood", ascending=False, inplace=True)
    scored_df.sort_values(by="likelihood", ascending=False, inplace=True)

    logger.info("KEGG annotation workflow completed successfully.")
    return KeggAnnotationWorkflowResult(
        high_score_recommendations=high_score_recommendations,
        kegg_recommendations=kegg_recommendations_df,
        scored_reactions=scored_df,
        updated_participants=updated_participants_df,
    )