#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KEGG Reaction Annotation Example

Demonstrates the workflow for annotating reactions in SBML models using KEGG
references: ChEBI→KEGG mapping, rule-based reaction matching, initial
likelihoods, and iterative participant updates (see core.reaction.amendment).
"""

import logging
import os
import sys

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from core.reaction.annotation_workflow import run_kegg_annotation_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

model_file = "tests/test_models/glycolysis_part1.xml"
kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
llm_model = "meta-llama/llama-3.1-8b-instruct"

# first annotate model using ChEBI to get a list of ChEBI recommendations
# In this example, a list of recommended ChEBI annotation is provided. 
recommendations_df = pd.read_csv("./examples/glycolysis_part1-recommendations.csv")
TOP_K=10


logger.info("Model file: %s", model_file)
logger.info("LLM model: %s", llm_model)
logger.info("")

logger.info("Analyzing model: %s", model_file)

logger.info(
    "1. Reaction Annotation Workflow (for models without reaction annotations)"
)
logger.info("-" * 65)

def main() -> None:
    
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)

    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API keys found in environment.")
        logger.warning(
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features."
        )
        return

    result = run_kegg_annotation_workflow(
        model_file=model_file,
        recommendations_df=recommendations_df,
        kegg_features_file=kegg_features_file,
        llm_model=llm_model,
    )
    if result is not None:
        logger.info(
            "Workflow outputs: ChEBI-to-KEGG rows=%s, KEGG reaction candidates=%s, "
            "scored candidates=%s, updated participants=%s",
            len(result.high_score_recommendations),
            len(result.kegg_recommendations),
            len(result.scored_reactions),
            len(result.updated_participants),
        )


if __name__ == "__main__":
    main()
