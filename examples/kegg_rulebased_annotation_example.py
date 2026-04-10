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
from core import annotate_model

from core.reaction.annotation_workflow import run_kegg_annotation_workflow_rulebased

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

model_file = "tests/test_models/glycolysis_part1.xml"
kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
llm_model = "Llama-3.3-70B-Instruct"


# first annotate model using ChEBI to get a list of ChEBI recommendations
# In this example, a list of recommended ChEBI annotation is provided. 
recommendations_df = pd.read_csv("./examples/glycolysis_part1-recommendations.csv")
TOP_K=10


def main() -> None:
    
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)

    annotation_result = annotate_model(
        model_file=model_file,
        llm_model=llm_model,
        method="rulebased",
        entity_type="reaction",
        database="kegg",
        top_k=TOP_K,
        species_recommendations_df=recommendations_df
    )



    """if result is not None:
        logger.info(
            "Workflow outputs: ChEBI-to-KEGG rows=%s, KEGG reaction candidates=%s, "
            "scored candidates=%s, updated participants=%s",
            len(result.high_score_recommendations),
            len(result.kegg_recommendations),
            len(result.scored_reactions),
            len(result.updated_participants),
        )"""


if __name__ == "__main__":
    main()
