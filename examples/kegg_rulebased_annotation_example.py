#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""KEGG Reaction Annotation Example

Demonstrates the workflow for annotating reactions in SBML models using KEGG
references: ChEBI→KEGG mapping, rule-based reaction matching, initial
likelihoods, and iterative participant updates (see core.reaction.amendment).
"""

from __future__ import annotations

import logging
import sys

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from core import annotate_model

from core.llm_interface import query_llm
from core.model_info import get_all_reaction_ids, map_reaction_ids_to_stoichiometry_strings
from core.reaction.kegg_features import KEGGReactionFeatures
from utils.constants import REACTION_ANNOTATION_RANKING_PROMPT

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
TOP_K = 10


def _strip_kegg_annotation_prefix(raw) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.upper().startswith("KEGG:"):
        return s[5:].strip()
    return s


def _reaction_annotation_choices_for_reaction(sub_df: pd.DataFrame) -> str:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for _, row in sub_df.iterrows():
        rid = _strip_kegg_annotation_prefix(row.get("annotation", ""))
        if not rid:
            continue
        definition = row.get("reaction_definition", "")
        if definition is None or (isinstance(definition, float) and definition != definition):
            definition = ""
        else:
            definition = str(definition).strip()
        key = (rid, definition)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{rid}: {definition}")
    return "\n".join(lines)


def main() -> None:

    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)

    ranked_responses: list[list[str]] = []
    ranked_reaction_ids: list[str] = []

    """_annotation_result, _metrics = annotate_model(
        model_file=model_file,
        llm_model=llm_model,
        method="rulebased",
        entity_type="reaction",
        database="kegg",
        top_k=TOP_K,
        species_recommendations_df=recommendations_df,
    )"""

    # Load the recommendations CSV written by annotate_model and add KEGG DEFINITION text.
    csv_path = Path(f"{Path(model_file).name}_recommendations.csv")
    result_df = pd.read_csv(csv_path)

    kegg_features = KEGGReactionFeatures.load_from_file(kegg_features_file)
    result_df["reaction_definition"] = result_df["annotation"].map(kegg_features.get_definition)

    # out_path = csv_path.with_name(csv_path.stem + "_with_kegg_definition.csv")
    result_df.to_csv(csv_path, index=False)
    logger.info(f"{csv_path} updated with with KEGG DEFINITIONs")

    reaction_ids = get_all_reaction_ids(model_file)
    id_to_equation = map_reaction_ids_to_stoichiometry_strings(model_file)

    for reaction_id in reaction_ids:
        model_reaction = id_to_equation.get(reaction_id, reaction_id)
        print(model_reaction)

        sub = result_df[result_df["id"] == reaction_id]
        reaction_annotation_choices = _reaction_annotation_choices_for_reaction(sub)
        if not reaction_annotation_choices.strip():
            continue

        prompt = REACTION_ANNOTATION_RANKING_PROMPT.format(
            model_reaction=model_reaction,
            reaction_annotation_choices=reaction_annotation_choices,
        )

        response_text = query_llm(prompt, model=llm_model, entity_type="reaction")
        response_lines = [ln.strip() for ln in (response_text or "").splitlines() if ln.strip()]

        print(reaction_id)
        print(response_lines)

        if len(response_lines) ==1 and response_lines[0] == "UNK":
            continue

        ranked_reaction_ids.append(reaction_id)
        ranked_responses.append(response_lines[:TOP_K])

    # ranked_responses is a list of lists aligned to ranked_reaction_ids.
    logger.info("Collected LLM rankings for %d reactions", len(ranked_responses))

    # Build a ranked DataFrame: only rows whose KEGG IDs appear in ranked_responses,
    # preserving order of appearance (reaction loop order, then LLM-ranked order).
    ranked_rows: list[pd.DataFrame] = []
    for reaction_id, kegg_ids in zip(ranked_reaction_ids, ranked_responses):
        for kegg_id in kegg_ids:
            if not kegg_id:
                continue
            # Match against the stored annotation format (e.g., "KEGG:R01068").
            mask = (result_df["id"] == reaction_id) & (result_df["annotation"].astype(str).str.upper() == f"KEGG:{kegg_id}".upper())
            rows = result_df[mask]
            if rows.empty:
                continue
            ranked_rows.append(rows.iloc[[0]])

    ranked_df = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else result_df.iloc[0:0].copy()
    ranked_out_path = csv_path.with_name(csv_path.stem + "_llm_ranked.csv")
    ranked_df.to_csv(ranked_out_path, index=False)
    logger.info("LLM-ranked recommendations saved to %s", ranked_out_path)

    return ranked_df


if __name__ == "__main__":
    main()
