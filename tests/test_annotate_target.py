"""Checks for annotate= species / reactions / both helpers."""

from pathlib import Path

import pandas as pd

from core.annotation_workflow import (
    _chebi_rows,
    _load_species_recommendations,
    _resolve_annotate,
    _species_recommendations_from_model,
)
from utils.constants import EntityType

ROOT = Path(__file__).resolve().parent.parent
MODEL_190 = ROOT / "tests" / "test_models" / "BIOMD0000000190.xml"
SPECIES_CSV = ROOT / "examples" / "glycolysis_part1-recommendations.csv"


def test_resolve_annotate():
    assert _resolve_annotate("species", EntityType.CHEMICAL, "direct") == "species"
    assert _resolve_annotate("reactions", EntityType.CHEMICAL, "direct") == "reactions"
    assert _resolve_annotate("reaction", EntityType.CHEMICAL, "direct") == "reactions"
    assert _resolve_annotate("both", EntityType.CHEMICAL, "direct") == "both"
    assert _resolve_annotate("species", EntityType.REACTION, "direct") == "reactions"
    assert _resolve_annotate("species", EntityType.CHEMICAL, "rulebased") == "reactions"
    assert _resolve_annotate("both", EntityType.REACTION, "rulebased") == "both"


def test_chebi_rows_filters_reason_and_non_chebi():
    df = pd.DataFrame([
        {"id": "Reason:", "annotation": "CHEBI:1"},
        {"id": "s1", "annotation": "CHEBI:4167"},
        {"id": "s2", "annotation": "UNIPROT:P12345"},
        {"id": "s3", "annotation": ""},
    ])
    out = _chebi_rows(df)
    assert list(out["id"]) == ["s1"]
    assert list(out["annotation"]) == ["CHEBI:4167"]


def test_species_recommendations_from_model_and_csv():
    from_model = _species_recommendations_from_model(str(MODEL_190))
    assert not from_model.empty
    assert {"id", "annotation"}.issubset(from_model.columns)
    assert from_model["annotation"].str.upper().str.startswith("CHEBI:").all()

    from_csv = _load_species_recommendations(str(MODEL_190), str(SPECIES_CSV))
    assert not from_csv.empty
    assert from_csv["annotation"].str.upper().str.startswith("CHEBI:").all()


if __name__ == "__main__":
    test_resolve_annotate()
    test_chebi_rows_filters_reason_and_non_chebi()
    test_species_recommendations_from_model_and_csv()
    print("ok")
