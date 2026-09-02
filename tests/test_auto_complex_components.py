"""Parser and per-component routing for auto-type complexes."""

from unittest.mock import patch

from core.annotation_workflow import _search_databases, rank_species_annotations_with_llm
from core.data_types import Recommendation
from core.llm_interface import parse_llm_response, parse_typed_components
from utils.constants import DatabaseID, EntityType
import pandas as pd


def test_parse_typed_complex_line():
    text = (
        'Ras_Raf1 (complex): "HRAS", "KRAS", "NRAS" (protein); "RAF1" (protein)\n'
        'Ras_GTP (complex): "HRAS", "KRAS", "NRAS" (protein); "GTP", "guanosine triphosphate" (chemical)\n'
        'A (chemical): "glucose", "D-glucose"\n'
        "Reason: mixed complex\n"
    )
    synonyms, types, reason, components = parse_llm_response(text, EntityType.AUTO)
    assert types["Ras_Raf1"] == "complex"
    assert types["Ras_GTP"] == "complex"
    assert types["A"] == "chemical"
    assert synonyms["Ras_Raf1"] == ["HRAS", "KRAS", "NRAS", "RAF1"]
    assert components["Ras_Raf1"] == [
        ("protein", ["HRAS", "KRAS", "NRAS"]),
        ("protein", ["RAF1"]),
    ]
    assert components["Ras_GTP"] == [
        ("protein", ["HRAS", "KRAS", "NRAS"]),
        ("chemical", ["GTP", "guanosine triphosphate"]),
    ]
    assert "A" not in components
    assert "mixed complex" in reason
    assert "(protein)" not in synonyms["Ras_Raf1"]


def test_parse_legacy_complex_has_no_components():
    text = 'D (complex): "glucose", "ATP", "Hexokinase-1"\nReason: old format\n'
    synonyms, types, _reason, components = parse_llm_response(text, EntityType.AUTO)
    assert types["D"] == "complex"
    assert synonyms["D"] == ["glucose", "ATP", "Hexokinase-1"]
    assert components == {}
    assert parse_typed_components('"glucose", "ATP", "Hexokinase-1"') == []


def _fake_search(species_list, synonyms_dict, database, method, top_k, tax_id=None, model_info=None, model_type=None):
    sid = species_list[0]
    names = list(synonyms_dict[sid])
    return [Recommendation(
        id=sid,
        synonyms=names,
        candidates=[f"{database}:{names[0]}"],
        candidate_names=[database],
        match_score=[1.0],
    )]


def test_complex_routes_each_component_to_one_db():
    with patch("core.annotation_workflow._search_one_database", side_effect=_fake_search) as mock_search:
        recs, _species_db, cand_dbs = _search_databases(
            ["X"],
            {"X": ["RAS", "GTP", "RAF1"]},
            EntityType.AUTO,
            [DatabaseID.CHEBI, DatabaseID.UNIPROT],
            "direct",
            3,
            entity_type_dict={"X": "complex"},
            component_dict={"X": [
                ("protein", ["RAS"]),
                ("chemical", ["GTP"]),
                ("protein", ["RAF1"]),
            ]},
        )
    dbs = [call.args[2] for call in mock_search.call_args_list]
    name_lists = [list(call.args[1]["X"]) for call in mock_search.call_args_list]
    assert dbs == ["uniprot", "chebi", "uniprot"]
    assert name_lists == [["RAS"], ["GTP"], ["RAF1"]]
    assert recs[0].candidates == ["uniprot:RAS", "chebi:GTP", "uniprot:RAF1"]
    assert cand_dbs[("X", "chebi:GTP")] == "chebi"
    assert cand_dbs[("X", "uniprot:RAS")] == "uniprot"


def test_untyped_complex_still_searches_all_dbs():
    with patch("core.annotation_workflow._search_one_database", side_effect=_fake_search) as mock_search:
        _search_databases(
            ["Ras_Raf1"],
            {"Ras_Raf1": ["RAS", "RAF1"]},
            EntityType.AUTO,
            [DatabaseID.CHEBI, DatabaseID.UNIPROT],
            "direct",
            3,
            entity_type_dict={"Ras_Raf1": "complex"},
        )
    dbs = [call.args[2] for call in mock_search.call_args_list]
    assert dbs == ["chebi", "uniprot"]


def test_rank_skips_complex_species():
    df = pd.DataFrame([
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "annotation": "CHEBI:1", "annotation_label": "rasagiline"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "annotation": "UNIPROT:P1", "annotation_label": "RAF1"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "annotation": "UNIPROT:P2", "annotation_label": "HRAS"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "annotation": "UNIPROT:P3", "annotation_label": "KRAS"},
    ])
    with patch("core.annotation_workflow.query_llm") as mock_llm:
        out = rank_species_annotations_with_llm("dummy.xml", df, n_return=2)
    mock_llm.assert_not_called()
    assert list(out["annotation"]) == ["CHEBI:1", "UNIPROT:P1", "UNIPROT:P2", "UNIPROT:P3"]
