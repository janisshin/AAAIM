"""Synthetic-only cache compatibility, dry-run and leakage tests."""
import copy
import json
from pathlib import Path

import pytest

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts import phase5_llm_plan as plan
from benchmark.scripts.phase5_perturb import digest


def fixture():
    record = {"evidence_id": "E01", "kegg_id": "R99991", "fused_rank": 1, "fused_score": .03,
              "bm25_rank": 1, "trained_biencoder_rank": 1, "name": "Synthetic", "definition": "Synthetic candidate",
              "equation": "C00001 <=> C00002", "enzyme": "", "rclass": "", "brite": "", "provenance": {"synthetic": True}}
    query = "Synthetic substrate => synthetic product"
    payload = {"model": "synthetic-only", "input": grounded.build_user_prompt({"query": query}, [grounded.EvidenceRecord(**record)]), "tools": []}
    return query, [record], payload


def test_clean_cache_only_reused_on_identical_query_evidence_and_configuration():
    query, records, original = fixture()
    rebuilt, compatible = plan.payload_for(query, records, original, records)
    assert compatible and rebuilt == original
    changed, compatible = plan.payload_for(query + " altered", records, original, records)
    assert not compatible and digest(changed) != digest(original)
    changed_records = copy.deepcopy(records)
    changed_records[0]["provenance"]["revision"] = "new"
    changed, compatible = plan.payload_for(query, changed_records, original, records)
    assert not compatible and digest(changed) != digest(original)
    assert original["model"] == changed["model"]


def test_paid_execution_fails_before_file_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input opened before execution denial")
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(SystemExit, match="disabled"):
        plan.main(["--execute", "--source-root", "SYNTHETIC", "--out", "SYNTHETIC"])
