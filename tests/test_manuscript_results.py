"""Offline checks for cohort safety, denominators, pairing and guarded plans."""
import copy
import json
from pathlib import Path

import libsbml
import pandas as pd
import pytest

from benchmark.scripts.manuscript_metrics import final_annotation_metrics, retrieval_values, paired_summary
from benchmark.scripts.manuscript_propagation import condition_queries, complete_prediction_table, load_model, alias_source_species
from benchmark.scripts.manuscript_plans import load_luna_runner, write_runner_plan
from core.reaction.grounded import provider_payload, digest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import socket
    import httpx
    def forbidden(*args, **kwargs):
        pytest.fail("network forbidden in manuscript tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)


@pytest.fixture
def source(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("not_a_benchmark_model")
    c = model.createCompartment()
    c.setId("c")
    c.setConstant(True)
    for sid in ["a", "b", "unused"]:
        s = model.createSpecies()
        s.setId(sid)
        s.setName("name " + sid)
        s.setMetaId("meta_" + sid)
        s.setCompartment("c")
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setConstant(False)
        cv = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
        cv.setBiologicalQualifierType(libsbml.BQB_IS)
        cv.addResource("https://identifiers.org/chebi/CHEBI:123")
        assert s.addCVTerm(cv) == 0
    reaction = model.createReaction()
    reaction.setId("r")
    reaction.setReversible(False)
    reaction.setFast(False)
    for callback, sid, coefficient in [(reaction.createReactant, "a", 2), (reaction.createProduct, "b", 1)]:
        ref = callback()
        ref.setSpecies(sid)
        ref.setStoichiometry(coefficient)
        ref.setConstant(True)
    path = tmp_path / "source.xml"
    assert libsbml.writeSBMLToFile(doc, str(path))
    return path


def test_no_reference_fallback_in_predicted_or_names_only(source):
    frame = pd.DataFrame([{"id": "a", "annotation": "CHEBI:456", "identity_status": "unresolved"}])
    rows = condition_queries(source, frame)
    assert "CHEBI:123" in rows["reference"][0]["query"]
    assert "ChEBI=" not in rows["predicted"][0]["query"]
    assert "ChEBI=" not in rows["names_only"][0]["query"]
    for condition in rows:
        assert rows[condition][0]["reaction_equation"] == "2 a => b"
        assert rows[condition][0]["species_evidence"]["b"]["name"] == "name b"


def test_prediction_alternatives_qualifiers_and_legacy_scope(source):
    frame = pd.DataFrame([{"id": "a", "annotation": "CHEBI:456", "qualifier": "is"},
                          {"id": "a", "annotation": "CHEBI:456", "qualifier": "hasPart"},
                          {"id": "a", "annotation": "KEGG:C00001", "qualifier": "is"}])
    rows = condition_queries(source, frame)["predicted"][0]
    assert "CHEBI:123" not in rows["query"]
    assert rows["query"].count("CHEBI:456") == 1 and "C00001" in rows["query"]
    assert rows["species_evidence"]["a"]["conflicting_identifiers"] == []
    assert {r["identity_status"] for r in rows["species_evidence"]["a"]["records"]} == {"legacy_unassessed"}
    frame["identity_status"] = ""
    assert "ChEBI=" not in condition_queries(source, frame)["predicted"][0]["query"]


def test_unknown_prediction_species_rejected(source):
    doc = load_model(source)
    with pytest.raises(ValueError, match="not in SBML"):
        complete_prediction_table(doc.getModel(), pd.DataFrame([{"id": "unknown", "annotation": "CHEBI:1"}]))


def test_complete_table_preserves_legacy_column_absence(source):
    model = load_model(source).getModel().clone()
    frame = complete_prediction_table(model, pd.DataFrame([{"id": "a", "annotation": "CHEBI:456"}]))
    assert set(frame.id) == {"a", "b", "unused"}
    assert "identity_status" not in frame


def test_uniform_aliases_remove_intrinsic_ids_and_preserve_condition_context(source, tmp_path):
    document = load_model(source)
    model = document.getModel()
    model.getSpecies("a").setId("C00001")
    reaction = model.getReaction("r")
    reaction.getReactant(0).setSpecies("C00001")
    law = reaction.createKineticLaw()
    law.setMath(libsbml.parseL3Formula("C00001 * 2"))
    assert libsbml.writeSBMLToFile(document, str(source))
    aliased = tmp_path / "aliased.xml"
    mapping = alias_source_species(source, aliased)
    assert set(mapping).isdisjoint(mapping.values())
    rows = condition_queries(aliased)
    assert "C00001" not in rows["names_only"][0]["query"]
    assert "name a" in rows["names_only"][0]["query"]
    assert rows["reference"][0]["reaction_equation"] == rows["names_only"][0]["reaction_equation"]
    parsed = load_model(aliased)
    assert mapping["C00001"] in libsbml.formulaToL3String(parsed.getModel().getReaction("r").getKineticLaw().getMath())


def outcome(rid, reference, status, predicted=(), model="m"):
    return {"model_id": model, "reaction_id": rid, "reference_ids": list(reference), "status": status, "prediction_ids": list(predicted)}


def test_multilabel_reference_denominators_abstentions_failures():
    rows = [outcome("a", ["R1", "R2"], "selected", ["R1"]),
            outcome("b", ["R3"], "selected", ["wrong"]),
            outcome("c", ["R4"], "abstained"), outcome("d", ["R5"], "incomplete"),
            outcome("e", [], "selected", ["R6"])]
    result = final_annotation_metrics(rows)
    assert result["all_reactions"] == 5 and result["evaluable_reactions"] == 4
    assert result["reference_agreement"] == .25
    assert result["pooled_micro_precision"] == .5 and result["selective_accuracy"] == .5
    assert result["pooled_micro_recall"] == 1 / 5
    assert result["mean_reaction_recall"] == .125
    assert result["answer_coverage_all"] == 3 / 5
    assert result["biological_abstentions"] == 1 and result["technical_failures"] == result["incomplete_outputs"] == 1
    assert result["undefined_precision_empty_predictions"] == 2


@pytest.mark.parametrize("status", ["planned", "not_attempted", "pending_species_evidence"])
def test_unexecuted_results_cannot_be_scored_as_failures(status):
    with pytest.raises(ValueError, match="pending"):
        final_annotation_metrics([outcome("a", ["R1"], status)])


def test_failed_output_never_asserts_identifiers():
    with pytest.raises(ValueError):
        final_annotation_metrics([outcome("a", ["R1"], "schema_invalid", ["R1"])])


def test_retrieval_candidates_are_hits_not_asserted_precision():
    values = retrieval_values(["x", "R1", "R2"], ["R1", "R2"])
    assert values["hit_at_1"] == 0 and values["hit_at_3"] == 1
    assert values["label_recall_at_3"] == 1 and values["mrr_at_10"] == .5
    with pytest.raises(ValueError):
        retrieval_values(["R1", "R1"], ["R1"])


def paired_row(mid, rid, cluster, value):
    return {"model_id": mid, "reaction_id": rid, "cluster_id": cluster,
            **{f"hit_at_{k}": value for k in [1, 3, 5, 10]}, "mrr_at_10": value}


def test_model_macro_and_cluster_pairing_reproducible():
    a = [paired_row("big", str(i), "c1", 0) for i in range(9)] + [paired_row("small", "x", "c2", 0)]
    b = [paired_row("big", str(i), "c1", 1) for i in range(9)] + [paired_row("small", "x", "c2", 0)]
    result = paired_summary(a, b, replicates=100)
    assert result == paired_summary(a[::-1], b[::-1], replicates=100)
    assert result["metrics"]["hit_at_1"]["micro_delta"] == .9
    assert result["metrics"]["hit_at_1"]["model_macro_delta"] == .5
    assert sum(result["metrics"]["hit_at_10"]["transitions"].values()) == 10


def test_unpaired_or_duplicate_reactions_rejected():
    a = [paired_row("m", "a", "c", 0)]
    with pytest.raises(ValueError):
        paired_summary(a + a, a + a)
    with pytest.raises(ValueError):
        paired_summary(a, [paired_row("m", "b", "c", 0)])
    with pytest.raises(ValueError):
        paired_summary(a, [paired_row("m", "a", "different", 0)])


def evidence():
    return [{"evidence_id": f"E{i:02d}", "kegg_id": f"R{i:05d}", "fused_rank": i, "fused_score": 1 / (60 + i),
             "bm25_rank": i, "trained_biencoder_rank": i, "name": "synthetic", "definition": "synthetic",
             "equation": "a => b", "enzyme": "", "rclass": "", "brite": ""} for i in range(1, 11)]


def test_complete_provider_input_change_always_changes_identity():
    original = provider_payload("query", evidence())
    for field, value in [("model", "other-model"), ("max_output_tokens", 512), ("reasoning", {"effort": "high"})]:
        changed = {**original, field: value}
        assert digest(original) != digest(changed)
    altered = evidence()
    altered[0]["fused_score"] += 1e-12
    assert digest(original) != digest(provider_payload("query", altered))


def test_catalog_cache_requires_frozen_configuration_bytes_and_environment(tmp_path):
    from benchmark.scripts.manuscript_propagation import (
        verify_catalog_cache, write_json, sha256, CONFIG, CATALOG_SHA256, MANIFEST_SHA256)
    matrix = tmp_path / "catalog_embeddings.npy"
    matrix.write_bytes(b"synthetic matrix, never loaded as numpy")
    protocol = {"selected_retrieval": {**CONFIG, "catalog_sha256": CATALOG_SHA256,
                                     "encoder_manifest_sha256": MANIFEST_SHA256}}
    write_json(tmp_path / "protocol.json", protocol)
    expected = {"device": "cpu", "dtype": "float32", "torch": "synthetic-version"}
    meta = {**expected, "embedding_sha256": sha256(matrix)}
    write_json(tmp_path / "catalog_embedding_cache.json", meta)
    write_json(tmp_path / "ranking_freeze.json", {
        "protocol_sha256": sha256(tmp_path / "protocol.json"), "catalog_cache": meta})
    assert verify_catalog_cache(tmp_path, expected) == meta
    with pytest.raises(ValueError, match="not compatible"):
        verify_catalog_cache(tmp_path, {**expected, "torch": "different-version"})
    matrix.write_bytes(b"changed document matrix")
    with pytest.raises(ValueError, match="not compatible"):
        verify_catalog_cache(tmp_path, expected)
    protocol["selected_retrieval"]["unapproved_document_transform"] = True
    write_json(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="configuration changed"):
        verify_catalog_cache(tmp_path, expected)


def test_existing_runner_plan_failure_resume_and_no_repurchase(tmp_path):
    luna = Path(__file__).resolve().parents[1].parent / "luna_public"
    if not (luna / "benchmark/scripts/luna_output_budget.py").is_file():
        pytest.skip("preserved optional runner unavailable")
    runner = load_luna_runner(luna)
    body = provider_payload("synthetic", evidence())
    rows = [{"unit": {"model_id": "unseen", "reaction_id": rid, "condition": "synthetic"}, "provider_payload": body}
            for rid in ["r1", "r2"]]
    directory = tmp_path / "plan"
    plan = write_runner_plan(directory, rows, runner)
    state = runner.RunState(directory, tmp_path / "state")
    key = state.requests[0]["unit_cache_key"]
    with pytest.raises(ValueError, match="not authorized"):
        state.run_one(key, lambda body: pytest.fail("must be denied"))
    result = state.run_one(key, lambda body: {"status": "incomplete", "model": "gpt-6.1-terra", "output": []},
                           authorized_plan_digest=state.plan_digest)
    assert result["terminal_status"] == "failure_incomplete" and result["biological_abstain"] is None
    resumed = runner.RunState(directory, tmp_path / "state")
    assert resumed.run_one(key, lambda body: pytest.fail("no retry"), authorized_plan_digest=state.plan_digest) == result
    assert len(resumed.coverage()) == 2 and resumed.total(resumed._read()) > 0
    second = state.requests[1]["unit_cache_key"]
    resumed.reserve(second)
    again = runner.RunState(directory, tmp_path / "state")
    interrupted = again.run_one(second, lambda body: pytest.fail("uncertain attempt must not resend"), authorized_plan_digest=again.plan_digest)
    assert interrupted["terminal_status"] == "failure_interrupted_or_uncertain"
    assert again.total(again._read()) == plan["reservation_nusd"]
