"""Synthetic reference/version safety tests; no biological judgments or network."""
import copy

import pandas as pd
import pytest

from benchmark.scripts import reference_sensitivity as sensitivity


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def review(verdict="label_correct", ids=""):
    return {"audit_id": "synthetic", "verdict": verdict, "proposed_kegg_ids": ids,
            "biological_rationale": "Synthetic reviewer rationale, no real case.", "supporting_sources": ""}


def context(**updates):
    return {"prediction_assessment": "prediction_correct", "recommended_label_action": "",
            "defensible_prediction_ids": "R00009", "failure_taxonomy": ["source_label_error"],
            "biological_rationale": "Synthetic later text.", **updates}


def test_explicit_rejection_replaces_not_unions_originals():
    result = sensitivity.derive_change(review("label_incorrect", "R00002; R00003"), context(), ["R00001", "R00004"])
    assert result["revised_reference_ids"] == ["R00002", "R00003"]
    assert result["operation"] == "replace" and result["source_pass"] == "Pass 1"


@pytest.mark.parametrize("verdict", ["label_correct_but_incomplete", "multiple_defensible_annotations"])
def test_explicit_addition_retains_every_original(verdict):
    result = sensitivity.derive_change(review(verdict, "R00002 R00003"), context(), ["R00001", "R00002"])
    assert result["revised_reference_ids"] == ["R00001", "R00002", "R00003"]
    assert result["operation"] == "add"


@pytest.mark.parametrize("verdict", ["label_correct", "label_incorrect", "multiple_defensible_annotations", "not_kegg_mappable"])
def test_outputs_taxonomy_and_abstention_never_supply_reference_ids(verdict):
    p1 = review(verdict)
    p2 = context(prediction_assessment="abstention_inappropriate", exact_matching_too_strict="yes")
    result = sensitivity.derive_change(p1, p2, ["R00001"])
    assert result["revised_reference_ids"] == ["R00001"] and not result["actionable"]


def test_conflicting_pass2_text_suspends_does_not_choose_replacement():
    p1, p2 = review("label_incorrect", "R00002"), context()
    hold = {"reason": "conflicting recorded reference claims", "evidence": [
        {"pass": 1, "field": "proposed_kegg_ids", "recorded_text": p1["proposed_kegg_ids"]},
        {"pass": 2, "field": "biological_rationale", "recorded_text": p2["biological_rationale"]}]}
    result = sensitivity.derive_change(p1, p2, ["R00001"], hold)
    assert result["operation"] == "hold" and result["revised_reference_ids"] == ["R00001"]
    assert result["reference_issue_unresolved"]
    p2["biological_rationale"] = "different export"
    with pytest.raises(ValueError, match="quotation"):
        sensitivity.derive_change(p1, p2, ["R00001"], hold)


@pytest.mark.parametrize("text", ["Maybe R00001?", "replace with R00001", "R1234", "R00001junk"])
def test_never_scrape_replacement_ids_from_prose(text):
    with pytest.raises(ValueError, match="explicit"):
        sensitivity.parse_proposal(text)


def test_layer_alignment_integrity_and_unreviewed_preservation():
    original = {("m", "a"): ["R00001"], ("m", "b"): ["R00004"]}
    before = copy.deepcopy(original)
    change = {"model_id": "m", "reaction_id": "a", "original_reference_ids": ["R00001"],
              "revised_reference_ids": ["R00002"], "actionable": True, "operation": "replace",
              "explicit_pass1_proposed_ids": ["R00002"]}
    result = sensitivity.apply_layer(original, [change])
    assert original == before and result[("m", "b")] == ["R00004"]
    for invalid in [[change, change], [{**change, "reaction_id": "unknown"}],
                    [{**change, "original_reference_ids": ["R00003"]}],
                    [{**change, "revised_reference_ids": []}], [{**change, "actionable": False}],
                    [{**change, "revised_reference_ids": ["R00009"]}], [{**change, "operation": "add"}]]:
        with pytest.raises(ValueError):
            sensitivity.apply_layer(original, invalid)


def prediction(rid, identifier, *, status="selected", answered=True):
    return {"model_id": "m", "reaction_id": rid, "cluster_id": "c", "predicted_id": identifier,
            "answered": answered, "status": status, "evidence_compliant": False}


def test_reference_changes_do_not_change_coverage_compliance_or_failure_states():
    predictions = [prediction("a", "R00002"), prediction("b", None, status="abstained", answered=False),
                   prediction("c", None, status="incomplete", answered=False),
                   prediction("d", None, status="no_candidate", answered=False)]
    old = [sensitivity.score_assertion(p, ["R00001"]) for p in predictions]
    new = [sensitivity.score_assertion(p, ["R00001", "R00002"]) for p in predictions]
    assert {k: v for k, v in old[0].items() if k != "correct"} == {k: v for k, v in new[0].items() if k != "correct"}
    a, b = sensitivity.final_summary(old), sensitivity.final_summary(new)
    assert (a["n"], a["answered"], a["correct"]) == (4, 1, 0)
    assert (b["n"], b["answered"], b["correct"]) == (4, 1, 1)
    assert b["technical_failures"] == b["incomplete_outputs"] == b["no_candidate"] == b["biological_abstentions"] == 1
    assert sensitivity.transitions(old, new)["incorrect_correct"] == 1


def test_multi_reference_hits_use_real_established_retrieval_evaluator(monkeypatch):
    from benchmark.scripts import phase3_retrieval as evaluator
    monkeypatch.setattr(evaluator, "match_kinds", lambda identifier, truth: {"brite_orthology": identifier in truth})
    rankings = [{"model_id": "m", "reaction_id": "a", "method": "synthetic", "ranked_ids": ["R00002", "R00003", "R00001"]}]
    truth = pd.DataFrame([{"model_id": "m", "reaction_id": "a", "cluster_id": "c", "stratum": "synthetic",
                           "seen_in_train": False, "truth": ["R00001", "R00002"]}])
    row = evaluator.score_rankings(rankings, truth).iloc[0]
    assert row.recall_at_1_exact and row.recall_at_3_exact and row.mrr_at_10_exact == 1
    truth.at[0, "truth"] = ["R00001"]
    row = evaluator.score_rankings(rankings, truth).iloc[0]
    assert not row.recall_at_1_exact and row.recall_at_3_exact and row.mrr_at_10_exact == 1 / 3


def test_paired_differences_keep_identical_cases_and_cluster_resampling():
    a = [sensitivity.score_assertion(prediction("a", "R00001"), ["R00001"]),
         sensitivity.score_assertion(prediction("b", "R00002"), ["R00001"])]
    b = [sensitivity.score_assertion(prediction("a", "R00002"), ["R00001"]),
         sensitivity.score_assertion(prediction("b", "R00001"), ["R00001"])]
    result = sensitivity.paired_comparison(a, b, replicates=100)
    assert result == sensitivity.paired_comparison(a[::-1], b[::-1], replicates=100)
    assert result["delta"] == 0 and result["n"] == 2
    assert result["transitions"]["correct_incorrect"] == result["transitions"]["incorrect_correct"] == 1
    with pytest.raises(ValueError):
        sensitivity.paired_comparison(a, b[:1])
    with pytest.raises(ValueError):
        sensitivity.paired_comparison(a, [{**r, "cluster_id": "other"} for r in b])


def test_immutable_destination_rejected_before_any_source_access(tmp_path):
    with pytest.raises(FileExistsError, match="immutable"):
        sensitivity.build(tmp_path / "missing source", tmp_path / "missing holds", tmp_path)


def test_retrieval_aggregation_is_order_independent_at_byte_precision():
    rows = [{"model_id": "m", "reaction_id": str(i), "mrr_at_10_exact": 1 / rank,
             **{f"recall_at_{k}_exact": rank <= k for k in [1, 3, 5, 10]}}
            for i, rank in enumerate([1, 3, 7, 9, 2, 8, 5, 10, 6, 4])]
    assert sensitivity.retrieval_summary(rows) == sensitivity.retrieval_summary(rows[::-1])
    with pytest.raises(ValueError, match="duplicate"):
        sensitivity.retrieval_summary(rows + rows[:1])


def test_readonly_verifier_rejects_mutated_version_artifact(tmp_path):
    from benchmark.scripts.manuscript_propagation import write_json, sha256
    artifact = tmp_path / "reference_view.jsonl"
    artifact.write_text("original synthetic reference\n", encoding="utf-8")
    write_json(tmp_path / "MANIFEST.json", {"files": [{"path": artifact.name, "sha256": sha256(artifact)}]})
    artifact.write_text("changed synthetic reference\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        sensitivity.verify_output(tmp_path)
