"""Explicit denominators for retrieval and asserted-annotation manuscript metrics."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def final_annotation_metrics(rows):
    """G is a supplied accepted-reference set; P is an asserted validated set.

    Candidate lists are never passed as P. Technical failures have no validated
    assertions and remain in all-reaction/recall denominators when G is known.
    """
    details = []
    for row in rows:
        truth = set(row["reference_ids"])
        status = row["status"]
        if status in {"planned", "not_attempted", "pending_species_evidence"}:
            raise ValueError("planned/unexecuted annotations must remain pending, not scored failures")
        predicted = set(row.get("prediction_ids", [])) if status == "selected" else set()
        if status != "selected" and row.get("prediction_ids"):
            raise ValueError("failed/abstained result cannot assert an annotation")
        overlap = len(truth & predicted)
        evaluable = bool(truth)
        details.append({"model_id": row["model_id"], "reaction_id": row["reaction_id"],
            "status": status, "evaluable": evaluable, "answered": bool(predicted),
            "correct": bool(overlap) if evaluable else None,
            "precision": ratio(overlap, len(predicted)) if evaluable else None,
            "recall": ratio(overlap, len(truth)), "set_exact": predicted == truth if evaluable else None,
            "hits": overlap, "asserted": len(predicted), "reference_count": len(truth)})
    scored = [row for row in details if row["evaluable"]]
    answered = [row for row in scored if row["answered"]]
    by_model = defaultdict(list)
    for row in scored:
        by_model[row["model_id"]].append(row)
    model_rows = []
    for model, values in sorted(by_model.items()):
        selected = [row for row in values if row["answered"]]
        precision = [row["precision"] for row in values if row["precision"] is not None]
        model_rows.append({"model_id": model, "n": len(values), "answered": len(selected),
            "agreement": sum(row["correct"] for row in values) / len(values),
            "precision": float(np.mean(precision)) if precision else None,
            "recall": float(np.mean([row["recall"] for row in values]))})
    total, n, n_answered = len(rows), len(scored), len(answered)
    return {"all_reactions": total, "evaluable_reactions": n, "missing_reference_reactions": total - n,
        "correct": sum(row["correct"] for row in scored), "answered_evaluable": n_answered,
        "answered_all": sum(row["answered"] for row in details),
        "reference_agreement": ratio(sum(row["correct"] for row in scored), n),
        "answer_coverage_all": ratio(sum(row["answered"] for row in details), total),
        "selective_accuracy": ratio(sum(row["correct"] for row in answered), n_answered),
        "pooled_micro_precision": ratio(sum(row["hits"] for row in scored), sum(row["asserted"] for row in scored)),
        "pooled_micro_recall": ratio(sum(row["hits"] for row in scored), sum(row["reference_count"] for row in scored)),
        "mean_reaction_precision": float(np.mean([row["precision"] for row in answered])) if answered else None,
        "mean_reaction_recall": float(np.mean([row["recall"] for row in scored])) if scored else None,
        "undefined_precision_empty_predictions": n - n_answered,
        "pooled_intersection_count": sum(row["hits"] for row in scored),
        "pooled_assertion_count": sum(row["asserted"] for row in scored),
        "pooled_reference_membership_count": sum(row["reference_count"] for row in scored),
        "biological_abstentions": sum(row["status"] == "abstained" for row in rows),
        "incomplete_outputs": sum(row["status"] == "incomplete" for row in rows),
        "technical_failures": sum(row["status"] not in {"selected", "abstained"} for row in rows),
        "model_macro_agreement": float(np.mean([row["agreement"] for row in model_rows])) if model_rows else None,
        "model_macro_precision": float(np.mean([row["precision"] for row in model_rows if row["precision"] is not None]))
            if any(row["precision"] is not None for row in model_rows) else None,
        "model_macro_recall": float(np.mean([row["recall"] for row in model_rows])) if model_rows else None,
        "n_models": len(model_rows),
        "model_macro_precision_defined_models": sum(row["precision"] is not None for row in model_rows),
        "model_rows": model_rows, "per_reaction": details}


def retrieval_values(ids, truth):
    if not truth:
        raise ValueError("retrieval metric requires nonempty frozen reference")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate retrieval identifiers")
    truth = set(truth)
    first = next((i for i, identifier in enumerate(ids[:10], 1) if identifier in truth), None)
    return {**{f"hit_at_{k}": float(bool(set(ids[:k]) & truth)) for k in [1, 3, 5, 10]},
            **{f"label_recall_at_{k}": len(set(ids[:k]) & truth) / len(truth) for k in [1, 3, 5, 10]},
            "mrr_at_10": 1 / first if first else 0.0}


def paired_summary(base, comparison, *, replicates=2000, seed=20261109):
    """Paired cluster bootstrap; resample whole clusters, keeping all model units."""
    key = lambda row: (row["model_id"], row["reaction_id"])
    a, b = {key(row): row for row in base}, {key(row): row for row in comparison}
    if len(a) != len(base) or len(b) != len(comparison) or a.keys() != b.keys():
        raise ValueError("paired populations must have identical unique reaction keys")
    if any(a[k]["cluster_id"] != b[k]["cluster_id"] for k in a):
        raise ValueError("cluster alignment mismatch")
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10"]
    clusters = sorted({r["cluster_id"] for r in base})
    model_clusters = {}
    for row in base:
        if row["model_id"] in model_clusters and model_clusters[row["model_id"]] != row["cluster_id"]:
            raise ValueError("model spans clusters")
        model_clusters[row["model_id"]] = row["cluster_id"]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    result = {"n_reactions": len(base), "n_models": len(model_clusters), "n_clusters": len(clusters),
              "replicates": replicates, "seed": seed, "direction": "comparison minus base", "metrics": {}}
    for metric in metrics:
        per_cluster = []
        for cluster in clusters:
            keys = [k for k in a if a[k]["cluster_id"] == cluster]
            model_deltas = []
            for mid in sorted({k[0] for k in keys}):
                model_keys = [k for k in keys if k[0] == mid]
                model_deltas.append(np.mean([b[k][metric] - a[k][metric] for k in model_keys]))
            per_cluster.append([sum(b[k][metric] - a[k][metric] for k in keys), len(keys),
                                sum(model_deltas), len(model_deltas)])
        matrix = np.asarray(per_cluster)
        sampled = matrix[draws].sum(axis=1)
        micro = sampled[:, 0] / sampled[:, 1]
        macro = sampled[:, 2] / sampled[:, 3]
        transitions = None
        if metric.startswith("hit_"):
            transitions = {"hit_hit": 0, "hit_miss": 0, "miss_hit": 0, "miss_miss": 0}
            for k in a:
                transitions[("hit" if a[k][metric] else "miss") + "_" + ("hit" if b[k][metric] else "miss")] += 1
        result["metrics"][metric] = {"base_micro": float(np.mean([a[k][metric] for k in a])),
            "comparison_micro": float(np.mean([b[k][metric] for k in a])),
            "micro_delta": float(matrix[:, 0].sum() / matrix[:, 1].sum()),
            "micro_ci95": list(map(float, np.quantile(micro, [.025, .975]))),
            "model_macro_delta": float(matrix[:, 2].sum() / matrix[:, 3].sum()),
            "model_macro_ci95": list(map(float, np.quantile(macro, [.025, .975]))),
            "transitions": transitions}
    return result
