from types import SimpleNamespace

from core.annotation_workflow import _aggregate_best_penalized_scores


def _rec(rid, score, reaction_type, failed_default_score=0.0):
    match_score = [] if score is None else [float(score)]
    return SimpleNamespace(
        id=rid,
        match_score=match_score,
        metadata={
            "reaction_type": reaction_type,
            "failed_default_score": float(failed_default_score),
        },
    )


def test_non_mappable_invariance_of_global_objective():
    # Base: one mappable + one failed mapping
    base = [
        _rec("R_map", 0.8, "mappable"),
        _rec("R_fail", None, "failed_mapping", failed_default_score=0.0),
    ]
    score_1 = _aggregate_best_penalized_scores(base)

    # Add non-mappable reactions; global objective must not change.
    with_non_mappable = base + [
        _rec("R_non_1", None, "non_mappable"),
        _rec("R_non_2", None, "non_mappable"),
    ]
    score_2 = _aggregate_best_penalized_scores(with_non_mappable)

    assert abs(score_1 - score_2) < 1e-8


def test_failed_mapping_contributes_exact_zero():
    recs = [
        _rec("R_map", 0.6, "mappable"),
        _rec("R_fail", None, "failed_mapping", failed_default_score=0.0),
    ]
    score = _aggregate_best_penalized_scores(recs)

    # Expected mean: (0.6 + 0.0) / 2
    assert abs(score - 0.3) < 1e-8


def test_denominator_uses_scored_reactions_only():
    # 2 mappable + 2 failed_mapping + 1 non_mappable
    recs = [
        _rec("R_m1", 0.8, "mappable"),
        _rec("R_m2", 0.4, "mappable"),
        _rec("R_f1", None, "failed_mapping", failed_default_score=0.0),
        _rec("R_f2", None, "failed_mapping", failed_default_score=0.0),
        _rec("R_nm", None, "non_mappable"),
    ]

    score = _aggregate_best_penalized_scores(recs)

    # scored length should be n_mappable + n_failed = 4
    expected = (0.8 + 0.4 + 0.0 + 0.0) / 4.0
    assert abs(score - expected) < 1e-8

