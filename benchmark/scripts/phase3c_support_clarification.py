"""Clarify frozen Phase 3C support counts without redefining historical metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.scripts.phase5_robustness import ROOT, read_rows, write


def support_counts(rows):
    n = len(rows)
    unretrieved = [r for r in rows if r["phase3a_unsupported_in_catalog"]]
    return {"n_validation": n,
            "direct_in_catalog_without_retrieved_evidence": len(unretrieved),
            "direct_reference_correct_without_retrieved_evidence": sum(bool(r["phase3a_target_only_exact"]) for r in unretrieved),
            "direct_reference_incorrect_without_retrieved_evidence": sum(not r["phase3a_target_only_exact"] for r in unretrieved),
            "direct_exact_correct_all": sum(bool(r["phase3a_target_only_exact"]) for r in rows),
            "grounded_answered": sum(bool(r["grounded_answered"]) for r in rows),
            "grounded_abstained": sum(bool(r["grounded_abstain"]) for r in rows),
            "grounded_out_of_supplied_evidence": sum(bool(r["grounded_unsupported"]) for r in rows),
            "grounded_in_evidence_reference_incorrect": sum(bool(r["grounded_answered"]) and not r["grounded_unsupported"] and not r["grounded_exact"] for r in rows),
            "grounded_full_evidence_compliant": sum(bool(r["grounded_evidence_compliant"]) for r in rows),
            "grounded_exact_correct_all": sum(bool(r["grounded_exact"]) for r in rows),
            "fusion_exact_correct_all": sum(bool(r["fusion_top1_exact"]) for r in rows),
            "support_outcomes_have_same_definition": False,
            "support_interval_establishes_reduced_biological_hallucination": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/phase5/review_fixes/phase3c_support_counts.json")
    args = parser.parse_args(argv)
    rows = read_rows(args.source_root / "benchmark/phase3/phase3c_validation/answer_key_join.jsonl")
    if len(rows) != 163 or len({r["sample_id"] for r in rows}) != 163:
        raise ValueError("exact frozen 163-row validation join required")
    result = support_counts(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
