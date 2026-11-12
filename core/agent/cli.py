"""CLI for the automatic model-annotation agent.

Example:
  python -m core.agent.cli model.xml --out _agent_out --backend mock
  python -m core.agent.cli model.xml --backend metabolm --encoder PATH --catalog data/kegg/kegg_reaction_features.lzma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running from repository root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent import AnnotationScope, ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to SBML model")
    parser.add_argument("--request", default="Annotate missing entities in this model")
    parser.add_argument("--out", default="_auto_annotate_out")
    parser.add_argument("--backend", choices=["mock", "metabolm"], default="mock")
    parser.add_argument("--encoder", help="Selected RRF encoder directory (required for real retrieval)")
    parser.add_argument("--catalog", default="data/kegg/kegg_reaction_features.lzma")
    parser.add_argument(
        "--hf-cache",
        default=None,
        help=(
            "Local MetaboLLM HF cache directory (authoritative). "
            "Else AAAIM_METABOLM_HF_CACHE, else documented discovered default. "
            "Invalid explicit/env paths do not fall back."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-planning-turns", type=int, default=12)
    parser.add_argument("--max-failure-revisions", type=int, default=6)
    parser.add_argument("--max-actions-per-turn", type=int, default=3)
    parser.add_argument("--planner-max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--allow-structural-fallback",
        action="store_true",
        help="Explicit labeled fallback workflow orchestration (off by default for metabolm)",
    )
    parser.add_argument("--scope", choices=["add_missing", "replace_existing"], default="add_missing")
    args = parser.parse_args(argv)

    if args.backend == "mock":
        backend = MockBackend()
        species_annotator = None
        allow_fallback = True if not args.allow_structural_fallback else True
    else:
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough
        from core.agent.species import make_real_species_annotator

        ensure_gpu_idle_enough()
        backend = MetaboLLMBackend(
            hf_cache=args.hf_cache,
            planner_max_new_tokens=args.planner_max_new_tokens,
        )
        species_annotator = make_real_species_annotator(backend)
        allow_fallback = bool(args.allow_structural_fallback)

    resources = None
    retriever = None
    if args.encoder:
        from core.reaction.retrieval import RetrievalResources

        resources = RetrievalResources(
            catalog_path=args.catalog,
            encoder_dir=args.encoder,
            device=args.device,
            batch_size=1,
        )
    elif args.backend == "metabolm":
        print(
            "warning: metabolm without --encoder will fail reaction retrieval "
            "(no TinyRetriever injection)",
            file=sys.stderr,
        )

    result = auto_annotate_model(
        args.model,
        request=args.request,
        backend=backend,
        resources=resources,
        reaction_retriever=retriever,
        output_dir=args.out,
        scope=AnnotationScope(args.scope),
        limits=ExecutionLimits(
            max_actions=args.max_actions,
            max_planning_turns=args.max_planning_turns,
            max_failure_revisions=args.max_failure_revisions,
            max_actions_per_turn=args.max_actions_per_turn,
            planner_max_new_tokens=args.planner_max_new_tokens,
        ),
        resume_from=args.resume_from,
        species_annotator=species_annotator,
        allow_structural_fallback=allow_fallback,
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    return 0 if result.report.status in {"completed", "partial", "interrupted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
