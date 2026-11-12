"""CLI for the automatic model-annotation agent.

Examples:
  python -m core.agent.cli model.xml --out _agent_out --backend mock
  python -m core.agent.cli model.xml --planner openai --inference metabolm \\
      --encoder PATH --catalog data/kegg/kegg_reaction_features.lzma \\
      --planner-spend-authorized --planner-spend-cap-usd 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Allow running from repository root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent import AnnotationScope, ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend


def _build_roles(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Return (backend, planner, inference) respecting precedence rules."""
    has_legacy = args.backend is not None
    has_roles = args.planner is not None or args.inference is not None
    if has_legacy and has_roles:
        raise SystemExit(
            "conflicting arguments: use either --backend OR --planner/--inference, not both"
        )

    if has_legacy:
        if args.backend == "mock":
            backend = MockBackend()
            return backend, None, None
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

        ensure_gpu_idle_enough()
        backend = MetaboLLMBackend(
            hf_cache=args.hf_cache,
            planner_max_new_tokens=args.planner_max_new_tokens,
        )
        return backend, None, None

    planner_name = args.planner or "openai"
    inference_name = args.inference or "metabolm"
    if planner_name == "mock" and inference_name == "mock":
        backend = MockBackend()
        return backend, None, None

    planner: Any
    inference: Any
    if planner_name == "openai":
        from core.agent.backends.openai_planner import OpenAIPlannerBackend

        planner = OpenAIPlannerBackend(
            authorized_annotation_models=["metabolm-local"],
        )
    elif planner_name == "metabolm":
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

        ensure_gpu_idle_enough()
        planner = MetaboLLMBackend(
            hf_cache=args.hf_cache,
            planner_max_new_tokens=args.planner_max_new_tokens,
        )
    elif planner_name == "mock":
        planner = MockBackend()
    else:
        raise SystemExit(f"unsupported --planner {planner_name!r}")

    if inference_name == "metabolm":
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

        if planner_name != "metabolm":
            ensure_gpu_idle_enough()
            inference = MetaboLLMBackend(
                hf_cache=args.hf_cache,
                planner_max_new_tokens=args.planner_max_new_tokens,
            )
        else:
            # Same MetaboLLM instance for both roles when explicitly requested.
            inference = planner
    elif inference_name == "mock":
        inference = MockBackend()
    else:
        raise SystemExit(f"unsupported --inference {inference_name!r}")

    return None, planner, inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to SBML model")
    parser.add_argument("--request", default="Annotate missing entities in this model")
    parser.add_argument("--out", default="_auto_annotate_out")
    parser.add_argument(
        "--backend",
        choices=["mock", "metabolm"],
        default=None,
        help="Legacy single-backend shorthand (conflicts with --planner/--inference)",
    )
    parser.add_argument(
        "--planner",
        choices=["openai", "metabolm", "mock"],
        default=None,
        help="Orchestration backend (default openai when role flags used)",
    )
    parser.add_argument(
        "--inference",
        choices=["metabolm", "mock"],
        default=None,
        help="Biological annotation backend (default metabolm when role flags used)",
    )
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
        "--planner-spend-authorized",
        action="store_true",
        help="Explicitly authorize paid OpenAI planner dispatches (default: false)",
    )
    parser.add_argument(
        "--planner-spend-cap-usd",
        type=float,
        default=0.0,
        help="Hard USD cap for planner-only spend (default 0 = zero paid dispatches)",
    )
    parser.add_argument(
        "--allow-structural-fallback",
        action="store_true",
        help="Explicit labeled fallback workflow orchestration (off by default for real planners)",
    )
    parser.add_argument("--scope", choices=["add_missing", "replace_existing"], default="add_missing")
    args = parser.parse_args(argv)

    # If neither legacy nor role flags given, default to offline mock.
    if args.backend is None and args.planner is None and args.inference is None:
        args.backend = "mock"

    backend, planner, inference = _build_roles(args)

    species_annotator = None
    if inference is not None and getattr(inference, "name", "") == "metabolm_baseline":
        from core.agent.species import make_real_species_annotator

        species_annotator = make_real_species_annotator(inference)
    elif backend is not None and getattr(backend, "name", "") == "metabolm_baseline":
        from core.agent.species import make_real_species_annotator

        species_annotator = make_real_species_annotator(backend)

    if args.backend == "mock" or (
        getattr(planner, "name", None) == "mock" and getattr(inference, "name", None) == "mock"
    ):
        allow_fallback = True
    else:
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
    elif args.backend == "metabolm" or (
        args.planner in {"openai", "metabolm"} or args.inference == "metabolm"
    ):
        print(
            "warning: real backends without --encoder will fail reaction retrieval "
            "(no TinyRetriever injection)",
            file=sys.stderr,
        )

    result = auto_annotate_model(
        args.model,
        request=args.request,
        backend=backend,
        planner=planner,
        inference=inference,
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
            planner_spend_authorized=bool(args.planner_spend_authorized),
            planner_spend_cap_usd=float(args.planner_spend_cap_usd),
            authorized_annotation_models=["metabolm-local"],
        ),
        resume_from=args.resume_from,
        species_annotator=species_annotator,
        allow_structural_fallback=allow_fallback,
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    if result.provenance.get("planner_usage"):
        print("planner_usage:", json.dumps(result.provenance["planner_usage"], indent=2))
    return 0 if result.report.status in {"completed", "partial", "interrupted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
