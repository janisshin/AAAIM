"""CLI for the automatic model-annotation agent.

Example:
  python -m core.agent.cli model.xml --out _agent_out --backend mock
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
    parser.add_argument("--encoder", help="Selected RRF encoder directory")
    parser.add_argument("--catalog", default="data/kegg/kegg_reaction_features.lzma")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--scope", choices=["add_missing", "replace_existing"], default="add_missing")
    args = parser.parse_args(argv)

    if args.backend == "mock":
        backend = MockBackend()
    else:
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

        ensure_gpu_idle_enough()
        backend = MetaboLLMBackend()

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

    result = auto_annotate_model(
        args.model,
        request=args.request,
        backend=backend,
        resources=resources,
        reaction_retriever=retriever,
        output_dir=args.out,
        scope=AnnotationScope(args.scope),
        limits=ExecutionLimits(max_actions=args.max_actions),
        resume_from=args.resume_from,
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    return 0 if result.report.status in {"completed", "partial", "interrupted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
