"""Runnable example for the automatic model-annotation agent.

Uses the deterministic MockBackend by default (no network, no GPU).

  python examples/auto_annotate_example.py
  python examples/auto_annotate_example.py --backend metabolm
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import libsbml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid importing the full core package __init__ (ChromaDB/RAG stack) so the
# local MetaboLLM environment can run the agent without those dependencies.
if "core" not in sys.modules:
    _core = types.ModuleType("core")
    _core.__path__ = [str(ROOT / "core")]
    sys.modules["core"] = _core

from core.agent import ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.species import make_real_species_annotator
from tests.agent.conftest import TinyRetriever


def build_demo_model(path: Path) -> Path:
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("agent_demo_mixed")
    cell = model.createCompartment()
    cell.setId("cell")
    cell.setConstant(True)
    for sid, name in [("a", "ATP"), ("b", "ADP"), ("amb", "unknown ambiguous molecule")]:
        s = model.createSpecies()
        s.setId(sid)
        s.setName(name)
        s.setCompartment("cell")
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setConstant(False)
        s.setMetaId("meta_" + sid)
    r = model.createReaction()
    r.setId("r1")
    r.setName("ATP hydrolysis")
    r.setReversible(False)
    r.setFast(False)
    for sid, stoich, create in [("a", 1, r.createReactant), ("b", 1, r.createProduct)]:
        ref = create()
        ref.setSpecies(sid)
        ref.setStoichiometry(stoich)
        ref.setConstant(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(doc, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "metabolm"], default="mock")
    parser.add_argument("--out", default="_auto_annotate_demo")
    args = parser.parse_args()

    model = build_demo_model(Path(args.out) / "demo_model.xml")
    species_annotator = None
    if args.backend == "mock":
        evidence = TinyRetriever().retrieve("ATP")
        backend = MockBackend(
            grounded_responses={
                "default": {
                    "abstain": False,
                    "predicted_kegg_id": evidence[0]["kegg_id"],
                    "selected_evidence_rank": 1,
                    "supporting_evidence_ids": ["E01"],
                    "confidence": 0.55,
                    "reasoning_summary": "Demo mock selection from synthetic evidence",
                    "abstention_reason": None,
                }
            }
        )
        # MockBackend explicitly selects the synthetic species annotator (tests/demos only).
    else:
        from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough

        print(json.dumps({"gpu_check": ensure_gpu_idle_enough()}, indent=2))
        backend = MetaboLLMBackend()
        species_annotator = make_real_species_annotator(backend)

    result = auto_annotate_model(
        str(model),
        request="Annotate missing entities in this model",
        backend=backend,
        reaction_retriever=TinyRetriever(),
        output_dir=args.out,
        limits=ExecutionLimits(max_actions=30, max_revisions=3),
        species_annotator=species_annotator,
    )
    print(
        "species_annotator:",
        json.dumps(result.provenance.get("species_annotator"), indent=2),
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    print("output_dir:", result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
