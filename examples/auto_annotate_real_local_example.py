"""Real local demonstration: MetaboLLM planning + verified full-catalog RRF.

Requirements:
  - CUDA free enough for MetaboLLM (serial GPU stages)
  - HF cache with pinned MetaboLLM baseline
  - Verified KEGG catalog + selected encoder (CPU retrieval)

Does not import from tests/. Does not inject SyntheticTinyRetriever.
Structural fallback is disabled unless --allow-structural-fallback is set.

  python examples/auto_annotate_real_local_example.py --out _auto_annotate_demo_real_local
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

if "core" not in sys.modules:
    _core = types.ModuleType("core")
    _core.__path__ = [str(ROOT / "core")]
    sys.modules["core"] = _core

from core.agent import ExecutionLimits, auto_annotate_model
from core.agent.species import make_real_species_annotator


def _default_encoder() -> Path:
    # Prefer the verified encoder installed beside the rrf_grounded_api worktree.
    candidates = [
        ROOT.parent / "rrf_grounded_api" / "_rrf_grounded_work" / "encoder",
        ROOT / "_rrf_grounded_work" / "encoder",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "verified encoder directory not found; expected "
        "../rrf_grounded_api/_rrf_grounded_work/encoder"
    )


def build_reaction_focused_model(path: Path) -> Path:
    """Small SBML with ChEBI already on participants (species tools skip under add_missing)."""
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("agent_real_local_rxn")
    cell = model.createCompartment()
    cell.setId("cell")
    cell.setConstant(True)
    for sid, name, chebi in [
        ("a", "ATP", "CHEBI:15422"),
        ("b", "ADP", "CHEBI:16761"),
    ]:
        s = model.createSpecies()
        s.setId(sid)
        s.setName(name)
        s.setCompartment("cell")
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setConstant(False)
        s.setMetaId("meta_" + sid)
        cv = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
        cv.setBiologicalQualifierType(libsbml.BQB_IS)
        cv.addResource(f"https://identifiers.org/chebi/{chebi}")
        assert s.addCVTerm(cv) == libsbml.LIBSBML_OPERATION_SUCCESS
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
    parser.add_argument("--out", default="_auto_annotate_demo_real_local")
    parser.add_argument("--catalog", default="data/kegg/kegg_reaction_features.lzma")
    parser.add_argument("--encoder", default=None)
    parser.add_argument(
        "--hf-cache",
        default=None,
        help=(
            "Local MetaboLLM HF cache (authoritative). "
            "Else AAAIM_METABOLM_HF_CACHE, else documented discovered default."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--planner-max-new-tokens", type=int, default=768)
    parser.add_argument("--max-actions-per-turn", type=int, default=2)
    parser.add_argument(
        "--allow-structural-fallback",
        action="store_true",
        help="Explicit labeled fallback workflow orchestration only",
    )
    args = parser.parse_args()

    catalog = Path(args.catalog)
    if not catalog.is_file():
        catalog = ROOT / args.catalog
    if not catalog.is_file():
        raise FileNotFoundError(f"catalog not found: {args.catalog}")
    encoder = Path(args.encoder) if args.encoder else _default_encoder()

    from core.agent.backends.metabolm import MetaboLLMBackend, ensure_gpu_idle_enough
    from core.reaction.retrieval import RetrievalResources

    print(json.dumps({"gpu_check": ensure_gpu_idle_enough()}, indent=2))
    backend = MetaboLLMBackend(
        hf_cache=args.hf_cache,
        planner_max_new_tokens=args.planner_max_new_tokens,
    )
    resources = RetrievalResources(
        catalog_path=str(catalog),
        encoder_dir=str(encoder),
        device=args.device,
        batch_size=1,
    )
    model = build_reaction_focused_model(Path(args.out) / "demo_model.xml")
    result = auto_annotate_model(
        str(model),
        request="Annotate missing reaction annotations using RRF grounded tools",
        backend=backend,
        resources=resources,
        reaction_retriever=None,
        species_annotator=make_real_species_annotator(backend),
        output_dir=args.out,
        limits=ExecutionLimits(
            max_actions=20,
            max_planning_turns=8,
            max_failure_revisions=2,
            max_actions_per_turn=args.max_actions_per_turn,
            planner_max_new_tokens=args.planner_max_new_tokens,
            allow_names_only_reactions=True,
            max_runtime_sec=1800.0,
        ),
        allow_structural_fallback=bool(args.allow_structural_fallback),
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2))
    print(
        "execution_mode:",
        result.provenance.get("execution_mode"),
        "llm_planning_succeeded:",
        result.provenance.get("llm_planning_succeeded"),
    )
    print("planner_meta turns:", len(result.provenance.get("planner_meta") or []))
    print("output_dir:", result.output_dir)
    print(
        "NOTE: schema/semantic validity does not certify biological accuracy; "
        "verify annotations only against retrieved evidence IDs."
    )
    return 0 if result.provenance.get("llm_planning_succeeded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
