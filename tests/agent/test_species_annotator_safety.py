"""Regression: real species path must not use synthetic ID fallbacks."""
from __future__ import annotations

import json
from pathlib import Path

import libsbml
import pandas as pd
import pytest

from core.agent import auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.inspect_model import inspect_sbml_model
from core.agent.schemas import FailureKind
from core.agent.species import (
    MOCK_SYNTHETIC_ANNOTATIONS,
    make_real_species_annotator,
    mock_species_annotator,
    resolve_species_annotator,
    species_database_available,
)
from core.agent.tools import AgentResources, ToolRegistry
from core.agent.schemas import AgentAction
from core.update_model import update_annotation


def _chem_model(tmp_path: Path) -> Path:
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("species_safety")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    for sid, name in [("water", "water"), ("atp", "ATP"), ("glucose", "D-glucose")]:
        s = model.createSpecies()
        s.setId(sid)
        s.setName(name)
        s.setCompartment("cell")
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setConstant(False)
        s.setMetaId("meta_" + sid)
    path = tmp_path / "species_safety.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return path


class _NameSynonymLLM:
    """Deterministic local LLM stub: echo display names as synonyms for direct search."""

    name = "name_synonym_stub"

    def identity(self):
        return {"backend": self.name, "synthetic": False}

    def generate(self, messages):
        # Produce a minimal JSON map of species_id -> synonyms from the user prompt text.
        blob = "\n".join(str(m.get("content") or "") for m in messages)
        # Fall back to common chemical synonyms present in the prompt.
        synonyms = {}
        for sid, name in [("water", "water"), ("atp", "ATP"), ("glucose", "D-glucose")]:
            if sid in blob or name in blob:
                synonyms[sid] = [name]
        # Many AAAIM prompts expect a specific structure; return a loose JSON object.
        payload = {
            "synonyms": synonyms,
            "entities": [
                {"id": sid, "type": "chemical", "synonyms": [name]}
                for sid, name in [("water", "water"), ("atp", "ATP"), ("glucose", "D-glucose")]
                if sid in blob or name in blob
            ],
        }
        return {
            "terminal_status": "generated",
            "raw_text": json.dumps(payload),
            "truncated": False,
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "error": None,
        }

    def unload(self):
        return None

    def grounded_transport(self, provider_payload):
        raise RuntimeError("not used in species safety tests")


def test_resolve_never_implicit_synthetic_for_real_backend():
    class Realish:
        name = "metabolm_baseline"

        def generate(self, messages):
            raise RuntimeError("should not be called in resolve")

    annotator, prov = resolve_species_annotator(Realish(), None)
    assert prov["kind"] == "real_pipeline"
    assert annotator is not mock_species_annotator
    assert getattr(annotator, "annotator_kind", None) == "real_pipeline"


def test_tool_registry_refuses_missing_annotator_without_synthetic(tmp_path, chemical_only_model):
    inv = inspect_sbml_model(str(chemical_only_model))
    resources = AgentResources(
        model_file=str(chemical_only_model),
        output_dir=tmp_path,
        inventory=inv,
        species_annotator=None,
        species_annotator_provenance={"kind": "none"},
    )
    registry = ToolRegistry(resources)
    obs = registry.execute(
        AgentAction(
            action_id="a1",
            tool="annotate_species_candidates",
            target_ids=["water"],
            arguments={"databases": ["chebi"]},
        )
    )
    assert obs.ok is False
    assert obs.failure_kind == FailureKind.TECHNICAL_FAILURE
    assert "synthetic" in (obs.error or "").lower() or "not configured" in (obs.error or "").lower()
    assert MOCK_SYNTHETIC_ANNOTATIONS["chebi"] not in (obs.error or "")


def test_mock_annotator_is_explicit_in_provenance(chemical_only_model, tmp_path):
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=tmp_path / "mock_prov",
    )
    assert result.provenance["species_annotator"]["kind"] == "mock_synthetic"
    assert result.provenance["species_annotator"]["explicit"] is True
    assert result.provenance["backend"]["backend"] == "mock"


def test_missing_resource_is_technical_failure_not_supported(tmp_path, monkeypatch):
    model = _chem_model(tmp_path)
    inv = inspect_sbml_model(str(model))

    def _missing(database="chebi"):
        return False, "/missing/cleannames2chebi.lzma"

    monkeypatch.setattr(
        "core.agent.species.species_database_available", _missing
    )
    backend = _NameSynonymLLM()
    annotator = make_real_species_annotator(backend, predict_qualifiers=False, validation=False)
    with pytest.raises(RuntimeError, match="resource unavailable"):
        annotator(str(model), ["water"], ["chebi"], inv)

    resources = AgentResources(
        model_file=str(model),
        output_dir=tmp_path / "out",
        inventory=inv,
        species_annotator=annotator,
        species_backend=backend,
        species_annotator_provenance={"kind": "real_pipeline"},
    )
    obs = ToolRegistry(resources).execute(
        AgentAction(
            action_id="a1",
            tool="annotate_species_candidates",
            target_ids=["water"],
            arguments={"databases": ["chebi"]},
        )
    )
    assert obs.ok is False
    assert obs.failure_kind == FailureKind.TECHNICAL_FAILURE
    assert obs.failure_kind != FailureKind.BIOLOGICAL_ABSTENTION
    recs = obs.result.get("recommendations") or []
    assert not any(str(r.get("identity_status", "")).lower() == "supported" for r in recs)


def test_unresolved_recommendations_are_not_written(tmp_path):
    model = _chem_model(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "id": "water",
                "display_name": "water",
                "annotation": "CHEBI:15377",
                "update_annotation": "add",
                "identity_status": "unresolved",
                "qualifier": "is",
                "Note": "unresolved identity",
            },
            {
                "id": "atp",
                "display_name": "ATP",
                "annotation": "CHEBI:15422",
                "update_annotation": "add",
                "identity_status": "supported",
                "qualifier": "is",
                "qualifier_status": "predicted",
            },
            {
                "id": "glucose",
                "display_name": "D-glucose",
                "annotation": "CHEBI:17634",
                "update_annotation": "add",
                "identity_status": "supported",
                "qualifier_status": "abstain",
                "qualifier": "is",
            },
        ]
    )
    out = tmp_path / "written.xml"
    update_annotation(str(model), recommendation_table=frame, new_model_path=str(out), add_all=True)
    doc = libsbml.SBMLReader().readSBML(str(out))
    m = doc.getModel()
    water_ann = m.getSpecies("water").getAnnotationString() if m.getSpecies("water").isSetAnnotation() else ""
    atp_ann = m.getSpecies("atp").getAnnotationString() if m.getSpecies("atp").isSetAnnotation() else ""
    glucose_ann = (
        m.getSpecies("glucose").getAnnotationString() if m.getSpecies("glucose").isSetAnnotation() else ""
    )
    assert "15377" not in water_ann  # unresolved not written as URI
    assert "15422" in atp_ann
    assert "17634" not in glucose_ann  # abstain not written
    # Source path unchanged
    assert Path(model).read_text(encoding="utf-8")


@pytest.mark.skipif(not species_database_available("chebi")[0], reason="ChEBI reference missing")
def test_direct_species_search_does_not_load_cygrpc():
    """method=direct must not import chromadb → grpc._cython.cygrpc (App Control risk)."""
    import sys

    # Drop modules if a prior test already imported them.
    for name in list(sys.modules):
        if name == "chromadb" or name.startswith("chromadb.") or "cygrpc" in name:
            sys.modules.pop(name, None)

    from core.database_search import get_species_recommendations_direct

    assert "chromadb" not in sys.modules
    assert "grpc._cython.cygrpc" not in sys.modules
    recs = get_species_recommendations_direct(
        ["water", "atp"], {"water": ["water"], "atp": ["ATP"]}, database="chebi", top_k=3
    )
    assert recs
    assert "chromadb" not in sys.modules
    assert "grpc._cython.cygrpc" not in sys.modules
    assert any(getattr(r, "candidates", None) for r in recs)


@pytest.mark.skipif(not species_database_available("chebi")[0], reason="ChEBI reference missing")
def test_real_direct_search_is_evidence_dependent(tmp_path, monkeypatch):
    """Different species names must not collapse to one hard-coded CHEBI ID."""
    from core.database_search import get_species_recommendations_direct

    model = _chem_model(tmp_path)
    inv = inspect_sbml_model(str(model))
    synonyms = {
        "water": ["water"],
        "atp": ["ATP"],
        "glucose": ["D-glucose"],
    }
    recs = get_species_recommendations_direct(
        ["water", "atp", "glucose"], synonyms, database="chebi", top_k=3
    )
    assert isinstance(recs, list) and recs
    by_id: dict[str, set[str]] = {}
    for rec in recs:
        sid = str(getattr(rec, "id", "") or getattr(rec, "species_id", ""))
        cands = getattr(rec, "candidates", None) or getattr(rec, "annotations", None) or []
        ids = set()
        for cand in cands:
            if isinstance(cand, dict):
                ids.add(str(cand.get("annotation") or cand.get("id") or ""))
            else:
                ids.add(str(getattr(cand, "annotation", None) or getattr(cand, "id", cand) or ""))
        if not sid and hasattr(rec, "to_dict"):
            payload = rec.to_dict()
            sid = str(payload.get("id") or "")
            ids.add(str(payload.get("annotation") or ""))
        by_id.setdefault(sid or "unknown", set()).update({i for i in ids if i})
    synthetic = MOCK_SYNTHETIC_ANNOTATIONS["chebi"]
    nonempty = {ann for anns in by_id.values() for ann in anns if ann}
    assert nonempty, "expected ChEBI direct hits for water/ATP/glucose synonyms"
    assert len(nonempty) >= 2
    assert not all(anns == {synthetic} for anns in by_id.values() if anns)

    # Real annotator without LLM backend must fail closed (no synthetic fill-in)
    with pytest.raises(RuntimeError, match="local LLM backend"):
        make_real_species_annotator(None)(str(model), ["water"], ["chebi"], inv)


def test_auto_annotate_with_real_backend_does_not_use_mock_ids_on_failure(tmp_path):
    class BrokenReal:
        name = "metabolm_baseline"

        def identity(self):
            return {"backend": "metabolm_baseline"}

        def generate(self, messages):
            return {
                "terminal_status": "context_overflow",
                "raw_text": None,
                "truncated": False,
                "error": "refused",
                "usage": {},
            }

        def unload(self):
            return None

        def grounded_transport(self, body):
            return {"status": "incomplete", "model": "metabolm-local", "output": [], "error": "n/a"}

        def plan(self, **kwargs):
            from core.agent.backends.mock import MockBackend

            return MockBackend()._default_plan(kwargs["inventory"])

    model = _chem_model(tmp_path)
    result = auto_annotate_model(
        str(model),
        backend=BrokenReal(),
        output_dir=tmp_path / "broken_real",
        reaction_retriever=None,
        allow_structural_fallback=True,
    )
    assert result.provenance["species_annotator"]["kind"] == "real_pipeline"
    # Recommendations must not be the hard-coded synthetic supported IDs
    species_path = Path(result.output_dir) / "species_recommendations.csv"
    if species_path.is_file():
        df = pd.read_csv(species_path)
        if not df.empty and "annotation" in df.columns:
            assert not (
                df["annotation"].astype(str) == MOCK_SYNTHETIC_ANNOTATIONS["chebi"]
            ).all()
    # Any completed species annotations from the default mock path are forbidden
    for obs in result.observations:
        if obs.tool == "annotate_species_candidates" and obs.ok:
            recs = obs.result.get("recommendations") or []
            kinds = {r.get("annotator_kind") for r in recs}
            assert "mock_synthetic" not in kinds
