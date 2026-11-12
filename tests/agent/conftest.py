"""Shared fixtures for automatic annotation agent tests."""
from __future__ import annotations

import libsbml
import pytest


def _write_doc(doc, path):
    assert libsbml.writeSBMLToFile(doc, str(path))
    return path


def _species(model, sid, name, compartment="cell"):
    s = model.createSpecies()
    s.setId(sid)
    s.setName(name)
    s.setCompartment(compartment)
    s.setBoundaryCondition(False)
    s.setHasOnlySubstanceUnits(False)
    s.setConstant(False)
    s.setMetaId("meta_" + sid)
    return s


def _reaction(model, rid, name, reactants, products):
    r = model.createReaction()
    r.setId(rid)
    r.setName(name)
    r.setReversible(False)
    r.setFast(False)
    for sid, stoich in reactants:
        ref = r.createReactant()
        ref.setSpecies(sid)
        ref.setStoichiometry(stoich)
        ref.setConstant(True)
    for sid, stoich in products:
        ref = r.createProduct()
        ref.setSpecies(sid)
        ref.setStoichiometry(stoich)
        ref.setConstant(True)
    return r


def _add_chebi(species, chebi_id):
    species.setMetaId(species.getMetaId() or "meta_" + species.getId())
    uri = f"https://identifiers.org/chebi/{chebi_id}"
    cv = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
    cv.setBiologicalQualifierType(libsbml.BQB_IS)
    cv.addResource(uri)
    assert species.addCVTerm(cv) == libsbml.LIBSBML_OPERATION_SUCCESS


@pytest.fixture
def chemical_only_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("chemical_only")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "water", "water")
    _species(model, "atp", "ATP")
    return _write_doc(doc, tmp_path / "chemical_only.xml")


@pytest.fixture
def reaction_only_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("reaction_only")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "a", "ATP")
    _species(model, "b", "ADP")
    _add_chebi(model.getSpecies("a"), "CHEBI:15422")
    _add_chebi(model.getSpecies("b"), "CHEBI:16761")
    _reaction(model, "r1", "ATP hydrolysis", [("a", 1)], [("b", 1)])
    return _write_doc(doc, tmp_path / "reaction_only.xml")


@pytest.fixture
def mixed_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("mixed")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "a", "ATP")
    _species(model, "b", "ADP")
    _species(model, "amb", "unknown ambiguous molecule")
    _species(model, "prot", "hexokinase protein")
    _reaction(model, "r1", "ATP to ADP", [("a", 1)], [("b", 1)])
    return _write_doc(doc, tmp_path / "mixed.xml")


@pytest.fixture
def partial_annotation_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("partial")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "a", "ATP")
    _species(model, "b", "ADP")
    _add_chebi(model.getSpecies("a"), "CHEBI:15422")
    _reaction(model, "r1", "rxn", [("a", 1)], [("b", 1)])
    return _write_doc(doc, tmp_path / "partial.xml")


@pytest.fixture
def complex_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("complex_model")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "cplx", "pyruvate dehydrogenase complex")
    return _write_doc(doc, tmp_path / "complex.xml")


@pytest.fixture
def gene_protein_model(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("gene_protein")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    _species(model, "g1", "TP53 gene")
    _species(model, "p1", "p53 protein")
    return _write_doc(doc, tmp_path / "gene_protein.xml")


@pytest.fixture
def invalid_model(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text("<not-sbml>", encoding="utf-8")
    return path


class TinyRetriever:
    provenance = {"resource": "synthetic", "verified_selected_checkpoint": False}
    catalog = {
        f"R{i:05d}": {
            "NAME": "ATP" if i == 1 else f"name{i}",
            "DEFINITION": "ATP hydrolysis",
            "EQUATION": "ATP => ADP",
        }
        for i in range(1, 13)
    }

    def __init__(self):
        self.retrieve_calls = 0
        # Instance copy so tests can mutate identity without cross-talk.
        self.provenance = dict(type(self).provenance)

    def retrieve(self, query):
        from core.reaction import retrieval

        self.retrieve_calls += 1
        docs = [
            {"kegg_id": kid, "text": retrieval.document_text(kid, fields)}
            for kid, fields in self.catalog.items()
        ]
        return retrieval.evidence_for(
            self.catalog, retrieval.BM25(docs).rank(query), list(self.catalog)[::-1]
        )
