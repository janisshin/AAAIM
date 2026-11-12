"""Deterministic SBML inspection for the automatic annotation agent."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import libsbml

from core.agent.schemas import (
    AnnotationStatus,
    BiologicalType,
    EntityRecord,
    EntityRepresentation,
    ExistingAnnotation,
    ModelInventory,
)
from core.model_info import detect_model_format
from utils.constants import (
    CHEBI_URI_PATTERNS,
    KEGG_REACTION_URI_PATTERNS,
    NCBIGENE_URI_PATTERNS,
    UNIPROT_URI_PATTERNS,
    ModelType,
)

_DB_PATTERNS = {
    "chebi": CHEBI_URI_PATTERNS,
    "ncbigene": NCBIGENE_URI_PATTERNS,
    "uniprot": UNIPROT_URI_PATTERNS,
    "kegg.reaction": KEGG_REACTION_URI_PATTERNS,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_annotations(element) -> list[ExistingAnnotation]:
    if element is None or not element.isSetAnnotation():
        return []
    text = element.getAnnotationString()
    found: list[ExistingAnnotation] = []
    for qualifier in re.findall(
        r"<bqbiol:([A-Za-z]+)[^>]*>(.*?)</bqbiol:\1>", text, flags=re.DOTALL
    ):
        qname, block = qualifier
        for uri in re.findall(r'rdf:resource="([^"]+)"', block):
            database = "unknown"
            identifier = uri.rsplit("/", 1)[-1]
            for db, patterns in _DB_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, uri):
                        database = db
                        m = re.search(pattern, uri)
                        if m:
                            identifier = m.group(1) if m.lastindex else identifier
                        break
            found.append(
                ExistingAnnotation(
                    database=database, identifier=str(identifier), qualifier=qname, uri=uri
                )
            )
    return found


def _status_for(annotations: list[ExistingAnnotation], *, supported: bool) -> AnnotationStatus:
    if not supported:
        return AnnotationStatus.UNSUPPORTED
    if not annotations:
        return AnnotationStatus.MISSING
    dbs = {a.database for a in annotations}
    ids = {(a.database, a.identifier) for a in annotations}
    if len(ids) > 1 and len({a.database for a in annotations if a.database != "unknown"}) <= 1:
        # Multiple distinct IDs for same/unknown DB → conflicting
        by_db: dict[str, set[str]] = {}
        for a in annotations:
            by_db.setdefault(a.database, set()).add(a.identifier)
        if any(len(v) > 1 for v in by_db.values()):
            return AnnotationStatus.CONFLICTING
    if "unknown" in dbs and len(dbs) == 1:
        return AnnotationStatus.PARTIAL
    return AnnotationStatus.COMPLETE


def _guess_species_type(name: str, annotations: list[ExistingAnnotation]) -> BiologicalType:
    dbs = {a.database for a in annotations}
    if "chebi" in dbs:
        return BiologicalType.CHEMICAL
    if "uniprot" in dbs:
        return BiologicalType.PROTEIN
    if "ncbigene" in dbs:
        return BiologicalType.GENE
    lowered = (name or "").lower()
    if any(tok in lowered for tok in ("complex", "assembly")):
        return BiologicalType.COMPLEX
    if any(tok in lowered for tok in ("gene", "mrna", "dna")):
        return BiologicalType.GENE
    if any(tok in lowered for tok in ("protein", "enzyme", "kinase", "ase")):
        return BiologicalType.PROTEIN
    if annotations:
        return BiologicalType.AMBIGUOUS
    # Name-only metabolites are treated as chemical candidates, not whole-file labels.
    if lowered and not any(tok in lowered for tok in ("unknown", "ambiguous", "unclear")):
        return BiologicalType.CHEMICAL
    return BiologicalType.AMBIGUOUS


def inspect_sbml_model(model_file: str, *, scope_add_missing: bool = True) -> ModelInventory:
    """Parse and inventory entities without classifying the whole file as one type."""
    path = str(Path(model_file))
    if not Path(path).is_file():
        return ModelInventory(
            model_file=path,
            parse_ok=False,
            parse_error=f"file not found: {path}",
            gaps=["SBML file missing"],
        )
    reader = libsbml.SBMLReader()
    document = reader.readSBML(path)
    if document.getNumErrors() > 0 and document.getModel() is None:
        err = document.getError(0).getMessage() if document.getNumErrors() else "unreadable SBML"
        return ModelInventory(
            model_file=path,
            parse_ok=False,
            parse_error=err,
            gaps=["SBML parse failure"],
            sha256=_sha256(path),
        )
    model = document.getModel()
    if model is None:
        return ModelInventory(
            model_file=path,
            parse_ok=False,
            parse_error="SBML document has no model",
            gaps=["SBML parse failure"],
            sha256=_sha256(path),
        )

    model_type, format_info = detect_model_format(path)
    entities: list[EntityRecord] = []
    unsupported: list[EntityRecord] = []
    gaps: list[str] = []
    participant_map: dict[str, list[str]] = {}

    # Reactions (SBML reaction elements only)
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        rid = reaction.getId()
        parts: list[str] = []
        for j in range(reaction.getNumReactants()):
            sid = reaction.getReactant(j).getSpecies()
            parts.append(sid)
            participant_map.setdefault(sid, []).append(rid)
        for j in range(reaction.getNumProducts()):
            sid = reaction.getProduct(j).getSpecies()
            parts.append(sid)
            participant_map.setdefault(sid, []).append(rid)
        for j in range(reaction.getNumModifiers()):
            sid = reaction.getModifier(j).getSpecies()
            parts.append(sid)
            participant_map.setdefault(sid, []).append(rid)
        anns = _extract_annotations(reaction)
        entities.append(
            EntityRecord(
                target_id=rid,
                representation=EntityRepresentation.SBML_REACTION,
                biological_type=BiologicalType.REACTION,
                display_name=reaction.getName() or rid,
                annotation_status=_status_for(anns, supported=True),
                existing_annotations=anns,
                participants=parts,
            )
        )

    # Ordinary species
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        sid = species.getId()
        anns = _extract_annotations(species)
        btype = _guess_species_type(species.getName() or sid, anns)
        entities.append(
            EntityRecord(
                target_id=sid,
                representation=EntityRepresentation.SBML_SPECIES,
                biological_type=btype,
                display_name=species.getName() or sid,
                annotation_status=_status_for(anns, supported=True),
                existing_annotations=anns,
                participant_of=sorted(set(participant_map.get(sid, []))),
            )
        )

    # FBC gene products — supported for species-side gene/protein writing
    fbc = model.getPlugin("fbc")
    if fbc is not None and hasattr(fbc, "getNumGeneProducts"):
        for i in range(fbc.getNumGeneProducts()):
            gp = fbc.getGeneProduct(i)
            gid = gp.getId()
            anns = _extract_annotations(gp)
            btype = BiologicalType.GENE
            if any(a.database == "uniprot" for a in anns):
                btype = BiologicalType.PROTEIN
            entities.append(
                EntityRecord(
                    target_id=gid,
                    representation=EntityRepresentation.FBC_GENE_PRODUCT,
                    biological_type=btype,
                    display_name=gp.getName() or gid,
                    annotation_status=_status_for(anns, supported=True),
                    existing_annotations=anns,
                    support_notes=["FBC gene-product read/write supported via update_annotation"],
                )
            )

    # SBML-qual transitions are not grounded reaction targets
    qual = model.getPlugin("qual")
    if qual is not None and hasattr(qual, "getNumTransitions"):
        for i in range(qual.getNumQualitativeSpecies()):
            qs = qual.getQualitativeSpecies(i)
            qid = qs.getId()
            anns = _extract_annotations(qs)
            entities.append(
                EntityRecord(
                    target_id=qid,
                    representation=EntityRepresentation.SBML_QUAL_SPECIES,
                    biological_type=_guess_species_type(qs.getName() or qid, anns),
                    display_name=qs.getName() or qid,
                    annotation_status=_status_for(anns, supported=True),
                    existing_annotations=anns,
                    support_notes=["SBML-qual qualitative species writing supported via update_annotation"],
                )
            )
        for i in range(qual.getNumTransitions()):
            tr = qual.getTransition(i)
            tid = tr.getId() or f"transition_{i}"
            rec = EntityRecord(
                target_id=tid,
                representation=EntityRepresentation.SBML_QUAL_TRANSITION,
                biological_type=BiologicalType.REACTION,
                display_name=tr.getName() or tid,
                annotation_status=AnnotationStatus.UNSUPPORTED,
                support_notes=[
                    "SBML-qual transitions are not targets of reaction_method=rrf_grounded"
                ],
            )
            unsupported.append(rec)
            entities.append(rec)
        gaps.append("SBML-qual transitions unsupported for grounded reaction annotation")

    if model_type == ModelType.SBML_FBC:
        gaps.append("FBC flux constraints are ignored; only gene-product annotations are considered")

    if scope_add_missing:
        # Preserve complete/conflicting existing annotations unless replace scope is chosen later.
        pass

    return ModelInventory(
        model_file=path,
        model_id=model.getId() or "",
        model_format=model_type.name,
        parse_ok=True,
        entities=entities,
        unsupported=unsupported,
        gaps=gaps,
        sha256=_sha256(path),
    )


def entities_needing_annotation(
    inventory: ModelInventory,
    *,
    scope_add_missing: bool = True,
) -> list[EntityRecord]:
    needed = []
    for ent in inventory.entities:
        if ent.annotation_status == AnnotationStatus.UNSUPPORTED:
            continue
        if scope_add_missing and ent.annotation_status in {
            AnnotationStatus.COMPLETE,
            AnnotationStatus.CONFLICTING,
        }:
            continue
        if ent.representation == EntityRepresentation.SBML_QUAL_TRANSITION:
            continue
        needed.append(ent)
    return needed
