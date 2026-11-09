"""Offline paired species-evidence protocol, alignment and selected retrieval.

Only the frozen validation query whitelist defines eligible reactions. No live
provider adapter is imported or called. Labels are joined only after ranking
files have been frozen. Private study outputs belong in an ignored directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import libsbml
import pandas as pd

from core.reaction.species_evidence import build_queries, load_model, species_evidence, CONTRACT_VERSION
from core.reaction.retrieval import CATALOG_SHA256, MANIFEST_SHA256, CONFIG, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def write_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def network_identity(model):
    """Strict named-species/reaction signature for detecting possible accession aliases.

    Excludes annotations and model accession, never reference judgments. A match
    is a potential alias only; accepting a nonidentical file requires review.
    """
    species = [{"id": s.getId(), "name": s.getName(), "compartment": s.getCompartment(),
                "boundary": s.getBoundaryCondition(), "formula": s.getPlugin("fbc").getChemicalFormula()
                if s.getPlugin("fbc") else ""} for s in model.getListOfSpecies()]
    reactions = [{"id": r.getId(), "name": r.getName(), "reversible": r.getReversible(),
        "reactants": [(x.getSpecies(), x.getStoichiometry()) for x in r.getListOfReactants()],
        "products": [(x.getSpecies(), x.getStoichiometry()) for x in r.getListOfProducts()],
        "modifiers": [x.getSpecies() for x in r.getListOfModifiers()]} for r in model.getListOfReactions()]
    return digest({"species": sorted(species, key=lambda s: s["id"]),
                   "reactions": sorted(reactions, key=lambda r: r["id"])})


def complete_prediction_table(model, frame):
    """Close the table over every SBML species, so no source URI can fall back.

    Preserve absence of the status column for legacy tables. Never fill missing
    modern status values with 'supported'. Unknown species IDs fail alignment.
    """
    if not {"id", "annotation"} <= set(frame.columns):
        raise ValueError("prediction table lacks id/annotation")
    ids = {s.getId() for s in model.getListOfSpecies()}
    supplied = set(frame.id.astype(str)) - {"", "Reason:"}
    if supplied - ids:
        raise ValueError("prediction IDs not in SBML: " + str(sorted(supplied - ids)))
    clean = frame[frame.id.astype(str).isin(ids)].copy()
    absent = sorted(ids - supplied)
    return pd.concat([clean, pd.DataFrame([{"id": sid, "annotation": ""} for sid in absent])],
                     ignore_index=True).fillna("")


def condition_queries(model_path, predictions=None):
    document = load_model(model_path)
    model = document.getModel()
    empty = pd.DataFrame([{"id": s.getId(), "annotation": ""} for s in model.getListOfSpecies()],
                         columns=["id", "annotation"])
    conditions = {"reference": build_queries(model_path), "names_only": build_queries(model_path, empty)}
    if predictions is not None:
        conditions["predicted"] = build_queries(model_path, complete_prediction_table(model, predictions))
    for values in conditions.values():
        for row in values:
            other = next(r for r in conditions["reference"] if r["reaction_id"] == row["reaction_id"])
            if row["reaction_equation"] != other["reaction_equation"]:
                raise ValueError("condition changed equation")
            for sid, evidence in row["species_evidence"].items():
                if evidence["name"] != other["species_evidence"][sid]["name"]:
                    raise ValueError("condition changed species names")
            if values is conditions["names_only"] and any(e["chebi"] or e["kegg_compound"] for e in row["species_evidence"].values()):
                raise ValueError("names-only identifier leakage")
    return conditions


def alias_source_species(model_path, destination):
    """Uniform bijection in every arm; original IDs stay only in sidecars."""
    document = load_model(model_path)
    model = document.getModel()
    species = list(model.getListOfSpecies())
    mapping = {sid: f"aaaim_sp_{i:05d}" for i, sid in enumerate(sorted(s.getId() for s in species), 1)}
    if set(mapping) & set(mapping.values()):
        raise ValueError("alias namespace collides with source IDs")
    names = {s.getId(): s.getName() for s in species}
    for s in species:
        s.setId(mapping[s.getId()])
    # Update mathematical/plugin references as well as reaction participants.
    elements = [model] + list(model.getListOfAllElements())
    for old, new in mapping.items():
        for element in elements:
            element.renameSIdRefs(old, new)
    for s in model.getListOfSpecies():
        old = next(sid for sid, alias in mapping.items() if alias == s.getId())
        if s.getName() != names[old]:
            raise ValueError("aliasing changed source names")
    if not libsbml.writeSBMLToFile(document, str(destination)):
        raise OSError(destination)
    return mapping


def prepare(source_root, prediction_root, out, *, alias_ids=False):
    source_root, prediction_root, out = Path(source_root).resolve(), Path(prediction_root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    queries_path = source_root / "benchmark/phase3/retrieval_baselines/queries.jsonl"
    frozen = read_jsonl(queries_path)
    if len(frozen) != 969 or any(row.get("split") != "validation" for row in frozen):
        raise ValueError("requires exact frozen validation-only universe")
    keys = {(r["model_id"], r["reaction_id"]) for r in frozen}
    if len(keys) != 969:
        raise ValueError("duplicate validation keys")
    models = sorted({key[0] for key in keys})
    # Partition metadata only: no held-out reaction IDs, query text or labels.
    allocation_path = source_root / "benchmark/phase3/splits.csv"
    allocation = pd.read_csv(allocation_path, usecols=["model_id", "cluster_id", "split"], dtype=str).drop_duplicates()
    if allocation.groupby("model_id").split.nunique().max() != 1:
        raise ValueError("model spans partitions; cannot safely inspect full validation SBML")
    selected = allocation[allocation.model_id.isin(models)]
    if len(selected) != len(models) or not selected.split.eq("validation").all():
        raise ValueError("validation model allocation mismatch")
    cluster = dict(zip(selected.model_id, selected.cluster_id))
    train_models = set(allocation.loc[allocation.split.eq("train"), "model_id"])
    train_clusters = set(allocation.loc[allocation.split.eq("train"), "cluster_id"])
    if set(models) & train_models or set(cluster.values()) & train_clusters:
        raise ValueError("train/validation model or cluster overlap")
    input_hashes = {str(queries_path): sha256(queries_path), str(allocation_path): sha256(allocation_path)}
    model_info = {}
    species_inputs = {}
    for mid in models:
        path = source_root / "benchmark/models" / (mid + ".xml")
        document = load_model(path)
        model = document.getModel()
        reaction_ids = {r.getId() for r in model.getListOfReactions()}
        expected = {rid for m, rid in keys if m == mid}
        if not expected <= reaction_ids:
            raise ValueError("validation reaction IDs missing from source SBML")
        info = {"model_id": mid, "path": str(path), "sbml_sha256": sha256(path),
            "normalized_file_sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "network_identity": network_identity(model), "n_species": model.getNumSpecies(),
            "n_source_reactions": model.getNumReactions(), "n_validation_reactions": len(expected), "cluster_id": cluster[mid]}
        model_info[mid] = info
        input_hashes[str(path)] = info["sbml_sha256"]
        species_inputs[mid] = [{"id": s.getId(), "display_name": s.getName() or s.getId(),
                               "compartment": s.getCompartment()} for s in model.getListOfSpecies()]
    if alias_ids:
        (out / "aliased_inputs").mkdir()
        for mid, info in model_info.items():
            path = out / "aliased_inputs" / (mid + ".xml")
            info["species_aliases"] = alias_source_species(info["path"], path)
            info["query_source_path"] = str(path)
            info["query_source_sha256"] = sha256(path)
    prediction_records = []
    matched_predictions = {}
    # Prespecified single variant, selected by file/configuration, never outcomes.
    for directory in sorted((prediction_root / "results").iterdir()):
        if not directory.is_dir():
            continue
        candidate = directory / "full_source_fidelity_species.csv"
        metrics = directory / "full_source_fidelity_metrics.json"
        if not candidate.is_file():
            candidate = directory / "full_species.csv"
            metrics = directory / "full_metrics.json"
        if not candidate.is_file():
            continue
        model_path = prediction_root / (directory.name + ".xml")
        info = {"prediction_model_id": directory.name, "prediction_path": str(candidate),
                "prediction_sha256": sha256(candidate), "metrics_path": str(metrics),
                "selected_variant": candidate.name + "; priority source_fidelity then baseline; no human-final or repeat selection"}
        input_hashes[str(candidate)] = info["prediction_sha256"]
        frame = pd.read_csv(candidate, dtype=str).fillna("")
        info.update(rows=len(frame), species_ids=frame.id.nunique(),
                    identity_status_policy="modern" if "identity_status" in frame else "legacy_unassessed")
        if metrics.is_file():
            provenance = read_json(metrics)
            input_hashes[str(metrics)] = sha256(metrics)
            info["configuration"] = provenance.get("configuration")
            info["usage"] = provenance.get("metrics", {}).get("llm_usage")
        if not model_path.is_file():
            info.update(status="missing_source_sbml", compatible_validation_models=[])
        else:
            model = load_model(model_path).getModel().clone()
            info.update(source_sbml=str(model_path), source_sha256=sha256(model_path),
                network_identity=network_identity(model), source_species=model.getNumSpecies(),
                source_reactions=model.getNumReactions(),
                training_accession_overlap=directory.name in train_models)
            input_hashes[str(model_path)] = info["source_sha256"]
            ordinary = {s.getId() for s in model.getListOfSpecies()}
            qual = model.getPlugin("qual")
            qualitative = {s.getId() for s in qual.getListOfQualitativeSpecies()} if qual else set()
            unknown = sorted(set(frame.id) - ordinary - qualitative - {"", "Reason:"})
            info["unknown_species_ids"] = unknown
            info["possible_network_aliases"] = [mid for mid, ref in model_info.items() if ref["network_identity"] == info["network_identity"]]
            normalized = hashlib.sha256(model_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            exact = [mid for mid, ref in model_info.items() if normalized == ref["normalized_file_sha256"]]
            info["compatible_validation_models"] = exact if not unknown else []
            info["status"] = "compatible" if info["compatible_validation_models"] else "outside_validation_cohort_or_content_mismatch"
            for mid in info["compatible_validation_models"]:
                if not info.get("usage") or not info.get("configuration"):
                    raise ValueError("matching prediction lacks model/usage provenance")
                matched_predictions[mid] = frame
        prediction_records.append(info)
    cohort = [{"model_id": mid, "reaction_id": rid, "cluster_id": cluster[mid],
               "actual_prediction_available": mid in matched_predictions} for mid, rid in sorted(keys)]
    eligible = [row for row in cohort if row["actual_prediction_available"]]
    protocol = {"schema": "species-propagation-protocol-v1", "frozen_before_new_comparative_outcomes": True,
        "cohort_rule": "all 969 frozen validation reactions; select prediction compatibility by source identity/species alignment/provenance only",
        "conditions": {"reference": "available source SBML whole-species CVTerms under the public contract; not established biological truth",
            "predicted": "actual machine output, complete table covering every source species; absent rows blank, no SBML identifier fallback",
            "names_only": "complete blank identifier table; identical source names/equations/stoichiometry"},
        "target_n": 969, "target_models": len(models), "target_clusters": len(set(cluster.values())),
        "currently_available_ABC_n": len(eligible), "control_only_policy": "if B missing, execute reference versus names-only on the fixed target cohort; label separately from pending actual-prediction propagation",
        "no_case_selection_by_outcomes": True, "no_new_model_variants": True,
        "selected_retrieval": {**CONFIG, "catalog_sha256": CATALOG_SHA256, "encoder_manifest_sha256": MANIFEST_SHA256},
        "species_contract": CONTRACT_VERSION, "bootstrap": {"unit": "paired model cluster", "replicates": 2000, "seed": 20261109},
        "metrics": ["accepted-reference hit Recall@1/3/5/10", "MRR@10", "label recovery recall@K separately", "reaction-micro", "model-macro", "paired differences and hit transitions"],
        "final_annotation": {"model": "gpt-6.1-terra", "reasoning_effort": "low", "max_output_tokens": 2048,
            "schema": "GroundedAnnotation", "technical_failures_are_not_abstentions": True, "paid_calls_authorized": 0},
        "training_contamination": {"validation_training_accession_overlap": 0, "validation_training_cluster_overlap": 0,
            "model_selection_on_validation": True, "hosted_pretraining_contamination": "unknown; cannot exclude from local artifacts",
            "notes": "names/frozen annotations are source baselines; human audit and prediction correctness never select cohort"},
        "source_query_whitelist_sha256": input_hashes[str(queries_path)], "cohort_digest": digest(cohort),
        "alignment_digest": digest(prediction_records)}
    protocol["prediction_file_policy"] = "source_fidelity machine export preferred, otherwise existing full baseline; no outcome-based variant/repeat selection"
    protocol["uniform_species_id_aliases"] = alias_ids
    protocol["source_id_policy"] = ("species IDs are mapped to nonsemantic aliases identically in all arms; original names and equation structure retained; original IDs only in sidecars"
                                    if alias_ids else "original source IDs retained identically; annotation fields only removed")
    protocol["input_identity_manifest_digest"] = digest(list(model_info.values()))
    write_json(out / "protocol.json", protocol)
    write_rows(out / "cohort.jsonl", cohort)
    write_rows(out / "available_ABC_cohort.jsonl", eligible)
    write_json(out / "protocol_freeze.json", {"protocol_sha256": sha256(out / "protocol.json"), "cohort_sha256": sha256(out / "cohort.jsonl"),
        "labels_read": False, "comparative_outcomes_read": False})
    write_json(out / "alignment.json", {"predictions": prediction_records, "validation_models": list(model_info.values()),
        "n_ABC_reactions": len(eligible), "n_missing_prediction_reactions": 969 - len(eligible)})
    write_json(out / "input_checksums.json", input_hashes)
    write_json(out / "species_generation_inputs.json", species_inputs)
    query_rows = []
    availability = []
    for mid in models:
        prediction = matched_predictions.get(mid)
        if prediction is not None and alias_ids:
            prediction = prediction.copy()
            prediction["id"] = prediction.id.map(lambda sid: model_info[mid]["species_aliases"].get(sid, sid))
        conditions = condition_queries(model_info[mid].get("query_source_path", model_info[mid]["path"]), prediction)
        for condition, rows in conditions.items():
            for row in rows:
                if (mid, row["reaction_id"]) not in keys:
                    continue
                row.update(model_id=mid, cluster_id=cluster[mid], condition=condition)
                query_rows.append(row)
                species = row["species_evidence"]
                availability.append({"model_id": mid, "reaction_id": row["reaction_id"], "condition": condition,
                    "participants": len(species), "participants_with_identifiers": sum(bool(v["chebi"] or v["kegg_compound"]) for v in species.values()),
                    "eligible_identifiers": sum(len(v["chebi"]) + len(v["kegg_compound"]) for v in species.values()),
                    "unresolved_rows": sum(r["identity_status"] == "unresolved" for v in species.values() for r in v["records"]),
                    "legacy_rows": sum(r["identity_status"] == "legacy_unassessed" for v in species.values() for r in v["records"]),
                    "conflicting_identifiers": sum(len(v["conflicting_identifiers"]) for v in species.values())})
    write_rows(out / "queries.jsonl", query_rows)
    write_csv(out / "identifier_availability.csv", availability)
    write_json(out / "query_freeze.json", {"sha256": sha256(out / "queries.jsonl"), "n": len(query_rows),
        "labels_read": False, "protocol_sha256": sha256(out / "protocol.json")})
    return {"protocol": protocol, "prediction_models": len(prediction_records), "source_unchanged": all(sha256(p) == value for p, value in input_hashes.items())}


def verify_catalog_cache(directory, expected):
    """Verify a document matrix's complete configuration and original freeze.

    This never confers compatibility on query embeddings or rankings.
    """
    directory = Path(directory)
    meta = read_json(directory / "catalog_embedding_cache.json")
    prior_protocol = read_json(directory / "protocol.json")
    prior_freeze = read_json(directory / "ranking_freeze.json")
    if prior_protocol["selected_retrieval"] != {**CONFIG, "catalog_sha256": CATALOG_SHA256, "encoder_manifest_sha256": MANIFEST_SHA256}:
        raise ValueError("cached document/encoder configuration changed")
    if sha256(directory / "protocol.json") != prior_freeze["protocol_sha256"] or prior_freeze["catalog_cache"] != meta:
        raise ValueError("catalog cache is not bound to its original freeze")
    if any(meta.get(key) != value for key, value in expected.items()) or sha256(directory / "catalog_embeddings.npy") != meta["embedding_sha256"]:
        raise ValueError("catalog embedding cache is not compatible")
    return meta


def rank(study, source_root, encoder_dir, *, catalog_cache=None):
    """Recompute every available arm with one CPU encoder/catalog; freeze before labels."""
    import importlib.metadata
    import time
    import numpy as np
    import torch
    from core.reaction.retrieval import ReactionRetriever, RetrievalResources, evidence_for
    study, source_root = Path(study), Path(source_root)
    if (study / "ranking_freeze.json").exists():
        raise FileExistsError("rankings already frozen")
    freeze = read_json(study / "protocol_freeze.json")
    if sha256(study / "protocol.json") != freeze["protocol_sha256"] or sha256(study / "cohort.jsonl") != freeze["cohort_sha256"]:
        raise ValueError("protocol/cohort changed after freeze")
    query_freeze = read_json(study / "query_freeze.json")
    if sha256(study / "queries.jsonl") != query_freeze["sha256"]:
        raise ValueError("query input changed")
    queries = read_jsonl(study / "queries.jsonl")
    torch.set_num_threads(4)
    resources = RetrievalResources(str(source_root / "data/kegg/kegg_reaction_features.lzma"), str(Path(encoder_dir).resolve()), device="cpu", batch_size=64)
    started = time.monotonic()
    print("Loading verified resources and computing one full catalog embedding matrix (CPU, no provider calls)", flush=True)
    if catalog_cache is None:
        retriever = ReactionRetriever(resources)
    else:
        import importlib.metadata
        import lzma, pickle
        from core.reaction.retrieval import verify_model_dir, load_encoder, BM25, document_text
        catalog_cache = Path(catalog_cache)
        expected = {"catalog_sha256": CATALOG_SHA256, "encoder_manifest_sha256": MANIFEST_SHA256,
                    "device": "cpu", "dtype": "float32", "batch_size": 64, "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers")}
        meta = verify_catalog_cache(catalog_cache, expected)
        if sha256(resources.catalog_path) != CATALOG_SHA256:
            raise ValueError("catalog hash mismatch")
        verify_model_dir(resources.encoder_dir)
        retriever = object.__new__(ReactionRetriever)
        retriever.resources = resources
        with lzma.open(resources.catalog_path, "rb") as handle:
            retriever.catalog = pickle.load(handle)
        retriever.ids = sorted(retriever.catalog)
        if digest(retriever.ids) != meta["document_ids_digest"]:
            raise ValueError("cached document order changed")
        retriever.bm25 = BM25([{"kegg_id": kid, "text": document_text(kid, retriever.catalog[kid])} for kid in retriever.ids])
        retriever.model, retriever.tokenizer = load_encoder(resources.encoder_dir, "cpu")
        retriever.embeddings = np.load(catalog_cache / "catalog_embeddings.npy", allow_pickle=False)
    np.save(study / "catalog_embeddings.npy", retriever.embeddings, allow_pickle=False)
    cache_meta = {"catalog_sha256": CATALOG_SHA256, "encoder_manifest_sha256": MANIFEST_SHA256,
        "device": "cpu", "dtype": "float32", "batch_size": 64, "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "embedding_sha256": sha256(study / "catalog_embeddings.npy"),
        "shape": list(retriever.embeddings.shape), "document_ids_digest": digest(retriever.ids)}
    write_json(study / "catalog_embedding_cache.json", cache_meta)
    unique = sorted({row["query"] for row in queries})
    print(f"Encoding {len(unique)} distinct query texts for {len(queries)} condition units", flush=True)
    vectors = retriever.encode(unique)
    by_text = {}
    for pos, query in enumerate(unique):
        scores = retriever.embeddings @ vectors[pos]
        dense = [retriever.ids[i] for i in sorted(range(len(retriever.ids)), key=lambda i: (-float(scores[i]), retriever.ids[i]))[:100]]
        bm = retriever.bm25.rank(query, 100)
        evidence = evidence_for(retriever.catalog, bm, dense)
        from core.reaction.retrieval import fusion
        by_text[query] = {"bm25": bm, "trained": dense, "ranked_ids": fusion([bm, dense], retriever.ids)[:100], "evidence": evidence}
        if pos % 200 == 0:
            print(f"Ranked {pos + 1}/{len(unique)} distinct inputs", flush=True)
    ranked, evidence_rows = [], []
    for row in queries:
        result = by_text[row["query"]]
        common = {key: row[key] for key in ["model_id", "reaction_id", "cluster_id", "condition"]}
        ranked.append({**common, "query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
                       "ranked_ids": result["ranked_ids"], "bm25": result["bm25"], "trained": result["trained"]})
        evidence_rows.append({**common, "records": result["evidence"]})
    write_rows(study / "rankings.jsonl", ranked)
    write_rows(study / "evidence.jsonl", evidence_rows)
    result = {"rankings_sha256": sha256(study / "rankings.jsonl"), "evidence_sha256": sha256(study / "evidence.jsonl"),
        "queries_sha256": query_freeze["sha256"], "protocol_sha256": freeze["protocol_sha256"], "units": len(queries),
        "unique_texts": len(unique), "elapsed_seconds": time.monotonic() - started, "catalog_cache": cache_meta,
        "labels_read": False, "provider_calls": 0, "historical_rankings_assigned_to_changed_inputs": False}
    result["catalog_cache_reused_from"] = str(catalog_cache) if catalog_cache else None
    write_json(study / "ranking_freeze.json", result)
    return result


def analyze(study, source_root):
    """Join only source-SBML labels for frozen validation keys, after freeze."""
    import re
    from core.model_info import find_reactions_with_kegg_annotations
    from benchmark.scripts.manuscript_metrics import retrieval_values, paired_summary
    study, source_root = Path(study), Path(source_root)
    freeze = read_json(study / "ranking_freeze.json")
    if sha256(study / "rankings.jsonl") != freeze["rankings_sha256"]:
        raise ValueError("rankings changed after freeze")
    for name, field in [("protocol.json", "protocol_sha256"), ("queries.jsonl", "queries_sha256"),
                        ("evidence.jsonl", "evidence_sha256")]:
        if sha256(study / name) != freeze[field]:
            raise ValueError(f"{name} changed after ranking freeze")
    if (study / "evaluated_rows.jsonl").exists():
        raise FileExistsError("evaluation already exists")
    cohort = read_jsonl(study / "cohort.jsonl")
    allowed = {(r["model_id"], r["reaction_id"]) for r in cohort}
    truth = {}
    # This is the extractor used by build_benchmark.py on the same pinned SBML.
    # No mixed-split reaction-label CSV is read, and no held-out SBML is opened.
    alignment = read_json(study / "alignment.json")
    for info in alignment["validation_models"]:
        path = Path(info["path"])
        if sha256(path) != info["sbml_sha256"]:
            raise ValueError("source SBML changed")
        values, _ = find_reactions_with_kegg_annotations(str(path))
        for (mid, rid) in allowed:
            if mid != info["model_id"]:
                continue
            ids = sorted({str(x).strip() for x in values.get(rid, []) if re.fullmatch(r"R\d{5}", str(x).strip())})
            if not ids:
                raise ValueError("frozen eligible reaction has no reconstructed source label")
            truth[(mid, rid)] = ids
    old_rank = {(r["model_id"], r["reaction_id"]): r["ranked_ids"] for r in
        read_jsonl(source_root / "benchmark/phase3/retrieval_baselines/rankings_bm25.jsonl")}
    old_score = pd.read_csv(source_root / "benchmark/phase3/retrieval_baselines/scored_bm25.csv").fillna("")
    checks = 0
    for row in old_score.to_dict("records"):
        key = row["model_id"], row["reaction_id"]
        if key not in allowed:
            raise ValueError("unexpected key in validation-only scored artifact")
        values = retrieval_values(old_rank[key], truth[key])
        for k in [1, 3, 5, 10]:
            if bool(values[f"hit_at_{k}"]) != bool(row[f"recall_at_{k}_exact"]):
                raise ValueError("source-label reconstruction disagrees with frozen validation outcomes")
        first = next((i for i, kid in enumerate(old_rank[key], 1) if kid in truth[key]), None)
        expected = int(row["first_hit_rank_exact"]) if row["first_hit_rank_exact"] != "" else None
        if first != expected:
            raise ValueError("reconstructed first-hit rank mismatch")
        checks += 1
    pilot = read_jsonl(source_root / "benchmark/phase3/phase3c_validation/answer_key_join.jsonl")
    if any(set(row["ground_truth_ids"]) != set(truth[(row["model_id"], row["reaction_id"])]) for row in pilot):
        raise ValueError("source labels disagree with frozen 163-case labels")
    write_rows(study / "source_reference_sets.jsonl", [{"model_id": mid, "reaction_id": rid, "reference_ids": ids}
                                                        for (mid, rid), ids in sorted(truth.items())])
    rows = []
    for row in read_jsonl(study / "rankings.jsonl"):
        key = row["model_id"], row["reaction_id"]
        if key not in allowed:
            raise ValueError("non-cohort row")
        rows.append({**{k: row[k] for k in ["model_id", "reaction_id", "cluster_id", "condition"]},
                     **retrieval_values(row["ranked_ids"], truth[key])})
    write_rows(study / "evaluated_rows.jsonl", rows)
    summary, per_model = [], []
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10",
               "label_recall_at_1", "label_recall_at_3", "label_recall_at_5", "label_recall_at_10"]
    for condition in sorted({row["condition"] for row in rows}):
        values = pd.DataFrame([r for r in rows if r["condition"] == condition])
        if len(values) != 969:
            raise ValueError("incomplete condition coverage")
        groups = values.groupby("model_id")[metrics].mean()
        for model, group in groups.iterrows():
            per_model.append({"condition": condition, "model_id": model, "n": int(values.model_id.eq(model).sum()), **group.to_dict()})
        summary.append({"condition": condition, "n_reactions": 969, "n_models": values.model_id.nunique(),
            "n_clusters": values.cluster_id.nunique(), "experiment": "available reference/names-only control; actual-prediction arm pending",
            **{metric: float(values[metric].mean()) for metric in metrics},
            **{metric + "_count": int(values[metric].sum()) for metric in metrics if metric.startswith("hit_")},
            **{metric + "_model_macro": float(groups[metric].mean()) for metric in metrics}})
    a = [r for r in rows if r["condition"] == "reference"]
    c = [r for r in rows if r["condition"] == "names_only"]
    paired = paired_summary(a, c)
    write_csv(study / "summary.csv", summary)
    write_csv(study / "per_model.csv", per_model)
    write_json(study / "paired_reference_names_only.json", paired)
    availability = pd.read_csv(study / "identifier_availability.csv")
    ref_av = availability[availability.condition.eq("reference")].copy()
    ref_av["stratum"] = ref_av.apply(lambda r: "none" if r.participants_with_identifiers == 0 else
                                    "all" if r.participants == r.participants_with_identifiers else "partial", axis=1)
    strata = {}
    for stratum in ["none", "partial", "all"]:
        selected = {(r.model_id, r.reaction_id) for r in ref_av[ref_av.stratum.eq(stratum)].itertuples()}
        if selected:
            strata[stratum] = paired_summary([r for r in a if (r["model_id"], r["reaction_id"]) in selected],
                                            [r for r in c if (r["model_id"], r["reaction_id"]) in selected])
    write_json(study / "predefined_missingness_strata.json", strata)
    queries = read_jsonl(study / "queries.jsonl")
    by_key = {(r["condition"], r["model_id"], r["reaction_id"]): r for r in queries}
    unchanged = sum(by_key[("reference", mid, rid)]["query"] == by_key[("names_only", mid, rid)]["query"] for mid, rid in allowed)
    result = {"n": 969, "n_ABC_observed": 0 if not any(r["condition"] == "predicted" for r in rows) else 969,
        "reference_names_queries_identical": unchanged, "changed_queries": 969 - unchanged,
        "reconstructed_labels_from_pinned_validation_SBML": len(truth), "frozen_BM25_score_checks": checks,
        "frozen_pilot_reference_set_checks": len(pilot), "historical_reference_labels_modified": False,
        "training_or_tuning": False, "held_out_content_read": False}
    write_json(study / "analysis_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "rank", "analyze"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--alias-source-ids", action="store_true")
    parser.add_argument("--catalog-cache", type=Path)
    args = parser.parse_args()
    result = (prepare(args.source_root, args.prediction_root, args.out, alias_ids=args.alias_source_ids) if args.stage == "prepare" else
              rank(args.out, args.source_root, args.encoder, catalog_cache=args.catalog_cache) if args.stage == "rank" else analyze(args.out, args.source_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
