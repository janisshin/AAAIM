"""Content-based retrieval resource identities for checkpoint/resume.

Absolute paths are returned separately for provenance/audit only. Identity
hashes catalog bytes and encoder manifest digests without loading GPU models
or computing catalog embeddings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.agent.schemas import CheckpointCompatibilityError

RESOURCE_IDENTITY_SCHEMA_VERSION = 1

_COMPLETE_KEYS = ("identity_schema_version", "identity_complete", "kind")


def _dtype_for_device(device: str) -> str:
    return "float16 autocast" if device == "cuda" else "float32"


def fingerprint_encoder_directory(encoder_dir: str | Path) -> dict[str, Any]:
    """Validate selected encoder via ``verify_model_dir`` and record digests."""
    from core.reaction import retrieval as retrieval_mod

    directory = Path(encoder_dir)
    retrieval_mod.verify_model_dir(directory)
    manifest = directory / "archive_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    return {
        "manifest_sha256": retrieval_mod.sha256(manifest),
        "files": {str(entry["path"]): str(entry["sha256"]) for entry in entries},
    }


def fingerprint_retrieval_resources(resources: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Build content+settings identity for ``RetrievalResources`` (no GPU/encode)."""
    from core.reaction.retrieval import CONFIG, RetrievalResources, sha256

    if not isinstance(resources, RetrievalResources):
        raise TypeError(f"expected RetrievalResources, got {type(resources).__name__}")
    catalog_path = Path(resources.catalog_path)
    encoder_dir = Path(resources.encoder_dir)
    if not catalog_path.is_file():
        raise CheckpointCompatibilityError(
            f"retrieval catalog not found for identity: {catalog_path}"
        )
    catalog_sha = sha256(catalog_path)
    encoder = fingerprint_encoder_directory(encoder_dir)
    identity = {
        "identity_schema_version": RESOURCE_IDENTITY_SCHEMA_VERSION,
        "identity_complete": True,
        "kind": "RetrievalResources",
        "catalog_sha256": catalog_sha,
        "encoder": encoder,
        "settings": {
            "device": resources.device,
            "batch_size": int(resources.batch_size),
            "dtype": _dtype_for_device(resources.device),
            "retrieval_config": dict(CONFIG),
        },
    }
    paths = {
        "catalog_path": str(catalog_path.resolve()),
        "encoder_dir": str(encoder_dir.resolve()),
    }
    return identity, paths


def _strip_path_like_keys(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {
        "path",
        "paths",
        "catalog_path",
        "encoder_dir",
        "hf_cache",
        "snapshot_dir",
        "model_file",
        "output_dir",
    }
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in blocked or key.endswith("_path") or key.endswith("_dir"):
            continue
        if isinstance(value, dict):
            out[key] = _strip_path_like_keys(value)
        else:
            out[key] = value
    return out


def _normalize_explicit_provenance(provenance: dict[str, Any], *, kind: str) -> dict[str, Any]:
    data = _strip_path_like_keys(dict(provenance))
    if data.get("identity_schema_version") != RESOURCE_IDENTITY_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"injected {kind} provenance missing identity_schema_version="
            f"{RESOURCE_IDENTITY_SCHEMA_VERSION}; cannot establish resume identity"
        )
    if not data.get("identity_complete"):
        raise CheckpointCompatibilityError(
            f"injected {kind} provenance is not identity_complete; "
            "provide content fingerprints or refuse resumable use"
        )
    if "kind" not in data:
        data["kind"] = kind
    if not (data.get("content_sha256") or data.get("catalog_sha256") or data.get("encoder")):
        raise CheckpointCompatibilityError(
            f"injected {kind} provenance lacks content identity fields "
            "(content_sha256 / catalog_sha256 / encoder)"
        )
    data["identity_complete"] = True
    data["identity_schema_version"] = RESOURCE_IDENTITY_SCHEMA_VERSION
    return data


def fingerprint_reaction_retriever(retriever: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Identity for a retriever instance without re-encoding the catalog."""
    from core.reaction.retrieval import ReactionRetriever, RetrievalResources

    paths: dict[str, str] = {}
    if isinstance(retriever, ReactionRetriever):
        identity, paths = fingerprint_retrieval_resources(retriever.resources)
        identity = {
            **identity,
            "kind": "ReactionRetriever",
            "loaded_from": "RetrievalResources",
        }
        return identity, paths

    if hasattr(retriever, "resources") and isinstance(
        getattr(retriever, "resources"), RetrievalResources
    ):
        identity, paths = fingerprint_retrieval_resources(retriever.resources)
        identity = {
            **identity,
            "kind": type(retriever).__name__,
            "loaded_from": "RetrievalResources",
        }
        return identity, paths

    provenance = getattr(retriever, "provenance", None)
    if isinstance(provenance, dict) and provenance:
        kind = type(retriever).__name__
        return _normalize_explicit_provenance(provenance, kind=kind), paths

    raise CheckpointCompatibilityError(
        f"cannot establish resume identity for injected retriever "
        f"{type(retriever).__name__}: provide RetrievalResources, ReactionRetriever, "
        "or explicit complete provenance (identity_schema_version, identity_complete, "
        "content fingerprints). Class-name equality is not sufficient."
    )


def build_resource_identity(
    *,
    reaction_retriever: Any = None,
    reaction_resources: Any = None,
    species_annotator_provenance: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (identity_for_resume_hash, absolute_paths_for_provenance)."""
    from core.reaction.retrieval import RetrievalResources

    paths: dict[str, str] = {}
    retriever_identity: Any = None
    resources_identity: Any = None

    if reaction_resources is not None:
        if isinstance(reaction_resources, RetrievalResources):
            resources_identity, res_paths = fingerprint_retrieval_resources(reaction_resources)
            paths.update({f"resources.{k}": v for k, v in res_paths.items()})
        elif isinstance(reaction_resources, dict):
            # Explicit dict payloads must be complete content identities.
            resources_identity = _normalize_explicit_provenance(
                reaction_resources, kind="reaction_resources_dict"
            )
        else:
            raise CheckpointCompatibilityError(
                f"unsupported reaction_resources type {type(reaction_resources).__name__}; "
                "expected RetrievalResources or complete identity dict"
            )

    if reaction_retriever is not None:
        retriever_identity, ret_paths = fingerprint_reaction_retriever(reaction_retriever)
        paths.update({f"retriever.{k}": v for k, v in ret_paths.items()})

    # Prefer resources identity when both describe the same selected stack; still
    # record retriever kind so swapping an injected wrapper is visible.
    envelope = {
        "identity_schema_version": RESOURCE_IDENTITY_SCHEMA_VERSION,
        "identity_complete": True,
        "reaction_retriever": retriever_identity,
        "reaction_resources": resources_identity,
        "species_annotator": dict(species_annotator_provenance or {}),
    }
    # Incomplete when a non-null side failed to produce complete identity (should not happen).
    for side in ("reaction_retriever", "reaction_resources"):
        payload = envelope[side]
        if payload is None:
            continue
        if not isinstance(payload, dict) or not payload.get("identity_complete"):
            raise CheckpointCompatibilityError(
                f"incomplete {side} identity; refusing silent class-name compatibility"
            )
        if payload.get("identity_schema_version") != RESOURCE_IDENTITY_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"{side} identity_schema_version unsupported; restart without resume_from"
            )
    return envelope, paths


def assert_resource_identity_compatible(saved: Any, current: dict[str, Any]) -> None:
    """Reject missing/legacy/incomplete resource identities explicitly."""
    if not isinstance(saved, dict):
        raise CheckpointCompatibilityError(
            "checkpoint resource_identity missing or not an object; "
            "pre-v2 / incomplete identities cannot be resumed"
        )
    if saved.get("identity_schema_version") != RESOURCE_IDENTITY_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "checkpoint resource_identity lacks supported identity_schema_version; "
            "restart without resume_from (resource identity format strengthened)"
        )
    if not saved.get("identity_complete"):
        raise CheckpointCompatibilityError(
            "checkpoint resource_identity is incomplete; refusing silent accept"
        )
    if saved != current:
        raise CheckpointCompatibilityError(
            "checkpoint incompatible with current run: resource_identity"
        )
