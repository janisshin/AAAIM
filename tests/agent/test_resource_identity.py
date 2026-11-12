"""Content-based retrieval resource identity + resume compatibility."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from core.agent.resource_identity import (
    RESOURCE_IDENTITY_SCHEMA_VERSION,
    build_resource_identity,
    fingerprint_retrieval_resources,
)
from core.agent.schemas import CheckpointCompatibilityError
from core.agent import auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.reaction import retrieval as retrieval_mod
from core.reaction.retrieval import RetrievalResources


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def build_synthetic_encoder(directory: Path, *, marker: str = "enc-v1") -> Path:
    """Minimal encoder dir accepted by verify_model_dir after SHA monkeypatch."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": f'{{"marker":"{marker}"}}\n',
        "tokenizer_config.json": '{"tokenizer_class":"X"}\n',
        "tokenizer.json": '{"model":{"vocab":{"a":0}}}\n',
        "pytorch_model.bin": f"weights-{marker}".encode("utf-8"),
    }
    entries = []
    for name, payload in files.items():
        target = directory / name
        if isinstance(payload, str):
            _write(target, payload)
        else:
            _write(target, payload)
        entries.append({"path": name, "sha256": _sha(target)})
    manifest = {"files": entries}
    _write(directory / "archive_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return directory


def build_synthetic_catalog(path: Path, *, marker: str = "cat-v1") -> Path:
    # Identity hashes raw bytes; contents need not be a real KEGG pickle.
    _write(path, f"synthetic-catalog:{marker}\n".encode("utf-8"))
    return path


@pytest.fixture
def synthetic_stack(tmp_path, monkeypatch):
    catalog = build_synthetic_catalog(tmp_path / "catalog.bin", marker="A")
    encoder = build_synthetic_encoder(tmp_path / "encoder", marker="A")
    monkeypatch.setattr(retrieval_mod, "CATALOG_SHA256", _sha(catalog))
    monkeypatch.setattr(retrieval_mod, "MANIFEST_SHA256", _sha(encoder / "archive_manifest.json"))
    resources = RetrievalResources(
        catalog_path=str(catalog),
        encoder_dir=str(encoder),
        device="cpu",
        batch_size=8,
    )
    return resources, catalog, encoder


def test_retrieval_resources_identity_is_content_based_not_typename(synthetic_stack):
    resources, catalog, encoder = synthetic_stack
    identity, paths = fingerprint_retrieval_resources(resources)
    assert identity["identity_complete"] is True
    assert identity["identity_schema_version"] == RESOURCE_IDENTITY_SCHEMA_VERSION
    assert identity["kind"] == "RetrievalResources"
    assert identity["catalog_sha256"] == _sha(catalog)
    assert identity["encoder"]["manifest_sha256"] == _sha(encoder / "archive_manifest.json")
    assert identity["settings"]["device"] == "cpu"
    assert identity["settings"]["batch_size"] == 8
    assert "catalog_path" not in identity
    assert "encoder_dir" not in identity
    assert paths["catalog_path"].endswith("catalog.bin")
    assert Path(paths["encoder_dir"]).name == "encoder"


def test_catalog_or_encoder_content_change_changes_identity(tmp_path, monkeypatch):
    cat_a = build_synthetic_catalog(tmp_path / "a" / "catalog.bin", marker="A")
    enc_a = build_synthetic_encoder(tmp_path / "a" / "encoder", marker="A")
    monkeypatch.setattr(retrieval_mod, "CATALOG_SHA256", _sha(cat_a))
    monkeypatch.setattr(retrieval_mod, "MANIFEST_SHA256", _sha(enc_a / "archive_manifest.json"))
    id_a, _ = fingerprint_retrieval_resources(
        RetrievalResources(str(cat_a), str(enc_a), device="cpu", batch_size=4)
    )

    cat_b = build_synthetic_catalog(tmp_path / "b" / "catalog.bin", marker="B")
    enc_b = build_synthetic_encoder(tmp_path / "b" / "encoder", marker="A")
    monkeypatch.setattr(retrieval_mod, "CATALOG_SHA256", _sha(cat_b))
    monkeypatch.setattr(retrieval_mod, "MANIFEST_SHA256", _sha(enc_b / "archive_manifest.json"))
    id_b, _ = fingerprint_retrieval_resources(
        RetrievalResources(str(cat_b), str(enc_b), device="cpu", batch_size=4)
    )
    assert id_a["catalog_sha256"] != id_b["catalog_sha256"]

    cat_c = build_synthetic_catalog(tmp_path / "c" / "catalog.bin", marker="A")
    enc_c = build_synthetic_encoder(tmp_path / "c" / "encoder", marker="C")
    monkeypatch.setattr(retrieval_mod, "CATALOG_SHA256", _sha(cat_c))
    monkeypatch.setattr(retrieval_mod, "MANIFEST_SHA256", _sha(enc_c / "archive_manifest.json"))
    id_c, _ = fingerprint_retrieval_resources(
        RetrievalResources(str(cat_c), str(enc_c), device="cpu", batch_size=4)
    )
    assert id_a["encoder"]["files"] != id_c["encoder"]["files"]


def test_device_or_batch_change_changes_identity_same_paths(synthetic_stack):
    resources, catalog, encoder = synthetic_stack
    cpu = RetrievalResources(str(catalog), str(encoder), device="cpu", batch_size=8)
    cuda = RetrievalResources(str(catalog), str(encoder), device="cuda", batch_size=8)
    batch = RetrievalResources(str(catalog), str(encoder), device="cpu", batch_size=16)
    id_cpu, _ = fingerprint_retrieval_resources(cpu)
    id_cuda, _ = fingerprint_retrieval_resources(cuda)
    id_batch, _ = fingerprint_retrieval_resources(batch)
    assert id_cpu["settings"]["device"] != id_cuda["settings"]["device"]
    assert id_cpu["settings"]["dtype"] != id_cuda["settings"]["dtype"]
    assert id_cpu["catalog_sha256"] == id_cuda["catalog_sha256"]
    assert id_cpu["settings"]["batch_size"] != id_batch["settings"]["batch_size"]


def test_relocated_identical_resources_keep_identity(tmp_path, monkeypatch):
    src_cat = build_synthetic_catalog(tmp_path / "src" / "catalog.bin", marker="same")
    src_enc = build_synthetic_encoder(tmp_path / "src" / "encoder", marker="same")
    dst_cat = tmp_path / "dst" / "catalog.bin"
    dst_enc = tmp_path / "dst" / "encoder"
    dst_cat.parent.mkdir(parents=True)
    shutil.copy2(src_cat, dst_cat)
    shutil.copytree(src_enc, dst_enc)
    monkeypatch.setattr(retrieval_mod, "CATALOG_SHA256", _sha(src_cat))
    monkeypatch.setattr(retrieval_mod, "MANIFEST_SHA256", _sha(src_enc / "archive_manifest.json"))
    id_src, paths_src = fingerprint_retrieval_resources(
        RetrievalResources(str(src_cat), str(src_enc), device="cpu", batch_size=8)
    )
    id_dst, paths_dst = fingerprint_retrieval_resources(
        RetrievalResources(str(dst_cat), str(dst_enc), device="cpu", batch_size=8)
    )
    assert id_src == id_dst
    assert paths_src != paths_dst


def test_identity_does_not_load_gpu_or_embeddings(synthetic_stack, monkeypatch):
    resources, _, _ = synthetic_stack
    calls: list[str] = []

    def boom_load_encoder(*args, **kwargs):
        calls.append("load_encoder")
        raise AssertionError("load_encoder must not run during identity")

    def boom_load_retriever(*args, **kwargs):
        calls.append("load_retriever")
        raise AssertionError("load_retriever must not run during identity")

    monkeypatch.setattr(retrieval_mod, "load_encoder", boom_load_encoder)
    monkeypatch.setattr(retrieval_mod, "load_retriever", boom_load_retriever)
    fingerprint_retrieval_resources(resources)
    build_resource_identity(reaction_resources=resources, reaction_retriever=None)
    assert calls == []


def test_class_name_only_retriever_rejected():
    class Nameless:
        pass

    with pytest.raises(CheckpointCompatibilityError, match="cannot establish resume identity"):
        build_resource_identity(reaction_retriever=Nameless())


def test_resume_rejects_changed_resources_before_annotation(chemical_only_model, tmp_path, monkeypatch):
    from tests.agent.conftest import TinyRetriever

    out = tmp_path / "run"
    auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        reaction_retriever=TinyRetriever(),
        output_dir=out,
        interrupt_after_actions=1,
    )
    # Mutate live retriever identity before resume.
    bad = TinyRetriever()
    bad.provenance = {
        **bad.provenance,
        "content_sha256": "0" * 64,
    }
    annotated = {"called": False}

    class GuardBackend(MockBackend):
        def plan(self, *args, **kwargs):
            annotated["called"] = True
            return super().plan(*args, **kwargs)

    with pytest.raises(CheckpointCompatibilityError, match="resource_identity"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=GuardBackend(),
            reaction_retriever=bad,
            output_dir=out,
            resume_from=out,
        )
    assert annotated["called"] is False


def test_legacy_typename_resource_identity_rejected(chemical_only_model, tmp_path):
    out = tmp_path / "legacy"
    auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out,
        interrupt_after_actions=1,
    )
    ckpt_path = out / "checkpoint.json"
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    payload["resource_identity"] = {
        "reaction_retriever": None,
        "reaction_resources": {"type": "RetrievalResources"},
        "species_annotator": {},
    }
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(CheckpointCompatibilityError, match="identity_schema_version|incomplete"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=out,
            resume_from=out,
        )
