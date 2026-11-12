"""HF cache discovery, validation, and resume-identity tests (no downloads)."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

# Avoid importing core/__init__.py (ChromaDB/RAG stack) in lean MetaboLLM envs.
_ROOT = Path(__file__).resolve().parents[2]
if "core" not in sys.modules:
    _core = types.ModuleType("core")
    _core.__path__ = [str(_ROOT / "core")]
    sys.modules["core"] = _core

from core.agent.backends import metabolm as mb  # noqa: E402


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _hub_snapshot(cache: Path, repo_id: str, revision: str) -> Path:
    snap = cache / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    return snap


def build_minimal_valid_cache(cache: Path) -> Path:
    """Tiny offline Hub layout matching pinned adapter + base revisions."""
    # Keep paths short on Windows (MAX_PATH) while preserving Hub layout names.
    adapter = _hub_snapshot(cache, mb.ADAPTER_ID, mb.ADAPTER_REVISION)
    _write(adapter / "adapter_config.json", '{"peft_type":"LORA"}\n')
    _write(adapter / "adapter_model.safetensors", b"adapter-weights-v1")

    base = _hub_snapshot(cache, mb.BASE_ID, mb.BASE_REVISION)
    _write(base / "config.json", '{"model_type":"qwen3"}\n')
    _write(base / "tokenizer_config.json", '{"tokenizer_class":"X"}\n')
    index = {
        "weight_map": {
            "a": "w1.safetensors",
            "b": "w2.safetensors",
        }
    }
    _write(base / "model.safetensors.index.json", json.dumps(index))
    _write(base / "w1.safetensors", b"base-shard-1")
    _write(base / "w2.safetensors", b"base-shard-2")
    return cache


def test_explicit_path_wins_over_env_and_default(tmp_path, monkeypatch):
    explicit = build_minimal_valid_cache(tmp_path / "ex")
    env_cache = build_minimal_valid_cache(tmp_path / "en")
    monkeypatch.setenv(mb.HF_CACHE_ENV, str(env_cache))
    resolved = mb.resolve_hf_cache(explicit, environ=os.environ)
    assert resolved == explicit.resolve()
    backend = mb.MetaboLLMBackend(hf_cache=explicit, lazy=True)
    assert backend.hf_cache == explicit.resolve()
    assert backend.identity()["weight_fingerprints"]["content_sha256"]


def test_env_wins_over_default_and_invalid_env_does_not_fall_back(tmp_path, monkeypatch):
    env_cache = build_minimal_valid_cache(tmp_path / "eok")
    monkeypatch.setenv(mb.HF_CACHE_ENV, str(env_cache))
    assert mb.resolve_hf_cache(None, environ=os.environ) == env_cache.resolve()

    missing = tmp_path / "miss path" / "nope"
    monkeypatch.setenv(mb.HF_CACHE_ENV, str(missing))
    resolved = mb.resolve_hf_cache(None, environ=os.environ)
    assert resolved == missing.resolve()
    with pytest.raises(mb.MetaboLLMCacheError, match="does not exist"):
        mb.validate_metabolm_hf_cache(resolved)
    # Authoritative env: constructing backend must fail without discovering another cache.
    with pytest.raises(mb.MetaboLLMCacheError):
        mb.MetaboLLMBackend(lazy=True)


def test_invalid_explicit_does_not_fall_back(tmp_path, monkeypatch):
    good = build_minimal_valid_cache(tmp_path / "g")
    monkeypatch.setenv(mb.HF_CACHE_ENV, str(good))
    bad = tmp_path / "bad"
    with pytest.raises(mb.MetaboLLMCacheError):
        mb.MetaboLLMBackend(hf_cache=bad, lazy=True)


def test_ordinary_and_nested_worktree_discovery(tmp_path):
    # Ordinary checkout: .../repo/core/agent/backends/metabolm.py
    ordinary = tmp_path / "repo"
    module = ordinary / "core" / "agent" / "backends" / "metabolm.py"
    module.parent.mkdir(parents=True)
    module.write_text("# stub\n", encoding="utf-8")
    (ordinary / "benchmark" / "phase3").mkdir(parents=True)
    cache = build_minimal_valid_cache(
        ordinary / "benchmark" / "phase3" / "metabolm_grounded_comparison" / "_hf_cache"
    )
    assert mb.discover_default_hf_cache(start=module) == cache.resolve()

    # Nested worktree: cache lives on parent AAAIM root, module under integration worktree.
    parent = tmp_path / "R"
    nested_module = (
        parent
        / "benchmark"
        / "integration_worktrees"
        / "wt"
        / "core"
        / "agent"
        / "backends"
        / "metabolm.py"
    )
    nested_module.parent.mkdir(parents=True)
    nested_module.write_text("# stub\n", encoding="utf-8")
    (nested_module.parents[3] / "benchmark").mkdir(parents=True, exist_ok=True)
    (nested_module.parents[3] / "core" / "agent").mkdir(parents=True, exist_ok=True)
    parent_cache = build_minimal_valid_cache(
        parent / "benchmark" / "phase3" / "metabolm_grounded_comparison" / "_hf_cache"
    )
    assert mb.discover_default_hf_cache(start=nested_module) == parent_cache.resolve()


def test_no_repository_root_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv(mb.HF_CACHE_ENV, raising=False)
    orphan = tmp_path / "o" / "d" / "metabolm.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("# stub\n", encoding="utf-8")
    with pytest.raises(mb.MetaboLLMCacheError, match="Could not locate an AAAIM repository root"):
        mb.discover_default_hf_cache(start=orphan)
    with pytest.raises(mb.MetaboLLMCacheError, match=mb.HF_CACHE_ENV):
        mb.resolve_hf_cache(None, environ={}, start=orphan)


def test_path_with_spaces(tmp_path):
    cache = build_minimal_valid_cache(tmp_path / "c space" / "hf")
    validated = mb.validate_metabolm_hf_cache(cache)
    assert "c space" in validated["hf_cache"]
    backend = mb.MetaboLLMBackend(hf_cache=cache, lazy=True)
    assert "c space" in backend.resource_paths["hf_cache"]


def test_missing_shard_and_broken_symlink(tmp_path):
    cache = build_minimal_valid_cache(tmp_path / "c1")
    base = _hub_snapshot(cache, mb.BASE_ID, mb.BASE_REVISION)
    (base / "w2.safetensors").unlink()
    with pytest.raises(mb.MetaboLLMCacheError, match="w2.safetensors"):
        mb.validate_metabolm_hf_cache(cache)

    cache2 = build_minimal_valid_cache(tmp_path / "c2")
    base2 = _hub_snapshot(cache2, mb.BASE_ID, mb.BASE_REVISION)
    shard = base2 / "w1.safetensors"
    shard.unlink()
    target = base2 / "missing-blob.bin"
    try:
        shard.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not available on this platform/permissions")
    with pytest.raises(mb.MetaboLLMCacheError, match="[Bb]roken symlink|does not resolve|Missing"):
        mb.validate_metabolm_hf_cache(cache2)


def test_content_change_changes_fingerprint_same_revision(tmp_path):
    cache_a = build_minimal_valid_cache(tmp_path / "a")
    cache_b = build_minimal_valid_cache(tmp_path / "b")
    fp_a = mb.validate_metabolm_hf_cache(cache_a)["content_sha256"]
    adapter = _hub_snapshot(cache_b, mb.ADAPTER_ID, mb.ADAPTER_REVISION)
    _write(adapter / "adapter_model.safetensors", b"adapter-weights-TAMPERED")
    fp_b = mb.validate_metabolm_hf_cache(cache_b)["content_sha256"]
    assert fp_a != fp_b
    id_a = mb.MetaboLLMBackend(hf_cache=cache_a, lazy=True).identity()
    id_b = mb.MetaboLLMBackend(hf_cache=cache_b, lazy=True).identity()
    assert id_a["adapter_revision"] == id_b["adapter_revision"]
    assert id_a["weight_fingerprints"]["content_sha256"] != id_b["weight_fingerprints"]["content_sha256"]


def test_relocated_identical_content_keeps_resume_identity(tmp_path):
    cache1 = build_minimal_valid_cache(tmp_path / "p1")
    cache2 = build_minimal_valid_cache(tmp_path / "p2")
    b1 = mb.MetaboLLMBackend(hf_cache=cache1, lazy=True)
    b2 = mb.MetaboLLMBackend(hf_cache=cache2, lazy=True)
    assert b1.resource_paths["hf_cache"] != b2.resource_paths["hf_cache"]
    assert b1.identity() == b2.identity()
    assert "hf_cache" not in b1.identity()
    assert "load_info" not in b1.identity()


def test_transient_load_info_not_in_identity(tmp_path):
    cache = build_minimal_valid_cache(tmp_path / "c")
    backend = mb.MetaboLLMBackend(hf_cache=cache, lazy=True)
    before = backend.identity()
    backend.load_info = {
        "vram_free_mb": 123.4,
        "cuda_device_name": "FakeGPU",
        "vram_allocated_mb": 9.9,
    }
    after = backend.identity()
    assert before == after
    assert "vram_free_mb" not in json.dumps(after)


def test_load_uses_local_files_only(tmp_path, monkeypatch):
    cache = build_minimal_valid_cache(tmp_path / "c")
    backend = mb.MetaboLLMBackend(hf_cache=cache, lazy=True)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        repo = kwargs["repo_id"]
        rev = kwargs["revision"]
        return str(_hub_snapshot(cache, repo, rev))

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return (1, 2)

        @staticmethod
        def get_device_name(_idx=0):
            return "Fake"

        @staticmethod
        def memory_allocated():
            return 0

        @staticmethod
        def empty_cache():
            return None

    class _FakeTorch:
        cuda = _FakeCuda
        bfloat16 = "bf16"

        class inference_mode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

    class _FakeTok:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get("local_files_only") is True
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return "x"

        def __call__(self, *args, **kwargs):
            return {"input_ids": type("T", (), {"shape": (1, 1)})()}

        def decode(self, *args, **kwargs):
            return "{}"

    class _FakeModel:
        config = type("C", (), {"max_position_embeddings": 100})()
        device = "cpu"
        peft_config = {"default": object()}

        def eval(self):
            return self

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get("local_files_only") is True
            return cls()

    class _FakePeft:
        @staticmethod
        def from_pretrained(base, adapter, is_trainable=False):
            return _FakeModel()

    class _FakeBNB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import sys
    import types

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    peft = types.ModuleType("peft")
    peft.PeftModel = _FakePeft
    monkeypatch.setitem(sys.modules, "peft", peft)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _FakeTok
    transformers.AutoModelForCausalLM = _FakeModel
    transformers.BitsAndBytesConfig = _FakeBNB
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    # Avoid mutating real CUDA path; force torch import inside load to see fakes.
    info = backend.load()
    assert calls and all(c.get("local_files_only") is True for c in calls)
    assert info["adapter_loaded"] is True
    # Ensure process-wide HF env was not required/set by load for correctness of paths.
    assert backend.identity()["weight_fingerprints"]["content_sha256"]
