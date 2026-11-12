"""Species annotation helpers for the automatic annotation agent.

Synthetic identifiers exist only in ``mock_species_annotator`` for tests.
Real backends must use ``make_real_species_annotator`` (normalization →
direct/RAG candidate retrieval → optional ranking → identity/qualifier).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import pandas as pd

from core.agent.schemas import AnnotationStatus, BiologicalType, ModelInventory

# Known placeholder IDs formerly used as an implicit default. Real paths must
# never emit these unless they happen to be genuine database hits for a name.
MOCK_SYNTHETIC_ANNOTATIONS = {
    "chebi": "CHEBI:15377",
    "ncbigene": "NCBIGene:3988",
    "uniprot": "UNIPROT:P0DP23",
}


def _data_dir() -> Path:
    """Resolve AAAIM data/ without importing database_search (avoids RAG/cygrpc)."""
    return Path(__file__).resolve().parents[2] / "data"


def species_database_available(database: str = "chebi") -> tuple[bool, str]:
    """Check local reference files required for offline direct retrieval."""
    root = _data_dir()
    db = database.lower()
    if db == "chebi":
        path = root / "chebi" / "cleannames2chebi.lzma"
        return (path.is_file(), str(path))
    if db == "ncbigene":
        path = root / "ncbigene" / "names2ncbigene_bigg_organisms_protein-coding.lzma"
        return (path.is_file(), str(path))
    if db == "uniprot":
        path = root / "uniprot" / "names2uniprot_human+mouse.lzma"
        return (path.is_file(), str(path))
    return False, f"unsupported database: {database}"


def mock_species_annotator(
    model_file: str,
    target_ids: list[str],
    databases: list[str],
    inventory: ModelInventory,
) -> pd.DataFrame:
    """Explicit test/mock annotator that emits synthetic identifiers.

    Must only be selected when the backend is mock (or tests inject it).
    Rows are labeled so provenance can prove mock behavior.
    """
    by_id = {e.target_id: e for e in inventory.entities}
    rows = []
    for tid in target_ids:
        ent = by_id[tid]
        if ent.biological_type == BiologicalType.AMBIGUOUS:
            rows.append(
                {
                    "id": tid,
                    "display_name": ent.display_name,
                    "annotation": "",
                    "database": "",
                    "type": ent.biological_type.value,
                    "qualifier": "is",
                    "update_annotation": "ignore",
                    "identity_status": "unresolved",
                    "status": "unresolved",
                    "Note": "ambiguous biological type",
                    "annotator_kind": "mock_synthetic",
                }
            )
            continue
        db = (databases or ["chebi"])[0].lower()
        annotation = MOCK_SYNTHETIC_ANNOTATIONS.get(db, MOCK_SYNTHETIC_ANNOTATIONS["chebi"])
        if ent.annotation_status == AnnotationStatus.COMPLETE:
            continue
        rows.append(
            {
                "id": tid,
                "display_name": ent.display_name,
                "annotation": annotation,
                "database": db,
                "type": ent.biological_type.value,
                "qualifier": "is",
                "update_annotation": "add",
                "identity_status": "supported",
                "status": "selected",
                "match_score": 0.5,
                "annotator_kind": "mock_synthetic",
                "Note": "mock synthetic annotation for tests only",
            }
        )
    return pd.DataFrame(rows)


def _entity_type_for_databases(databases: Sequence[str]) -> str:
    dbs = {d.lower() for d in databases}
    if dbs == {"chebi"}:
        return "chemical"
    if dbs == {"ncbigene"}:
        return "gene"
    if dbs == {"uniprot"}:
        return "protein"
    return "auto"


@contextmanager
def llm_backend_hook(backend: Any) -> Iterator[dict[str, Any]]:
    """Route core.llm_interface calls through a local backend.generate when present.

    Hosted OpenAI/OpenRouter paths are not used while the hook is installed.
    """
    import core.llm_interface as llm

    meta: dict[str, Any] = {
        "hook_installed": False,
        "calls": 0,
        "stages": [],
        "errors": [],
    }
    if backend is None or not hasattr(backend, "generate"):
        yield meta
        return

    original = llm.query_llm_message_with_history

    def _hook(
        messages,
        model: str = "metabolm-local",
        max_retries: int = 1,
        initial_delay: float = 0.0,
        max_completion_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        usage_callback=None,
    ):
        meta["calls"] += 1
        meta["stages"].append({"model": model, "n_messages": len(messages)})
        prompt_messages = [dict(m) for m in messages]
        if response_format is not None:
            schema_text = (
                "\n\nLOCAL_TRANSPORT_JSON_SCHEMA\n"
                "Return exactly one JSON object matching this schema and no other text:\n"
                + json.dumps(response_format, sort_keys=True)
            )
            if prompt_messages and prompt_messages[0].get("role") == "system":
                prompt_messages[0] = {
                    **prompt_messages[0],
                    "content": str(prompt_messages[0].get("content") or "") + schema_text,
                }
            else:
                prompt_messages.insert(0, {"role": "system", "content": schema_text})
        # Prefer the caller's completion budget; fall back above the planner default
        # so structured species normalization JSON is less likely to truncate.
        budget = max_completion_tokens if max_completion_tokens is not None else 1536
        try:
            generated = backend.generate(prompt_messages, max_new_tokens=int(budget))
        except TypeError:
            # Test stubs may only accept messages.
            generated = backend.generate(prompt_messages)
        if generated.get("terminal_status") != "generated" or not generated.get("raw_text"):
            raise RuntimeError(
                f"local species LLM failed: {generated.get('error') or generated.get('terminal_status')}"
            )
        if generated.get("truncated"):
            raise RuntimeError("local species LLM output truncated; refusing silent acceptance")
        text = generated["raw_text"]
        if usage_callback and generated.get("usage"):
            usage_callback({"model": model, **generated["usage"], "local": True})
        return {"role": "assistant", "content": text}

    llm.query_llm_message_with_history = _hook  # type: ignore[assignment]
    meta["hook_installed"] = True
    try:
        yield meta
    finally:
        llm.query_llm_message_with_history = original  # type: ignore[assignment]


def _species_frame_from_annotate_result(result: Any) -> pd.DataFrame:
    """Normalize annotate_single_model success/error shapes into a DataFrame.

    Success may return AnnotationResult; failure paths often return
    ``(DataFrame, metrics_dict)`` without raising.
    """
    if result is None:
        raise RuntimeError("species annotation returned no result")
    if hasattr(result, "species_recommendations_df"):
        frame = result.species_recommendations_df
        if frame is None:
            raise RuntimeError("species annotation returned no recommendations table")
        return frame
    if isinstance(result, tuple) and len(result) >= 1:
        frame = result[0]
        metrics = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        err = metrics.get("error") if isinstance(metrics, dict) else None
        if err:
            raise RuntimeError(f"species annotation failed: {err}")
        if frame is None:
            raise RuntimeError("species annotation returned empty failure payload")
        if isinstance(frame, pd.DataFrame):
            if frame.empty and err:
                raise RuntimeError(f"species annotation failed: {err}")
            return frame
    raise RuntimeError(
        f"unexpected species annotation result type: {type(result).__name__}"
    )


def make_real_species_annotator(
    backend: Any = None,
    *,
    method: str = "direct",
    predict_qualifiers: bool = True,
    effort: str = "low",
    validation: bool = True,
    llm_model: str = "metabolm-local",
    max_completion_tokens: int = 1536,
    call_log: Optional[dict[str, Any]] = None,
) -> Callable[..., pd.DataFrame]:
    """Build a species annotator that uses the real AAAIM species pipeline."""

    def annotator(
        model_file: str,
        target_ids: list[str],
        databases: list[str],
        inventory: ModelInventory,
    ) -> pd.DataFrame:
        from core.annotation_workflow import annotate_single_model

        dbs = [d.lower() for d in (databases or ["chebi"])]
        for db in dbs:
            ok, path = species_database_available(db)
            if not ok:
                raise RuntimeError(
                    f"species database resource unavailable for {db}: {path}"
                )
        if not target_ids:
            return pd.DataFrame()
        entity_type = _entity_type_for_databases(dbs)
        database: Any = dbs[0] if entity_type != "auto" else dbs

        def _run():
            return annotate_single_model(
                model_file,
                annotate="species",
                species_ids=list(target_ids),
                method=method,
                database=database,
                entity_type=entity_type,
                predict_qualifiers=predict_qualifiers,
                effort=effort,
                validation=validation,
                llm_model=llm_model,
                max_completion_tokens=max_completion_tokens,
                top_k=3,
                n_return=1,
                save_to=None,
                verbose=False,
            )

        if backend is not None and hasattr(backend, "generate"):
            with llm_backend_hook(backend) as meta:
                if call_log is not None:
                    call_log.setdefault("species_llm", []).append(meta)
                result = _run()
        else:
            # Without a local generate() backend, refuse hosted fallback here.
            raise RuntimeError(
                "real species annotator requires a local LLM backend with generate(); "
                "refusing hosted-provider fallback and synthetic identifiers"
            )

        frame = _species_frame_from_annotate_result(result)
        frame = frame.copy()
        if "id" in frame.columns:
            frame = frame[frame["id"].astype(str).isin({str(t) for t in target_ids})].copy()
        frame["annotator_kind"] = "real_pipeline"
        # Never promote rows to supported merely because an ID string is well-formed.
        if "identity_status" not in frame.columns:
            frame["identity_status"] = ""
        return frame

    annotator.annotator_kind = "real_pipeline"  # type: ignore[attr-defined]
    return annotator


def resolve_species_annotator(
    backend: Any,
    species_annotator: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Choose annotator and provenance. Never silently fall back to synthetic IDs."""
    if species_annotator is not None:
        kind = getattr(species_annotator, "annotator_kind", None) or "injected"
        if species_annotator is mock_species_annotator or kind == "mock_synthetic":
            return species_annotator, {
                "kind": "mock_synthetic",
                "explicit": True,
                "note": "Synthetic species annotations are mock/test-only",
            }
        return species_annotator, {
            "kind": kind if isinstance(kind, str) else "injected",
            "explicit": True,
        }

    backend_name = getattr(backend, "name", type(backend).__name__)
    if backend_name == "mock" or type(backend).__name__ == "MockBackend":
        mock_species_annotator.annotator_kind = "mock_synthetic"  # type: ignore[attr-defined]
        return mock_species_annotator, {
            "kind": "mock_synthetic",
            "explicit": True,
            "note": "MockBackend selected explicit synthetic species annotator",
        }

    call_log: dict[str, Any] = {}
    annotator = make_real_species_annotator(backend, call_log=call_log)
    return annotator, {
        "kind": "real_pipeline",
        "explicit": True,
        "method": "direct",
        "identity_and_qualifiers": True,
        "call_log": call_log,
        "note": "Uses annotate_single_model + local LLM hook; no synthetic ID default",
    }
