"""
ChEBI ontology traversal and ChEBI → KEGG compound normalization with optional
relaxation along is_a parents.

Used by the KEGG reaction–mapping pipeline to widen mappings only when a
metabolite is unmatched or drives a weak reaction match.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

# Baseline vs leave-one-ChEBI-out scores for ``detect_problematic_metabolites``.
# Call with ``exclude_chebi=None`` for full reaction; with a ChEBI id to drop
# that term's contribution on both sides of the fingerprint.
ReactionScoreMatcher = Callable[[Optional[str]], float]

logger = logging.getLogger(__name__)

ParentMap = Mapping[str, Set[str]]
ChebiToKegg = Mapping[str, Any]


def parse_chebi_obo(obo_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """
    Parse a ChEBI OBO file and return directed is_a edges: child → parents.

    Returns:
        Mapping from ChEBI ID (e.g. CHEBI:17234) to a set of parent ChEBI IDs.
    """
    parent_map: Dict[str, Set[str]] = defaultdict(set)
    current_id: Optional[str] = None
    path = Path(obo_path)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
                continue
            if line.startswith("id:"):
                current_id = line.split("id:", 1)[1].strip()
            elif line.startswith("is_a:") and current_id:
                parent_id = line.split("is_a:", 1)[1].split("!")[0].strip()
                if parent_id:
                    parent_map[current_id].add(parent_id)

    return {k: set(v) for k, v in parent_map.items()}


def get_ancestors(
    chebi_id: str,
    parent_map: ParentMap,
    depth: Optional[int] = None,
) -> Set[str]:
    """
    Ancestors of chebi_id along is_a, excluding chebi_id itself.

    Args:
        chebi_id: ChEBI identifier.
        parent_map: Child → parent sets from parse_chebi_obo (or equivalent).
        depth: If None, full transitive closure. If a non-negative int, include
            only nodes within that many edges from chebi_id (parents at 1 hop,
            grandparents at 2, ...).
    """
    visited: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(chebi_id, 0)]

    while frontier:
        current, d = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        if depth is not None and d >= depth:
            continue

        for parent in parent_map.get(current, ()):
            frontier.append((parent, d + 1))

    visited.discard(chebi_id)
    return visited


def _coerce_kegg_values(raw: Any) -> Set[str]:
    """Normalize pickle/CSV values to a set of KEGG compound strings."""
    if raw is None:
        return set()
    if isinstance(raw, float) and str(raw) == "nan":
        return set()
    if isinstance(raw, str):
        s = raw.strip()
        return {s} if s else set()
    if isinstance(raw, (set, frozenset)):
        return {str(x).strip() for x in raw if str(x).strip()}
    if isinstance(raw, (list, tuple)):
        out: Set[str] = set()
        for x in raw:
            out.update(_coerce_kegg_values(x))
        return out
    return {str(raw).strip()} if str(raw).strip() else set()


def merge_chebi_to_kegg_mapping(raw: Mapping[str, Any]) -> Dict[str, Set[str]]:
    """
    Build ChEBI → set(KEGG) from a flat ChEBI→KEGG map (string or list values).
    Duplicate ChEBI keys in the source are merged.
    """
    merged: Dict[str, Set[str]] = defaultdict(set)
    for chebi, val in raw.items():
        if chebi is None:
            continue
        cid = str(chebi).strip()
        if not cid:
            continue
        merged[cid].update(_coerce_kegg_values(val))
    return {k: set(v) for k, v in merged.items()}


def _chebi_lookup_keys(chebi_id: str) -> Iterator[str]:
    """Try common key variants for cross-reference dicts."""
    c = chebi_id.strip()
    if not c:
        return
    yield c
    upper = c.upper()
    if upper.startswith("CHEBI:"):
        num = c.split(":", 1)[-1]
        if num.isdigit():
            yield num
            yield f"CHEBI:{num}"
    elif c.isdigit():
        yield f"CHEBI:{c}"


def kegg_ids_for_chebi_term(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
) -> Set[str]:
    """Direct KEGG compounds for this ChEBI term only (no ontology walk)."""
    for key in _chebi_lookup_keys(chebi_id):
        if key in chebi_to_kegg:
            return _coerce_kegg_values(chebi_to_kegg[key])
    return set()


def normalize_chebi(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
) -> Set[str]:
    """
    KEGG compound IDs reachable from a ChEBI term at a given relaxation level.

    Level 0: only direct ChEBI→KEGG hits.
    Level L≥1: union of KEGG hits over this term and is_a ancestors within
    min(L, max_depth) hops.
    """
    if level <= 0:
        return kegg_ids_for_chebi_term(chebi_id, chebi_to_kegg)

    hop_cap = max(0, min(int(level), int(max_depth)))
    ancestors = get_ancestors(chebi_id, parent_map, depth=hop_cap)
    expanded = {chebi_id} | ancestors

    kegg_ids: Set[str] = set()
    for cid in expanded:
        kegg_ids.update(kegg_ids_for_chebi_term(cid, chebi_to_kegg))
    return kegg_ids


def normalize_reaction(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
    *,
    as_union: bool = False,
) -> Union[Set[str], Dict[str, Set[str]]]:
    """
    Map each ChEBI in chebi_metabolites to relaxed KEGG compound sets.

    Args:
        chebi_metabolites: ChEBI IDs participating in one reaction (any order).
        as_union: If True, return the union of all per-metabolite sets; if False,
            return a dict ChEBI → set(KEGG).

    Returns:
        Dict mapping each distinct input ChEBI string to KEGG IDs, or a single
        set if as_union is True.
    """
    per: Dict[str, Set[str]] = {}
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c:
            continue
        per[c] = normalize_chebi(c, chebi_to_kegg, parent_map, level=level, max_depth=max_depth)
    if as_union:
        out: Set[str] = set()
        for s in per.values():
            out.update(s)
        return out
    return per


def progressive_normalization(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    max_level: int = 2,
    max_depth: int = 2,
) -> Tuple[Set[str], int]:
    """
    Try relaxation levels 0 … max_level until at least one KEGG ID is found.

    Returns:
        (kegg_ids, level_used). If nothing matches at any level, returns
        (empty set, max_level).
    """
    for level in range(max_level + 1):
        kegg_ids = normalize_chebi(
            chebi_id, chebi_to_kegg, parent_map, level=level, max_depth=max_depth
        )
        if kegg_ids:
            return kegg_ids, level
    return set(), max_level


_CACHED_PARENT_MAP: Optional[Dict[str, Set[str]]] = None
_CACHED_PARENT_MAP_SOURCE: Optional[str] = None


def load_chebi_parent_map(
    *,
    obo_path: Optional[Union[str, Path]] = None,
    gz_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> Dict[str, Set[str]]:
    """
    Load ChEBI is_a parent map from gzipped JSON (list values) or from OBO.

    Search order: explicit gz_path → data_dir/chebi_parent_map.json.gz →
    explicit obo_path → data_dir/chebi.obo.
    """
    global _CACHED_PARENT_MAP, _CACHED_PARENT_MAP_SOURCE

    base = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data" / "chebi"
    gz = Path(gz_path) if gz_path else base / "chebi_parent_map.json.gz"
    obo = Path(obo_path) if obo_path else base / "chebi.obo"

    chosen: Optional[Path] = None
    if gz_path:
        chosen = Path(gz_path)
    elif gz.exists():
        chosen = gz
    elif obo_path:
        chosen = Path(obo_path)
    elif obo.exists():
        chosen = obo

    if chosen is None or not chosen.exists():
        raise FileNotFoundError(
            f"No ChEBI parent source found. Tried gz={gz} and obo={obo}. "
            "Place chebi.obo or chebi_parent_map.json.gz under data/chebi/."
        )

    src_key = str(chosen.resolve())
    if use_cache and _CACHED_PARENT_MAP is not None and _CACHED_PARENT_MAP_SOURCE == src_key:
        return _CACHED_PARENT_MAP

    if chosen.suffix == ".gz" or str(chosen).endswith(".json.gz"):
        with gzip.open(chosen, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        parent_map = {
            str(k): set(v) if not isinstance(v, set) else set(v)
            for k, v in raw.items()
        }
    else:
        parent_map = dict(parse_chebi_obo(chosen))

    if use_cache:
        _CACHED_PARENT_MAP = parent_map
        _CACHED_PARENT_MAP_SOURCE = src_key

    logger.info("Loaded ChEBI parent map (%d terms) from %s", len(parent_map), chosen)
    return parent_map


def build_kegg_mapping_dataframe(
    species_ids: Iterable[str],
    species_to_chebi: Mapping[str, str],
    relax_level: Mapping[str, int],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    max_ancestor_depth: int = 2,
) -> Any:
    """
    Long-form DataFrame columns [id, KEGG_ID] for map_reactions_to_kegg.

    One row per (species, KEGG) pair; duplicate indices are OK for the
    existing lookup logic in map_metabolites_to_kegg.
    """
    import pandas as pd

    rows: List[Dict[str, str]] = []
    for sid in species_ids:
        chebi = species_to_chebi.get(sid)
        if not chebi or not str(chebi).strip():
            continue
        lvl = int(relax_level.get(sid, 0))
        keggs = normalize_chebi(
            str(chebi).strip(),
            chebi_to_kegg,
            parent_map,
            level=lvl,
            max_depth=max_ancestor_depth,
        )
        if not keggs:
            # Omit species with no KEGG hit so map_reactions_to_kegg skips them (unmapped).
            continue
        for k in sorted(keggs):
            rows.append({"id": sid, "KEGG_ID": k})
    if not rows:
        return pd.DataFrame(columns=["id", "KEGG_ID"])
    return pd.DataFrame(rows)


def detect_unmapped_species_ids(
    species_ids: Iterable[str],
    species_to_chebi: Mapping[str, str],
    relax_level: Mapping[str, int],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    max_ancestor_depth: int = 2,
) -> Set[str]:
    """
    Species (model metabolite IDs) whose ChEBI annotation yields no KEGG compounds
    at that species' current relaxation level.
    """
    unmapped: Set[str] = set()
    for sid in species_ids:
        ch = str(species_to_chebi.get(sid, "")).strip()
        if not ch:
            continue
        lvl = int(relax_level.get(sid, 0))
        if not normalize_chebi(
            ch, chebi_to_kegg, parent_map, level=lvl, max_depth=max_ancestor_depth
        ):
            unmapped.add(sid)
    return unmapped


def detect_unmapped_metabolites(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
) -> Set[str]:
    """
    ChEBI terms that have no KEGG compound mapping at the given relaxation level.

    Uses ``normalize_chebi`` so semantics match the main normalization pipeline.
    """
    unmapped: Set[str] = set()
    seen: Set[str] = set()
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if not normalize_chebi(c, chebi_to_kegg, parent_map, level=level, max_depth=max_depth):
            unmapped.add(c)
    return unmapped


def detect_problematic_metabolites(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    reaction_matcher: ReactionScoreMatcher,
    level: int = 0,
    max_depth: int = 2,
    threshold: float = 0.0,
) -> Set[str]:
    """
    ChEBI terms whose removal **increases** the reaction match score vs a fixed
    reference (leave-one-out on the ChEBI annotation).

    ``reaction_matcher`` must be provided by the integration layer. The
    contract is:

    - ``reaction_matcher(None)``: score for the full reaction fingerprint.
    - ``reaction_matcher(chebi_id)``: score when species carrying this ChEBI
      are omitted from substrate/product KEGG counters.

    ``chebi_to_kegg`` and ``parent_map`` are included for API symmetry with
    ``detect_unmapped_metabolites``; the matcher typically closes over the
    current mapping state so these may be unused here.

    Args:
        threshold: Minimum score gain (vs baseline) required to flag a term.
        level / max_depth: API symmetry with ``normalize_chebi`` / callers; the
            matcher closure from the pipeline encodes the active level.
        chebi_to_kegg / parent_map: API symmetry with ``detect_unmapped_metabolites``.
    """
    _ = (chebi_to_kegg, parent_map, level, max_depth)
    baseline = reaction_matcher(None)
    problematic: Set[str] = set()
    seen: Set[str] = set()
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if reaction_matcher(c) > baseline + threshold:
            problematic.add(c)
    return problematic


def select_metabolites_to_relax(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    reaction_matcher: Optional[ReactionScoreMatcher] = None,
    *,
    level: int = 0,
    max_depth: int = 2,
    score_threshold: float = 0.0,
    participant_species: Optional[Iterable[str]] = None,
    species_to_chebi: Optional[Mapping[str, str]] = None,
    relax_level: Optional[Mapping[str, int]] = None,
) -> Set[str]:
    """
    Union of ChEBI annotation terms to relax: unmapped and/or score-sensitive.

    Unmapped detection uses ``detect_unmapped_species_ids`` when
    ``participant_species``, ``species_to_chebi``, and ``relax_level`` are
    provided (per-species relaxation levels). Otherwise uses
    ``detect_unmapped_metabolites`` with a single ``level`` for all ChEBI ids.

    Score-sensitive detection runs only if ``reaction_matcher`` is not None
    (see ``detect_problematic_metabolites``).
    """
    if (
        participant_species is not None
        and species_to_chebi is not None
        and relax_level is not None
    ):
        um_s = detect_unmapped_species_ids(
            participant_species,
            species_to_chebi,
            relax_level,
            chebi_to_kegg,
            parent_map,
            max_ancestor_depth=max_depth,
        )
        unmapped = {
            str(species_to_chebi[s]).strip()
            for s in um_s
            if str(species_to_chebi.get(s, "")).strip()
        }
    else:
        unmapped = detect_unmapped_metabolites(
            chebi_metabolites, chebi_to_kegg, parent_map, level=level, max_depth=max_depth
        )

    if reaction_matcher is None:
        return unmapped

    score_sensitive = detect_problematic_metabolites(
        chebi_metabolites,
        chebi_to_kegg,
        parent_map,
        reaction_matcher,
        level=level,
        max_depth=max_depth,
        threshold=score_threshold,
    )
    return unmapped | score_sensitive


def select_relaxations_by_global_improvement(
    candidate_species: Iterable[str],
    relaxation_levels: Mapping[str, int],
    compute_global_score: Callable[[Mapping[str, int]], float],
    *,
    max_relax_level: int,
    delta_threshold: float = 0.0,
):
    """
    Select species to relax only when the global objective improves.

    Args:
        candidate_species: Species IDs eligible for one-step trial relaxation.
        relaxation_levels: Current species -> level mapping.
        compute_global_score: Callable that evaluates the full-model objective for a
            given relaxation-level mapping.
        max_relax_level: Upper bound for any species relaxation level.
        delta_threshold: Minimum strictly-positive improvement required.
    """
    to_relax: List[str] = []

    current_score = compute_global_score(relaxation_levels)

    for s in candidate_species:
        if int(relaxation_levels.get(s, 0)) >= int(max_relax_level):
            continue

        trial_levels = relaxation_levels.copy()
        trial_levels[s] = int(trial_levels.get(s, 0)) + 1

        new_score = compute_global_score(trial_levels)

        if float(new_score) - float(current_score) > float(delta_threshold):
            to_relax.append(s)

    return to_relax


def unified_reaction_objective(
    base_score: float,
    relaxation_levels: Optional[Mapping[str, Any]],
    *,
    lam: float = 0.1,
    max_relax_level: int = 1,
) -> float:
    """
    Single objective for ranking KEGG reaction matches: similarity minus relaxation penalty.

    ``base_score`` must be the raw similarity (unchanged). All ranking / top-k selection
    should use only this return value.

    Args:
        base_score: Raw reaction similarity (e.g. from ``score_model_against_kegg_reaction``).
        relaxation_levels: Entity id (e.g. model species id) → relaxation level. If None or
            empty, or ``lam == 0``, returns ``base_score`` unchanged.
        lam: Penalty weight λ.
        max_relax_level: Maximum allowed relaxation level used to normalize penalty
            to ``[0, 1]``.
    """
    if lam == 0:
        return float(base_score)
    if not relaxation_levels:
        return float(base_score)

    levels = [int(v) for v in relaxation_levels.values() if v is not None]
    if not levels:
        return float(base_score)

    # Max-only penalty (participant-count independent), normalized to [0, 1].
    max_lvl = max(1, int(max_relax_level))
    penalty = float(max(levels))
    normalized_penalty = penalty / float(max_lvl)
    return float(base_score) - float(lam) * normalized_penalty


def unified_reaction_objective_weighted(
    base_score: float,
    relaxation_levels: Optional[Mapping[str, Any]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    lam: float = 0.1,
    max_relax_level: int = 1,
) -> float:
    """
    Weighted penalty: aggregate ``weight(entity) * relaxation_level(entity)`` then apply λ.
    """
    if lam == 0:
        return float(base_score)
    if not relaxation_levels:
        return float(base_score)

    if weights is None:
        weights = {}

    terms: List[float] = []
    for ent, lvl in relaxation_levels.items():
        if lvl is None:
            continue
        w = float(weights.get(ent, 1.0))
        terms.append(w * float(int(lvl)))

    if not terms:
        return float(base_score)

    # Max-only weighted term, normalized to keep penalty bounded.
    max_lvl = max(1, int(max_relax_level))
    penalty = float(max(terms))
    normalized_penalty = penalty / float(max_lvl)
    return float(base_score) - float(lam) * normalized_penalty

def should_continue_iteration(
    current_best_score: float,
    previous_best_score: Optional[float],
    relaxation_levels: Mapping[str, Any],
    to_relax: Union[Set[str], Iterable[str]],
    *,
    score_tolerance: float = 1e-3,
) -> bool:
    """
    Whether to run another relaxation iteration after scoring.

    Continue while there are entities in ``to_relax`` **or** the aggregate penalized
    score still moves by at least ``score_tolerance`` vs ``previous_best_score``.

    Stop when ``to_relax`` is empty **and** either ``previous_best_score`` is still
    the initial sentinel (``None`` or ``-inf``, nothing to relax on first pass) **or**
    the score change is below tolerance (stable).

    Callers should initialize ``previous_best_score`` to ``float("-inf")`` before
    the loop and assign it to the current penalized aggregate at the end of each
    iteration that continues.

    ``relaxation_levels`` is accepted for API symmetry; default rule uses ``to_relax``.
    """
    _ = relaxation_levels
    need_relax = bool(to_relax)
    if need_relax:
        return True
    if previous_best_score is None or previous_best_score == float("-inf"):
        return False
    return abs(float(current_best_score) - float(previous_best_score)) >= float(
        score_tolerance
    )