"""
Reaction eligibility classification for scoring/coverage accounting.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pandas as pd

from ..database_search import (
    cancel_spectators,
    load_chebi2kegg_dict,
    score_model_against_kegg_reaction,
    _get_kegg_recommendations_rulebased,
)
from .hierarchy_relaxation import (
    build_kegg_mapping_dataframe,
    iter_chebi_for_species,
    load_chebi_child_map,
    load_chebi_parent_map,
    merge_chebi_to_kegg_mapping,
    normalize_chebi,
    select_relaxations_by_global_improvement,
    select_metabolites_to_relax,
    should_continue_iteration,
    unified_reaction_objective,
)

logger = logging.getLogger(__name__)


def normalize_reactions(model_reactions):
    """
    Prepare KEGG-mapped reaction dicts for comparison: drop cofactors and keep
    substrate/product multisets (Counters).

    **Not** ``hierarchy_relaxation.normalize_reaction``: that function maps
    ChEBI terms to (relaxed) KEGG compound ID sets using the ontology. This
    function assumes ``model_reactions`` already carry KEGG IDs in substrate /
    product lists.

    Args:
        model_reactions: List of reaction dictionaries
        
    Returns:
        List of normalized reaction dictionaries

    See Also:
        ``hierarchy_relaxation.normalize_reaction`` — ChEBI → KEGG compound sets.
    """
    normalized_reactions = []
    
    for rxn in model_reactions:
        subs = count_metabolites(rxn.get('substrates', []))
        prods = count_metabolites(rxn.get('products', []))
        
        normalized_reactions.append({
            'reaction_name_in_model': rxn.get('id', 'Unknown'),
            'substrate_counter': subs,
            'product_counter': prods,
        })
                
    return normalized_reactions
            
def count_metabolites(kegg_list):

    counter = Counter()

    for kegg_id in kegg_list:
        if kegg_id is None:
            continue  # skip unmapped
        if kegg_id:
            counter[kegg_id] += 1  # track stoichiometry
    return counter


def parse_metabolites(side: str) -> Counter:
    side = side.strip()
    if not side:
        return Counter()
    result = Counter()
    for term in side.split("+"):
        parts = term.strip().split()
        if len(parts) == 1:
            coeff = 1
            met = parts[0]
        else:
            try:
                coeff = float(parts[0])
            except ValueError:
                coeff = 1
                met = term.strip()
            else:
                met = parts[-1]
        met = met.lstrip("$")
        result[met] += coeff
    return result


def parse_reaction_equation(rxn_str: str) -> Tuple[Counter, Counter]:
    """
    Parse a reaction equation string into reactant and product metabolite Counters.

    Same rules as ``map_reactions_to_kegg`` (``+`` terms, optional stoichiometry).
    """
    if "=>" in rxn_str or "->" in rxn_str:
        lhs, rhs = re.split(r"=>|->", rxn_str)
    else:
        return Counter(), Counter()

    reactants = parse_metabolites(lhs)
    products = parse_metabolites(rhs)
    return reactants, products


def collect_species_ids_from_rxn_list(rxn_list: List[str], spectators: bool = False) -> Set[str]:
    """All metabolite species IDs appearing in ``rxn_list`` after optional spectator cancellation."""
    out: Set[str] = set()
    for rxn in rxn_list:
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn
        reactants, products = parse_reaction_equation(rxn_str)
        if not spectators:
            reactants, products = cancel_spectators(reactants, products)
        out.update(reactants.keys())
        out.update(products.keys())
    return out


def map_reactions_to_kegg(rxn_list: List[str], reaction_ids: List[str], id_df: pd.DataFrame, spectators=False) -> List[Dict[str, Any]]:
    """
    Map reaction strings to KEGG reaction identifiers.
    
    This function processes a list of reaction strings and maps the metabolites
    in each reaction to their corresponding KEGG IDs using the provided mapping DataFrame.
    
    Args:
        rxn_list: List of reaction strings in the format "id: reactants -> products"
        id_df: DataFrame with columns 'id' and 'KEGG_ID' mapping metabolite IDs to KEGG IDs
        
    Returns:
        List of dictionaries containing mapped reaction information:
        - id: Reaction identifier
        - reaction_string: Original reaction string
        - substrates: List of Counter objects with mapped substrate KEGG IDs and stoichiometry
        - products: List of Counter objects with mapped product KEGG IDs and stoichiometry
    """
    
    # Keep full rows so we can propagate relaxation metadata when available.
    id_lookup = id_df.copy()
    id_grouped = dict(list(id_lookup.groupby("id")))

    # Process each reaction
    output = []
       
    for idx, rxn in enumerate(rxn_list):
        # Extract reaction string (remove ID prefix if present)
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn

        # Parse reaction equation into reactants and products
        reactants, products = parse_reaction_equation(rxn_str)

        if not spectators: 
            # Stoichiometric cancellation -- eliminate specatators
            reactants, products = cancel_spectators(reactants, products)

        # Map metabolite CHEBI IDs to KEGG IDs
        substrates_mapped = map_metabolites_to_kegg(reactants, id_lookup, _grouped=id_grouped)
        products_mapped = map_metabolites_to_kegg(products, id_lookup, _grouped=id_grouped)

        # Store mapped reaction
        output.append({
            "id": reaction_ids[idx],
            "reaction_string": rxn_str,
            "substrates": substrates_mapped,
            "products": products_mapped
        })
    
    return output

def map_metabolites_to_kegg(
        counter: Counter,
        mapping_df: pd.DataFrame,
        *,
        _grouped: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Counter]:
        """
        Map metabolite IDs to KEGG IDs while preserving stoichiometry.
        
        For each metabolite in the counter, finds all possible KEGG IDs and
        generates all possible combinations of mappings.
        
        Args:
            counter: Counter mapping metabolite IDs to stoichiometric coefficients
            mapping_df: DataFrame mapping metabolite IDs to KEGG IDs (+ optional metadata)
            _grouped: Pre-computed ``mapping_df.groupby("id")`` dict for hot-path callers.
            
        Returns:
            List of Counter objects representing all possible KEGG ID mappings
        """
        has_relax_cols = "direction" in mapping_df.columns and "distance" in mapping_df.columns
        grouped = _grouped if _grouped is not None else dict(list(mapping_df.groupby("id")))

        id_choices = dict()
        for met, coeff in counter.items():
            try:
                met_rows = grouped.get(met)
                if met_rows is None or met_rows.empty:
                    raise KeyError(met)

                if has_relax_cols:
                    choices = []
                    for tup in met_rows.itertuples():
                        kid = tup.KEGG_ID
                        if pd.isna(kid) or str(kid).strip() == "":
                            continue
                        choices.append(
                            {
                                "kegg_id": str(kid).strip(),
                                "canonical_id": str(getattr(tup, "canonical_id", "")).strip(),
                                "direction": str(getattr(tup, "direction", "exact")).strip(),
                                "distance": int(getattr(tup, "distance", 0) or 0),
                            }
                        )
                else:
                    choices = [str(kid).strip() for kid in met_rows["KEGG_ID"].tolist() if str(kid).strip()]

                id_choices[met] = {
                    'species_id': met,
                    'coeff': coeff,
                    'candidates': choices
                }
            except (KeyError, IndexError):
                logger.debug(f"No KEGG mapping found for metabolite: {met}")
                continue
        
        if not id_choices:
            return []
                       
        return id_choices


def _participant_species_from_normalized_reaction(nr: Dict[str, Any]) -> Set[str]:
    s: Set[str] = set()
    for side in ("substrates", "products"):
        block = nr.get(side, {})
        if isinstance(block, dict):
            s.update(block.keys())
    return s


def _per_reaction_best_scores(match_results: List[Any]) -> Dict[str, Optional[float]]:
    """Per-reaction best eligible score.  ``None`` for non_mappable (excluded from mean)."""
    best_by_rxn: Dict[str, float] = {}
    classification_by_rxn: Dict[str, str] = {}
    ambiguous_default_by_rxn: Dict[str, float] = {}
    failed_default_by_rxn: Dict[str, float] = {}
    for rec in match_results:
        rid = rec.id
        meta = getattr(rec, "metadata", None) or {}
        rtype = str(meta.get("reaction_type", "mappable"))
        classification_by_rxn[rid] = rtype
        if rtype == "ambiguous_mapping":
            ambiguous_default_by_rxn[rid] = float(meta.get("ambiguous_default_score", 0.0))
        if rtype == "failed_mapping":
            failed_default_by_rxn[rid] = float(meta.get("failed_default_score", 0.0))
        if rtype != "mappable":
            continue
        if not rec.match_score:
            continue
        sc = float(rec.match_score[0])
        prev = best_by_rxn.get(rid)
        if prev is None or sc > prev:
            best_by_rxn[rid] = sc

    out: Dict[str, Optional[float]] = {}
    for rid, rtype in classification_by_rxn.items():
        if rtype == "non_mappable":
            out[rid] = None
            continue
        if rtype == "ambiguous_mapping":
            out[rid] = float(ambiguous_default_by_rxn.get(rid, 0.0))
            continue
        if rtype == "failed_mapping":
            out[rid] = float(failed_default_by_rxn.get(rid, 0.0))
            continue
        if rid in best_by_rxn:
            out[rid] = float(best_by_rxn[rid])
    return out


def _aggregate_best_penalized_scores(match_results: List[Any]) -> float:
    """
    Mean penalized score over score-eligible reactions:
    - include mappable reactions by best penalized match
    - include ambiguous_mapping reactions with default low score
    - exclude non_mappable reactions
    """
    per_rxn = _per_reaction_best_scores(match_results)
    scored = [v for v in per_rxn.values() if v is not None]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def _aggregate_from_per_reaction_scores(per_rxn: Dict[str, Optional[float]]) -> float:
    """Mean of non-None per-reaction scores (same semantics as ``_aggregate_best_penalized_scores``)."""
    scored = [v for v in per_rxn.values() if v is not None]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def _build_species_to_rxn_indices(
    rxn_list: List[str], spectators: bool = False,
) -> Dict[str, Set[int]]:
    """Map each species ID to the set of reaction-list indices it participates in."""
    species_to_indices: Dict[str, Set[int]] = {}
    for i, rxn in enumerate(rxn_list):
        rxn_str = rxn.split(":", 1)[1] if ":" in rxn else rxn
        reactants, products = parse_reaction_equation(rxn_str)
        if not spectators:
            reactants, products = cancel_spectators(reactants, products)
        for sid in set(reactants.keys()) | set(products.keys()):
            species_to_indices.setdefault(sid, set()).add(i)
    return species_to_indices


def _reaction_coverage_stats(match_results: List[Any]) -> Dict[str, Any]:
    """Coverage counts and percentages by reaction classification."""
    reaction_type_by_id: Dict[str, str] = {}
    for rec in match_results:
        rid = rec.id
        meta = getattr(rec, "metadata", None) or {}
        reaction_type_by_id[rid] = str(meta.get("reaction_type", "mappable"))
    total = len(reaction_type_by_id)
    counts = {
        "mappable": 0,
        "ambiguous_mapping": 0,
        "non_mappable": 0,
    }
    for rtype in reaction_type_by_id.values():
        if rtype in counts:
            counts[rtype] += 1
    successful_mapped = counts["mappable"]
    denom = float(total) if total else 1.0
    return {
        "counts": counts,
        "percent_mappable": round(100.0 * counts["mappable"] / denom, 2),
        "percent_successfully_mapped": round(100.0 * max(successful_mapped, 0) / denom, 2),
        "percent_ambiguous_mapping": round(100.0 * counts["ambiguous_mapping"] / denom, 2),
        "percent_non_mappable": round(100.0 * counts["non_mappable"] / denom, 2),
    }


def _top_kegg_reference_from_matches(match_results: List[Any]) -> Dict[str, str]:
    """Best-scoring KEGG reaction id per model reaction id from split recommendations."""
    best: Dict[str, Tuple[str, float]] = {}
    for rec in match_results:
        if not rec.candidates or not rec.match_score:
            continue
        rid = rec.id
        kid = rec.candidates[0]
        sc = float(rec.match_score[0])
        prev = best.get(rid)
        if prev is None or sc > prev[1]:
            best[rid] = (kid, sc)
    return {rid: t[0] for rid, t in best.items()}


def _species_ids_for_chebi_relax_targets(
    chebi_hit: Set[str],
    species_ids: Iterable[str],
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    max_relax_level: int,
) -> Set[str]:
    """Map ChEBI ids from ``select_metabolites_to_relax`` to relaxable model species ids."""
    ch_norm = {str(c).strip() for c in chebi_hit if str(c).strip()}
    out: Set[str] = set()
    for sid in species_ids:
        if int(relax_level.get(sid, 0)) >= max_relax_level:
            continue
        for ch in iter_chebi_for_species(species_to_chebi, str(sid)):
            if str(ch).strip() in ch_norm:
                out.add(sid)
                break
    return out


def _leave_one_out_penalized_matcher_factory(
    nr: Dict[str, Any],
    ref_kegg: str,
    part: Set[str],
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    merged_kegg: Mapping[str, Set[str]],
    parent_map: Mapping[str, Set[str]],
    max_ancestor_depth: int,
    cofactors: Set[str],
    spectators: bool,
    penalty_lam: float,
    max_relax_level: int,
):
    """
    Leave-one-ChEBI-out matcher returning **penalized** objective only (no raw scores
    in control flow that uses this closure).
    """
    reaction_relax_levels = {sid: int(relax_level.get(sid, 0) or 0) for sid in part}

    def reaction_matcher(exclude_chebi: Optional[str]) -> float:
        sub_c = _kegg_counters_from_normalized_block(
            nr.get("substrates"),
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth,
            exclude_chebi,
        )
        prod_c = _kegg_counters_from_normalized_block(
            nr.get("products"),
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth,
            exclude_chebi,
        )
        base = score_model_against_kegg_reaction(
            sub_c,
            prod_c,
            ref_kegg,
            cofactors_to_ignore=cofactors,
            spectators=spectators,
        )[0]
        return unified_reaction_objective(
            base,
            reaction_relax_levels if reaction_relax_levels else None,
            lam=penalty_lam,
            max_relax_level=max_relax_level,
        )

    return reaction_matcher


def _kegg_counters_from_normalized_block(
    block: Any,
    species_to_chebi: Mapping[str, Any],
    relax_level: Mapping[str, int],
    merged_kegg: Mapping[str, Set[str]],
    parent_map: Mapping[str, Set[str]],
    max_ancestor_depth: int,
    exclude_chebi: Optional[str],
) -> Counter:
    """
    Rebuild KEGG compound counters for one side of a normalized reaction using
    ``normalize_chebi`` at each species' current relaxation level.

    When ``exclude_chebi`` is set, species annotated with that ChEBI are omitted
    (leave-one-ChEBI-out for problematic-metabolite detection).
    """
    ctr: Counter = Counter()
    if not isinstance(block, dict):
        return ctr
    ex = (exclude_chebi or "").strip()
    for met_id, v in block.items():
        if not isinstance(v, dict):
            continue
        coeff = float(v.get("coeff", 1))
        lvl = int(relax_level.get(met_id, 0))
        keggs_union: Set[str] = set()
        for ch in iter_chebi_for_species(species_to_chebi, str(met_id)):
            c = str(ch).strip()
            if not c:
                continue
            if ex and c == ex:
                continue
            keggs_union.update(
                normalize_chebi(
                    c, merged_kegg, parent_map, level=lvl, max_depth=max_ancestor_depth
                )
            )
        for kid in keggs_union:
            if kid:
                ctr[kid] += coeff
    return ctr


def _species_to_chebi_from_recommendations(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build species id → list of distinct ChEBI annotation strings.

    All recommendation rows are retained (no single-row-per-id collapse). Order
    follows ``match_score`` descending when that column exists, so higher-ranked
    candidates appear first in each list.
    """
    if "match_score" in df.columns:
        sorted_df = df.sort_values("match_score", ascending=False)
    else:
        sorted_df = df
    out: Dict[str, List[str]] = {}
    for _, row in sorted_df.iterrows():
        sid = str(row["id"]).strip()
        ann = str(row["annotation"]).strip()
        if not sid or not ann:
            continue
        out.setdefault(sid, [])
        if ann not in out[sid]:
            out[sid].append(ann)
    return out


def map_reactions_to_kegg_with_relaxation(
    rxn_list: List[str],
    reaction_ids: List[str],
    species_recommendations_df: pd.DataFrame,
    *,
    parent_map: Optional[Mapping[str, Set[str]]] = None,
    chebi_to_kegg: Optional[Mapping[str, Any]] = None,
    obo_path: Optional[str] = None,
    parent_map_gz: Optional[str] = None,
    spectators: bool = False,
    max_relax_level: int = 2,
    max_ancestor_depth: int = 2,
    max_descendant_depth: Optional[int] = None,
    score_gain_threshold: float = 0.0,
    score_tolerance: float = 1e-3,
    max_relaxation_rounds: int = 8,
    cofactors_to_ignore: Optional[Set[str]] = None,
    top_k: Optional[int] = None,
    penalty_lam: float = 0.1,
    run_matching: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, int]]:
    """
    Single iterative loop: normalize → penalized KEGG matching → relax targets → converge.

    **Per iteration**

    1. Build ``normalized_reactions`` via ``map_reactions_to_kegg`` (ChEBI→KEGG at
       current ``relax_level`` per species).
    2. ``_get_kegg_recommendations_rulebased`` computes raw similarity internally and
       ranks only on ``unified_reaction_objective``; aggregate ``score`` for control
       flow is the mean best penalized score per model reaction.
    3. ``select_metabolites_to_relax`` (unmapped ∪ score-sensitive) yields ChEBI terms;
       these map back to **species ids** in ``to_relax`` only (no global relaxation).
    4. ``should_continue_iteration`` stops when ``to_relax`` is empty and the penalized
       aggregate is stable vs ``previous_best_score`` (initialized to ``-inf``).
    5. Increment ``relax_level`` only for species in ``to_relax`` (capped at
       ``max_relax_level``).

    Args:
        max_descendant_depth: Maximum downward traversal depth used for
            relaxation-aware ChEBI expansion. If None, defaults to
            ``max_ancestor_depth`` for backward-compatible behavior.
        rxn_list: Reaction strings ``"RID: lhs -> rhs"`` as for map_reactions_to_kegg.
        species_recommendations_df: Must include ``id`` (species) and ``annotation``
            (ChEBI ID). Optional ``match_score`` ranks rows; **all** distinct
            ``annotation`` values per ``id`` are kept (not only the top row).
        parent_map: Optional precomputed child→parents map; otherwise loaded via
            ``load_chebi_parent_map`` (gz or OBO under data/chebi/).
        chebi_to_kegg: Optional raw ChEBI→KEGG dict; defaults to
            ``load_chebi2kegg_dict()``.
        score_gain_threshold: Minimum score increase (leave-one-out minus baseline)
            required to flag a ChEBI term as score-sensitive.
        score_tolerance: Stop when there is nothing to relax and the mean best penalized
            score per reaction changes by less than this vs the previous iteration.
        run_matching: If False, performs a single mapping pass and returns an empty
            match list (no refinement).

    Returns:
        (normalized_reactions, kegg_match_results, species_relax_level_by_id)
    """

    species_to_rxn_idx = _build_species_to_rxn_indices(rxn_list, spectators=spectators)

    # Mutable state for the incremental scorer (reset each outer iteration).
    _incr_base_levels: Dict[str, int] = {}
    _incr_base_per_rxn: Dict[str, Optional[float]] = {}
    _incr_base_id_kegg_df: Optional[pd.DataFrame] = None

    def _set_incremental_baseline(
        levels: Mapping[str, int],
        per_rxn_scores: Dict[str, Optional[float]],
        base_df: pd.DataFrame,
    ) -> None:
        nonlocal _incr_base_levels, _incr_base_per_rxn, _incr_base_id_kegg_df
        _incr_base_levels = dict(levels)
        _incr_base_per_rxn = dict(per_rxn_scores)
        _incr_base_id_kegg_df = base_df

    def compute_global_score(levels: Mapping[str, int]) -> float:
        """Incremental global objective — only rescores reactions affected by changed species."""
        changed = {
            s for s in set(levels) | set(_incr_base_levels)
            if levels.get(s, 0) != _incr_base_levels.get(s, 0)
        }

        if not changed or _incr_base_id_kegg_df is None:
            return float(_aggregate_from_per_reaction_scores(_incr_base_per_rxn))

        affected_idx: Set[int] = set()
        for s in changed:
            affected_idx |= species_to_rxn_idx.get(s, set())

        if not affected_idx:
            return float(_aggregate_from_per_reaction_scores(_incr_base_per_rxn))

        # Rebuild mapping DF only for changed species, keep rest from baseline.
        trial_partial_df = build_kegg_mapping_dataframe(
            changed,
            species_to_chebi,
            levels,
            merged_kegg,
            parent_map,
            max_ancestor_depth=max_ancestor_depth,
            child_map=child_map,
            max_descendant_depth=down_depth,
        )
        trial_id_kegg_df = pd.concat(
            [_incr_base_id_kegg_df[~_incr_base_id_kegg_df['id'].isin(changed)], trial_partial_df],
            ignore_index=True,
        )

        # Re-normalize only affected reactions.
        affected_sorted = sorted(affected_idx)
        affected_rxn_list = [rxn_list[i] for i in affected_sorted]
        affected_rxn_ids = [reaction_ids[i] for i in affected_sorted]
        trial_normalized_affected = map_reactions_to_kegg(
            affected_rxn_list, affected_rxn_ids, trial_id_kegg_df, spectators=spectators
        )

        # Re-score only affected reactions.
        trial_match_affected = _get_kegg_recommendations_rulebased(
            trial_normalized_affected,
            cofactors_to_ignore=cofactors,
            top_k=top_k,
            spectators=spectators,
            relaxation_levels_by_entity=levels,
            penalty_lam=penalty_lam,
            max_relax_level=max_relax_level,
            species_to_chebi=species_to_chebi,
            parent_map=parent_map,
            child_map=child_map,
            chebi_to_kegg=merged_kegg,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=down_depth,
        )

        # Merge: baseline scores for unaffected reactions + new scores for affected ones.
        affected_rxn_id_set = set(affected_rxn_ids)
        merged_per_rxn = {
            rid: sc for rid, sc in _incr_base_per_rxn.items()
            if rid not in affected_rxn_id_set
        }
        merged_per_rxn.update(_per_reaction_best_scores(trial_match_affected))
        return float(_aggregate_from_per_reaction_scores(merged_per_rxn))


    if species_recommendations_df is None or species_recommendations_df.empty:
        return [], [], {}

    required_cols = {"id", "annotation"}
    if not required_cols.issubset(species_recommendations_df.columns):
        raise ValueError(
            f"species_recommendations_df must contain columns {required_cols}, "
            f"got {set(species_recommendations_df.columns)}"
        )

    df = species_recommendations_df.dropna(subset=["annotation"])
    df = df[df["annotation"].astype(str).str.strip() != ""]
    if df.empty:
        return [], [], {}

    species_to_chebi = _species_to_chebi_from_recommendations(df)

    if parent_map is None:
        parent_map = load_chebi_parent_map(obo_path=obo_path, gz_path=parent_map_gz)
    child_map = load_chebi_child_map(parent_map=parent_map)
    if chebi_to_kegg is None:
        chebi_to_kegg = load_chebi2kegg_dict()

    merged_kegg = merge_chebi_to_kegg_mapping(chebi_to_kegg)
    down_depth = int(max_ancestor_depth) if max_descendant_depth is None else int(max_descendant_depth)

    relax_level: Dict[str, int] = {sid: 0 for sid in species_to_chebi}

    cofactors = cofactors_to_ignore if cofactors_to_ignore is not None else set()
    normalized_reactions: List[Dict[str, Any]] = []
    match_results: List[Any] = []

    max_iterations = 1 if not run_matching else max(1, int(max_relaxation_rounds))
    previous_best_score: float = float("-inf")

    for _iteration in range(max_iterations):
        # --- Step 1: build normalized reactions ---
        id_kegg_df = build_kegg_mapping_dataframe(
            species_to_chebi.keys(),
            species_to_chebi,
            relax_level,
            merged_kegg,
            parent_map,
            max_ancestor_depth=max_ancestor_depth,
            child_map=child_map,
            max_descendant_depth=down_depth,
        )
        normalized_reactions = map_reactions_to_kegg(
            rxn_list, reaction_ids, id_kegg_df, spectators=spectators
        )

        if not run_matching:
            match_results = []
            break

        # --- Step 2: KEGG matching (raw similarity inside matcher; ranking = penalized only) ---
        match_results = _get_kegg_recommendations_rulebased(
            normalized_reactions,
            cofactors_to_ignore=cofactors,
            top_k=top_k,
            spectators=spectators,
            relaxation_levels_by_entity=relax_level,
            penalty_lam=penalty_lam,
            max_relax_level=max_relax_level,
            species_to_chebi=species_to_chebi,
            parent_map=parent_map,
            child_map=child_map,
            chebi_to_kegg=merged_kegg,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=down_depth,
        )
        score = _aggregate_best_penalized_scores(match_results)
        coverage = _reaction_coverage_stats(match_results)
        logger.info(f"Reaction coverage: {coverage}")

        # --- Step 3: build candidate species (unmapped + optionally problematic) ---
        participants_union: Set[str] = set()
        for nr in normalized_reactions:
            participants_union |= _participant_species_from_normalized_reaction(nr)
        if not participants_union:
            participants_union = collect_species_ids_from_rxn_list(rxn_list, spectators=spectators)
        participants_union &= set(species_to_chebi.keys())

        candidate_species: Set[str] = set()
        top_ref = _top_kegg_reference_from_matches(match_results)
        species_in_any_part: Set[str] = set()

        for nr in normalized_reactions:
            part = _participant_species_from_normalized_reaction(nr) & set(species_to_chebi.keys())
            if not part:
                continue
            species_in_any_part |= part
            chebi_union = sorted(
                {
                    c
                    for s in part
                    for c in iter_chebi_for_species(species_to_chebi, str(s))
                    if c
                }
            )
            if not chebi_union:
                continue

            ref_kegg = top_ref.get(nr.get("id"))
            if ref_kegg:
                matcher = _leave_one_out_penalized_matcher_factory(
                    nr,
                    ref_kegg,
                    part,
                    species_to_chebi,
                    relax_level,
                    merged_kegg,
                    parent_map,
                    max_ancestor_depth,
                    cofactors,
                    spectators,
                    penalty_lam,
                    max_relax_level,
                )
            else:
                matcher = None

            chebi_to_relax = select_metabolites_to_relax(
                chebi_union,
                merged_kegg,
                parent_map,
                matcher,
                score_threshold=score_gain_threshold,
                participant_species=part,
                species_to_chebi=species_to_chebi,
                relax_level=relax_level,
                max_depth=max_ancestor_depth,
            )
            candidate_species |= _species_ids_for_chebi_relax_targets(
                chebi_to_relax,
                part,
                species_to_chebi,
                relax_level,
                max_relax_level,
            )

        orphan = participants_union - species_in_any_part
        if orphan:
            orch_chebi = sorted(
                {
                    c
                    for s in orphan
                    for c in iter_chebi_for_species(species_to_chebi, str(s))
                    if c
                }
            )
            if orch_chebi:
                chebi_orphan = select_metabolites_to_relax(
                    orch_chebi,
                    merged_kegg,
                    parent_map,
                    None,
                    participant_species=orphan,
                    species_to_chebi=species_to_chebi,
                    relax_level=relax_level,
                    max_depth=max_ancestor_depth,
                )
                candidate_species |= _species_ids_for_chebi_relax_targets(
                    chebi_orphan,
                    orphan,
                    species_to_chebi,
                    relax_level,
                    max_relax_level,
                )

        # Seed incremental scorer with this iteration's baseline before trialing.
        _set_incremental_baseline(relax_level, _per_reaction_best_scores(match_results), id_kegg_df)

        # Global objective gate: relax only species that improve full-model score.
        to_relax: Set[str] = set(
            select_relaxations_by_global_improvement(
                sorted(candidate_species),
                relax_level,
                compute_global_score,
                max_relax_level=max_relax_level,
                delta_threshold=score_gain_threshold,
            )
        )

        # --- Step 4: convergence (penalized score + relaxation state) ---
        if not should_continue_iteration(
            score,
            previous_best_score,
            relax_level,
            to_relax,
            score_tolerance=score_tolerance,
        ):
            previous_best_score = score
            break

        previous_best_score = score

        # --- Step 5: apply relaxation (only entities in to_relax) ---
        for entity in to_relax:
            relax_level[entity] = min(relax_level.get(entity, 0) + 1, max_relax_level)

    return normalized_reactions, match_results, relax_level