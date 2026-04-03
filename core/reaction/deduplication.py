"""
Reaction deduplication by canonical signatures.

This module is intentionally pure and deterministic: it does not depend on
input ordering and only merges reactions that are chemically identical after
species normalization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


logger = logging.getLogger(__name__)

CanonicalSignature = Tuple[Tuple[str, int], ...]


@dataclass
class Reaction:
    """
    Canonical reaction representation for deduplication output.

    Notes:
    - `reactants` and `products` are expanded lists (stoichiometric counts
      repeated as separate entries) using canonical species IDs.
    - `score` is the aggregated score for the merged group.
    """

    reaction_id: str
    reaction_equation: str
    reactants: List[str]
    products: List[str]
    score: float
    metadata: Optional[Dict[str, Any]] = None


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    """
    Retrieve a field from either a dict-like object or an attribute-based object.
    """

    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _normalize_species(
    species: Sequence[str],
    species_to_canonical: Mapping[str, str],
    *,
    log_missing: bool = True,
    missing_log_limit: int = 20,
) -> List[str]:
    """
    Map each raw species to a canonical equivalence-class ID.

    If a species is missing from the mapping, the raw species is kept and the
    missing key is logged internally.
    """

    out: List[str] = []
    missing_logged: set[str] = set()

    for s in species:
        raw = str(s).strip()
        if not raw:
            continue
        mapped = species_to_canonical.get(raw)
        if mapped is None or str(mapped).strip() == "":
            if log_missing and raw not in missing_logged and len(missing_logged) < missing_log_limit:
                logger.debug(
                    "Species missing from species_to_canonical; keeping raw value: %r", raw
                )
                missing_logged.add(raw)
            out.append(raw)
        else:
            out.append(str(mapped).strip())
    return out


def _build_multiset(items: Iterable[str]) -> Counter:
    """
    Build a multiset Counter from items (duplicates count towards multiplicity).
    """

    ctr: Counter = Counter()
    for x in items:
        key = str(x).strip()
        if key:
            ctr[key] += 1
    return ctr


def _signature_from_counters(reactant_ctr: Counter, product_ctr: Counter) -> CanonicalSignature:
    """
    Deterministic canonical signature:
    - represent reaction as unordered multisets of reactants and products
    - serialize sorted reactant items + sorted product items as tuple of pairs
    """

    reactant_items = sorted(reactant_ctr.items(), key=lambda kv: kv[0])
    product_items = sorted(product_ctr.items(), key=lambda kv: kv[0])
    sig_list: List[Tuple[str, int]] = []
    sig_list.extend((k, int(v)) for k, v in reactant_items if int(v) != 0)
    sig_list.extend((k, int(v)) for k, v in product_items if int(v) != 0)
    return tuple(sig_list)


def _build_signature(
    reactants: Sequence[str],
    products: Sequence[str],
    *,
    collapse_reversible: bool = False,
) -> CanonicalSignature:
    """
    Build canonical signature from already-normalized species identifiers.

    - unordered multisets for reactants/products (stoichiometry via multiplicity)
    - optional reversible collapse via lexicographically smallest signature
    """

    r_ctr = _build_multiset(reactants)
    p_ctr = _build_multiset(products)
    forward = _signature_from_counters(r_ctr, p_ctr)
    if not collapse_reversible:
        return forward
    reverse = _signature_from_counters(p_ctr, r_ctr)
    return forward if forward <= reverse else reverse


def _signature_for_reaction(
    reactants: Sequence[str],
    products: Sequence[str],
    *,
    species_to_canonical: Mapping[str, str],
    collapse_reversible: bool = False,
) -> CanonicalSignature:
    """
    Build canonical signature after species normalization.
    """
    norm_reactants = _normalize_species(reactants, species_to_canonical, log_missing=True)
    norm_products = _normalize_species(products, species_to_canonical, log_missing=True)
    return _build_signature(
        norm_reactants,
        norm_products,
        collapse_reversible=collapse_reversible,
    )


def _expand_from_counter(counter: Counter) -> List[str]:
    """
    Expand a multiset Counter back to a deterministic sorted list.
    """

    out: List[str] = []
    for sp, count in sorted(counter.items(), key=lambda kv: kv[0]):
        out.extend([sp] * int(count))
    return out


def _equation_from_sides(reactants: Sequence[str], products: Sequence[str]) -> str:
    """
    Build a deterministic reaction-equation string from expanded canonical sides.
    """

    def join_side(items: Sequence[str]) -> str:
        # Inputs are expanded lists; joining preserves multiplicity.
        return " + ".join(items) if items else ""

    return f"{join_side(reactants)} -> {join_side(products)}"


def _logsumexp(values: Sequence[float]) -> float:
    """
    Numerically stable log-sum-exp for real numbers.

    Returns -inf if all values are -inf.
    """

    if not values:
        return float("-inf")

    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    total = 0.0
    for v in values:
        total += math.exp(v - m)
    return m + math.log(total)


def _aggregate_scores(
    scores: Sequence[float],
    *,
    score_aggregation: str = "logsumexp",
    scores_are_log_probs: bool = True,
) -> float:
    """
    Aggregate group scores.

    - `max`: maximum score
    - `logsumexp`: log(sum(exp(score_i))) if `scores_are_log_probs=True`
      (default). If scores are probabilities (`scores_are_log_probs=False`),
      this computes log(sum(score_i)) when possible.
    """

    if not scores:
        return float("-inf")

    agg = str(score_aggregation).lower().strip()
    if agg == "max":
        return float(max(scores))
    if agg != "logsumexp":
        raise ValueError(f"Unsupported score_aggregation={score_aggregation!r}")

    if scores_are_log_probs:
        return float(_logsumexp([float(s) for s in scores]))

    # Interpret scores as probabilities. Prefer a safe log domain conversion:
    # log(sum(p_i)). If p_i are already in linear space, combining in log-space
    # yields log(p_total), which preserves probabilistic meaning.
    probs = [float(s) for s in scores]
    if any(p < 0 for p in probs):
        raise ValueError("scores_are_log_probs=False but at least one score < 0")
    if any(p == 0 for p in probs) and all(p == 0 for p in probs):
        return float("-inf")
    s = sum(p for p in probs if p > 0)
    if s <= 0:
        return float("-inf")
    return float(math.log(s))


def _merge_group(
    sig: CanonicalSignature,
    group: List[Dict[str, Any]],
    *,
    species_to_canonical: Mapping[str, str],
    collapse_reversible: bool,
    score_aggregation: str,
    scores_are_log_probs: bool,
) -> Tuple[Reaction, List[str], float]:
    """
    Merge a single signature group into one canonical `Reaction`.

    Returns:
    - canonical_reaction
    - merged_ids (sorted)
    - aggregated_score
    """

    merged_ids = sorted([g["reaction_id"] for g in group])
    rep_id = merged_ids[0]
    rep_entry = next(g for g in group if g["reaction_id"] == rep_id)

    # Aggregate scores deterministically (sort by reaction_id).
    score_by_id = {g["reaction_id"]: float(g["score"]) for g in group}
    ordered_scores = [score_by_id[rid] for rid in merged_ids]
    aggregated = _aggregate_scores(
        ordered_scores,
        score_aggregation=score_aggregation,
        scores_are_log_probs=scores_are_log_probs,
    )

    # Reconstruct canonical representative structure using the signature.
    norm_reactants = _normalize_species(rep_entry["reactants"], species_to_canonical, log_missing=True)
    norm_products = _normalize_species(rep_entry["products"], species_to_canonical, log_missing=True)
    r_ctr = _build_multiset(norm_reactants)
    p_ctr = _build_multiset(norm_products)

    if collapse_reversible:
        forward = _signature_from_counters(r_ctr, p_ctr)
        reverse = _signature_from_counters(p_ctr, r_ctr)
        if sig == reverse and sig != forward:
            # Representative needed the reversed canonical orientation.
            r_ctr, p_ctr = p_ctr, r_ctr

    canon_reactants = _expand_from_counter(r_ctr)
    canon_products = _expand_from_counter(p_ctr)
    canon_eq = _equation_from_sides(canon_reactants, canon_products)

    canonical_reaction = Reaction(
        reaction_id=rep_id,
        reaction_equation=canon_eq,
        reactants=canon_reactants,
        products=canon_products,
        score=aggregated,
        metadata={
            "merged_ids": merged_ids,
            "canonical_signature": sig,
            **(rep_entry["metadata"] or {}),
        }
        if rep_entry.get("metadata") is not None
        else {"merged_ids": merged_ids, "canonical_signature": sig},
    )
    return canonical_reaction, merged_ids, aggregated


def deduplicate_reactions(
    reactions: Sequence[Any],
    species_to_canonical: Mapping[str, str],
    *,
    collapse_reversible: bool = False,
    score_aggregation: str = "logsumexp",
    scores_are_log_probs: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[CanonicalSignature, List[str]]]:
    """
    Deduplicate reactions by canonical multiset signature after species normalization.

    Input reaction objects must provide:
    - reaction_id (str)
    - reactants (List[str])
    - products (List[str])
    - score (float)
    Optional:
    - reaction_equation (str) and metadata

    Returns:
    - deduplicated_reactions: list of dicts with keys:
        - canonical_reaction: Reaction
        - merged_ids: List[str]
        - score: float (aggregated)
    - merge_map: Dict[canonical_signature, List[reaction_id]]
    """

    groups: MutableMapping[CanonicalSignature, List[Dict[str, Any]]] = defaultdict(list)

    # Normalize species and build signatures for grouping.
    for rxn in reactions:
        rid = _get_field(rxn, "reaction_id", "id", default=None)
        if rid is None:
            raise ValueError("Each reaction must have `reaction_id` (or `id`) field.")
        rid = str(rid)

        reactants = _get_field(rxn, "reactants", default=[])
        products = _get_field(rxn, "products", default=[])

        if reactants is None:
            reactants = []
        if products is None:
            products = []

        # Normalize signature after species normalization.
        sig = _signature_for_reaction(
            reactants,
            products,
            species_to_canonical=species_to_canonical,
            collapse_reversible=collapse_reversible,
        )

        score_val = _get_field(rxn, "score", default=float("-inf"))
        score = float(score_val) if score_val is not None else float("-inf")

        eq = _get_field(rxn, "reaction_equation", "equation", default=None)
        metadata = _get_field(rxn, "metadata", default=None)
        if metadata is not None and not isinstance(metadata, dict):
            metadata = {"metadata": metadata}

        groups[sig].append(
            {
                "reaction_id": rid,
                "reactants": list(reactants),
                "products": list(products),
                "score": score,
                "reaction_equation": eq,
                "metadata": metadata,
            }
        )

    # Build deterministic output.
    deduplicated_reactions: List[Dict[str, Any]] = []
    merge_map: Dict[CanonicalSignature, List[str]] = {}

    # Sort groups deterministically (by signature tuple).
    for sig in sorted(groups.keys()):
        group = groups[sig]
        canonical_reaction, merged_ids, aggregated = _merge_group(
            sig,
            group,
            species_to_canonical=species_to_canonical,
            collapse_reversible=collapse_reversible,
            score_aggregation=score_aggregation,
            scores_are_log_probs=scores_are_log_probs,
        )
        merge_map[sig] = merged_ids

        deduplicated_reactions.append(
            {
                "canonical_reaction": canonical_reaction,
                "merged_ids": merged_ids,
                "score": aggregated,
            }
        )

    return deduplicated_reactions, merge_map

