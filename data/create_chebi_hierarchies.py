import gzip, json
from rdkit import Chem  # :contentReference[oaicite:0]{index=0}
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

ALLOWED_RELATIONS = {
    "is_a",
    "has_part",
    "is_conjugate_acid_of",
    "is_conjugate_base_of",
    "has_functional_parent",
}




def build_equivalence_map(structure_map: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    """
    Groups ChEBI IDs by canonical InChIKey (optionally stereochemistry-stripped).
    """
    eq_map = defaultdict(set)

    for chebi_id, props in structure_map.items():
        inchikey = props.get("inchikey")
        if not inchikey:
            continue

        # canonicalization: ignore stereochemistry
        key = inchikey.split("-")[0]
        eq_map[key].add(chebi_id)

    return dict(eq_map)


def expand_chebi(
    seed_chebi_ids: Set[str],
    relation_map: Dict[str, List[Tuple[str, str]]],
    structure_map: Dict[str, Dict[str, str]],
    max_up_depth: int = 1,
    max_down_depth: int = 1,
) -> Tuple[Set[str], Set[str]]:
    """
    Returns:
        expanded_ids: all reachable IDs under bounded traversal
        equivalence_ids: closure under InChIKey equivalence
    """

    # ---------- build reverse graph (child map) ----------
    child_map = defaultdict(list)
    parent_map = defaultdict(list)

    for node, edges in relation_map.items():
        for rel, tgt in edges:
            if rel == "is_a":
                parent_map[node].append(tgt)
                child_map[tgt].append(node)
            elif rel in ALLOWED_RELATIONS:
                child_map[tgt].append(node)

    # ---------- upward expansion ----------
    def bfs_up(start_ids: Set[str]) -> Set[str]:
        visited = set(start_ids)
        q = deque([(n, 0) for n in start_ids])

        while q:
            node, depth = q.popleft()
            if depth >= max_up_depth:
                continue

            for parent in parent_map.get(node, []):
                if parent not in visited:
                    visited.add(parent)
                    q.append((parent, depth + 1))

        return visited

    # ---------- downward expansion ----------
    def bfs_down(start_ids: Set[str]) -> Set[str]:
        visited = set(start_ids)
        q = deque([(n, 0) for n in start_ids])

        while q:
            node, depth = q.popleft()
            if depth >= max_down_depth:
                continue

            for child in child_map.get(node, []):
                if child not in visited:
                    visited.add(child)
                    q.append((child, depth + 1))

        return visited

    # run expansions
    up = bfs_up(seed_chebi_ids)
    down = bfs_down(seed_chebi_ids)

    expanded = up | down

    # ---------- equivalence closure ----------
    eq_map = build_equivalence_map(structure_map)

    equiv_expanded = set()
    for cid in expanded:
        inchikey = structure_map.get(cid, {}).get("inchikey")
        if not inchikey:
            continue

        key = inchikey.split("-")[0]
        equiv_expanded.update(eq_map.get(key, {cid}))

    return expanded, equiv_expanded
# --- ChEBI OBO parser (parent map) ---

def parse_chebi_obo(filepath):
    relation_map = {}
    current_id = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_id = None
                continue

            if line.startswith("id: CHEBI:"):
                current_id = line.split("id: ")[1]
                relation_map.setdefault(current_id, [])

            elif current_id and line.startswith("is_a: CHEBI:"):
                parent = line.split("is_a: ")[1].split(" !")[0]
                relation_map[current_id].append(("is_a", parent))

            elif current_id and line.startswith("relationship:"):
                parts = line.split("relationship: ")[1]
                rel_type, target = parts.split(" CHEBI:")
                target = "CHEBI:" + target.split(" !")[0]
                if rel_type in ALLOWED_RELATIONS:
                    relation_map[current_id].append((rel_type, target))

    return relation_map


# --- ChEBI SDF parser (structure map) ---

def parse_chebi_sdf(filepath):
    structure_map = {}

    suppl = Chem.SDMolSupplier(filepath)
    for mol in suppl:
        if mol is None:
            continue

        try:
            chebi_id = mol.GetProp("ChEBI ID")
        except KeyError:
            continue

        structure_map[chebi_id] = {
            "smiles": Chem.MolToSmiles(mol) if mol else None,
            "inchi": mol.GetProp("InChI") if mol.HasProp("InChI") else None,
            "inchikey": mol.GetProp("InChIKey") if mol.HasProp("InChIKey") else None,
        }

    return structure_map


def build_child_map(parent_map):
    child_map = defaultdict(list)
    for child, parents in parent_map.items():
        for p in parents:
            child_map[p].append(child)
    return child_map


def build_equivalence_classes(structure_map):
    eq_classes = defaultdict(set)

    for chebi_id, props in structure_map.items():
        inchikey = props.get("inchikey")
        if not inchikey:
            continue

        # canonicalize: ignore stereochemistry if desired
        key = inchikey.split("-")[0]

        eq_classes[key].add(chebi_id)

    return eq_classes

# --- Usage ---

parent_map = parse_chebi_obo("chebi/chebi.obo")
child_map = build_child_map(parent_map)
structure_map = parse_chebi_sdf("chebi/chebi.sdf")
equivalence_classes = build_equivalence_classes(structure_map)

# --- Example sanity check ---

example_id = "CHEBI:17234"  # glucose-related example

print("Parents:", parent_map.get(example_id, []))
print("Structure:", structure_map.get(example_id, {}))



with gzip.open("chebi/chebi_parent_map.json.gz", "wt") as f:
    json.dump(parent_map, f)

with gzip.open("chebi/chebi_child_map.json.gz", "wt") as f:
    child_map_out = [{"parent": k[0], "child": k[1], "value": v} for k, v in child_map.items()]
    json.dump(child_map_out, f)

with gzip.open("chebi/chebi_structure_map.json.gz", "wt") as f:
    json.dump(structure_map, f)

with gzip.open("chebi/chebi_equivalence_classes.json.gz", "wt") as f:
    json.dump(equivalence_classes, f)