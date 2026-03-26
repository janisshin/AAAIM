import gzip, json
from rdkit import Chem  # :contentReference[oaicite:0]{index=0}
from collections import defaultdict

# --- ChEBI OBO parser (parent map) ---

def parse_chebi_obo(filepath):
    parent_map = {}
    current_id = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_id = None

            elif line.startswith("id: CHEBI:"):
                current_id = line.split("id: ")[1]
                parent_map[current_id] = []

            elif line.startswith("is_a: CHEBI:") and current_id:
                parent = line.split("is_a: ")[1].split(" !")[0]
                parent_map[current_id].append(parent)

    return parent_map


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
# --- Usage ---

parent_map = parse_chebi_obo("chebi/chebi.obo")
child_map = build_child_map(parent_map)
structure_map = parse_chebi_sdf("chebi/chebi.sdf")


# --- Example sanity check ---

example_id = "CHEBI:17234"  # glucose-related example

print("Parents:", parent_map.get(example_id, []))
print("Structure:", structure_map.get(example_id, {}))



with gzip.open("chebi/chebi_parent_map.json.gz", "wt") as f:
    json.dump(parent_map, f)

with gzip.open("chebi/chebi_structure_map.json.gz", "wt") as f:
    json.dump(structure_map, f)