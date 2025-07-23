import pandas as pd
import libsbml
import re
from core.model_info import detect_model_format
import logging

# XML helpers (adapted from AMAS)
def divide_existing_annotation(inp_str, qualifier):
    """
    Parse the annotation string to extract items in <rdf:Bag> for the specified qualifier.
    Returns dict with 'container' (annotation with empty block) and 'items' (list of <rdf:li> lines).
    """
    qualifier_pattern = rf"(<{qualifier}[^>]*?>\s*<rdf:Bag>.*?</rdf:Bag>\s*</{qualifier}>)"
    qualifier_matches = re.findall(qualifier_pattern, inp_str, re.DOTALL)
    if not qualifier_matches:
        return None
    rdf_li_pattern = r"<rdf:li[^>]*/>"
    items = []
    for block in qualifier_matches:
        items.extend(re.findall(rdf_li_pattern, block))
    match_prefix = re.match(rf"<{qualifier}.*?>", qualifier_matches[0])
    if match_prefix:
        qualifier_opening = match_prefix.group()
    else:
        qualifier_opening = f"<{qualifier}>"
    empty_qualifier_block = (
        f"      {qualifier_opening}\n        <rdf:Bag>\n        </rdf:Bag>\n      </{qualifier}>"
    )
    stripped_annotation = re.sub(qualifier_pattern, "", inp_str, flags=re.DOTALL).strip()
    container = stripped_annotation.replace(
        "</rdf:Description>",
        f"{empty_qualifier_block}\n    </rdf:Description>"
    )
    return {"container": container, "items": items}

def insert_items_back_to_container(container, items, qualifier):
    """
    Insert <rdf:li> items back into the <rdf:Bag> of the specified qualifier in the container.
    """
    bag_pattern = rf"(<{qualifier}[^>]*?>\s*<rdf:Bag>).*?</rdf:Bag>\s*</{qualifier}>"
    match = re.search(bag_pattern, container, re.DOTALL)
    items_str = "\n".join([f"          {item}" for item in items])
    if match:
        qualifier_opening = match.group(1)
        updated_bag = f"{qualifier_opening}\n{items_str}\n        </rdf:Bag>"
        updated_container = re.sub(bag_pattern, updated_bag + f"\n      </{qualifier}>", container, flags=re.DOTALL)
    else:
        qualifier_opening = f"<{qualifier} xmlns:bqbiol=\"http://biomodels.net/biology-qualifiers/\">"
        updated_bag = f"      {qualifier_opening}\n        <rdf:Bag>\n{items_str}\n        </rdf:Bag>\n      </{qualifier}>"
        updated_container = container.replace(
            "</rdf:Description>",
            f"{updated_bag}\n    </rdf:Description>"
        )
    return updated_container

def extract_ontology_from_items(items_list):
    """
    Extract ontology from items and return a flat list of (ontology type, full identifier, full_item_str).
    ontology_type: e.g., 'chebi'
    full identifier: e.g., 'CHEBI:15414'
    """
    result = []
    for item in items_list:
        # Try identifiers.org/TYPE/ID
        m = re.match(r'.*identifiers.org/([^/]+)/([^/\"]+)', item)
        if m:
            ontology_type, ontology_id = m.group(1), m.group(2).replace('"', '')
            result.append((ontology_type, ontology_id, item))
        else:
            # If not identifiers.org, preserve the item as is (e.g., kegg.compound)
            result.append((None, None, item))
    return result

def create_annotation_item(knowledge_resource, identifier):
    return f'<rdf:li rdf:resource="http://identifiers.org/{knowledge_resource}/{identifier}"/>'

def clean_items(items):
    # Only deduplicate by full item string
    return sorted(set(items))

# Main update logic
def update_annotation(
    original_model_path, 
    recommendation_table = None, 
    new_model_path = None, 
    qualifier='is'
    ):
    """
    Update SBML model annotations based on a recommendation table.
    Only add/delete the specified ontology (e.g., CHEBI), preserving all other annotation terms (e.g., KEGG).
    """
    if recommendation_table is None:
        recommendation_table = original_model_path + '_recommendations.csv'
    if new_model_path is None:
        new_model_path = original_model_path + '_updated.xml'
    # Load model
    reader = libsbml.SBMLReader()
    document = reader.readSBML(original_model_path)
    model = document.getModel()
    if model is None:
        raise ValueError(f"Could not load SBML model: {original_model_path}")
    # Load table
    if isinstance(recommendation_table, str):
        df = pd.read_csv(recommendation_table)
    else:
        df = recommendation_table.copy()
    # Get model type
    model_type, _ = detect_model_format(original_model_path)
    # Group by species
    grouped = df.groupby('id')
    added, removed, updated_species = 0, 0, set()
    for species_id, group in grouped:
        to_add, to_delete = [], []
        for _, row in group.iterrows():
            action = str(row.get('update_annotation', 'ignore')).lower()
            annotation = row.get('annotation', '')
            typ = row.get('type', '')
            if action == 'add' and annotation:
                if ':' in annotation:
                    knowledge_resource, identifier = annotation.split(':', 1)
                    to_add.append((knowledge_resource.lower(), annotation))
            elif action == 'delete' and annotation:
                if ':' in annotation:
                    knowledge_resource, identifier = annotation.split(':', 1)
                    to_delete.append((knowledge_resource.lower(), annotation))
        sbml_elem = None
        if model_type.name.startswith('SBML'):
            sbml_elem = model.getElementBySId(species_id)
        elif model_type.name == 'SBML_FBC':
            sbml_elem = model.getElementBySId(species_id)
            if sbml_elem is None and hasattr(model, 'getPlugin'):
                fbc = model.getPlugin('fbc')
                if fbc:
                    sbml_elem = fbc.getGeneProduct(species_id)
        elif model_type.name == 'SBML_QUAL':
            qual = model.getPlugin('qual')
            if qual:
                sbml_elem = qual.getQualitativeSpecies(species_id)
        if sbml_elem is None:
            continue
        if sbml_elem.isSetAnnotation():
            annotation_str = sbml_elem.getAnnotation().toXMLString()
        else:
            annotation_str = ''
        ann_dict = divide_existing_annotation(annotation_str, f'bqbiol:{qualifier}') if annotation_str else None
        existing_items = ann_dict['items'] if ann_dict else []
        # Parse all items, keep non-target ontologies, operate only on target ontology
        keep_items = []
        removed_this_species = 0
        for ont_type, ont_id, item in extract_ontology_from_items(existing_items):
            # Only match for the correct ontology type and full identifier
            if ont_type is not None and (ont_type.lower(), ont_id) in to_delete:
                removed += 1
                removed_this_species += 1
                continue
            keep_items.append(item)
        # Add new terms (avoid duplicates)
        existing_set = set((ont_type.lower(), ont_id) for ont_type, ont_id, _ in extract_ontology_from_items(keep_items) if ont_type)
        for ont in to_add:
            if ont not in existing_set:
                keep_items.append(create_annotation_item(ont[0], ont[1]))
                added += 1
        final_items = clean_items(keep_items)
        if ann_dict:
            new_anno = insert_items_back_to_container(ann_dict['container'], final_items, f'bqbiol:{qualifier}')
        else:
            meta_id = species_id
            qualifier_tag = f'bqbiol:{qualifier}'
            new_anno = f'<annotation>\n  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n    <rdf:Description rdf:about="#' + meta_id + '">\n      <' + qualifier_tag + '>\n        <rdf:Bag>\n' + '\n'.join(f'          {item}' for item in final_items) + f'\n        </rdf:Bag>\n      </' + qualifier_tag + '>\n    </rdf:Description>\n  </rdf:RDF>\n</annotation>'
        sbml_elem.setAnnotation(new_anno)
        if to_add or removed_this_species:
            updated_species.add(species_id)
    writer = libsbml.SBMLWriter()
    writer.writeSBMLToFile(document, new_model_path)
    print(f"Update complete. {len(updated_species)} species updated. {added} annotations added, {removed} removed.")
