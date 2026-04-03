import pandas as pd
import libsbml
import re
from core.model_info import detect_model_format

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
    # Skip Reason row if present
    if not df.empty and 'id' in df.columns:
        reason_rows = df[df['id'] == 'Reason:']
        if not reason_rows.empty:
            reason_text = reason_rows.iloc[0].get('display_name', '')
            # if reason_text:
            #     print(f"LLM Reason: {reason_text}")
        df = df[df['id'] != 'Reason:'].copy()
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
            # Get qualifier from the row, fall back to default if not specified
            row_qualifier = row.get('qualifier', qualifier)
            if row_qualifier == 'NA' or not row_qualifier:
                row_qualifier = qualifier  # Use default qualifier
            
            if action == 'add' and annotation:
                if ':' in annotation:
                    knowledge_resource, identifier = annotation.split(':', 1)
                    to_add.append((knowledge_resource.lower(), annotation, row_qualifier))
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
        # Handle existing annotations - preserve all qualifier blocks
        existing_items = []
        if annotation_str:
            # Extract all bqbiol qualifier blocks
            qualifier_blocks = re.findall(r'<bqbiol:([^>]+)[^>]*?>(.*?)</bqbiol:\1>', annotation_str, flags=re.DOTALL)
            for qual_name, block_content in qualifier_blocks:
                # Extract items from this qualifier block
                rdf_li_pattern = r"<rdf:li[^>]*/>"
                items = re.findall(rdf_li_pattern, block_content)
                for item in items:
                    existing_items.append((item, qual_name))
        
        # Parse all items, keep non-target ontologies, operate only on target ontology
        keep_items = []
        removed_this_species = 0
        
        # Group existing items by qualifier
        existing_by_qualifier = {}
        for item, qual_name in existing_items:
            if qual_name not in existing_by_qualifier:
                existing_by_qualifier[qual_name] = []
            existing_by_qualifier[qual_name].append(item)
        
        # Remove items to delete
        for ont_type, ont_id, item in extract_ontology_from_items([item for item, _ in existing_items]):
            # Only match for the correct ontology type and full identifier
            if ont_type is not None and (ont_type.lower(), ont_id) in to_delete:
                removed += 1
                removed_this_species += 1
                # Remove from all qualifier blocks
                for qual_name in list(existing_by_qualifier.keys()):
                    existing_by_qualifier[qual_name] = [i for i in existing_by_qualifier[qual_name] if i != item]
                continue
        
        # Add new terms (avoid duplicates)
        existing_set = set((ont_type.lower(), ont_id) for ont_type, ont_id, _ in extract_ontology_from_items([item for item, _ in existing_items]) if ont_type)
        
        # Group new items by qualifier
        new_by_qualifier = {}
        for ont in to_add:
            if len(ont) == 3:  # New format with qualifier
                knowledge_resource, annotation, row_qualifier = ont
            else:  # Old format without qualifier
                knowledge_resource, annotation = ont
                row_qualifier = qualifier
            
            if (knowledge_resource, annotation) not in existing_set:
                if row_qualifier not in new_by_qualifier:
                    new_by_qualifier[row_qualifier] = []
                new_by_qualifier[row_qualifier].append(create_annotation_item(knowledge_resource, annotation))
                added += 1
        
        # Merge existing and new items by qualifier
        final_by_qualifier = {}
        for qual_name in set(list(existing_by_qualifier.keys()) + list(new_by_qualifier.keys())):
            final_by_qualifier[qual_name] = []
            if qual_name in existing_by_qualifier:
                final_by_qualifier[qual_name].extend(existing_by_qualifier[qual_name])
            if qual_name in new_by_qualifier:
                final_by_qualifier[qual_name].extend(new_by_qualifier[qual_name])
        
        # Build new annotation string
        if final_by_qualifier:
            meta_id = species_id
            new_anno = f'<annotation>\n  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n    <rdf:Description rdf:about="#{meta_id}">\n'
            
            for qual_name, items in final_by_qualifier.items():
                if items:  # Only add qualifier blocks with items
                    qualifier_tag = f'bqbiol:{qual_name}'
                    new_anno += f'      <{qualifier_tag}>\n        <rdf:Bag>\n'
                    for item in items:
                        new_anno += f'          {item}\n'
                    new_anno += f'        </rdf:Bag>\n      </{qualifier_tag}>\n'
            
            new_anno += '    </rdf:Description>\n  </rdf:RDF>\n</annotation>'
        else:
            new_anno = ''
        sbml_elem.setAnnotation(new_anno)
        if to_add or removed_this_species:
            updated_species.add(species_id)
    writer = libsbml.SBMLWriter()
    writer.writeSBMLToFile(document, new_model_path)
    print(f"Update complete. {len(updated_species)} species updated. {added} annotations added, {removed} removed.")
