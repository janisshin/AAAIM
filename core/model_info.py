"""
Model Information Extraction for AAAIM

Extracts model information and context for annotation
"""

import re
import libsbml
import antimony
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import for SBML-qual support
try:
    import biolqm
    import pyboolnet
    QUAL_SUPPORT = True
except ImportError:
    QUAL_SUPPORT = False
    logging.warning("biolqm and/or pyboolnet not available - SBML-qual models not supported")

from utils.constants import ModelType, MODEL_FORMAT_PLUGINS, NCBIGENE_URI_PATTERNS, CHEBI_URI_PATTERNS

logger = logging.getLogger(__name__)

def detect_model_format(model_file: str) -> Tuple[ModelType, Dict[str, Any]]:
    """
    Detect the format of an SBML model (regular SBML, SBML-fbc, or SBML-qual).
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Tuple of (model_type, format_info) where format_info contains relevant plugin information
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return ModelType.SBML, {}
    
    format_info = {
        "has_fbc": False,
        "has_qual": False,
        "num_species": model.getNumSpecies(),
        "num_reactions": model.getNumReactions()
    }
    
    # Check for FBC plugin
    fbc_plugin = model.getPlugin("fbc")
    if fbc_plugin and fbc_plugin.getNumGeneProducts() > 0:
        format_info["has_fbc"] = True
        format_info["num_gene_products"] = fbc_plugin.getNumGeneProducts()
        return ModelType.SBML_FBC, format_info
    
    # Check for qual plugin
    qual_plugin = model.getPlugin("qual")
    if qual_plugin and qual_plugin.getNumQualitativeSpecies() > 0:
        format_info["has_qual"] = True
        format_info["num_qualitative_species"] = qual_plugin.getNumQualitativeSpecies()
        format_info["num_transitions"] = qual_plugin.getNumTransitions()
        return ModelType.SBML_QUAL, format_info
    
    # Default to regular SBML
    return ModelType.SBML, format_info

def extract_id_from_annotation(annotation_str: str, uri_patterns: List[str]) -> List[str]:
    """Helper function to extract IDs from annotation string."""
    ids = []
    for pattern in uri_patterns:
        matches = re.findall(pattern, annotation_str)
        ids.extend(matches)
    return list(set(ids))  # Remove duplicates

def find_species_with_chebi_annotations(model_file: str) -> Dict[str, List[str]]:
    """
    Find species with existing ChEBI annotations.
    Supports both identifiers.org URIs and URN formats.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary mapping species IDs to their ChEBI annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    chebi_annotations = {}
    
    # Extract annotations from species
    for species in model.getListOfSpecies():
        species_id = species.getId()
        
        if species.isSetAnnotation():
            annotation = species.getAnnotation()
            annotation_str = annotation.toXMLString()
            
            # Look for ChEBI URIs using all supported patterns
            chebi_ids = extract_id_from_annotation(annotation_str, CHEBI_URI_PATTERNS)

            if chebi_ids:
                chebi_annotations[species_id] = chebi_ids
    
    return chebi_annotations
    
def find_species_with_ncbigene_annotations(model_file: str) -> Dict[str, List[str]]:
    """
    Find species with existing NCBI gene annotations.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary mapping species IDs to their NCBI gene annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    model_type, format_info = detect_model_format(model_file)
    ncbigene_annotations = {}
    
    if model_type == ModelType.SBML_FBC:
        # Extract annotations from FBC gene products
        fbc_plugin = model.getPlugin("fbc")
        if fbc_plugin:
            for gene_product in fbc_plugin.getListOfGeneProducts():
                gene_product_id = gene_product.getId()
                
                if gene_product.isSetAnnotation():
                    annotation = gene_product.getAnnotation()
                    annotation_str = annotation.toXMLString()
                    gene_ids = extract_id_from_annotation(annotation_str, NCBIGENE_URI_PATTERNS)
                    
                    if gene_ids:
                        ncbigene_annotations[gene_product_id] = gene_ids
    
    elif model_type == ModelType.SBML_QUAL:
        # Extract annotations from qual qualitative species
        qual_plugin = model.getPlugin("qual")
        if qual_plugin:
            for qual_species in qual_plugin.getListOfQualitativeSpecies():
                qual_species_id = qual_species.getId()
                
                if qual_species.isSetAnnotation():
                    annotation = qual_species.getAnnotation()
                    annotation_str = annotation.toXMLString()
                    gene_ids = extract_id_from_annotation(annotation_str, NCBIGENE_URI_PATTERNS)
                    
                    if gene_ids:
                        ncbigene_annotations[qual_species_id] = gene_ids
    
    return ncbigene_annotations

def get_species_display_names(model_file: str, entity_type: str = "chemical") -> Dict[str, str]:
    """
    Get the display names for all species in the model.
    Supports regular species, FBC gene products, and qual qualitative species.
    
    Args:
        model_file: Path to the SBML model file
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        Dictionary mapping species/gene IDs to their display names
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    model_type, format_info = detect_model_format(model_file)
    
    if entity_type == "gene":
        names = {}
        
        if model_type == ModelType.SBML_FBC:
            # Use FBC plugin for gene products
            fbc_plugin = model.getPlugin("fbc")
            if fbc_plugin:
                for gene_product in fbc_plugin.getListOfGeneProducts():
                    gene_id = gene_product.getIdAttribute()
                    
                    # Try to get name in order of preference: name > label > id
                    if gene_product.isSetName() and gene_product.getName():
                        gene_name = gene_product.getName()
                    elif gene_product.isSetLabel() and gene_product.getLabel():
                        gene_name = gene_product.getLabel()
                    else:
                        gene_name = gene_id
                    
                    names[gene_id] = gene_name
        
        elif model_type == ModelType.SBML_QUAL:
            # Use qual plugin for qualitative species
            qual_plugin = model.getPlugin("qual")
            if qual_plugin:
                for qual_species in qual_plugin.getListOfQualitativeSpecies():
                    qual_id = qual_species.getId()
                    
                    # Try to get name in order of preference: name > id
                    if qual_species.isSetName() and qual_species.getName():
                        qual_name = qual_species.getName()
                    else:
                        qual_name = qual_id
                    
                    names[qual_id] = qual_name
        
        return names
    else:
        # Use regular species for chemical entities
        names = {val.getId(): val.getName() for val in model.getListOfSpecies()}
        return names

def get_all_species_ids(model_file: str, entity_type: str = "chemical") -> List[str]:
    """
    Get all species IDs from an SBML model.
    Supports regular species, FBC gene products, and qual qualitative species.
    
    Args:
        model_file: Path to SBML model file
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        List of species/gene IDs
    """
    display_names = get_species_display_names(model_file, entity_type)
    return list(display_names.keys())

def extract_qual_transitions(model_file: str, species_ids: List[str]) -> List[str]:
    """
    Extract boolean transitions from SBML-qual models using biolqm and pyboolnet.
    Self loops are ignored.
    
    Args:
        model_file: Path to the SBML-qual model file
        species_ids: List of species IDs to filter transitions for
        
    Returns:
        List of transition strings in the format "target, rule"
    """
    if not QUAL_SUPPORT:
        logger.warning("biolqm/pyboolnet not available - cannot extract qual transitions")
        return []
    
    try:
        # Load model with biolqm
        model_lqm = biolqm.load(model_file)
        
        # Convert to pyboolnet format and get primes
        primes = biolqm.to_pyboolnet(model_lqm)
        
        # Convert primes to boolean network format
        bnet_string = pyboolnet.file_exchange.primes2bnet(primes)
        
        # Parse the bnet string to extract transitions
        transitions = []
        for line in bnet_string.strip().split('\n'):
            line = line.strip()
            if line and ',' in line:
                target, rule = line.split(',', 1)
                target = target.strip()
                rule = rule.strip()

                # Remove self loops: skip if left_side and right_side are the same (ignoring whitespace)
                if target.strip() == rule.strip():
                    # print(f"Skipping self loop: {target.strip()} = {rule.strip()}")
                    continue
                
                # Check if this transition involves any of our target species
                if target in species_ids or any(species_id in rule for species_id in species_ids):
                    transitions.append(f"{target} = {rule}")
        
        return transitions
        
    except Exception as e:
        logger.error(f"Error extracting qual transitions: {e}")
        return []

def extract_model_info(model_file: str, species_ids: List[str], entity_type: str = "chemical") -> Dict[str, Any]:
    """
    Extract display names and reactions/transitions for the specified species.
    Supports regular SBML, SBML-fbc, and SBML-qual models.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        Dictionary with model name, model type, display names, and reactions/transitions
    """
    ########## MODEL DETECTION AND BASIC INFO ##########
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        logger.error(f"Error loading SBML file: {model_file}")
        return {}
    
    model_type, format_info = detect_model_format(model_file)
    
    # Extract model name
    model_name = model.getName() if model.isSetName() else model.getId() if model.isSetId() else ""
    
    # Extract model notes
    model_notes = model.getNotesString() if model.isSetNotes() else ""
    if model_notes != "":
        # Remove HTML tags from model notes
        model_notes = re.sub(r'<[^>]*>', '', model_notes)

        # Split by newlines and/or multiple spaces
        lines = re.split(r'\n|\s{2,}', model_notes)
        
        # List of keywords/fragments that indicate boilerplate text
        boilerplate_keywords = [
            'copyright', 'public domain', 'rights', 'CC0', 'dedication', 
            'please refer', 'BioModels Database', 'cite', 'citing',
            'terms of use', 'Li C', 'BMC Syst', 'encoded model', 
            'entitled to use', 'redistribute', 'commercially', 'restricted way', 'verbatim',
            'BIOMD', 'resource'
        ]
        
        # Filter out lines containing boilerplate keywords
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(keyword.lower() in line.lower() for keyword in boilerplate_keywords):
                filtered_lines.append(line)
        
        # Reassemble the filtered content with proper spacing
        model_notes = '\n'.join(filtered_lines)

    ########## DISPLAY NAMES ##########
    all_display_names = get_species_display_names(model_file, entity_type)
    # filter to only include species_ids
    filtered_display_names = {id: all_display_names.get(id, "") for id in species_ids if id in all_display_names}

    ########## REACTIONS/TRANSITIONS ##########
    reactions = []
    
    if entity_type == "gene" and model_type == ModelType.SBML_QUAL:
        # For SBML-qual gene models, extract boolean transitions
        reactions = extract_qual_transitions(model_file, species_ids)
        
        # Update display names to include all species mentioned in transitions
        related_species = set(species_ids)
        for transition in reactions:
            # Extract species IDs from transition rules
            all_ids_in_transition = re.findall(r'\b([A-Za-z0-9_]+)\b', transition)
            related_species.update(all_ids_in_transition)
        
        # Filter display names to include our target species and all related species
        filtered_display_names = {species_id: all_display_names.get(species_id, "") for species_id in related_species if species_id in all_display_names}
        
    elif entity_type == "gene" and model_type == ModelType.SBML_FBC:
        # For SBML-fbc gene models, reactions are empty (genes don't participate in reactions directly)
        reactions = []
        
    else:
        # For chemical entities or regular SBML models, use antimony
        antimony.clearPreviousLoads()
        sbml_model = antimony.loadSBMLFile(model_file)
        if sbml_model == -1:
            print(f"Error loading SBML file: {antimony.getLastError()}")
            return {} 
        
        antimony_string = antimony.getAntimonyString()
        
        # Parse the antimony_string to extract reactions
        # Look for lines with => symbols which indicate reactions
        reaction_pattern = re.compile(r'// Reactions:.*?(?=//|$)', re.DOTALL)
        reactions_section = reaction_pattern.search(antimony_string)
        
        reaction_matches = []
        if reactions_section:
            reactions_text = reactions_section.group(0).replace("// Reactions:", "").strip()
            reaction_pattern = re.compile(r'([^;]+)(=>)([^;]+);', re.MULTILINE)
            reaction_matches = reaction_pattern.findall(reactions_text)

        # If no matches found with '=>', try with '=' instead
        if not reaction_matches:
            reaction_pattern = re.compile(r'// Rate Rules:.*?(?=//|$)', re.DOTALL)
            reactions_section = reaction_pattern.search(antimony_string)
            
            reaction_matches = []
            if reactions_section:
                reactions_text = reactions_section.group(0).replace("// Rate Rules:", "").strip()
                reaction_pattern = re.compile(r'([^;]+)(=)([^;]+);', re.MULTILINE)
                reaction_matches = reaction_pattern.findall(reactions_text)

        # Keep track of all species involved in reactions with our target species
        related_species = set(species_ids)
        
        # Filter reactions to only include those involving our species
        for match in reaction_matches:
            left_side, arrow, right_side = match
            reaction_str = f"{left_side.strip()} {arrow} {right_side.strip()}"
            
            # Check if any of our species IDs are in this reaction
            if any(re.search(r'\b' + re.escape(species_id) + r'\b', left_side + ' ' + right_side) for species_id in species_ids):
                reactions.append(reaction_str)
                
                # Extract all species IDs from this reaction
                all_ids_in_reaction = re.findall(r'\b([A-Za-z0-9_]+)\b', left_side + ' ' + right_side)
                related_species.update(all_ids_in_reaction)
    
        # Filter display names to include our target species and all related species
        filtered_display_names = {species_id: all_display_names.get(species_id, "") for species_id in related_species if species_id in all_display_names}
    
    return {
        "model_name": model_name,
        "model_type": model_type,
        "format_info": format_info,
        "display_names": filtered_display_names,
        "reactions": reactions,
        "model_notes": model_notes
    }

def format_prompt(model_file: str, species_ids: List[str], entity_type: str = "chemical") -> str:
    """
    Format the information for the LLM prompt.
    Adapts format based on model type (SBML, SBML-fbc, SBML-qual).
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include in the prompt
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        Formatted prompt string
    """
    model_info = extract_model_info(model_file, species_ids, entity_type)
    if model_info == {}:
        return ""
    
    model_type = model_info.get("model_type", ModelType.SBML)
    
    # For gene entities, format prompt differently based on model type
    if entity_type == "gene":
        if model_type == ModelType.SBML_QUAL:
            # SBML-qual models have boolean transitions
            prompt = f"""Now annotate these:
{entity_type.title()} to annotate: {", ".join(species_ids)}
Model: "{model_info["model_name"]}" 
// Display Names:
{model_info["display_names"]}
// Boolean Transitions:
{chr(10).join(model_info["reactions"])}
// Notes:
{model_info["model_notes"]}

Return up to 3 standardized names or common synonyms for each {entity_type}, ranked by likelihood.
Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type}s after 'Reason:' by the end.

SpeciesA: "name1", "name2", …
SpeciesB:  …
Reason: …
            """
        else:
            # SBML-fbc models don't have reactions for genes
            prompt = f"""Now annotate these:
{entity_type.title()} to annotate: {", ".join(species_ids)}
Model: "{model_info["model_name"]}" 
// Display Names:
{model_info["display_names"]}
// Notes:
{model_info["model_notes"]}

Return up to 3 standardized names or common synonyms for each {entity_type}, ranked by likelihood.
Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type}s after 'Reason:' by the end.

SpeciesA: "name1", "name2", …
SpeciesB:  …
Reason: …
            """
    else:
        # Original format for chemical entities
        prompt = f"""Now annotate these:
Species to annotate: {", ".join(species_ids)}
Model: "{model_info["model_name"]}"
// Display Names:
{model_info["display_names"]}
// Reactions:
{chr(10).join(model_info["reactions"])}
// Notes:
{model_info["model_notes"]}

Return up to 3 standardized names or common synonyms for each species, ranked by likelihood.
Use the below format, do not include any other text except the synonyms, and give short reasons for all species after 'Reason:' by the end.

SpeciesA: "name1", "name2", …
SpeciesB:  …
Reason: …
        """
    return prompt 