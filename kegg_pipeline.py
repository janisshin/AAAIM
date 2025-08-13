#!/usr/bin/env python3
"""
KEGG Pipeline for SBML Model Annotation

This script implements a pipeline to annotate reactions in SBML models:
1. Load an SBML model file
2. Use an LLM + ChEBI RAG to identify species and assign ChEBI IDs
3. Map ChEBI IDs to KEGG compound IDs
4. Parse reactions from the model and extract substrates and products
5. Find matching KEGG reactions for each model reaction
6. Insert annotations into the model
7. Return an annotated SBML model

Usage:
    python kegg_pipeline.py --model <path_to_sbml_model> [--output <output_file>]
"""

import os
import sys
import argparse
import logging
import json
import lzma
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Add the project root to the Python path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import AAAIM modules
import libsbml
import antimony
from core.model_info import extract_model_info, get_all_species_ids, detect_model_format
from core.database_search import get_species_recommendations_rag
from core.llm_interface import query_llm, parse_llm_response
from utils.constants import ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kegg_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def match_reactions_to_kegg(input_reaction, kegg_reactions, match_mode="substrate_product", min_match=2):
    """
    input_reaction: dict with 'substrates_kegg' and 'products_kegg' (list of Cxxxxx)
    kegg_reactions: list of parsed KEGG reactions (from your JSON)
    match_mode: one of 'substrate_only', 'substrate_product', or 'any'
    min_match: minimum number of matching compounds (total) to consider a match

    Returns: ranked list of matching KEGG reactions
    """
    matches = []

    for rxn in kegg_reactions:
        subs = set(rxn.get("substrates", []))
        prods = set(rxn.get("products", []))

        in_subs = set(input_reaction.get("substrates_kegg", []))
        in_prods = set(input_reaction.get("products_kegg", []))

        if match_mode == "substrate_only":
            score = len(in_subs & subs)
        elif match_mode == "substrate_product":
            score = len(in_subs & subs) + len(in_prods & prods)
        elif match_mode == "any":
            score = len((in_subs | in_prods) & (subs | prods))
        else:
            raise ValueError("Invalid match mode")

        if score >= min_match:
            matches.append((score, rxn))

    return sorted(matches, key=lambda x: -x[0])  # highest score first






def load_sbml_model(model_file: str) -> Tuple[libsbml.SBMLDocument, ModelType, Dict[str, Any]]:
    """
    Load an SBML model file and detect its format.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Tuple containing:
        - SBML document
        - Model type (SBML, SBML-fbc, SBML-qual)
        - Format information dictionary
    
    Raises:
        ValueError: If the model file cannot be loaded
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    
    # Check for errors
    if document.getNumErrors() > 0:
        error_msg = f"Error loading SBML model: {document.getError(0).getMessage()}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Detect model format
    model_type, format_info = detect_model_format(model_file)
    logger.info(f"Loaded model of type {model_type} with {format_info.get('num_species', 0)} species and {format_info.get('num_reactions', 0)} reactions")
    
    return document, model_type, format_info

def extract_species_info(model_file: str) -> Dict[str, Any]:
    """
    Extract species information from an SBML model.
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Dictionary containing species information
    """
    # Get all species IDs
    species_ids = get_all_species_ids(model_file, entity_type="chemical")
    
    if not species_ids:
        logger.warning("No species found in model")
        return {}
    
    logger.info(f"Found {len(species_ids)} species in model")
    
    # Extract model context for the species
    model_info = extract_model_info(model_file, species_ids, entity_type="chemical")
    
    if not model_info:
        logger.error("Failed to extract model context")
        return {}
    
    logger.info(f"Extracted context for model: {model_info['model_name']}")
    
    return {
        "species_ids": species_ids,
        "model_info": model_info
    }

def identify_species_with_llm(model_file: str, species_ids: List[str], llm_model: str = "claude-3-sonnet-20240229") -> Dict[str, List[str]]:
    """
    Use an LLM to identify species and generate potential names/synonyms.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to identify
        llm_model: LLM model to use
        
    Returns:
        Dictionary mapping species IDs to lists of potential names/synonyms
    """
    from core.model_info import format_prompt
    
    # Format prompt for LLM
    prompt = format_prompt(model_file, species_ids, entity_type="chemical")
    
    if not prompt:
        logger.error("Failed to format prompt")
        return {}
    
    logger.info(f"Querying LLM ({llm_model})...")
    
    try:
        # Query LLM
        result = query_llm(prompt, model=llm_model, entity_type="chemical")
        
        if not result:
            logger.error("No response from LLM")
            return {}
        
        # Parse LLM response
        synonyms_dict, reason = parse_llm_response(result)
        
        if not synonyms_dict:
            logger.error("Failed to parse LLM response")
            return {}
        
        logger.info(f"Parsed synonyms for {len(synonyms_dict)} species")
        logger.info(f"LLM reasoning: {reason}")
        
        return synonyms_dict
        
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return {}

def map_species_to_chebi(species_ids: List[str], synonyms_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map species to ChEBI IDs using RAG.
    
    Args:
        species_ids: List of species IDs
        synonyms_dict: Dictionary mapping species IDs to lists of potential names/synonyms
        
    Returns:
        Dictionary mapping species IDs to ChEBI IDs
    """
    logger.info("Mapping species to ChEBI IDs using RAG...")
    
    try:
        # Use RAG to get recommendations
        recommendations = get_species_recommendations_rag(
            species_ids=species_ids,
            synonyms_dict=synonyms_dict,
            database="chebi",
            top_k=1  # Get only the top match for each species
        )
        
        # Create mapping from species ID to ChEBI ID
        species_to_chebi = {}
        
        for rec in recommendations:
            if rec.candidates:
                # Get the top candidate (highest similarity score)
                top_candidate = rec.candidates[0]
                species_to_chebi[rec.id] = f"CHEBI:{top_candidate}"
            else:
                species_to_chebi[rec.id] = ""
        
        logger.info(f"Mapped {len([v for v in species_to_chebi.values() if v])} species to ChEBI IDs")
        
        return species_to_chebi
        
    except Exception as e:
        logger.error(f"Error mapping species to ChEBI IDs: {e}")
        return {}

def load_chebi_to_kegg_mapping(mapping_file: str = "data/kegg/chebi_to_kegg_map.lzma") -> Dict[str, str]:
    """
    Load the ChEBI to KEGG compound ID mapping from a compressed JSON file.
    
    Args:
        mapping_file: Path to the compressed JSON mapping file
        
    Returns:
        Dictionary mapping ChEBI IDs to KEGG compound IDs
    """
    logger.info(f"Loading ChEBI to KEGG mapping from {mapping_file}")
    
    try:
        with lzma.open(mapping_file, "rt", encoding="utf-8") as f:
            mapping = json.load(f)
        
        logger.info(f"Loaded mapping for {len(mapping)} ChEBI IDs")
        return mapping
    
    except Exception as e:
        logger.error(f"Error loading ChEBI to KEGG mapping: {e}")
        return {}

def map_chebi_to_kegg_ids(species_to_chebi: Dict[str, str]) -> Dict[str, str]:
    """
    Map ChEBI IDs to KEGG compound IDs.
    
    Args:
        species_to_chebi: Dictionary mapping species IDs to ChEBI IDs
        
    Returns:
        Dictionary mapping species IDs to KEGG compound IDs
    """
    logger.info("Mapping ChEBI IDs to KEGG compound IDs")
    
    # Load the ChEBI to KEGG mapping
    chebi_to_kegg_map = load_chebi_to_kegg_mapping()
    
    if not chebi_to_kegg_map:
        logger.error("Failed to load ChEBI to KEGG mapping")
        return {}
    
    # Map species IDs to KEGG compound IDs
    species_to_kegg = {}
    
    for species_id, chebi_id in species_to_chebi.items():
        if chebi_id and chebi_id in chebi_to_kegg_map:
            species_to_kegg[species_id] = chebi_to_kegg_map[chebi_id]
        else:
            species_to_kegg[species_id] = ""
    
    logger.info(f"Mapped {len([v for v in species_to_kegg.values() if v])} species to KEGG compound IDs")
    
    return species_to_kegg

def load_kegg_reactions(kegg_reactions_file: str = "data/kegg/parsed_kegg_reactions.json") -> List[Dict[str, Any]]:
    """
    Load KEGG reactions from a JSON file.
    
    Args:
        kegg_reactions_file: Path to the KEGG reactions JSON file
        
    Returns:
        List of KEGG reaction dictionaries
    """
    logger.info(f"Loading KEGG reactions from {kegg_reactions_file}")
    
    try:
        with open(kegg_reactions_file, 'r') as f:
            kegg_reactions = json.load(f)
        
        logger.info(f"Loaded {len(kegg_reactions)} KEGG reactions")
        return kegg_reactions
    
    except Exception as e:
        logger.error(f"Error loading KEGG reactions: {e}")
        return []

def find_matching_kegg_reactions(parsed_reactions: List[Dict[str, Any]], min_match: int = 2) -> List[Dict[str, Any]]:
    """
    Find matching KEGG reactions for each model reaction.
    
    Args:
        parsed_reactions: List of parsed model reactions
        min_match: Minimum number of matching compounds to consider a match
        
    Returns:
        List of model reactions with matching KEGG reactions
    """
    logger.info("Finding matching KEGG reactions...")
    
    # Load KEGG reactions
    kegg_reactions = load_kegg_reactions()
    
    if not kegg_reactions:
        logger.error("Failed to load KEGG reactions")
        return parsed_reactions
    
    # Find matching KEGG reactions for each model reaction
    for reaction in parsed_reactions:
        # Skip reactions without KEGG compound IDs
        if not any(reaction["substrates_kegg"]) and not any(reaction["products_kegg"]):
            reaction["kegg_matches"] = []
            continue
        
        # Create input reaction for matching
        input_reaction = {
            "substrates_kegg": [kegg_id for kegg_id in reaction["substrates_kegg"] if kegg_id],
            "products_kegg": [kegg_id for kegg_id in reaction["products_kegg"] if kegg_id]
        }
        
        # Find matching KEGG reactions
        matches = match_reactions_to_kegg(
            input_reaction=input_reaction,
            kegg_reactions=kegg_reactions,
            match_mode="substrate_product",
            min_match=min_match
        )
        
        # Add matching KEGG reactions to the model reaction
        reaction["kegg_matches"] = [
            {
                "score": score,
                "reaction_id": rxn.get("reaction_id", ""),
                "name": rxn.get("name", ""),  # Enzyme name
                "equation": rxn.get("raw_equation", ""),
                "substrates": rxn.get("substrates", []),
                "products": rxn.get("products", []),
                "ec_numbers": rxn.get("ec_numbers", []),
                "pathways": rxn.get("pathways", [])  # Pathway links
            }
            for score, rxn in matches[:5]  # Limit to top 5 matches
        ]
    
    # Count reactions with matches
    reactions_with_matches = sum(1 for reaction in parsed_reactions if reaction.get("kegg_matches", []))
    logger.info(f"Found matching KEGG reactions for {reactions_with_matches} out of {len(parsed_reactions)} reactions")
    
    return parsed_reactions

def annotate_sbml_model(model_file: str, llm_model: str = "claude-3-sonnet-20240229") -> Dict[str, str]:
    """
    Annotate an SBML model by identifying species and assigning ChEBI IDs.
    
    Args:
        model_file: Path to the SBML model file
        llm_model: LLM model to use
        
    Returns:
        Dictionary mapping species IDs to ChEBI IDs
    """
    logger.info(f"Starting annotation for model: {model_file}")
    
    try:
        # Load SBML model
        _, model_type, _ = load_sbml_model(model_file)
        
        # Extract species information
        species_info = extract_species_info(model_file)
        
        if not species_info:
            logger.error("Failed to extract species information")
            return {}
        
        species_ids = species_info["species_ids"]
        
        # Identify species with LLM
        synonyms_dict = identify_species_with_llm(model_file, species_ids, llm_model)
        
        if not synonyms_dict:
            logger.error("Failed to identify species with LLM")
            return {}
        
        # Map species to ChEBI IDs
        species_to_chebi = map_species_to_chebi(species_ids, synonyms_dict)
        
        return species_to_chebi
        
    except Exception as e:
        logger.error(f"Error annotating model: {e}")
        return {}

def save_results(species_to_chebi: Dict[str, str], output_file: str):
    """
    Save annotation results to a file.
    
    Args:
        species_to_chebi: Dictionary mapping species IDs to ChEBI IDs
        output_file: Path to the output file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'species_id': list(species_to_chebi.keys()),
        'chebi_id': list(species_to_chebi.values())
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

def parse_reactions(model_file: str, species_to_chebi: Dict[str, str], species_to_kegg: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Parse reactions from an SBML model and extract substrates and products.
    
    Args:
        model_file: Path to the SBML model file
        species_to_chebi: Dictionary mapping species IDs to ChEBI IDs
        species_to_kegg: Dictionary mapping species IDs to KEGG compound IDs
        
    Returns:
        List of dictionaries containing reaction information
    """
    logger.info("Parsing reactions from model...")
    
    # Extract model info to get reactions
    species_ids = get_all_species_ids(model_file, entity_type="chemical")
    model_info = extract_model_info(model_file, species_ids, entity_type="chemical")
    
    if not model_info or "reactions" not in model_info:
        logger.error("Failed to extract reactions from model")
        return []
    
    reactions = model_info["reactions"]
    logger.info(f"Found {len(reactions)} reactions in model")
    
    # Parse each reaction to extract substrates and products
    parsed_reactions = []
    
    for i, reaction_str in enumerate(reactions):
        try:
            # Parse reaction equation
            substrates, products = parse_reaction_equation(reaction_str)
            
            # Map species IDs to ChEBI IDs and KEGG compound IDs
            substrates_chebi = []
            products_chebi = []
            substrates_kegg = []
            products_kegg = []
            
            for substrate in substrates:
                if substrate in species_to_chebi and species_to_chebi[substrate]:
                    substrates_chebi.append(species_to_chebi[substrate])
                else:
                    substrates_chebi.append("")
                
                if substrate in species_to_kegg and species_to_kegg[substrate]:
                    substrates_kegg.append(species_to_kegg[substrate])
                else:
                    substrates_kegg.append("")
            
            for product in products:
                if product in species_to_chebi and species_to_chebi[product]:
                    products_chebi.append(species_to_chebi[product])
                else:
                    products_chebi.append("")
                
                if product in species_to_kegg and species_to_kegg[product]:
                    products_kegg.append(species_to_kegg[product])
                else:
                    products_kegg.append("")
            
            # Create reaction dictionary
            reaction_dict = {
                "id": f"R{i+1}",
                "equation": reaction_str,
                "substrates": substrates,
                "products": products,
                "substrates_chebi": substrates_chebi,
                "products_chebi": products_chebi,
                "substrates_kegg": substrates_kegg,
                "products_kegg": products_kegg
            }
            
            parsed_reactions.append(reaction_dict)
            
        except Exception as e:
            logger.warning(f"Error parsing reaction '{reaction_str}': {e}")
            continue
    
    logger.info(f"Successfully parsed {len(parsed_reactions)} reactions")
    return parsed_reactions

def insert_annotations_into_model(model_file: str, reactions: List[Dict[str, Any]], output_file: str = None) -> str:
    """
    Insert annotations into the Antimony model.
    
    Args:
        model_file: Path to the SBML model file
        reactions: List of reactions with KEGG annotations
        output_file: Path to the output file (if None, return the annotated model as a string)
        
    Returns:
        Annotated model as a string (if output_file is None) or path to the output file
    """
    logger.info("Inserting annotations into the model...")
    
    # Load the SBML model using Antimony
    antimony.clearPreviousLoads()
    sbml_model = antimony.loadSBMLFile(model_file)
    
    if sbml_model == -1:
        logger.error(f"Error loading SBML file: {antimony.getLastError()}")
        return ""
    
    # Get the Antimony string
    antimony_string = antimony.getAntimonyString()
    
    # Create a mapping from reaction ID to annotations
    reaction_annotations = {}
    for reaction in reactions:
        reaction_id = reaction["id"]
        annotations = reaction.get("kegg_annotations", {})
        
        if not any(annotations.values()):
            continue
        
        reaction_annotations[reaction_id] = annotations
    
    # Parse the Antimony string to identify reactions
    reaction_pattern = re.compile(r'// Reactions:.*?(?=//|$)', re.DOTALL)
    reactions_section = reaction_pattern.search(antimony_string)
    
    if not reactions_section:
        logger.warning("No reactions section found in Antimony string")
        return antimony_string
    
    reactions_text = reactions_section.group(0).replace("// Reactions:", "").strip()
    
    # Split reactions by semicolon
    reaction_statements = reactions_text.split(';')
    
    # Create a new reactions section with annotations
    new_reactions_section = "// Reactions:\n"
    
    for i, statement in enumerate(reaction_statements):
        if not statement.strip():
            continue
        
        # Add the original reaction statement
        new_reactions_section += statement.strip() + ";\n"
        
        # Add annotations if available
        reaction_id = f"R{i+1}"
        if reaction_id in reaction_annotations:
            annotations = reaction_annotations[reaction_id]
            
            # Add EC numbers
            if annotations.get("ec_numbers"):
                ec_numbers = ", ".join(annotations["ec_numbers"])
                new_reactions_section += f"  // EC Number: {ec_numbers}\n"
            
            # Add enzyme names
            if annotations.get("enzyme_names"):
                enzyme_names = "; ".join(annotations["enzyme_names"])
                new_reactions_section += f"  // Enzyme: {enzyme_names}\n"
            
            # Add KEGG reaction IDs
            if annotations.get("kegg_reaction_ids"):
                kegg_ids = ", ".join(annotations["kegg_reaction_ids"])
                new_reactions_section += f"  // KEGG Reaction: {kegg_ids}\n"
            
            # Add pathways
            if annotations.get("pathways"):
                pathways = "; ".join(annotations["pathways"])
                new_reactions_section += f"  // Pathway: {pathways}\n"
            
            new_reactions_section += "\n"
    
    # Replace the reactions section in the Antimony string
    new_antimony_string = antimony_string.replace(reactions_section.group(0), "// Reactions:\n" + new_reactions_section)
    
    # If output file is provided, save the annotated model
    if output_file:
        # Convert back to SBML
        antimony.clearPreviousLoads()
        antimony.loadAntimonyString(new_antimony_string)
        sbml_string = antimony.getSBMLString()
        
        with open(output_file, 'w') as f:
            f.write(sbml_string)
        
        logger.info(f"Annotated model saved to {output_file}")
        return output_file
    
    return new_antimony_string

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="KEGG Pipeline for SBML Model Annotation")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the SBML model file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="species_to_chebi.csv",
        help="Path to the output file for species mapping (default: species_to_chebi.csv)"
    )
    
    parser.add_argument(
        "--kegg-output",
        type=str,
        default="species_to_kegg.csv",
        help="Path to the output file for KEGG mapping (default: species_to_kegg.csv)"
    )
    
    parser.add_argument(
        "--reactions-output",
        type=str,
        default="reactions.json",
        help="Path to the output file for reactions (default: reactions.json)"
    )
    
    parser.add_argument(
        "--annotated-model-output",
        type=str,
        default="annotated_model.xml",
        help="Path to the output file for the annotated model (default: annotated_model.xml)"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-3-sonnet-20240229",
        help="LLM model to use (default: claude-3-sonnet-20240229)"
    )
    
    parser.add_argument(
        "--min-match",
        type=int,
        default=2,
        help="Minimum number of matching compounds to consider a match (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Annotate model with ChEBI IDs
    species_to_chebi = annotate_sbml_model(args.model, args.llm_model)
    
    if not species_to_chebi:
        logger.error("Annotation failed")
        sys.exit(1)
    
    # Map ChEBI IDs to KEGG compound IDs
    species_to_kegg = map_chebi_to_kegg_ids(species_to_chebi)
    
    # Parse reactions
    reactions = parse_reactions(args.model, species_to_chebi, species_to_kegg)
    
    # Find matching KEGG reactions
    reactions = find_matching_kegg_reactions(reactions, args.min_match)
    
    # Save species to ChEBI mapping results
    save_results(species_to_chebi, args.output)
    
    # Save species to KEGG mapping results
    df = pd.DataFrame({
        'species_id': list(species_to_kegg.keys()),
        'kegg_id': list(species_to_kegg.values())
    })
    df.to_csv(args.kegg_output, index=False)
    logger.info(f"KEGG mapping saved to {args.kegg_output}")
    
    # Save reaction results with annotations
    annotated_reactions = []
    for reaction in reactions:
        # Extract annotations from KEGG matches
        kegg_annotations = {
            "ec_numbers": set(),
            "enzyme_names": set(),
            "kegg_reaction_ids": set(),
            "pathways": set()
        }
        
        for match in reaction.get("kegg_matches", []):
            # Add EC numbers
            for ec in match.get("ec_numbers", []):
                kegg_annotations["ec_numbers"].add(ec)
            
            # Add enzyme name
            if match.get("name"):
                kegg_annotations["enzyme_names"].add(match.get("name"))
            
            # Add KEGG reaction ID
            if match.get("reaction_id"):
                kegg_annotations["kegg_reaction_ids"].add(match.get("reaction_id"))
            
            # Add pathways
            for pathway in match.get("pathways", []):
                kegg_annotations["pathways"].add(pathway)
        
        # Convert sets to lists for JSON serialization
        reaction["kegg_annotations"] = {
            "ec_numbers": list(kegg_annotations["ec_numbers"]),
            "enzyme_names": list(kegg_annotations["enzyme_names"]),
            "kegg_reaction_ids": list(kegg_annotations["kegg_reaction_ids"]),
            "pathways": list(kegg_annotations["pathways"])
        }
        
        annotated_reactions.append(reaction)
    
    # Save reaction results
    with open(args.reactions_output, 'w') as f:
        json.dump(annotated_reactions, f, indent=2)
    logger.info(f"Reaction information saved to {args.reactions_output}")
    
    # Insert annotations into the model
    annotated_model = insert_annotations_into_model(args.model, annotated_reactions, args.annotated_model_output)
    
    logger.info("Annotation completed successfully")

if __name__ == "__main__":
    main()