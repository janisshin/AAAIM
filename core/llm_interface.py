"""
LLM Interface for AAAIM

Handles LLM interactions for annotation.
"""

import os
import re
import time
from typing import Dict, List, Tuple, Any
from openai import OpenAI
import logging
from utils.constants import EntityType

logger = logging.getLogger(__name__)

# Helper function to get entity type options from enum
def _get_entity_type_options() -> str:
    """Get comma-separated list of entity types from EntityType enum."""
    # Exclude REACTION from auto-detection as they're not applicable to species
    exclude_types = ['reaction']
    types = [e.value for e in EntityType if e.value not in exclude_types]
    return ', '.join(types)

# Automatic entity type detection
SYSTEM_PROMPT_AUTO = f"""You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized or canonical names for ontology lookup, and determine the entity type for each species.
For each species, identify whether it is chemical, protein, or complex. Specify the entity type in parentheses after the species ID, followed by synonyms.
Entity types should be one of: [{_get_entity_type_options()}]. Note that amino acids and tRNAs are considered as chemical. Try your best to give the most likely names without modifications (e.g., no "phosphorylated") or extra information (e.g., no “protein”, “complex”, or localization terms like “nuclear”).
For complexes, consider both the protein and chemical components, and list them all separately with commas.

Here is one example:
Species to annotate: A, B, E
Model: "glycolysis model"
// Display Names:
A is "glucose";
B is "ATP";
C is "glucose-6-phosphate";
D is "ADP";
E is "hexokinase";

// Reactions:
A + B -> C + D; kcat * E * A * B

This should return:
A (chemical): "glucose", "D-glucose"
B (chemical): "ATP", "adenosine triphosphate"
E (protein): "hexokinase", "ATP:D-hexose 6-phosphotransferase"
Reason: This reaction represents the first step of glycolysis, where hexokinase (E) catalyzes the phosphorylation of glucose (A) by ATP (B) to form glucose-6-phosphate (C) and ADP (D).
"""

# System prompt for chemical annotation
SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized or canonical chemical names for ontology lookup on ChEBI. 
All given species are chemical entities. For complexes, only consider the chemical components. If lacking information about details, try your best to give the most likely general name.

Here is one example:
Species: A, B, D
Model: "citric acid cycle model"
 // Display Names:
A is "acetyl-CoA";
B is "citrate";
C is "CoA";
 // Reactions:
A + oxaloacetate => B + C;
E + F => D;

This should return:
A: "acetyl-CoA", "acetyl coenzyme A"
B: "citric acid", "sodium citrate", "citrate(4−)"
D: "UNK"
Reason: the reaction is likely to be the TCA cycle, where A is the substrate and B is an intermediate. D is unknown because no display names are given for its reactants."""

# SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized or canonical chemical names for ontology lookup on ChEBI. 
# All given species are chemical entities. For complexes, only consider the chemical components.
# Return "UNK" if not or unsure.

# Here is one example:
# Species: A, B, D
# Model: "citric acid cycle model"
#  // Display Names:
# A is "acetyl-CoA";
# B is "citrate";
# C is "CoA";
#  // Reactions:
# A + oxaloacetate => B + C;
# E + F => D;

# This should return:
# A: "acetyl-CoA", "acetyl coenzyme A"
# B: "citric acid", "sodium citrate", "citrate(4−)"
# D: "UNK"
# Reason: the reaction is likely to be the TCA cycle, where A is the substrate and B is an intermediate. D is unknown because no display names are given for its reactants."""

# System prompt for gene annotation
SYSTEM_PROMPT_GENE = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized gene names or common gene symbols for ontology lookup on NCBI Gene. 
All given species are genes. For complexes, only consider the gene components. If lacking information about details, try your best to give the most likely general name.

Here is one example:
Species: G1, G2, G3
Model: "NF-κB signaling pathway"
 // Display Names:
G1 is "p65";
G2 is "p50";
G3 is "IKK";
 // Reactions:
G1 = G1 | (G3 & !(G1 & G2))
G2 = G1
G3 = G3

This should return:
G1: "RELA", "p65", "NFKB3"
G2: "NFKB1", "KBF1", "NF-kB"
G3: "CHUK", "IKK1", "BPS2"
Reason: This appears to be a regulatory motif in the NF-κB signaling pathway. G1 is the p65 subunit (RELA), G2 is the p50 subunit (NFKB1), and G3 is IKK, a kinase that phosphorylates p50."""

# System prompt for protein annotation
SYSTEM_PROMPT_PROTEIN = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized protein names for ontology lookup on UniProt.
All given species are proteins. For complexes, only consider the protein components and separate their names with commas. Try your best to give the most likely protein names without modifications (e.g., no "phosphorylated") or extra information (e.g., no “protein”, “complex”, or localization terms like “nuclear”).
Here is one example:
Species: C1, C2
Model: "NF-κB signaling pathway"
// Display Names:  
C1 is "NFκB (nuclear)";  
C2 is "IKK complex";  
// Reactions:  
C2 => phosphorylates C1;  
C1 (cytoplasmic) => C1 (nuclear);  

This should return:
C1: NFKB1, RELA  
C2: CHUK, IKBKB, IKBKG
Reason: “NFkB (nuclear)” refers to the activated NF-κB complex, typically composed of NFKB1 (p50) and RELA (p65). The “IKK complex” consists of CHUK (IKKα), IKBKB (IKKβ), and IKBKG (NEMO). Extra terms like “nuclear” are ignored, and only the UniProt protein names of the components are listed, separated by commas."""

# System prompt for reaction and enzyme annotation
SYSTEM_PROMPT_REACTION = """You are a biomedical knowledge assistant. Your task is to normalize reaction and enzyme names from biochemical models into standardized or canonical reaction or enzyme names for ontology lookup on KEGG. 
Examine each reaction's label, and its substrates and products to determine the enzyme or process responsible for the reaction. If lacking information about details, try your best to give the most likely description. Return "UNK" if not or unsure.

Here is one example:
Species: A, B, D
Model: "citric acid cycle model"
 // Display Names:
J1 is "CS";
J2 is "ACON";
J3 is "IDH";
 // Reactions:
J1: AcetylCoA + OAA -> Citrate + CoA; 
J2: Citrate <-> Isocitrate;
J3: Isocitrate + NAD -> AKG + CO2 + NADH;

This should return:
J1: "Citrate synthase",
J2: "Aconitase"
J3: "Isocitrate dehydrogenase"
Reason: these reactions match the reactions found in the TCA cycle """

SYSTEM_PROMPT = SYSTEM_PROMPT_AUTO

def get_system_prompt(entity_type: str = "chemical") -> str:
    """
    Get the appropriate system prompt based on entity type.
    
    Args:
        entity_type: Type of entity ("chemical", "gene", "protein", "auto")
        
    Returns:
        System prompt string
    """
    if entity_type == "auto":
        return SYSTEM_PROMPT_AUTO
    elif entity_type == "chemical":
        return SYSTEM_PROMPT_CHEMICAL
    elif entity_type == "gene":
        return SYSTEM_PROMPT_GENE
    elif entity_type == "protein":
        return SYSTEM_PROMPT_PROTEIN
    elif entity_type == "reaction":
        return SYSTEM_PROMPT_REACTION
    else:
        logger.warning(f"Unknown entity type {entity_type}, using chemical prompt")
        return SYSTEM_PROMPT_CHEMICAL

def query_llm(prompt: str, developer_prompt: str = None, model="gpt-4o-mini", entity_type: str = "chemical"):
    """
    Query the OpenAI LLM with the formatted prompt.
    Exact replication of query_llm from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        prompt: The formatted prompt to send to the LLM
        developer_prompt: The system prompt (if None, will use appropriate prompt for entity_type)
        model: The model to use for the LLM, e.g., "meta-llama/llama-3.3-70b-instruct:free"
        entity_type: Type of entity for prompt selection if developer_prompt is None

    Returns:
        String response from LLM or empty string on error
    """
    if developer_prompt is None:
        developer_prompt = get_system_prompt(entity_type)
    
    response = None
    if model.startswith("gpt"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": developer_prompt},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0.2,
                # max_tokens=10000 # this changes to max_completion_tokens for gpt-5
            )
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return ""
    elif model.startswith("meta-llama"):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": developer_prompt},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0.2,
                # max_tokens=10000
            )
        except Exception as e:
            print(f"Error querying OpenRouter: {e}")
            return ""
    elif model.startswith("Llama"):
        client = OpenAI(base_url="https://api.llama.com/compat/v1", api_key=os.getenv("LLAMA_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": developer_prompt},
                    {"role": "user", "content": prompt}
                ],
                # max_tokens=10000,
                # temperature=0.2
            )
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return ""
    else:
        raise ValueError(f"Model {model} not supported")
    
    if response is not None and hasattr(response, "choices") and response.choices:
        return response.choices[0].message.content
    else:
        print("No response or empty response from LLM.")
        return ""

def parse_llm_response(response, entity_type: str = "auto") -> Tuple[Dict[str, List[str]], Dict[str, str], str]:
    """
    Parse the LLM response to extract species synonyms and entity types in the format:
    SpeciesA (chemical): "name1", "name2", ...
    SpeciesB (gene): "name1", name2, ...
    Reason: ...
    
    Extended to support automatic entity type detection.
    
    Args:
        response: The raw response string from the LLM
        entity_type: The entity type being used ("auto" for automatic detection, 
                     or specific type like "chemical", "gene", "protein")
        
    Returns:
        Tuple containing:
        - Dictionary mapping species IDs to lists of synonyms
        - Dictionary mapping species IDs to entity types
        - Reason string
    """
    # Remove markdown code block syntax if present
    response = re.sub(r'```.*?\n', '', response)
    response = re.sub(r'```\s*$', '', response)
    
    # Initialize the dictionaries and reason
    synonyms_dict = {}
    entity_type_dict = {}
    reason = ""
    
    # Split response into lines
    lines = response.strip().split('\n')
    reason_start = None

    # Find the line where 'Reason:' starts
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith('reason:'):
            reason_start = idx
            break

    if reason_start is not None:
        # Everything after 'Reason:' is the reason, including the rest of the lines
        reason_lines = lines[reason_start:]
        if reason_lines:
            # Remove the 'Reason:' prefix from the first line
            first_line = reason_lines[0]
            reason = first_line[first_line.lower().find('reason:') + len('reason:'):].strip()
            # Add the rest of the lines (if any)
            if len(reason_lines) > 1:
                reason += '\n' + '\n'.join(l.strip() for l in reason_lines[1:])
        # Only parse synonym lines before 'Reason:'
        lines = lines[:reason_start]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse with entity type format: "SpeciesA (entity_type): names..."
        entity_type_pattern = r'^([A-Za-z0-9_]+)\s*\((\w+)\):\s*(.+)$'
        entity_type_match = re.match(entity_type_pattern, line)
        
        if entity_type_match:
            # Format with entity type
            species_id = entity_type_match.group(1).strip()
            detected_type = entity_type_match.group(2).strip().lower() 
            names_str = entity_type_match.group(3).strip()
            # Only use detected type if in auto mode, otherwise use specified entity_type
            if entity_type == "auto":
                entity_type_dict[species_id] = detected_type
            else:
                entity_type_dict[species_id] = entity_type
        else:
            # Standard format without entity type: "SpeciesA: names..."
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            species_id = parts[0].strip()
            names_str = parts[1].strip()
            # Use specified entity_type, or "unknown" only if in auto mode
            if entity_type == "auto":
                entity_type_dict[species_id] = "unknown"
            else:
                entity_type_dict[species_id] = entity_type

        # Extract all synonyms, handling both quoted and unquoted names
        names = []

        # First, extract all quoted items
        quoted_items = re.findall(r'"([^"]*)"', names_str)
        names.extend(quoted_items)

        # Remove quoted parts from the string for further processing
        processed_str = names_str
        for item in quoted_items:
            processed_str = processed_str.replace(f'"{item}"', '')

        # Now extract unquoted items by splitting on commas
        unquoted_parts = [part.strip() for part in processed_str.split(',')]
        for part in unquoted_parts:
            if part and not part.isspace():
                names.append(part)

        # Remove any empty strings that might have been added
        names = [name for name in names if name and not name.isspace()]

        if names:
            synonyms_dict[species_id] = names

    # Handle case where parsing failed
    if not synonyms_dict and not reason:
        print("Failed to parse response:")
        print(response)
        # Save the response with timestamp
        timestamp = int(time.time())
        error_file = f"error_response_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(str(response))
        print(f"Error response saved to: {error_file}")

    return synonyms_dict, entity_type_dict, reason 