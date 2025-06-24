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

logger = logging.getLogger(__name__)

# System prompt from AMAS for chemical annotation
SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized or canonical chemical names or common synonyms for ontology lookup on ChEBI. Only consider them as chemical entities, return "UNK" if not or unsure.

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

# System prompt for gene annotation
SYSTEM_PROMPT_GENE = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized gene names or common gene symbols for ontology lookup on NCBI Gene. Only consider them as gene entities, return "UNK" if not or unsure.

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

# Backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_CHEMICAL

def get_system_prompt(entity_type: str = "chemical") -> str:
    """
    Get the appropriate system prompt based on entity type.
    
    Args:
        entity_type: Type of entity ("chemical", "gene")
        
    Returns:
        System prompt string
    """
    if entity_type == "chemical":
        return SYSTEM_PROMPT_CHEMICAL
    elif entity_type == "gene":
        return SYSTEM_PROMPT_GENE
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
                temperature=0.2,
                max_tokens=10000
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
                temperature=0.2,
                max_tokens=10000
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
                temperature=0.2
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

def parse_llm_response(response) -> Tuple[Dict[str, List[str]], str]:
    """
    Parse the LLM response to extract species synonyms in the format:
    SpeciesA: "name1", "name2", ...
    SpeciesB: "name1", name2, ...
    Reason: ...
    
    Exact replication of parse_llm_response from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        response: The raw response string from the LLM
        
    Returns:
        Tuple containing:
        - Dictionary mapping species IDs to lists of synonyms
        - Reason string
    """
    # Remove markdown code block syntax if present
    response = re.sub(r'```.*?\n', '', response)
    response = re.sub(r'```\s*$', '', response)
    
    # Initialize the dictionary and reason
    synonyms_dict = {}
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

        # Parse species lines (in format "SpeciesA: "name1", "name2", ...)
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        species_id = parts[0].strip()
        names_str = parts[1].strip()

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

    return synonyms_dict, reason 