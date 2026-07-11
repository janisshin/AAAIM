"""
LLM Interface for AAAIM

Handles LLM interactions for annotation.
"""

import os
import re
import time
import requests
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI, RateLimitError, APIError
import logging
from utils.constants import EntityType

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Default retry settings for rate limit errors (429)
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 10  # seconds
DEFAULT_MAX_DELAY = 120  # seconds (2 minutes max wait)

logger = logging.getLogger(__name__)

# Helper function to get entity type options from enum
def _get_entity_type_options() -> str:
    """Get comma-separated list of entity types from EntityType enum."""
    # Exclude REACTION from auto-detection as they're not applicable to species
    exclude_types = ['reaction']
    types = [e.value for e in EntityType if e.value not in exclude_types]
    return ', '.join(types)

# Automatic entity type detection
SYSTEM_PROMPT_AUTO = f"""You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized names for ontology lookup, and determine the entity type for each species.
For each species, identify entity type from the following options: [{_get_entity_type_options()}]. Specify the entity type in parentheses after the species ID, followed by synonyms. Note that amino acids and tRNAs are considered as chemical. 
For complexes, do not give the name of the complex, only list standardized names of the chemical and protein components, separated by commas (no other symbols like “:” or “-”). E.g., for "EGF-EGFR^2", return "EGF", "EGFR".
Try your best to give the most likely terminology without modifications (e.g., no "phosphorylated") or extra information (e.g., no “protein”, “complex”, or localization terms like “nuclear”).

Here is one example:
Species to annotate: A, B, C, D
Model: "hexokinase reaction"
// Display Names:
A is "glucose";
B is "ATP";
C is "hexokinase (cytoplasmic)";
D is "glucose-ATP-hexokinase complex (active)";

// Reactions:
A + B + C -> D;
D -> products;

This should return:
A (chemical): "glucose", "D-glucose"
B (chemical): "ATP", "adenosine triphosphate"
C (protein): "Hexokinase-1", "HK1"
D (complex): "glucose", "ATP", "Hexokinase-1"
Reason: A and B are small-molecule substrates (chemicals), C is the enzyme (protein), and D represents the enzyme–substrate complex. For the complex D, the complex name and extra info (“complex”, “active”) are removed, and only the standardized names of its components are listed.
"""

# System prompt for chemical annotation
# SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized names for ontology lookup on ChEBI. 
# All given species are chemical entities. For complexes, only consider the chemical components. If lacking information about details, try your best to give the most likely general name.
# Do not include modifications or extra information (e.g., no “dissolved”, “anion”, or localization terms like “nuclear”).
# """

SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized names for ontology lookup on ChEBI. 
All given species are chemical entities. For complexes, only consider the chemical components. If lacking information about details, try your best to give the most likely general name.
Do not include modifications or extra information (e.g., no “dissolved”, “anion”, or localization terms like “nuclear”).

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
All given species are proteins. For complexes, only consider the protein components and separate their names with commas. E.g., for "EGF-EGFR^2", return "EGF", "EGFR".
Try your best to give the most likely standardized terminology without any extra information. E.g., a model may contain various states (e.g., phosphorylated, nuclear, or transcribed) of the same protein, you should only return the most likely standard name like "BMAL1" but not "BMAL1_phosphorylated".
For protein names that represent a family or ambiguous label, return all reasonable subtype or isoform candidates. E.g., “AKT” → AKT1, AKT2, AKT3; “RAS” → KRAS, NRAS, HRAS

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

def _make_api_call_with_retry(client, model: str, messages: list, 
                               max_retries: int = DEFAULT_MAX_RETRIES,
                               initial_delay: float = DEFAULT_INITIAL_DELAY,
                               max_delay: float = DEFAULT_MAX_DELAY,
                               api_name: str = "API"):
    """
    Make an API call with retry logic for rate limit errors (429).
    
    Args:
        client: OpenAI client instance
        model: Model name
        messages: List of message dicts for the API
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        api_name: Name of the API for logging
        
    Returns:
        API response or None on failure
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            # print(response)
            return response
            
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries:
                # Extract wait time from error message if available
                wait_time = delay
                error_msg = str(e)
                
                # Try to parse retry-after from error message
                if "retry after" in error_msg.lower():
                    try:
                        # Look for patterns like "retry after X seconds" or "try again in X"
                        import re
                        match = re.search(r'(\d+)\s*(?:seconds?|s)', error_msg.lower())
                        if match:
                            suggested_wait = int(match.group(1))
                            wait_time = max(wait_time, suggested_wait + 1)  # Add 1s buffer
                    except:
                        pass
                
                # Cap at max_delay
                wait_time = min(wait_time, max_delay)
                
                print(f"Rate limit error (429) from {api_name}. Attempt {attempt + 1}/{max_retries + 1}. "
                      f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                
                # Exponential backoff for next attempt
                delay = min(delay * 2, max_delay)
            else:
                print(f"Rate limit error (429) from {api_name}. Max retries ({max_retries}) exceeded.")
                
        except APIError as e:
            # Handle other API errors (500, 502, 503, etc.)
            last_exception = e
            status_code = getattr(e, "status_code", None)
            if status_code in [500, 502, 503, 504] and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(f"API error ({status_code}) from {api_name}. Attempt {attempt + 1}/{max_retries + 1}. "
                      f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
            else:
                print(f"API error from {api_name}: {e}")
                break
                
        except Exception as e:
            # Non-retryable error
            print(f"Error querying {api_name}: {e}")
            return None
    
    # All retries exhausted
    if last_exception:
        print(f"All retries exhausted for {api_name}. Last error: {last_exception}")
    return None


def _make_openrouter_api_call_with_retry(
    model: str,
    messages: list,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
):
    """
    Make an OpenRouter chat completion request.

    Uses raw HTTP so provider-specific fields like ``reasoning_details`` are
    preserved and can be passed back unchanged in follow-up requests.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter models")

    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
    }

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            if response.status_code == 429 and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(
                    f"Rate limit error (429) from OpenRouter. Attempt {attempt + 1}/{max_retries + 1}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
                continue

            if response.status_code in [500, 502, 503, 504] and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(
                    f"API error ({response.status_code}) from OpenRouter. Attempt {attempt + 1}/{max_retries + 1}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            last_exception = e
            print(f"API error from OpenRouter: {e}")
            break

    if last_exception:
        print(f"All retries exhausted for OpenRouter. Last error: {last_exception}")
    return None


def _extract_response_text(response: Any) -> Optional[str]:
    """
    Extract assistant text from the response shapes used by supported chat APIs.

    OpenAI-compatible chat completions return ``choices[0].message.content``.
    """
    if response is None:
        return None

    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)

    return None


def _extract_assistant_message(response: Any) -> Optional[Dict[str, Any]]:
    """Extract an assistant message dict, preserving OpenRouter reasoning details."""
    if response is None:
        return None

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        assistant_message = {
            "role": "assistant",
            "content": message.get("content") or "",
        }
        if message.get("reasoning_details") is not None:
            assistant_message["reasoning_details"] = message.get("reasoning_details")
        return assistant_message

    text = _extract_response_text(response)
    if text is None:
        return None
    return {"role": "assistant", "content": text}


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("meta-llama") or model.startswith("openrouter/")


def query_llm_message(
    prompt: str,
    developer_prompt: str = None,
    model="gpt-4o-mini",
    entity_type: str = "chemical",
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
) -> Dict[str, Any]:
    """
    Query the configured LLM and return the assistant message dict.

    For OpenRouter reasoning requests, the returned dict includes
    ``reasoning_details`` when the provider returns it; pass that dict back
    unchanged in later message history.
    """
    if developer_prompt is None:
        developer_prompt = get_system_prompt(entity_type)

    messages = [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": prompt}
    ]

    return query_llm_message_with_history(
        messages,
        model=model,
        max_retries=max_retries,
        initial_delay=initial_delay,
    )


def query_llm(prompt: str, developer_prompt: str = None, model="gpt-4o-mini", entity_type: str = "chemical",
              max_retries: int = DEFAULT_MAX_RETRIES, initial_delay: float = DEFAULT_INITIAL_DELAY):
    """
    Query the configured LLM with the formatted prompt.
    Includes automatic retry with exponential backoff for rate limit errors (429).
    
    Args:
        prompt: The formatted prompt to send to the LLM
        developer_prompt: The system prompt (if None, will use appropriate prompt for entity_type)
        model: The model to use, e.g. "gpt-4o-mini" or
            "meta-llama/llama-3.3-70b-instruct:free"
        entity_type: Type of entity for prompt selection if developer_prompt is None
        max_retries: Maximum number of retry attempts for rate limit errors (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 10)

    Returns:
        String response from LLM or empty string on error
    """
    assistant_message = query_llm_message(
        prompt,
        developer_prompt,
        model=model,
        entity_type=entity_type,
        max_retries=max_retries,
        initial_delay=initial_delay,
    )
    text = assistant_message.get("content") if assistant_message else None
    if text:
        return text
    else:
        print("No response or empty response from LLM.")
        return ""

def query_llm_message_with_history(
    messages: list,
    model: str = "gpt-4o-mini",
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
) -> Dict[str, Any]:
    """Query the LLM with full history and return the assistant message dict."""
    response = None
    if model.startswith("gpt"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries, initial_delay=initial_delay,
            api_name="OpenAI"
        )
    elif _is_openrouter_model(model):
        response = _make_openrouter_api_call_with_retry(
            model,
            messages,
            max_retries=max_retries,
            initial_delay=initial_delay,
        )
    else:
        raise ValueError(
            f"Model {model} not supported. Use an OpenAI model starting with "
            "'gpt' or an OpenRouter model starting with 'meta-llama' or 'openrouter/'."
        )

    return _extract_assistant_message(response) or {}


def query_llm_with_history(messages: list, model: str = "gpt-4o-mini",
                           max_retries: int = DEFAULT_MAX_RETRIES,
                           initial_delay: float = DEFAULT_INITIAL_DELAY) -> str:
    """
    Query the LLM with a full conversation history (multi-turn).
    
    Used by the feedback loop to send the original prompt, the LLM's prior
    response, and user feedback as a coherent conversation so the LLM can
    revise its output.
    
    Args:
        messages: List of message dicts (role/content) representing the full
                  conversation so far, including system, user, assistant, and
                  feedback turns.
        model: LLM model identifier.
        max_retries: Retry attempts for rate-limit / transient errors.
        initial_delay: Initial backoff delay in seconds.

    Returns:
        The assistant's response text, or empty string on failure.
    """
    assistant_message = query_llm_message_with_history(
        messages,
        model=model,
        max_retries=max_retries,
        initial_delay=initial_delay,
    )
    text = assistant_message.get("content") if assistant_message else None
    if text:
        return text
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
