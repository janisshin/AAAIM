import os
import sys
import pandas as pd
from itertools import chain
import lzma
import pickle

from dotenv import load_dotenv
load_dotenv()

import re
from collections import Counter
from rapidfuzz import fuzz

from pathlib import Path
# Add parent directory to path to import AAAIM modules
sys.path.append(str(Path(__file__).parent.parent))

from core import annotate_model, curate_model, get_available_databases, database_search
from core import normalize_reactions
from core import load_chebi2kegg_dict, load_kegg_reaction_features_dict
from core.update_model import update_annotation
from core.model_info import extract_reactions_from_sbml, extract_model_info
from core.annotation_workflow import map_reactions_to_kegg, _generate_recommendation_table
from core.model_info import get_all_reaction_ids


print("AAAIM KEGG Reaction Annotation Example")
print("=" * 50)

# Configuration
model_file = "../tests/glycolysis_part1.xml"
# model_file = "tests/test_models/BIOMD0000000190.xml"
file_name = model_file.split('.')[0]

# llm_model = "meta-llama/llama-3.3-70b-instruct:free"  # or "gpt-4o-mini"
llm_model = "meta-llama/llama-3.1-8b-instruct"
top_k = 10

entity_type='reaction'
database='kegg'
cofactors_to_ignore = {
    'C00001',  # H2O
    'C00080',  # H+
    'C00007',  # O2
    'C00027',  # H2O2
    'C00009',  # Phosphate
    'C00013',  # Diphosphate
    'C00008',  # ADP
    'C00002',  # ATP
    'C00003',  # NAD+
    'C00004',  # NADH
    'C00005',  # NADPH
    'C00006',  # NADP+
}


## ----- HELPER METHODS
def extract_classifications(raw_text, classification):
    """
    classification (str): either 'brite' or 'orthology'
    Extracts only the BRITE hierarchy (excluding [BR:...] tags, EC leaf nodes, 
    and reaction entries).
    """
    lines = raw_text.splitlines()
    clean_lines = []

    if classification == 'brite':
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # Skip lines with [BR:...] tags
            if "[BR:" in stripped:
                continue
            # Skip EC leaf numbers (pure numbers like 2.2.1.6)
            if re.fullmatch(r"(\d+\.)+\d+", stripped):
                continue
            # Skip lines that start with an R number (reaction ID)
            if re.match(r"R\d{5}", stripped):
                continue
            
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1:
                clean_lines.append(parts[1].strip())
            else:
                clean_lines.append(stripped)
    
    elif classification == 'orthology':
        for line in lines:
        # Split once on spaces to remove the Kxxxxx ID
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                # Remove the EC info if present
                name = parts[1].split(" [EC:")[0].strip()
                clean_lines.append(name)

    elif classification == 'definition':
        parts = []
        buf = ""
        paren_level = 0  # Track nested parentheses

        i = 0
        while i < len(raw_text):
            c = raw_text[i]

            # Track parentheses
            if c == '(':
                paren_level += 1
            elif c == ')':
                paren_level -= 1

            # Split points: + outside parentheses or <=>
            if c == '+' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
            elif raw_text[i:i+3] == '<=>' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 2  # skip the next two chars of <=>
            elif raw_text[i:i+2] == '->' and paren_level == 0:
                parts.append(buf.strip())
                buf = ""
                i += 1  
            else:
                buf += c

            i += 1

        # Add remaining buffer
        if buf:
            parts.append(buf.strip())
        # parts = [p for p in parts if p]
        strip_dollars = [p.lstrip("$") for p in parts if p]
        clean_lines = [
            re.sub(r'^(?:\(?[0-9nmt+\-*/]+\)?\s+)+', '', p).strip()
            for p in strip_dollars
        ]
    
    return "; ".join(set(clean_lines))

def normalize(prob_dict):
    total = sum(prob_dict.values())
    if total > 0:
        for key in prob_dict:
            prob_dict[key] /= total
    return prob_dict

def update_species_probs(query_species, candidate_reactions, candidate_probs):
    """
    Update the species match probabilities for a single query species
    based on the candidate reaction probabilities.
    
    Returns a dictionary mapping candidate species to updated probabilities.
    """
    updated_probs = {}

    # Loop over each candidate reaction
    for candidate in candidate_reactions:
        prob_candidate = candidate_probs.get(candidate, 0.0)
        
        # Loop over each species in the candidate reaction
        for cand_species in candidate.participants:
            # Check if this candidate species could match the query species
            if is_plausible_match(query_species, cand_species):
                # Accumulate probability weighted by candidate reaction probability
                if cand_species not in updated_probs:
                    updated_probs[cand_species] = 0.0
                updated_probs[cand_species] += prob_candidate

    # Normalize the probabilities so they sum to 1
    total = sum(updated_probs.values())
    if total > 0:
        for species in updated_probs:
            updated_probs[species] /= total

    return updated_probs

def choose_best_annotation(species_probs):
    """
    Choose the best annotation for a query species based on
    the current species match probabilities.
    
    Parameters:
        species_probs (dict): Mapping from candidate species to probabilities.
        
    Returns:
        str: The candidate species with the highest probability.
             Returns None if no candidates are available.
    """
    if not species_probs:
        return None  # no plausible matches
    
    # Find the candidate with the maximum probability
    best_species = max(species_probs, key=species_probs.get)
    return best_species

def has_converged(updated_annotations, previous_annotations):
    """
    Check if the EM algorithm has converged for a query reaction.
    
    Convergence occurs when the annotations of all species do not change
    from the previous iteration.
    
    Parameters:
        updated_annotations (dict): Mapping from query species to current annotation.
        previous_annotations (dict): Mapping from query species to previous annotation.
        
    Returns:
        bool: True if converged, False otherwise.
    """
    # If there were no previous annotations, we haven't converged yet
    if not previous_annotations:
        return False
    
    # Compare updated vs previous for all species
    for species, new_annotation in updated_annotations.items():
        old_annotation = previous_annotations.get(species)
        if new_annotation != old_annotation:
            return False  # At least one species changed, not converged
    
    return True  # All species unchanged, converged

def is_plausible_match(query_species, cand_species, threshold=80):
    """
    Decide if a candidate species could match the query species.
    For now, match if names are identical or similar.
    """
    max_score = fuzz.partial_ratio(query_species.lower(), cand_species.lower())
    return max_score >= threshold
    # return query_species.lower() == cand_species.lower()


def init_species_probs_from_dict(reaction_participants, counters):
    """
    Initialize species match probabilities for each reaction.

    Parameters
    ----------
    reaction_participants : dict
        reaction_id -> list of query participants
    counters : pd.Series
        reaction_id -> Counter of candidate species across candidate reactions

    Returns
    -------
    dict
        reaction_id -> {query_species: {candidate_species: prob}}
    """
    species_match_probs = {}

    for rxn_id, query_species_list in reaction_participants.items():
        candidate_counter = counters[rxn_id]
        species_probs_for_rxn = {}

        for query_species in query_species_list:
            # keep only plausible matches
            plausible = {cand: count for cand, count in candidate_counter.items()
                         if is_plausible_match(query_species, cand)}

            if plausible:
                total = sum(plausible.values())
                species_probs_for_rxn[query_species] = {
                    cand: count / total for cand, count in plausible.items()
                }
            else:
                species_probs_for_rxn[query_species] = {}

        species_match_probs[rxn_id] = species_probs_for_rxn

    return species_match_probs

def get_participants(annotation):
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("DEFINITION", "")
    return extract_classifications(definition, 'definition')

def get_participant_ids(annotation):
    kegg_id = annotation.split(':')[1] if ':' in annotation else annotation
    definition = kegg_reaction_features.get(kegg_id, {}).get("EQUATION", "")
    return extract_classifications(definition, 'definition')

def compute_reaction_likelihoods(init_probs, kegg_recommendations_df):
    """
    Compute likelihood scores for each candidate reaction given initial species probabilities.

    Parameters
    ----------
    init_probs : dict
        reaction_id -> {query_species: {candidate_species: prob}}
    kegg_recommendations_df : pd.DataFrame
        Columns: ['id', 'annotation', 'participants', 'participant_ids']

    Returns
    -------
    pd.DataFrame
        Same as input, with an extra column 'likelihood' containing the computed likelihood score
    """

    def fuzzy_jaccard(set_a, set_b, threshold=70):
        """
        Compute fuzzy Jaccard similarity between two sets of strings.
        threshold: minimum similarity to count as a match (0-100).
        """
        overlap = 0
        for a in set_a:
            best = max((fuzz.ratio(a, b) / 100 for b in set_b), default=0)
            if best * 100 >= threshold:
                overlap += best  # fractional overlap
        denom = len(set_a) + len(set_b) - overlap
        return overlap / denom if denom > 0 else 0

    likelihoods = []
    
    for _, row in kegg_recommendations_df.iterrows():
        rxn_id = row['id']
        species_annotations=dict()
        candidate_participants = set(row['participants'].split("; "))

        # --- 1. Compute species-level match probability ---
        prob_product = 1.0
        if rxn_id in init_probs:
            query_participants = set([i[0] for i in init_probs[rxn_id].items()])
            species_probs = init_probs[rxn_id]
            for query_species, cand_dict in species_probs.items():
                match_probs = [cand_dict[cand] for cand in candidate_participants if cand in cand_dict]
                prob_product *= max(match_probs) if match_probs else 1e-6

        else:
            prob_product = 1e-6

        # --- 2. Compute Jaccard penalty for extra participants ---
        filtered_candidate_participants = candidate_participants.copy()
        if not (query_participants & cofactors_to_ignore):
            filtered_candidate_participants -= cofactors_to_ignore

        jaccard_score = fuzzy_jaccard(query_participants, filtered_candidate_participants)

        # --- 3. Combine both scores ---
        likelihood = prob_product * jaccard_score
        likelihoods.append(likelihood)
    

    kegg_recommendations_df = kegg_recommendations_df.copy()
    kegg_recommendations_df['likelihood'] = likelihoods

    # Rescale so each group of candidate reaction likelihoods sums to 1
    group_sums = kegg_recommendations_df.groupby('id')['likelihood'].transform('sum')
    kegg_recommendations_df['likelihood'] = kegg_recommendations_df['likelihood'] / group_sums
    return kegg_recommendations_df






## ----- STEP 1
all_entity_ids = get_all_reaction_ids(model_file)
model_info = extract_model_info(model_file, all_entity_ids, entity_type)

# Check if KEGG reaction database is available
available_dbs = get_available_databases()
print(f"Available databases: {available_dbs}")

if "chebi" not in available_dbs:
    print("ERROR: ChEBI chemical database not available!")
    print("Please ensure ChEBI reference files are present in data/chebi/")
    

if "kegg" not in available_dbs:
    print("ERROR: KEGG reaction database not available!")
    print("Please ensure KEGG reference files are present in data/kegg/")
    

# Check if model file exists
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    print("Please provide a valid SBML model file.")
    

# Check API keys
if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    print("Warning: No API keys found in environment.")
    print("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
    

print(f"Model file: {model_file}")
print(f"LLM model: {llm_model}")
print()

print(f"\nAnalyzing model: {model_file}")

# Example 1: Reaction Annotation Workflow
print("\nEXAMPLE 1. Reaction Annotation Workflow (for models without reaction annotations)")
print("-" * 60)

## this line below should be deleted eventually
recommendations_df = pd.read_csv('../recommendations_correctedChEBI.csv')

## ----- STEP 2
print("\nStep 2: Map ChEBI IDs to KEGG Compound IDs")
print("-" * 60)  

# Load ChEBI to KEGG mapping
chebi_to_kegg_map = load_chebi2kegg_dict()

# Add KEGG IDs to chemical recommendations if available
if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
    # Map ChEBI IDs to KEGG IDs
    recommendations_df['KEGG_ID'] = recommendations_df['annotation'].apply(
        lambda x: chebi_to_kegg_map.get(x, "")
    )

# Filter out rows with empty KEGG_ID
filtered_df = recommendations_df[recommendations_df['KEGG_ID'].notna() & (recommendations_df['KEGG_ID'] != '')]

# Keep rows that have the max match_score per id
high_score_recommendations = filtered_df[
    filtered_df['match_score'] == filtered_df.groupby('id')['match_score'].transform('max')
].reset_index(drop=True)

print("\nSample of ChEBI to KEGG mapping:")
print(high_score_recommendations[['id', 'display_name', 'annotation', 'KEGG_ID', 'match_score']].head().to_string(index=False))

print("\nStep 3: Begin rule-based matching to identify reactions")
reactions, _ = extract_reactions_from_sbml(model_file, list(high_score_recommendations['id'].unique()))
normalized_reactions = map_reactions_to_kegg(reactions, high_score_recommendations[['id', 'KEGG_ID']], spectators=False)

# Get KEGG recommendations
match_results = database_search._get_kegg_recommendations_rulebased(
    normalized_reactions, cofactors_to_ignore = cofactors_to_ignore,
    spectators=False)

# Build recommendation table
kegg_recommendations_df = _generate_recommendation_table(model_file, 
                                                    match_results, 
                                                    {}, 
                                                    model_info, 
                                                    entity_type, 
                                                    database, 
                                                    {})






# 2. Initialize probabilities for each candidate reaction
#    (e.g., uniform distribution to start)
kegg_recommendations_df['match_score_norm'] = (
    kegg_recommendations_df['match_score'] /
    kegg_recommendations_df.groupby('id')['match_score'].transform('sum')
    )
kegg_recommendations_df[['id','match_score', 'match_score_norm']]




reaction_participants = dict()
for i in model_info['reactions']:
    participant_str = extract_classifications(i.split(':')[1].strip(), 'definition')
    participant_names = []
    for participant in participant_str.split('; '):
        participant_names.append(recommendations_df[recommendations_df['id']==participant]['annotation_label'].values[0])
    reaction_participants[i.split(':')[0].strip()]= participant_names


import lzma
import pickle
with lzma.open('../data/kegg/kegg_reaction_features.lzma', 'rb') as f:
    kegg_reaction_features = pickle.load(f)
# kegg_reaction_features



# Create new column in the dataframe
kegg_recommendations_df['participants'] = kegg_recommendations_df['annotation'].apply(get_participants)
kegg_recommendations_df['participant_ids'] = kegg_recommendations_df['annotation'].apply(get_participant_ids)
kegg_recommendations_df.head()


merged_participants = (
    kegg_recommendations_df
    .groupby("id")["participants"]
    .agg("; ".join)  # concatenate strings with "; "
)

# - build a set of counters to account for all the species in the list
counters = merged_participants.apply(
    lambda s: Counter(p.strip() for p in s.split(";") if p.strip())
)





init_probs = init_species_probs_from_dict(reaction_participants, counters)

scored_df2 = compute_reaction_likelihoods(init_probs, kegg_recommendations_df)
scored_df2.groupby("id")[["annotation", "likelihood"]].apply(lambda g: g.sort_values(by="likelihood", ascending=False))