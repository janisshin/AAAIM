"""
AAAIM Constants

Defines constants used throughout the AAAIM system.
"""

from enum import Enum
from typing import Dict, List

# Entity Types
class EntityType(Enum):
    """Types of biological entities that can be annotated."""
    CHEMICAL = "chemical"
    GENE = "gene" 
    PROTEIN = "protein"
    COMPLEX = "complex"
    # REACTION = "reaction"
    UNKNOWN = "unknown"

# Model Types
class ModelType(Enum):
    """Types of SBML models supported."""
    SBML = "SBML"
    SBML_QUAL = "SBML-qual"
    SBML_FBC = "SBML-fbc"

# Database Identifiers
class DatabaseID(Enum):
    """Supported biological databases."""
    CHEBI = "chebi"
    NCBIGENE = "ncbigene"
    UNIPROT = "uniprot"
    RHEA = "rhea"
    GO = "go"
    PUBMED = "pubmed"
    KEGG = "kegg"
    EC = "ec"

# Database Prefixes and URIs
DATABASE_PREFIXES: Dict[DatabaseID, str] = {
    DatabaseID.CHEBI: "CHEBI:",
    DatabaseID.NCBIGENE: "NCBIGENE:",
    DatabaseID.UNIPROT: "UNIPROT:",
    DatabaseID.RHEA: "RHEA:",
    DatabaseID.GO: "GO:",
    DatabaseID.PUBMED: "PUBMED:",
    DatabaseID.KEGG: "KEGG:",
    DatabaseID.EC: "EC:",
}

DATABASE_URIS: Dict[DatabaseID, str] = {
    DatabaseID.CHEBI: "https://identifiers.org/chebi/CHEBI:",
    DatabaseID.NCBIGENE: "https://identifiers.org/ncbigene:",
    DatabaseID.UNIPROT: "https://identifiers.org/uniprot:",
    DatabaseID.RHEA: "https://identifiers.org/rhea:",
    DatabaseID.GO: "https://identifiers.org/GO:",
    DatabaseID.PUBMED: "https://identifiers.org/pubmed:",
    DatabaseID.KEGG: "https://identifiers.org/kegg.reaction:",
    DatabaseID.EC: "https://identifiers.org/ec-code:",

}

# Entity Type to Database Mapping
ENTITY_DATABASE_MAPPING: Dict[EntityType, List[DatabaseID]] = {
    EntityType.CHEMICAL: [DatabaseID.CHEBI],
    EntityType.GENE: [DatabaseID.NCBIGENE],
    EntityType.PROTEIN: [DatabaseID.UNIPROT],
    EntityType.COMPLEX: [DatabaseID.CHEBI, DatabaseID.UNIPROT, DatabaseID.NCBIGENE],
    # EntityType.REACTION: [DatabaseID.RHEA, DatabaseID.EC, DatabaseID.KEGG],
}

# Confidence Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.3

# Batch Processing
DEFAULT_BATCH_SIZE = 50
MAX_BATCH_SIZE = 200

# LLM Settings
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 30

# Cache Settings
DEFAULT_CACHE_TTL_HOURS = 24
MAX_CACHE_SIZE_MB = 1000 

# REF files
REF_CHEBI2LABEL = "chebi2label.lzma"
REF_NAMES2CHEBI = "cleannames2chebi.lzma"
REF_CHEBI2FORMULA = "chebi_shortened_formula.lzma"
REF_NCBIGENE2LABEL = "ncbigene2label_bigg_organisms_protein-coding_added.lzma"
REF_NAMES2NCBIGENE = "names2ncbigene_bigg_organisms_protein-coding.lzma"
REF_UNIPROT2LABEL = "uniprot2label_human+mouse+rat.lzma"
REF_NAMES2UNIPROT = "names2uniprot_human+mouse+rat.lzma"
REF_CHEBI2KEGG_COMPOUND = "chebi_to_kegg_map.lzma" 
REF_KEGG_REACTION2NAME = "reactionnames2kegg.lzma"
REF_KEGG2EC = "kegg2ec.lzma"
REF_KEGG_REACTION_FEATURES = "kegg_reaction_features.lzma"
REF_KEGG_PARSED_REACTIONS = "parsed_kegg_reactions.lzma"

# Model Format Detection
MODEL_FORMAT_PLUGINS = {
    "fbc": ModelType.SBML_FBC,
    "qual": ModelType.SBML_QUAL
}

# Annotation URI Patterns
CHEBI_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/chebi/CHEBI:(\d+)',
    r'urn:miriam:chebi:CHEBI:(\d+)'
]

NCBIGENE_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/ncbigene/(\d+)',
    r'urn:miriam:ncbigene:(\d+)'
]

UNIPROT_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/uniprot/(\w+)',
    r'urn:miriam:uniprot:(\w+)'
]

KEGG_REACTION_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.reaction:(R\d+)',
    r'urn:miriam:kegg\.reaction:(R\d+)'
]

KEGG_COMPOUND_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.compound:(C\d+)',
    r'urn:miriam:kegg\.compound:(C\d+)'
]

KEGG_PATHWAY_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.pathway:(map\d+)',
    r'urn:miriam:kegg\.pathway:(map\d+)'
]

KEGG_ENZYME_URI_PATTERNS = [
    r'https?://identifiers\.org/ec-code:(\d+\.\d+\.\d+\.\d+)',
    r'urn:miriam:ec-code:(\d+\.\d+\.\d+\.\d+)'
]

KEGG_GENE_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.gene:([\w]+:[\w]+)',
    r'urn:miriam:kegg\.gene:([\w]+:[\w]+)'
]
