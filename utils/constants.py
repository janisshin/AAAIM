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
    REACTION = "reaction"
    TRANSITION = "transition"
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
}

# Entity Type to Database Mapping
ENTITY_DATABASE_MAPPING: Dict[EntityType, List[DatabaseID]] = {
    EntityType.CHEMICAL: [DatabaseID.CHEBI],
    EntityType.GENE: [DatabaseID.NCBIGENE, DatabaseID.GO],
    EntityType.PROTEIN: [DatabaseID.UNIPROT, DatabaseID.GO],
    EntityType.REACTION: [DatabaseID.RHEA, DatabaseID.EC, DatabaseID.KEGG],
    EntityType.TRANSITION: [DatabaseID.PUBMED, DatabaseID.GO],
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
REF_NCBIGENE2LABEL = "ncbigene2label_bigg_organisms_protein-coding_updated.lzma"
REF_NAMES2NCBIGENE = "names2ncbigene_bigg_organisms_protein-coding_updated.lzma"

# Model Format Detection
MODEL_FORMAT_PLUGINS = {
    "fbc": ModelType.SBML_FBC,
    "qual": ModelType.SBML_QUAL
}

# Annotation URI Patterns
NCBIGENE_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/ncbigene/(\d+)',
    r'urn:miriam:ncbigene:(\d+)'
]

CHEBI_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/chebi/CHEBI:(\d+)',
    r'urn:miriam:chebi:CHEBI:(\d+)'
]