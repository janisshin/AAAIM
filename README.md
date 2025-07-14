# AAAIM (Auto-Annotator via AI for Modeling)

AAAIM is a LLM-powered system for annotating biosimulation models with standardized ontology terms. It supports both chemical and gene entity annotation.

## Installation

```bash
# python = 3.12

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Environment Variables

Set up your LLM provider API keys:

```bash
# For OpenAI models (gpt-4o-mini, gpt-4.1-nano)
export OPENAI_API_KEY="your-openai-key"

# For OpenRouter models (meta-llama/llama-3.3-70b-instruct:free)
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Usage

AAAIM currently provides two main workflows for both chemical and gene annotation:

### 1. Annotation Workflow (for new models)

- **Purpose**: Annotate models with no or limited existing annotations
- **Input**: All species in the model
- **Output**: Annotation recommendations for all species
- **Metrics**: Accuracy is NA when no existing annotations available

#### Chemical Annotation (ChEBI)

```python
from core import annotate_model

# Annotate all chemical species in a model
recommendations_df, metrics = annotate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)

# Save results
recommendations_df.to_csv("chemical_annotation_results.csv", index=False)
```

#### Gene Annotation (NCBI Gene)

```python
from core import annotate_model

# Annotate all gene species in a model
recommendations_df, metrics = annotate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene"
)

# Save results
recommendations_df.to_csv("gene_annotation_results.csv", index=False)
```

### 2. Curation Workflow (for models with existing annotations)

- **Purpose**: Evaluate and improve existing annotations
- **Input**: Only species that already have annotations
- **Output**: Validation and improvement recommendations
- **Metrics**: Accuracy calculated against existing annotations

#### Chemical Curation

```python
from core import curate_model

# Curate existing chemical annotations
curations_df, metrics = curate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)

print(f"Chemical entities with existing annotations: {metrics['total_entities']}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

#### Gene Curation

```python
from core import curate_model

# Curate existing gene annotations
curations_df, metrics = curate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene"
)

print(f"Gene entities with existing annotations: {metrics['total_entities']}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

### Advanced Usage

```python
# More control over parameters
recommendations_df, metrics = annotate_model(
    model_file = "path/to/model.xml",
    llm_model = "meta-llama/llama-3.3-70b-instruct:free",       # the LLM model used to predict annotations
    max_entities = 100,					 # maximum number of entities to annotate (None for all)
    entity_type = "gene",				 # type of entities to annotate ("chemical", "gene")
    database = "ncbigene",				 # database to use ("chebi", "ncbigene")
    method = "direct",					 # method used to find the ontology ID ("direct", "rag")
    top_k = 3						 # number of top candidates to return per entity (based on scores)
)
```

### Example

```python
# Using "tests/test_models/BIOMD0000000190.xml"
python examples/simple_example.py
```

## Methods

### Direct matching

After LLM performs synonym normalization, use direct dictionary matching to find ontology ID and report hit counting. Returns the top_k candidates with the highest hit counts.

### Retrival augmented generation (RAG)

After LLM performs synonym normalization, use RAG with embeddings to find the top_k most similar ontology terms based on cosine similarity.

To use RAG, create embeddings of the ontology first:

```bash
cd data
# for ChEBI:
python load_data.py --database chebi --model default
# for NCBI gene, specify the taxnomy id:
python load_data.py --database ncbigene --model default --tax_id 9606
```

## Databases

### Currently Supported

- **ChEBI**: Chemical Entities of Biological Interest

  - **Entity Type**: `chemical`
  - **Direct**: Dictionary of standard names to ontology ID. Returns top_k candidates with highest hit counts.
  - **RAG**: Embeddings of ontology terms. Returns top_k most similar terms.
- **NCBI Gene**: Gene annotation

  - **Entity Type**: `gene`
  - **Direct**: Dictionary of gene names to NCBI gene IDs. Returns top_k candidates with highest hit counts.
  - **RAG**: Not yet implemented.

### Future Support

- **UniProt**: Protein annotation
- **Rhea**: Reaction annotation
- **GO**: Gene Ontology terms

## Data Files

### ChEBI Data

- **Location**: `data/chebi/`
- **Files**:
  - `cleannames2chebi.lzma`: Mapping from clean names to ChEBI IDs
  - `chebi2label.lzma`: Mapping from ChEBI IDs to labels
  - `chebi2names.lzma`: ChEBI synonyms used for RAG approach
- **Source**: ChEBI ontology downloaded from https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz.

### NCBI gene Data

- **Location**: `data/ncbigene/`
- **Files**:
  - `names2ncbigene_bigg_organisms_protein-coding.lzma`: Mapping from names to NCBI gene IDs, only include protein-coding genes from 18 species covered in Bigg models for file size considerations
  - `ncbigene2label_bigg_organisms_protein-coding.lzma`: Mapping from NCBI gene IDs to labels
  - `ncbigene2names_tax{tax_id}_protein-coding.lzma`: NCBI gene synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the NCBI gene FTP site: https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/.

## File Structure

```
aaaim/
├── core/
│   ├── __init__.py              # Main interface exports
│   ├── annotation_workflow.py   # Annotation workflow (models without annotations)
│   ├── curation_workflow.py     # Curation workflow (models with annotations)
│   ├── model_info.py           # Model parsing and context
│   ├── llm_interface.py        # LLM interaction
│   └── database_search.py      # Database search functions
├── utils/
│   ├── constants.py
│   ├── evaluation.py 		# functions for evaluation
├── examples/
│   ├── simple_example.py    	# Simple usage demo
├── data/
│   ├── chebi/                   # ChEBI compressed dictionaries
│   ├── ncbigene/                # NCBIgene compressed dictionaries
│   ├── chroma_storage/          # Database embeddings for RAG
└── tests/
    ├── test_models     	 # Test models
    └── aaaim_evaluation.ipynb   # evaluation notebook
```

## Future Development

### Planned Features

- **Multi-Database Support**: UniProt, GO, Rhea
- **Improve RAG for NCBI Gene**: Test on other embedding models for genes
- **Web Interface**: User-friendly annotation tool
