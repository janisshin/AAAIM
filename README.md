# AAAIM (Auto-Annotator via AI for Modeling)

AAAIM is a LLM-powered system for annotating biosimulation models with standardized ontology terms.

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

AAAIM currently provides two main workflows:

### 1. Annotation Workflow (for new models)

- **Purpose**: Annotate models with no or limited existing annotations
- **Input**: All species in the model
- **Output**: Annotation recommendations for all species
- **Metrics**: Accuracy is NA when no existing annotations available

```python
from core import annotate_model

# Annotate all species in a model
recommendations_df, metrics = annotate_model(model_file="path/to/model.xml")

# Save results
recommendations_df.to_csv("annotation_results.csv", index=False)
```

### 2. Curation Workflow (for models with existing annotations)

- **Purpose**: Evaluate and improve existing annotations
- **Input**: Only species that already have annotations
- **Output**: Validation and improvement recommendations
- **Metrics**: Accuracy calculated against existing annotations

```python
from core import curate_model

# Curate existing annotations
curations_df, metrics = curate_model(model_file="path/to/model.xml")

print(f"Entities with existing annotations: {metrics['total_entities']}")
print(f"Accuracy: {metrics['accuracy']:.1%}")

# Save results
curations_df.to_csv("curation_results.csv", index=False)
```

### Advanced Usage

```python
# More control over parameters
recommendations_df, metrics = annotate_model(
    model_file = "path/to/model.xml",
    llm_model = "meta-llama/llama-3.3-70b-instruct:free",       # the LLM model used to predict annotations
    max_entities = 100,					 # maximum number of entities to annotate (None for all)
    entity_type = "chemical",				 # type of entities to annotate ("chemical", "gene", "protein")
    database = "chebi",					 # database to use ("chebi", "ncbigene", "uniprot")
    method = "direct"					 # method used to find the ontology ID ("direct", "rag")
)
```

### Example

```python
# Using "tests/test_models/BIOMD0000000190.xml"
python examples/simple_example.py
```

## Methods

### Direct matching

After LLM performs synonym normalization, use direct dictionary matching to find ontology ID and report hit counting.

### Retrival augmented generation (RAG)

After LLM performs synonym normalization, use direct dictionary matching to find ontology ID and report hit counting.

## Databases

### Currently Supported

- **ChEBI**: Chemical Entities of Biological Interest
  - **Direct**: Dictionary of standard names to ontology ID.
  - **RAG**: Embeddings of ontology terms.

### Future Support

- **NCBI Gene**: Gene annotation
- **UniProt**: Protein annotation
- **Rhea**: Reaction annotation
- **GO**: Gene Ontology terms

## Data Files

### ChEBI Data

- **Location**: `data/chebi/`
- **Files**:
  - `cleannames2chebi.lzma`: Mapping from clean names to ChEBI IDs
  - `chebi2label.lzma`: Mapping from ChEBI IDs to labels
- **Source**: ChEBI ontology downloaded from https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz.

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
│   ├── simple_example.py    # Simple usage demo
├── data/
│   └── chebi/                   # ChEBI compressed dictionaries
└── tests/
    └── test_models     	 # Test models
    └── aaaim_evaluation.ipynb   # evaluation notebook
```

## Future Development

### Planned Features

- **Multi-Database Support**: NCBI Gene, UniProt, GO, Rhea
- **RAG improvement**: Reduce vector embedding size
- **Web Interface**: User-friendly annotation tool
