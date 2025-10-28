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

Alternatively, you can setup an `.env` file that looks like the following:

```bash
LLAMA_API_KEY=<your-llama-api-key-here>
OPENROUTER_API_KEY=<your-openrouter-api-key-here>
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
    database="ncbigene",
    tax_id="9606"  # for human
)

# Save results
recommendations_df.to_csv("gene_annotation_results.csv", index=False)
```

#### Protein Annotation (UniProt)

```python
from core import annotate_model

# Annotate all gene species in a model
recommendations_df, metrics = annotate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id="9606"  # for human
)

# Save results
recommendations_df.to_csv("protein_annotation_results.csv", index=False)
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

#### Protein Curation

```python
from core import curate_model

# Curate existing gene annotations
curations_df, metrics = curate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id=9606  # for human
)

print(f"Gene entities with existing annotations: {metrics['total_entities']}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

### 3. Updating Model Annotations After Review

After running `annotate_model` or `curate_model`, you can review the resulting CSV file and edit the `update_annotation` column for each entity:

- `add`: Add the recommended annotation to the model for that entity.
- `delete`: Remove the annotation for that entity.
- `ignore` or `keep`: Leave the annotation unchanged. Whether keep the existing one, or ignore the new suggestion.

To apply your changes and save a new SBML model:

```python
from core.update_model import update_annotation

update_annotation(
    original_model_path="path/to/original_model.xml",
    recommendation_table="recommendations.csv",  # or a pandas DataFrame
    new_model_path="path/to/updated_model.xml",
    qualifier="is"  # (optional) bqbiol qualifier, default is 'is'
)
```

A summary of added/removed annotations will be printed after update.

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

# Direct access to qualifier tracking functions
from core.model_info import find_species_with_annotations_and_qualifiers

# Get annotations and qualifiers for any supported database
annotations, qualifiers = find_species_with_annotations_and_qualifiers(
    model_file="path/to/model.xml",
    database="chebi",  # or "ncbigene", "uniprot"
    bqbiol_qualifiers=['is', 'isVersionOf']  # optional: filter by specific qualifiers
)

print(f"Found {len(annotations)} species with annotations")
for species_id, annotation_ids in annotations.items():
    if species_id in qualifiers:
        print(f"{species_id}: {annotation_ids}")
        for ann_id, qualifier in qualifiers[species_id].items():
            print(f"  {ann_id} -> {qualifier}")
    else:
        print(f"{species_id}: {annotation_ids} (no qualifier info)")
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
# for uniprot, specify the taxnomy id:
python load_data.py --database uniprot --model default --tax_id 9606
# for KEGG:
python load_data.py --database kegg --model default
```

## Databases

### Currently Supported

- **ChEBI**: Chemical Entities of Biological Interest

  - **Entity Type**: `chemical`
  - All terms in ChEBI are included.
- **NCBI Gene**: Gene annotation

  - **Entity Type**: `gene`
  - Only genes for common species are supported (those included in bigg models).
- **UniProt**: Protein annotation

  - **Entity Type**: `uniprot`
  - Only proteins for human (9606) and mouse (10090) are supported for now.
- **KEGG**: Compound/reaction annotation

  - For reaction substrates and products.

### Future Support

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
  - `ncbigene2label_bigg_organisms_protein-coding.lzma`: Mapping from NCBI gene IDs to labels (primary name)
  - `ncbigene2names_tax{tax_id}_protein-coding.lzma`: NCBI gene synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the NCBI gene FTP site: https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/.

### UniProt Data

- **Location**: `data/uniprot/`
- **Files**:
  - `names2uniprot_human+mouse.lzma`: Mapping from synonyms to UniProt IDs, only include human and mouse proteins for now
  - `uniprot2label_human+mouse.lzma`: Mapping from UniProt IDs to labels (primary name)
  - `uniprot2names_tax{tax_id}.lzma`: Uniprot synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the UniProt site: https://www.uniprot.org/help/downloads (Reviewed (Swiss-Prot) xml).

### KEGG Data

- **Location**: `data/kegg/`
- **Files**:
  - `chebi_to_kegg_map.lzma`: Mapping from ChEBI IDs to KEGG compound IDs.
  - `parsed_kegg_reactions.lzma`: Dict of KEGG reactions and their attributes
- **Source**: Data are obtained from the KEGG site: https://rest.kegg.jp.

## File Structure

```
aaaim/
├── core/
│   ├── __init__.py              # Main interface exports
│   ├── annotation_workflow.py   # Annotation workflow (models without annotations)
│   ├── curation_workflow.py     # Curation workflow (models with annotations)
│   ├── model_info.py           # Model parsing and context
│   ├── llm_interface.py        # LLM interaction
│   ├── database_search.py      # Database search functions
│   └── update_model.py         # put annotations into model
├── utils/
│   ├── constants.py
│   ├── evaluation.py 		# functions for evaluation
├── examples/
│   ├── simple_example.py    	# Simple usage demo
├── data/
│   ├── chebi/                   # ChEBI compressed dictionaries
│   ├── ncbigene/                # NCBIgene compressed dictionaries
│   ├── uniprot/                 # UniProt compressed dictionaries
│   ├── chroma_storage/          # Database embeddings for RAG
└── tests/
    ├── test_models     	 # Test models
    └── aaaim_evaluation.ipynb   # evaluation notebook
```

## Future Development

### Planned Features

- **Multi-Database Support**: GO, Rhea, mapping between ontologies
- **Improve RAG for NCBI Gene**: Test on other embedding models for genes
- **Web Interface**: User-friendly annotation tool
