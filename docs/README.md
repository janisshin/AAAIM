# AAAIM (Auto-Annotator via AI for Modeling)

AAAIM is a LLM-powered system for annotating biosimulation models with standardized ontology terms. It currently supports chemical, gene and protein annotation for species, and KEGG annotation for reactions in SBML models.

## Installation

```bash
# python = 3.12

# Install dependencies
pip install -r requirements.txt
```

Set up your LLM provider API keys:

```bash
# For OpenAI models (gpt-4o-mini, gpt-4.1-nano)
export OPENAI_API_KEY="your-openai-key"

# For OpenRouter models (meta-llama/llama-3.3-70b-instruct:free)
export OPENROUTER_API_KEY="your-openrouter-key"

# For LlaMa models (Llama-3.3-70B-Instruct)
export LLAMA_API_KEY="your-llama-key"
```

Alternatively, you can setup an `.env` file that looks like the following:

```bash
OPENAI_API_KEY=<your-openai-api-key-here>
OPENROUTER_API_KEY=<your-openrouter-api-key-here>
LLAMA_API_KEY=<your-llama-api-key-here>
```

## Usage

AAAIM provides two main workflows plus an interactive feedback mechanism.

Both `annotate_model` and `curate_model` return an `AnnotationResult` object.
It holds `recommendations_df` and `metrics` as attributes and also supports
tuple unpacking for backward compatibility:

```python
# Both styles work:
result = annotate_model("model.xml", ...)
recommendations_df, metrics = annotate_model("model.xml", ...)
```

A recommendations CSV is saved automatically after every run and after every
feedback revision. The filename printed to the console tells you where it was
written.

### 1. Annotation Workflow (for new models)

- **Purpose**: Annotate models with no or limited existing annotations
- **Input**: All species in the model
- **Output**: Annotation recommendations for all species
- **Metrics**: Accuracy is NA when no existing annotations available
- **Large Models**: Automatically splits models with >50 entities into chunks to avoid LLM context limits

#### Chemical Annotation (ChEBI)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)
```

#### Gene Annotation (NCBI Gene)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene",
    tax_id="9606"  # for human
)
```

#### Protein Annotation (UniProt)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id="9606"  # for human
)
```

#### Automatic Entity Type Detection

AAAIM supports automatic detection of entity types (chemical, gene, protein, complex, or unknown) for models with mixed entity types:

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="auto",
    database=["chebi", "uniprot"]  # choose from these databases
)

# The results will include a 'type' column indicating the detected entity type
print(result.recommendations_df[['species_id', 'type', 'synonyms_LLM', 'predictions_names']])
```

**How it works:**

- The LLM analyzes each species in context (display names, reactions, model notes) to determine its type
- Detected types: `chemical`, `gene`, `protein`, `complex`, or `unknown`
- Database matching is performed using the appropriate database for each detected type
- The `database` parameter accepts a list to specify which databases to use:
  - Chemicals → ChEBI
  - Genes → NCBI Gene
  - Proteins → UniProt
  - Complexes → ChEBI, UniProt, or NCBI Gene
- Species with `unknown` type are included in results with their LLM-suggested synonyms but no database matches

### 2. Curation Workflow (for models with existing annotations)

- **Purpose**: Evaluate and improve existing annotations
- **Input**: Only species that already have annotations
- **Output**: Validation and improvement recommendations
- **Metrics**: Accuracy calculated against existing annotations
- **Large Models**: Automatically splits models with >50 entities into chunks to avoid LLM context limits

#### Chemical Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)

print(f"Chemical entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

#### Gene Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene"
)

print(f"Gene entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

#### Protein Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id=9606  # for human
)

print(f"Protein entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

### 3. User Feedback

After reviewing the initial recommendations you can iteratively refine them.
The previous prompt, LLM response, and your feedback are sent back to the
LLM for a revised set of synonyms, which then go through the retrieval
pipeline again.

#### Single revision

```python
result = annotate_model("model.xml", entity_type="chemical", database="chebi")

# Provide feedback
result = result.revise("Species X should be glucose-6-phosphate, not glucose")
```

#### Interactive loop

```python
result.feedback_loop()          # console prompt; press Enter to accept
```

A custom callback can replace the console prompt (useful for GUIs or
notebooks):

```python
def my_feedback(df, iteration):
    # display df in your UI, collect text
    return user_text   # return "" or None to stop

result.feedback_loop(get_feedback_fn=my_feedback)
```

#### Versioned CSV output

Every revision writes a new file so that earlier recommendations are
never lost:

```
model.xml_recommendations.csv       ← initial run
model.xml_recommendations_v1.csv    ← after 1st revision
model.xml_recommendations_v2.csv    ← after 2nd revision
```

#### Advanced control

`result.revise()` and `result.feedback_loop()` use the same database,
method, LLM model, and other parameters from the initial run by default.
For advanced use you can also call the lower-level helpers directly; see
`core/feedback.py` for the full API.

### 4. Updating Model Annotations After Review

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

### 5. Advanced Usage

```python
# More control over parameters
result = annotate_model(
    model_file = "path/to/model.xml",
    llm_model = "meta-llama/llama-3.3-70b-instruct:free",       # the LLM model used to predict annotations
    max_entities = 100,					 # maximum number of entities to annotate (None for all)
    entity_type = "gene",				 # type of entities to annotate ("chemical", "gene", "protein", "auto")
    database = "ncbigene",				 # database to use ("chebi", "ncbigene", "uniprot") or list for auto mode
    method = "direct",					 # method used to find the ontology ID ("direct", "rag")
    top_k = 3,						 # number of LLM synonyms and top database candidates to return per entity
    chunk_size = 50					 # split large models into chunks of 50 entities (None for no chunking)
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

### 6. Evaluation and Results Analysis

AAAIM provides tools for evaluating annotation quality and analyzing results:

```python
from utils.evaluation import evaluate_single_model, print_evaluation_results

# Evaluate a single model and get detailed results with a 'type' column
result_df = evaluate_single_model(
    model_file="path/to/model.xml",
    llm_model="gpt-4o-mini",
    method="direct",
    top_k=3,
    entity_type="auto",  # or "chemical", "gene", "protein"
    database=["chebi", "uniprot"]  # for auto mode, or single database for specific type
)

# The result DataFrame includes a 'type' column showing detected entity types
print(result_df[['species_id', 'type', 'synonyms_LLM', 'predictions_names', 'accuracy']])

# Print summary statistics from a results CSV file
print_evaluation_results(
    results_csv="results.csv",
    ref_results_csv="reference_results.csv",  # optional: filter to only species in reference
    bqbiol_qualifiers=['is', 'isVersionOf'],  # optional: filter by annotation qualifiers
    entity_types=['chemical', 'protein']  # optional: filter by detected entity types
)
```

**Output columns:**

- `detected_entity_type`: Detected entity type (chemical, gene, protein, complex, or unknown)
- `synonyms_LLM`: LLM-suggested synonyms for the species
- `predictions`: Top-k database IDs matched for this species
- `predictions_names`: Corresponding names for the predicted IDs
- `exist_annotation_id`: Existing annotation IDs from the model (if any)
- `exist_annotation_name`: Names of existing annotations
- `accuracy`: Match accuracy between predictions and existing annotations

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
  - Used for: small molecules, metabolites, compounds
- **NCBI Gene**: Gene annotation
  - **Entity Type**: `gene`
  - Only genes for common species are supported (those included in bigg models).
  - Used for: genes, DNA sequences, gene symbols
- **UniProt**: Protein annotation
  - **Entity Type**: `protein`
  - Only proteins for human (9606) and mouse (10090) are supported for now.
  - Used for: proteins, enzymes
- **KEGG**: Compound/reaction annotation
  - For reaction substrates and products.

### Entity Type to Database Mapping

When using `entity_type="auto"`, AAAIM automatically selects the appropriate database(s) based on the detected entity type:


| Detected Type | Default Databases         | Usage                                          |
| ------------- | ------------------------- | ---------------------------------------------- |
| `chemical`    | ChEBI                     | Small molecules, metabolites, compounds        |
| `gene`        | NCBI Gene                 | Genes, DNA sequences, gene symbols             |
| `protein`     | UniProt                   | Proteins, enzymes                              |
| `complex`     | ChEBI, UniProt, NCBI Gene | Protein complexes, chemical complexes          |
| `unknown`     | None                      | LLM synonyms included but no database matching |


You can restrict which databases are used by providing a `database` list parameter. For example, `database=["chebi", "uniprot"]` will only use ChEBI for chemicals and UniProt for proteins, but will not search NCBI Gene even if genes are detected.

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
- **Source**: ChEBI ontology downloaded from [https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz](https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz).

### NCBI gene Data

- **Location**: `data/ncbigene/`
- **Files**:
  - `names2ncbigene_bigg_organisms_protein-coding.lzma`: Mapping from names to NCBI gene IDs, only include protein-coding genes from 18 species covered in Bigg models for file size considerations
  - `ncbigene2label_bigg_organisms_protein-coding.lzma`: Mapping from NCBI gene IDs to labels (primary name)
  - `ncbigene2names_tax{tax_id}_protein-coding.lzma`: NCBI gene synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the NCBI gene FTP site: [https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/](https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/).

### UniProt Data

- **Location**: `data/uniprot/`
- **Files**:
  - `names2uniprot_human+mouse.lzma`: Mapping from synonyms to UniProt IDs, only include human and mouse proteins for now
  - `uniprot2label_human+mouse.lzma`: Mapping from UniProt IDs to labels (primary name)
  - `uniprot2names_tax{tax_id}.lzma`: Uniprot synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the UniProt site: [https://www.uniprot.org/help/downloads](https://www.uniprot.org/help/downloads) (Reviewed (Swiss-Prot) xml).

### KEGG Data

- **Location**: `data/kegg/`
- **Files**:
  - `chebi_to_kegg_map.lzma`: Mapping from ChEBI IDs to KEGG compound IDs.
  - `parsed_kegg_reactions.lzma`: Dict of KEGG reactions and their attributes
- **Source**: Data are obtained from the KEGG site: [https://rest.kegg.jp](https://rest.kegg.jp).

## File Structure

```
aaaim/
├── core/
│   ├── __init__.py              # Main interface exports
│   ├── annotation_workflow.py   # Annotation workflow (models without annotations)
│   ├── curation_workflow.py     # Curation workflow (models with annotations)
│   ├── feedback.py             # Feedback module (AnnotationResult, revise, feedback_loop)
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

