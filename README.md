# AAAIM — Auto-Annotator via AI for Modeling

AAAIM is an LLM-powered tool that annotates biosimulation models (SBML) with standardized ontology terms from ChEBI, NCBI Gene, UniProt, and KEGG.

![AAAIM Workflow](docs/AAAIM%20workflow.png)

---

## Installation

**Requirements**: Python 3.12

```bash
pip install -r requirements.txt
```

Set at least one LLM API key (in your shell or a `.env` file):

```bash
OPENAI_API_KEY=<your-openai-key>          # gpt-4o-mini, gpt-4.1-nano
OPENROUTER_API_KEY=<your-openrouter-key>  # llama-3.3-70b (free tier available)
LLAMA_API_KEY=<your-llama-key>            # Llama-3.3-70B-Instruct
```

---

## Quick Start

```python
from core import annotate_model

# Annotate all species — entity types are detected automatically
recommendations_df, metrics = annotate_model(
    model_file="path/to/model.xml",
    entity_type="auto",               # detects chemical / gene / protein / complex
    database=["chebi", "uniprot"]     # databases to search
)

recommendations_df.to_csv("recommendations.csv", index=False)
```

Run the bundled example (uses a test SBML model):

```bash
python examples/simple_example.py
```

For models with existing annotations (curation/validation workflow):

```python
from core import curate_model

curations_df, metrics = curate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

---

## Applying Annotation Recommendations

After reviewing the output CSV, edit the `update_annotation` column for each row:

| Value                 | Effect                         |
| --------------------- | ------------------------------ |
| `add`               | Add the recommended annotation |
| `delete`            | Remove the existing annotation |
| `ignore` / `keep` | Leave unchanged                |

Then write the updated model:

```python
from core.update_model import update_annotation

update_annotation(
    original_model_path="model.xml",
    recommendation_table="recommendations.csv",
    new_model_path="model_updated.xml"
)
```

---

## Optional: RAG-based Search

By default, AAAIM uses direct dictionary matching (`method="direct"`). For semantic (embedding-based) search, use `method="rag"` — but you must build the vector index first.

**One-time setup** (builds embeddings for all databases, this may take a while depending on the size of your database):

```bash
python setup_rag.py                        # all databases, human (tax_id=9606)
python setup_rag.py --databases chebi      # ChEBI only
python setup_rag.py --tax_id 10090         # mouse
```

Then pass `method="rag"` to `annotate_model()` or `curate_model()`.

---

## Full Documentation

See [docs/README.md](docs/README.md) for:

- All parameters for `annotate_model` / `curate_model`
- Per-database annotation examples
- Evaluation utilities
- Data file descriptions
- Supported embedding models for RAG
