# ChEBI RAG-based Entity Linking

This directory contains scripts for implementing RAG (Retrieval-Augmented Generation) based entity linking for ChEBI chemical entities. The system uses ChromaDB for vector storage and supports multiple embedding models.

## Overview

The RAG entity linking system works in two phases:

1. **Data Preparation & Indexing**: Load ChEBI reference data and create embeddings
2. **Entity Linking**: Use embeddings to retrieve candidate entities and link chemical mentions to ChEBI IDs

## Files

- `load_data.py` - Main script for loading ChEBI data and creating embeddings
- `rag_entity_linking_example.py` - Example script demonstrating entity linking
- `README_ChEBI_RAG.md` - This documentation file

## Prerequisites

1. **ChEBI Reference Data**: The system expects `data/chebi/chebi2names.lzma` file containing a dictionary mapping ChEBI IDs to lists of names/synonyms.
2. **Required Packages**: Install dependencies:

   ```bash
   pip install chromadb==1.0.11 compress-pickle==2.1.0 tqdm
   ```
3. **OpenAI API Key** (optional): For OpenAI embeddings, set the environment variable:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Step 1: Create Embeddings

Create embeddings using the default sentence transformer model:

```bash
cd data
python load_data.py --model default --collection chebi_default --test
```

Create embeddings using OpenAI's text-embedding-ada-002:

```bash
python load_data.py --model openai --collection chebi_openai --test
```

Create embeddings using Llama (requires Ollama server running):

```bash
python load_data.py --model llama --collection chebi_llama --test
```

#### Command Line Options

- `--chebi_file`: Path to ChEBI reference file (default: `data/chebi/chebi2names.lzma`)
- `--collection`: ChromaDB collection name (default: `chebi_entities`)
- `--model`: Embedding model type (`default`, `openai`, `llama`)
- `--persist_directory`: ChromaDB storage directory (default: `chroma_storage`)
- `--batch_size`: Batch size for indexing (default: 500)
- `--test`: Run test queries after creating embeddings
- `--test_only`: Only run test queries (skip embedding creation)

### Step 2: Entity Linking Examples

#### Single Chemical Query

Search for a specific chemical:

```bash
python rag_entity_linking_example.py --query "aspirin" --collection chebi_default
```

Search with OpenAI embeddings:

```bash
python rag_entity_linking_example.py --query "glucose" --model openai --collection chebi_openai
```

#### Text Processing

Process text containing chemical mentions:

```bash
python rag_entity_linking_example.py --text "The patient was treated with glucose and caffeine" --collection chebi_default
```

#### Show LLM Prompt

See the prompt that would be sent to an LLM for final entity linking decision:

```bash
python rag_entity_linking_example.py --query "aspirin" --show_prompt --collection chebi_default
```

## System Architecture

### Data Preparation

1. **Load ChEBI Data**: The system loads the compressed pickle file containing ChEBI ID to names/synonyms mappings
2. **Document Creation**: Each ChEBI entry is converted into multiple documents - one for each name/synonym
3. **Embedding Creation**: Documents are embedded using the selected model and stored in ChromaDB

### Entity Linking Process

1. **Query Processing**: Chemical mention is converted to embedding
2. **Candidate Retrieval**: Top-k most similar entities are retrieved from ChromaDB
3. **LLM Decision**: Retrieved candidates are formatted as a prompt for LLM final decision
4. **Result**: Best matching ChEBI ID with confidence score

### Document Structure

Each document stored in ChromaDB contains minimal metadata to reduce file size:

```python
{
    "chebi_id": "CHEBI:12345",       # ChEBI identifier
    "name": "glucose"                # Specific name/synonym
}
```

The document text is simply the chemical name itself, making the system efficient and compact.

## Embedding Models

### Default Model

- **Model**: `all-MiniLM-L6-v2`
- **Provider**: Sentence Transformers
- **Pros**: Fast, free, works offline
- **Cons**: General-purpose, not specialized for chemistry

### OpenAI Model

- **Model**: `text-embedding-ada-002`
- **Provider**: OpenAI
- **Pros**: High quality, large training data
- **Cons**: Requires API key, costs money, online only

### Llama Model

- **Model**: `nomic-embed-text`
- **Provider**: Ollama (local)
- **Pros**: Free, works offline, good quality
- **Cons**: Requires local Ollama server setup

## Example Workflow

1. **Prepare embeddings** (one-time setup):

   ```bash
   python load_data.py --model default --collection chebi_default
   python load_data.py --model openai --collection chebi_openai
   ```
2. **Test different models**:

   ```bash
   python rag_entity_linking_example.py --query "caffeine" --model default --collection chebi_default
   python rag_entity_linking_example.py --query "caffeine" --model openai --collection chebi_openai
   ```
3. **Compare results** and choose the best model for your use case
4. **Integrate with your LLM pipeline** using the generated prompts

## References

- ChEBI Database: https://www.ebi.ac.uk/chebi/
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
