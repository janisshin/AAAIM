#!/usr/bin/env python3
"""
RAG Setup Script for AAAIM

Creates vector embeddings for ontology databases used in RAG-based annotation.
Run this once before using method="rag" in annotate_model() or curate_model().

Usage:
    python setup_rag.py                        # set up all databases (human, 9606)
    python setup_rag.py --databases chebi      # ChEBI only
    python setup_rag.py --tax_id 10090         # mouse instead of human
    python setup_rag.py --databases chebi uniprot --tax_id 9606
"""

import subprocess
import sys
import argparse
from pathlib import Path

SUPPORTED_DATABASES = ["chebi", "ncbigene", "uniprot"]
LOAD_SCRIPT = Path(__file__).parent / "data" / "load_data.py"


def build_embeddings(databases, tax_id, model="default"):
    for db in databases:
        cmd = [sys.executable, str(LOAD_SCRIPT), "--database", db, "--model", model]
        if db in ("ncbigene", "uniprot"):
            cmd += ["--tax_id", str(tax_id)]
        print(f"\n{'='*60}")
        print(f"Building embeddings for: {db}" + (f" (tax_id={tax_id})" if db != "chebi" else ""))
        print(f"{'='*60}")
        result = subprocess.run(cmd, cwd=str(LOAD_SCRIPT.parent))
        if result.returncode != 0:
            print(f"ERROR: Failed to build embeddings for {db}. Check the output above.")
            sys.exit(1)
    print(f"\nDone! Embeddings saved to data/chroma_storage/")


def main():
    parser = argparse.ArgumentParser(description="Set up RAG embeddings for AAAIM")
    parser.add_argument(
        "--databases", nargs="+", choices=SUPPORTED_DATABASES, default=SUPPORTED_DATABASES,
        help="Databases to build embeddings for (default: all)"
    )
    parser.add_argument(
        "--tax_id", default="9606",
        help="NCBI taxonomy ID for species-specific databases (default: 9606 = human)"
    )
    parser.add_argument(
        "--model", default="default", choices=["default", "openai", "llama"],
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )
    args = parser.parse_args()

    print("AAAIM RAG Setup")
    print(f"Databases : {', '.join(args.databases)}")
    print(f"Tax ID    : {args.tax_id}")
    print(f"Emb model : {args.model}")
    build_embeddings(args.databases, args.tax_id, args.model)


if __name__ == "__main__":
    main()
