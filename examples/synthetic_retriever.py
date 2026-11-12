"""Synthetic reaction retriever for offline mock demos/tests.

Not a verified catalog. Real local demos must use RetrievalResources +
load_retriever over data/kegg/kegg_reaction_features.lzma and a selected encoder.
"""
from __future__ import annotations


class SyntheticTinyRetriever:
    """12-entry BM25 fixture for mock/synthetic demonstrations only."""

    provenance = {
        "resource": "synthetic",
        "verified_selected_checkpoint": False,
        "kind": "synthetic_tiny_retriever",
    }
    catalog = {
        f"R{i:05d}": {
            "NAME": "ATP" if i == 1 else f"name{i}",
            "DEFINITION": "ATP hydrolysis",
            "EQUATION": "ATP => ADP",
        }
        for i in range(1, 13)
    }

    def retrieve(self, query):
        from core.reaction import retrieval

        docs = [
            {"kegg_id": kid, "text": retrieval.document_text(kid, fields)}
            for kid, fields in self.catalog.items()
        ]
        return retrieval.evidence_for(
            self.catalog, retrieval.BM25(docs).rank(query), list(self.catalog)[::-1]
        )
