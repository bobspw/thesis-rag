"""
Vector Store Module for Thesis Research Assistant
=================================================

Handles embedding and similarity search across research papers using:
- Ollama's nomic-embed-text for local embeddings (no API costs)
- FAISS for fast vector similarity search

Author: Ajay Pravin Mahale
Thesis: Explainable AI for LLMs

WHY LOCAL EMBEDDINGS?
- Privacy: Unpublished thesis work stays on my machine
- Cost: No API fees for embedding 12+ research papers
- Speed: No network latency for queries
- Reliability: Works offline

TECHNICAL NOTES:
- nomic-embed-text produces 768-dim vectors (vs OpenAI's 1536)
- Quality is sufficient for academic paper retrieval
- FAISS IndexFlatIP gives exact search (no approximation errors)
"""

import json
import os
import requests
from dataclasses import asdict

import faiss
import numpy as np

from ingestion import DocumentChunk


class OllamaEmbeddings:
    """
    Local embeddings via Ollama's nomic-embed-text model.

    This runs entirely on my M1 Mac - no data sent to external servers.
    Embedding dimension is 768 (smaller than OpenAI but works well).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.embedding_dim = 768

    def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for a single text."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.

        Ollama processes one at a time (no native batching),
        but it's still fast enough for ~100 chunks.
        """
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 20 == 0:
                print(f"  Embedding chunk {i + 1}/{len(texts)}...")
            embedding = self.embed(text)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)


class VectorStore:
    """
    FAISS-backed vector store for research paper retrieval.

    Architecture:
    - FAISS index: Stores embedding vectors for similarity search
    - Metadata list: Stores paper names, page numbers, text for citations

    The two structures are parallel: index i in FAISS corresponds
    to metadata[i]. This enables returning full citation info.
    """

    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embedder = OllamaEmbeddings(model=embedding_model)
        self.embedding_dim = self.embedder.embedding_dim

        # FAISS index using inner product (cosine sim on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Parallel metadata storage for citations
        self.metadata: list[dict] = []

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        """
        Add paper chunks to the index.

        This embeds all chunks and stores both vectors (in FAISS)
        and metadata (in self.metadata) for retrieval.
        """
        if not chunks:
            return

        print(f"Embedding {len(chunks)} chunks from your research papers...")

        # Extract texts and embed
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store metadata for citations
        for chunk in chunks:
            self.metadata.append(asdict(chunk))

        print(f"✓ Indexed {len(chunks)} chunks")
        print(f"  Total vectors in index: {self.index.ntotal}")

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        """
        Search for paper chunks relevant to a research question.

        Returns list of (metadata, score) tuples where metadata
        contains source_file, page_number, and text for citation.
        """
        if self.index.ntotal == 0:
            return []

        # Embed and normalize query
        query_embedding = self.embedder.embed(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Build results with citation metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            metadata = self.metadata[idx]
            results.append((metadata, float(score)))

        return results

    def save(self, directory: str) -> None:
        """Save index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Saved index to {directory}")

    def load(self, directory: str) -> None:
        """Load index and metadata from disk."""
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        print(f"✓ Loaded index from {directory}")
        print(f"  Total vectors: {self.index.ntotal}")


if __name__ == "__main__":
    print("Testing Ollama connection...")
    try:
        embedder = OllamaEmbeddings()
        test = embedder.embed("mechanistic interpretability")
        print(f"✓ Ollama working! Embedding dim: {len(test)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure 'ollama serve' is running")
