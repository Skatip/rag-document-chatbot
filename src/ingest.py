from __future__ import annotations

import os
import uuid
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_documents, chunk_text, save_json, l2_normalize, Chunk


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    resp = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vectors


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.")

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    storage_dir = os.path.join(os.path.dirname(__file__), "..", "storage")
    os.makedirs(storage_dir, exist_ok=True)

    docs = load_documents(data_dir)
    if not docs:
        raise SystemExit(f"No documents found in {data_dir}. Add .txt or .pdf files.")

    # Build chunks
    all_chunks: List[Chunk] = []
    for source, text in docs:
        for part in chunk_text(text, chunk_size=900, chunk_overlap=150):
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    source=source,
                    text=part
                )
            )

    print(f"Loaded {len(docs)} documents, created {len(all_chunks)} chunks.")

    # Embed chunks
    client = OpenAI()
    texts = [c.text for c in all_chunks]
    vectors = embed_texts(client, embed_model, texts)

    # Normalize for cosine similarity using inner product
    vectors = l2_normalize(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save FAISS index + chunks metadata
    index_path = os.path.join(storage_dir, "index.faiss")
    chunks_path = os.path.join(storage_dir, "chunks.json")

    faiss.write_index(index, index_path)
    save_json(chunks_path, [c.__dict__ for c in all_chunks])

    print("✅ Saved:")
    print(f"- {index_path}")
    print(f"- {chunks_path}")
    print("\nNext: python src/query.py \"your question here\"")


if __name__ == "__main__":
    main()
