import argparse
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction as ChromaSentenceTransformerEmbeddingFunction,
)


def build_embedding_fn(model_name: str):
    return ChromaSentenceTransformerEmbeddingFunction(model_name=model_name)


def main():
    parser = argparse.ArgumentParser(description="Query a local Chroma vector DB")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory of Chroma DB")
    parser.add_argument("--collection", type=str, default="default", help="Chroma collection name")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model for retrieval")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist dir not found: {persist_dir}")

    embedding_fn = build_embedding_fn(args.model)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=args.collection, embedding_function=embedding_fn)

    results = collection.query(query_texts=[args.query], n_results=args.top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    print("Top matches:\n")
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        src = meta.get("source")
        page = meta.get("page")
        chunk = meta.get("chunk")
        print(f"{idx}. score={dist:.4f} | {src} (page {page}, chunk {chunk})")
        print(doc[:500].replace("\n", " ") + ("..." if len(doc) > 500 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
