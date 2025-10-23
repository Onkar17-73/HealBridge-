import argparse
import os
from pathlib import Path
from typing import List, Tuple
import uuid

from pypdf import PdfReader
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction as ChromaSentenceTransformerEmbeddingFunction,
)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def extract_pdf_chunks(pdf_path: Path, chunk_size: int, overlap: int) -> Tuple[List[str], List[dict]]:
    reader = PdfReader(str(pdf_path))
    documents: List[str] = []
    metadatas: List[dict] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        page_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(page_chunks):
            documents.append(chunk)
            metadatas.append({
                "source": str(pdf_path.name),
                "page": page_index,
                "chunk": chunk_idx,
            })
    return documents, metadatas


def build_embedding_fn(model_name: str):
    # Use Chroma's EmbeddingFunction wrapper to satisfy its interface
    return ChromaSentenceTransformerEmbeddingFunction(model_name=model_name)


def ensure_collection(client: chromadb.ClientAPI, name: str, embedding_fn):
    # Try to get; if not exists, create
    try:
        col = client.get_collection(name=name, embedding_function=embedding_fn)
        # If existing collection has different embedding, recreate if requested upstream.
        return col
    except Exception:
        return client.create_collection(name=name, embedding_function=embedding_fn)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into a local Chroma vector DB")
    parser.add_argument("--pdf", type=str, required=False, help="Path to the PDF file to ingest")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory to store Chroma DB")
    parser.add_argument("--collection", type=str, default="default", help="Chroma collection name")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in characters")
    parser.add_argument("--reset", action="store_true", help="If set, delete and recreate the collection before ingesting")

    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]

    pdf_path = Path(args.pdf) if args.pdf else workspace / "comprehensive-clinical-nephrology.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Pass --pdf <path-to-file.pdf>.")

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using persist dir: {persist_dir}")

    embedding_fn = build_embedding_fn(args.model)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'")
        except Exception:
            pass

    collection = ensure_collection(client, args.collection, embedding_fn)

    print(f"Reading and chunking PDF: {pdf_path.name}")
    documents, metadatas = extract_pdf_chunks(pdf_path, args.chunk_size, args.overlap)

    if not documents:
        print("No text extracted from the PDF. Nothing to ingest.")
        return

    print(f"Adding {len(documents)} chunks to collection '{args.collection}'...")

    # Add in batches to reduce memory pressure
    batch_size = 256
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
        docs_batch = documents[i:i+batch_size]
        meta_batch = metadatas[i:i+batch_size]
        ids_batch = [str(uuid.uuid4()) for _ in range(len(docs_batch))]
        collection.add(documents=docs_batch, metadatas=meta_batch, ids=ids_batch)

    print("Persisting database...")
    # chromadb PersistentClient persists automatically; just a log message for clarity
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
