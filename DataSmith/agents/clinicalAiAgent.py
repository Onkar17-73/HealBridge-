import argparse
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction as ChromaSentenceTransformerEmbeddingFunction,
)
from duckduckgo_search import DDGS
from dotenv import load_dotenv


def load_env():
    # Load .env if present and environment variables
    load_dotenv(override=False)


def build_embedding_fn(model_name: str):
    return ChromaSentenceTransformerEmbeddingFunction(model_name=model_name)


def get_collection(persist_dir: Path, collection_name: str, embedding_fn):
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return client.get_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception:
        return None


def retrieve(
    collection,
    query: str,
    top_k: int = 5,
) -> Tuple[List[str], List[dict], List[float]]:
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return docs, metas, dists


def web_search(query: str, max_results: int = 5) -> List[dict]:
    # duckduckgo-search returns dicts with keys like 'title', 'href', 'body'
    out: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    return out


def make_prompt(query: str, context_blocks: List[Tuple[str, dict]], fallback_blocks: List[dict] | None = None) -> str:
    lines: List[str] = []
    if context_blocks:
        lines.append("You are a helpful assistant. Answer using the provided context. If unsure, say you are unsure.")
        lines.append("\nContext from the knowledge base:")
        for i, (text, meta) in enumerate(context_blocks, start=1):
            src = meta.get("source")
            page = meta.get("page")
            chunk = meta.get("chunk")
            lines.append(f"[KB {i}] Source={src}, page={page}, chunk={chunk}\n{text}")
    elif fallback_blocks:
        lines.append("You are a helpful assistant. You have some web search snippets. Answer concisely and cite URLs.")
        lines.append("\nWeb search snippets:")
        for i, r in enumerate(fallback_blocks, start=1):
            title = r.get("title")
            url = r.get("url")
            snippet = r.get("snippet")
            lines.append(f"[Web {i}] {title} — {url}\n{snippet}")
    else:
        lines.append("You are a helpful assistant. No context is available. Answer concisely based on general knowledge.")

    lines.append("\nQuestion:")
    lines.append(query)
    lines.append("\nRequirements:\n- Be concise (4-8 sentences).\n- Cite sources inline like [KB 1] or with URLs [Web 2].\n- If you're not sure, state the uncertainty clearly.")

    return "\n".join(lines)


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 512) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Create a .env file or set the environment variable.")
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("groq package is not installed. Please pip install groq.") from e

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Clinical AI Agent: Answer with local RAG, fallback to web search, generate with Groq")
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory of Chroma DB")
    parser.add_argument("--collection", type=str, default="default", help="Chroma collection name")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for retrieval")
    parser.add_argument("--llm", type=str, default="llama-3.1-8b-instant", help="Groq model name for generation")
    parser.add_argument("--top_k", type=int, default=5, help="Number of KB chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.45, help="Max cosine distance to accept KB context (lower is more similar)")
    parser.add_argument("--web_max", type=int, default=5, help="Max web results to fetch when falling back")

    args = parser.parse_args()

    load_env()

    persist_dir = Path(args.persist_dir)
    context_blocks: List[Tuple[str, dict]] = []

    embedding_fn = build_embedding_fn(args.model)
    collection = None
    if persist_dir.exists():
        collection = get_collection(persist_dir, args.collection, embedding_fn)

    best_distance = None
    if collection is not None:
        docs, metas, dists = retrieve(collection, args.query, top_k=args.top_k)
        # Select only those under threshold
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is None:
                continue
            if best_distance is None or dist < best_distance:
                best_distance = dist
            if dist <= args.threshold:
                context_blocks.append((doc, meta))

    use_web = len(context_blocks) == 0

    fallback_blocks: List[dict] | None = None
    if use_web:
        fallback_blocks = web_search(args.query, max_results=args.web_max)

    prompt = make_prompt(args.query, context_blocks, fallback_blocks)

    if best_distance is not None:
        print(f"[debug] best KB distance: {best_distance:.4f} (threshold {args.threshold})")
    print("[debug] using KB context:" if not use_web else "[debug] using Web search fallback")

    answer = call_groq(prompt, model=args.llm)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
