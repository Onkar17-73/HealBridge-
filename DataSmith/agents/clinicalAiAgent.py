import os
import json
import re
import logging
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction as ChromaSentenceTransformerEmbeddingFunction,
)
from ddgs import DDGS
from dotenv import load_dotenv


def load_env():
    # Load .env if present and environment variables
    load_dotenv(override=False)
    # Ensure main logger is set up when this module is imported standalone
    try:
        from utils.logging_setup import setup_logging
        setup_logging()
    except Exception:
        pass


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
    log = logging.getLogger("datasmith.rag")
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    log.debug("retrieve n=%d top_k=%d best=%.4f", len(docs), top_k, dists[0] if dists else -1)
    return docs, metas, dists


def web_search(query: str, max_results: int = 5) -> List[dict]:
    log = logging.getLogger("datasmith.rag")
    # duckduckgo-search returns dicts with keys like 'title', 'href', 'body'
    out: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    log.debug("web_search qlen=%d results=%d", len(query), len(out))
    return out


def format_patient_section(patient_info: dict | None) -> List[str]:
    if not patient_info:
        return []
    lines = ["\nPatient context (use when relevant):"]
    def grab(key):
        val = patient_info.get(key)
        if isinstance(val, list):
            return ", ".join(val[:6])
        return val
    items = [
        ("Name", grab("patient_name")),
        ("Discharge date", grab("discharge_date")),
        ("Primary diagnosis", grab("primary_diagnosis")),
        ("Medications", grab("medications")),
        ("Dietary restrictions", grab("dietary_restrictions")),
        ("Follow-up", grab("follow_up")),
        ("Warning signs", grab("warning_signs")),
        ("Instructions", grab("discharge_instructions")),
    ]
    for k, v in items:
        if v:
            lines.append(f"- {k}: {v}")
    return lines


def _format_history(chat_history: Optional[List[Dict[str, str]]], max_messages: int = 6) -> List[str]:
    if not chat_history:
        return []
    # Take the last N messages to keep prompt concise
    recent = chat_history[-max_messages:]
    lines = ["\nPrior conversation (use only if relevant):"]
    for m in recent:
        role = m.get("role", "user").capitalize()
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"- {role}: {content}")
    return lines


def make_prompt(query: str, context_blocks: List[Tuple[str, dict]], fallback_blocks: List[dict] | None = None, patient_info: dict | None = None, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    log = logging.getLogger("datasmith.rag")
    lines: List[str] = []
    if context_blocks:
        lines.append("You are a helpful assistant. Prefer the knowledge base (KB) context below and incorporate patient context when relevant. If unsure, say you are unsure.")
        lines.append("\nContext from the knowledge base:")
        for i, (text, meta) in enumerate(context_blocks, start=1):
            src = meta.get("source")
            page = meta.get("page")
            chunk = meta.get("chunk")
            lines.append(f"[KB {i}] Source={src}, page={page}, chunk={chunk}\n{text}")
    elif fallback_blocks:
        lines.append(
            "You are a helpful assistant. You have some web search snippets; also incorporate patient context when relevant. "
            "Answer concisely and cite sources readably. Begin your answer with a brief disclosure that you used web search because the KB context was insufficient "
            "(e.g., 'I couldn't find this in our internal knowledge base, so I checked recent web sources.')."
        )
        lines.append("\nWeb search snippets:")
        for i, r in enumerate(fallback_blocks, start=1):
            title = r.get("title")
            url = r.get("url")
            snippet = r.get("snippet")
            lines.append(f"[Web {i}] {title} — {url}\n{snippet}")
    else:
        lines.append("You are a helpful assistant. No KB/web context is available; use patient context and general knowledge.")

    # Optional patient section
    if patient_info:
        lines.append("\nPatient context (do not cite as KB; cite as [Patient] if needed):")
        lines.extend(format_patient_section(patient_info))

    # Optional chat history
    hist_lines = _format_history(chat_history)
    if hist_lines:
        lines.extend(hist_lines)

    lines.append("\nQuestion:")
    lines.append(query)
    lines.append(
        "\nRequirements:\n"
        "- Be concise (3-5 sentences) and answer the CURRENT user question directly.\n"
        "- Avoid repeating prior assistant content verbatim; if already covered, refer briefly and add new, relevant details.\n"
        "- Provide practical guidance: monitoring steps and when to escalate care; do NOT prescribe or adjust medications.\n"
        "- Do NOT include inline citations; sources will be shown separately.\n"
        "- Only include a brief web-search disclosure when web snippets are provided below; if no web snippets are provided, do NOT mention web search.\n"
        "- If referencing patient information, weave it naturally (no special tags).\n"
        "- If you're not sure, state the uncertainty clearly.\n"
        "- Respect the patient's context (medications, restrictions) if it affects the answer."
    )

    prompt = "\n".join(lines)
    log.debug("prompt.built len=%d has_kb=%s has_web=%s", len(prompt), bool(context_blocks), bool(fallback_blocks))
    return prompt


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 512) -> str:
    log = logging.getLogger("datasmith.llm")
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
    text = resp.choices[0].message.content
    log.debug("groq.call model=%s len=%d", model, len(text) if text else 0)
    return text


def generate_answer(
    query: str,
    persist_dir: str = "chroma_db",
    collection: str = "default",
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm: str = "llama-3.1-8b-instant",
    top_k: int = 5,
    threshold: float = 0.45,
    web_max: int = 5,
    patient_info: dict | None = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    min_kb_hits: int = 1,
    strict_threshold: Optional[float] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> tuple[str, dict]:
    load_env()
    log = logging.getLogger("datasmith.rag")

    pdir = Path(persist_dir)
    context_blocks: List[Tuple[str, dict]] = []

    embedding_fn = build_embedding_fn(emb_model)
    coll = None
    if pdir.exists():
        coll = get_collection(pdir, collection, embedding_fn)

    best_distance = None
    if coll is not None:
        docs, metas, dists = retrieve(coll, query, top_k=top_k)
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is None:
                continue
            if best_distance is None or dist < best_distance:
                best_distance = dist
            if dist <= threshold:
                context_blocks.append((doc, meta))

    # Decide whether to trust KB context using strict gating if provided
    if strict_threshold is not None:
        use_kb = (len(context_blocks) >= max(1, min_kb_hits)) and (best_distance is not None and best_distance <= strict_threshold)
    else:
        use_kb = len(context_blocks) > 0

    use_web = not use_kb
    log.info("retrieval.decision use_kb=%s use_web=%s best=%.4f hits=%d", use_kb, use_web, best_distance or -1, len(context_blocks))
    fallback_blocks: List[dict] | None = None
    if use_web:
        fallback_blocks = web_search(query, max_results=web_max)
        # Clear KB blocks to avoid mixing weak KB with web fallback
        context_blocks = []

    prompt = make_prompt(query, context_blocks, fallback_blocks, patient_info, chat_history)
    if llm_fn is not None:
        answer = llm_fn(prompt)
    else:
        answer = call_groq(prompt, model=llm)

    # Post-process citations: replace [KB i] with friendly source names, e.g., (Source: <filename>, p.<page>)
    # Build index -> (source, page) map consistent with the numbering used in the prompt
    kb_index_map: Dict[int, Dict[str, Optional[str]]] = {}
    for idx, (_doc, meta) in enumerate(context_blocks, start=1):
        src = meta.get("source") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        if src:
            kb_index_map[idx] = {"source": src, "page": page}

    def _kb_repl(match: re.Match) -> str:
        try:
            i = int(match.group(1))
        except Exception:
            return match.group(0)
        meta = kb_index_map.get(i)
        if not meta:
            return match.group(0)
        src = meta.get("source") or "knowledge base"
        # Use only the basename if it's a path
        src_name = os.path.basename(src)
        page = meta.get("page")
        page_txt = f", p.{page}" if page is not None else ""
        return f"(Source: {src_name}{page_txt})"

    answer = re.sub(r"\[KB\s*(\d+)\]", _kb_repl, answer, flags=re.IGNORECASE)

    # Post-process [Web i] to (Source: <domain>) using the same ordering used in the prompt
    web_index_map: Dict[int, Dict[str, Optional[str]]] = {}
    if fallback_blocks:
        for idx, r in enumerate(fallback_blocks, start=1):
            u = r.get("url")
            t = r.get("title")
            if u:
                web_index_map[idx] = {"url": u, "title": t}

    def _web_repl(match: re.Match) -> str:
        try:
            i = int(match.group(1))
        except Exception:
            return match.group(0)
        meta = web_index_map.get(i)
        if not meta:
            return match.group(0)
        raw = (meta.get("url") or "").strip()
        try:
            domain = urlparse(raw).netloc or raw
        except Exception:
            domain = raw
        if domain.startswith("www."):
            domain = domain[4:]
        return f"(Source: {domain})"

    answer = re.sub(r"\[Web\s*(\d+)\]", _web_repl, answer, flags=re.IGNORECASE)

    # If web was used, ensure a clear disclosure line at the start (in case the model omitted it)
    if use_web:
        lead = "I checked recent web sources to supplement our internal knowledge base. "
        low = (answer or "").strip().lower()
        if not any(phrase in low for phrase in [
            "i checked recent web sources",
            "i checked the web",
            "i searched the web",
            "according to recent",
            "to supplement our internal knowledge base",
        ]):
            answer = lead + answer
    else:
        # If web was NOT used, strip any spurious web-disclosure phrasing the model might add
        ans = answer.strip()
        # Consider only the first sentence or line for disclosure-like phrases
        first_split = re.split(r"(\.|\n)\s+", ans, maxsplit=1)
        head = first_split[0].lower()
        disclosure_patterns = [
            "i checked recent web sources",
            "i checked the web",
            "i searched the web",
            "according to recent web",
            "i couldn't find this in our internal knowledge base",
            "since the internal knowledge base was insufficient",
        ]
        if any(p in head for p in disclosure_patterns):
            # Drop the first sentence/line
            if len(first_split) > 2:
                answer = first_split[2].lstrip()
            else:
                # No clear sentence boundary; remove the phrase occurrences from the head
                for p in disclosure_patterns:
                    ans = re.sub(re.escape(p), "", ans, flags=re.IGNORECASE)
                answer = ans.strip()

    # Build a separate user-friendly references list and strip inline citations
    # KB references (unique by source+page)
    kb_refs_struct: List[Dict[str, Optional[str]]] = []
    seen_kb = set()
    for _, meta in context_blocks:
        src = (meta or {}).get("source")
        page = (meta or {}).get("page")
        if not src:
            continue
        key = (src, str(page) if page is not None else "")
        if key in seen_kb:
            continue
        seen_kb.add(key)
        kb_refs_struct.append({
            "source": src,
            "page": str(page) if page is not None else None,
            "label": f"{os.path.basename(src)}" + (f", p.{page}" if page is not None else ""),
        })

    # Web references (unique by domain)
    web_refs_struct: List[Dict[str, Optional[str]]] = []
    seen_domains = set()
    if fallback_blocks:
        for r in fallback_blocks:
            url = (r or {}).get("url") or ""
            title = (r or {}).get("title")
            try:
                domain = urlparse(url).netloc or url
            except Exception:
                domain = url
            if domain.startswith("www."):
                domain = domain[4:]
            if not domain:
                continue
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            web_refs_struct.append({
                "domain": domain,
                "url": url or None,
                "title": title or None,
                "label": domain,
            })

    # Strip inline citations like (Source: ...)
    answer = re.sub(r"\s*\(Source:[^)]+\)", "", answer)
    # Remove any leftover bracket tags if present
    answer = re.sub(r"\[(KB|Web)\s*\d+\]", "", answer, flags=re.IGNORECASE)
    answer = answer.replace("[Patient]", "").replace("  ", " ").strip()

    info = {
        "used_kb": not use_web,
        "used_web": use_web,
        "best_distance": best_distance,
        "strict_threshold": strict_threshold,
        "min_kb_hits": min_kb_hits,
        "kb_refs": [
            {"source": m.get("source"), "page": m.get("page"), "chunk": m.get("chunk")}
            for _, m in context_blocks
        ],
        "user_references": {
            "kb": kb_refs_struct,
            "web": web_refs_struct,
        },
    }

    log.info("answer.ready used_kb=%s used_web=%s answer_len=%d", info["used_kb"], info["used_web"], len(answer or ""))
    return answer, info


"""
Module exports callable functions for retrieval-augmented answering.
This module has no CLI entrypoint by design; use app/main.py to run the system.
"""
