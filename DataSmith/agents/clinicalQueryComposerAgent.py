import json
import logging
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Callable

from dotenv import load_dotenv

# Local agents (same folder)
from agents.clinicalDbAgent import fetch_by_name
from agents.clinicalAiAgent import call_groq


def load_env():
    load_dotenv(override=False)


def build_patient_info(
    name: Optional[str],
    table: str,
    schema: str,
    match: str,
    patient_info_json: Optional[str],
) -> Optional[Dict[str, Any]]:
    if patient_info_json:
        try:
            return json.loads(patient_info_json)
        except Exception as e:
            raise RuntimeError(f"Invalid --patient_info_json: {e}")

    if not name:
        return None

    results = fetch_by_name(name=name, table=table, schema=schema, limit=1, match=match)
    if not results:
        return None
    return asdict(results[0])


def make_composer_prompt(user_question: str, patient_info: Optional[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    parts = [
        "You are a clinical query composer for retrieval over a nephrology textbook (KB).",
        "Your task: rewrite the user's question into ONE compact, retrieval-optimized query that will maximize KB recall.",
        "Rules:",
        "- Combine the user's question with any clinically relevant patient context (diagnosis, meds, restrictions).",
        "- Include key medical terms and synonyms (e.g., edema = swelling).",
        "- Keep it short (<= 25 words), unambiguous, and specific to the clinical topic.",
        "- Do NOT include patient identifiers, instructions, citations, or commentary.",
        "- Output ONLY the composed query text (no quotes, no prefix/suffix).",
        "",
        f"User question: {user_question}",
    ]
    if patient_info:
        # Small, structured context
        def g(k):
            v = patient_info.get(k)
            if isinstance(v, list):
                return ", ".join([str(x) for x in v[:6]])
            return v
        parts += [
            "Patient context:",
            f"- Primary diagnosis: {g('primary_diagnosis') or 'n/a'}",
            f"- Medications: {g('medications') or 'n/a'}",
            f"- Dietary restrictions: {g('dietary_restrictions') or 'n/a'}",
            f"- Discharge date: {g('discharge_date') or 'n/a'}",
        ]
    if chat_history:
        recent = chat_history[-4:]
        parts += ["\nRecent conversation (for context only):"]
        for m in recent:
            role = (m.get("role") or "user").capitalize()
            content = (m.get("content") or "").strip()
            if content:
                parts.append(f"- {role}: {content}")
    return "\n".join(parts)


def compose_query(
    user_question: str,
    patient_info: Optional[Dict[str, Any]],
    compose_llm: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    log = logging.getLogger("datasmith.composer")
    prompt = make_composer_prompt(user_question, patient_info, chat_history)
    if llm_fn is not None:
        query = llm_fn(prompt).strip()
    else:
        query = call_groq(prompt, model=compose_llm, temperature=0.1, max_tokens=128).strip()
    # Post-process: keep it single line and trimmed
    query = query.replace("\n", " ").strip()
    # Safety: prevent overly long queries
    if len(query.split()) > 32:
        query = " ".join(query.split()[:32])
    log.info("compose.result len=%d q=%s", len(query), query)
    return query


"""
Query composer utilities for building retrieval-optimized queries.
No CLI entrypoint here; use app/main.py for running the application.
"""
