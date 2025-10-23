import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any, List

# Ensure project root on sys.path so 'agents' package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import logging
from agents.clinicalDbAgent import fetch_by_name
from agents.clinicalAiAgent import generate_answer, call_groq as groq_fallback_call
from agents.receptionistAgent import generate_opening_message, generate_handoff_line, classify_smalltalk
from agents.clinicalQueryComposerAgent import compose_query
from typing import Optional, Callable


# Defaults for RAG + LLM
DEFAULTS = {
    "persist_dir": "chroma_db",
    "collection": "default",
    "embed_model": "sentence-transformers/all-mpnet-base-v2",
    "top_k": 8,
    "threshold": 0.55,
    "web_max": 5,
    "compose_llm": "llama-3.1-8b-instant",
    "answer_llm": "llama-3.1-8b-instant",
    # Stricter gating for KB usage: require at least 2 good hits and best distance <= 0.40
    "min_kb_hits": 2,
    "strict_threshold": 0.40,
}


def load_env():
    load_dotenv(override=False)
    # Initialize logging once
    try:
        from utils.logging_setup import setup_logging
        setup_logging()
    except Exception:
        pass


def prompt(text: str) -> str:
    try:
        return input(text).strip()
    except KeyboardInterrupt:
        print("\nGoodbye")
        sys.exit(0)


"""
Thin orchestrator: delegates greeting/handoff to receptionistAgent,
query composition to clinicalQueryComposerAgent, and answering to clinicalAiAgent.
"""


def fetch_patient_interactive() -> Optional[Dict[str, Any]]:
    # Ask for name until found or user quits
    while True:
        name = prompt("Hello! I'm your post-discharge care assistant. What's your name? ")
        if not name:
            print("I didn't catch your name. Please try again.")
            continue
        rows = fetch_by_name(name=name, table="patient_discharges", schema="public", limit=1, match="exact")
        if rows:
            row = rows[0]
            # Normalize to plain dict
            if hasattr(row, "__dataclass_fields__"):
                return asdict(row)
            elif isinstance(row, dict):
                return row
            else:
                try:
                    return dict(row)  # type: ignore[arg-type]
                except Exception:
                    # Fallback minimal mapping
                    return {"patient_name": str(row)}
        else:
            retry = prompt("I couldn't find your record. Try full name as in records, or press Enter to cancel: ")
            if not retry:
                return None


 


def main():
    load_env()
    print("Welcome! Let's check in and get you the information you need.\n")

    log = logging.getLogger("datasmith.main")
    patient = fetch_patient_interactive()
    if not patient:
        print("No record selected. Exiting.")
        log.info("no patient record selected; exiting")
        return

    # Build a shared LangChain LLM callable if available (clean, optional)
    llm_fn: Optional[Callable[[str], str]] = None
    try:
        from langchain_groq import ChatGroq  # type: ignore
        chat = ChatGroq(model=DEFAULTS["answer_llm"], temperature=0.2)
        # Simple client-side rate limiter to avoid 429s; also fallback to Groq SDK on error
        rate_state = {"last": 0.0}
        min_interval_s = 1.2

        def _safe_llm(p: str) -> str:
            # Ensure a minimal spacing between calls
            now = time.time()
            elapsed = now - rate_state["last"]
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            try:
                # Hint a token cap to reduce load; kwargs are forwarded to client
                out = chat.invoke(p, max_tokens=512)
                txt = getattr(out, "content", "")
                txt = txt if isinstance(txt, str) else str(txt)
                log.debug("llm.invoke success; len=%d", len(txt))
                return txt.strip()
            except Exception:
                # Fast fallback to Groq SDK to avoid long internal retry sleeps
                log.warning("llm.invoke failed; falling back to groq SDK", exc_info=True)
                return groq_fallback_call(p, model=DEFAULTS["answer_llm"], temperature=0.2, max_tokens=512).strip()
            finally:
                rate_state["last"] = time.time()

        llm_fn = _safe_llm
    except Exception:
        llm_fn = None

    # Greeting + check-in (single natural message)
    first_name = (patient.get("patient_name") or "there").split()[0]
    log.info("session.start name=%s dx=%s", patient.get("patient_name"), patient.get("primary_diagnosis"))
    opening = generate_opening_message(patient, llm=DEFAULTS["answer_llm"], llm_fn=llm_fn)
    log.info("receptionist.greeting output=%s", opening)

    print(opening)

    # Capture user's concern
    user_question = prompt("\nPlease tell me what you'd like help with: ")
    log.info("user.input first=%s", user_question)
    if not user_question:
        print("Thanks! If you have a question later, just let me know.")
        return

    # Receptionist handoff message (LLM-generated) before connecting to the clinical AI
    handoff_msg = generate_handoff_line(first_name, user_question, llm=DEFAULTS["answer_llm"], llm_fn=llm_fn)
    print("\nReceptionist:\n")
    print(handoff_msg)
    log.info("receptionist.handoff output=%s", handoff_msg)

    # Compose retrieval-optimized query
    composed = compose_query(user_question, patient, DEFAULTS["compose_llm"], chat_history=[
        {"role": "assistant", "content": opening},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": handoff_msg},
    ], llm_fn=llm_fn)
    log.info("composer.query first=%s", composed)

    # Answer via Clinical AI Agent (RAG + patient context)
    # Initialize chat history including the assistant opening and first user message
    chat_history: List[Dict[str, str]] = [
        {"role": "assistant", "content": opening},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": handoff_msg},
    ]

    answer, info = generate_answer(
        query=composed,
        persist_dir=DEFAULTS["persist_dir"],
        collection=DEFAULTS["collection"],
        emb_model=DEFAULTS["embed_model"],
        llm=DEFAULTS["answer_llm"],
        top_k=DEFAULTS["top_k"],
        threshold=DEFAULTS["threshold"],
        web_max=DEFAULTS["web_max"],
        patient_info=patient,
        chat_history=chat_history,
        min_kb_hits=DEFAULTS["min_kb_hits"],
        strict_threshold=DEFAULTS["strict_threshold"],
        llm_fn=llm_fn,
    )
    log.info("answer.generated used_kb=%s used_web=%s best_distance=%s kb_refs=%d", info.get("used_kb"), info.get("used_web"), info.get("best_distance"), len(info.get("kb_refs") or []))

    print("\nAssistant:\n")
    print(answer)
    # Friendly sources list (if available)
    try:
        refs = info.get("user_references", {}) or {}
        kb_refs = refs.get("kb") or []
        web_refs = refs.get("web") or []
        if kb_refs or web_refs:
            print("\nSources:\n")
            for r in kb_refs:
                print(f"- {r.get('label')}")
            for r in web_refs:
                label = r.get("label") or r.get("domain") or "web source"
                print(f"- {label}")
            try:
                kb_labels = [r.get("label") for r in kb_refs]
                web_labels = [(r.get("label") or r.get("domain")) for r in web_refs]
                log.info("sources.list kb=%s web=%s", kb_labels, web_labels)
            except Exception:
                pass
    except Exception:
        pass
    # Optional brief debug to understand retrieval behavior
    try:
        used_kb = info.get("used_kb")
        used_web = info.get("used_web")
        best = info.get("best_distance")
        log.debug("retrieval.debug used_kb=%s used_web=%s best=%s", used_kb, used_web, best)
        if used_kb or used_web:
            print("\n[debug] retrieval:")
            if best is not None:
                print(f"- best KB distance: {best:.4f} (threshold {DEFAULTS['threshold']})")
            print(f"- used KB: {used_kb}")
            print(f"- used Web fallback: {used_web}")
            st = info.get("strict_threshold")
            mk = info.get("min_kb_hits")
            if st is not None:
                print(f"- strict gating: min_kb_hits={mk}, strict_threshold={st}")
            if used_kb:
                refs = info.get("kb_refs", [])
                for i, r in enumerate(refs, start=1):
                    src = r.get("source")
                    page = r.get("page")
                    chunk = r.get("chunk")
                    print(f"  KB {i}: {src} (page {page}, chunk {chunk})")
    except Exception:
        pass

    # Enter multi-turn loop
    chat_history.append({"role": "assistant", "content": answer})
    last_user_norm: Optional[str] = None
    while True:
        try:
            next_q = prompt("\nAsk another question (or type 'exit' to finish): ")
        except KeyboardInterrupt:
            print("\nGoodbye")
            log.info("session.end interrupt")
            return
        if not next_q or next_q.lower() in {"exit", "quit", "q"}:
            print("Take care!")
            log.info("session.end user_exit")
            return

        # Handle small-talk without querying KB/web
        st = classify_smalltalk(next_q)
        if st == "gratitude":
            msg = f"You’re very welcome, {first_name}. If anything else comes up, I’m here."
            print("\nAssistant:\n")
            print(msg)
            log.info("smalltalk.gratitude reply=%s", msg)
            chat_history.append({"role": "assistant", "content": msg})
            continue
        if st == "goodbye":
            print("\nAssistant:\n")
            print("Wishing you a smooth recovery. Take care!")
            log.info("smalltalk.goodbye")
            return
        if st == "affirm":
            msg = "Noted. Is there anything else you’d like to discuss?"
            print("\nAssistant:\n")
            print(msg)
            log.info("smalltalk.affirm reply=%s", msg)
            chat_history.append({"role": "assistant", "content": msg})
            continue

        # Prevent loops on repeated identical questions
        norm = (next_q or "").strip().lower()
        if last_user_norm and norm == last_user_norm:
            msg = "We just covered that. Is there a new detail or change you'd like to discuss?"
            print("\nAssistant:\n")
            print(msg)
            chat_history.append({"role": "assistant", "content": msg})
            continue

        chat_history.append({"role": "user", "content": next_q})
        composed_next = compose_query(next_q, patient, DEFAULTS["compose_llm"], chat_history=chat_history, llm_fn=llm_fn)
        log.info("composer.query next=%s", composed_next)
        answer_next, info_next = generate_answer(
            query=composed_next,
            persist_dir=DEFAULTS["persist_dir"],
            collection=DEFAULTS["collection"],
            emb_model=DEFAULTS["embed_model"],
            llm=DEFAULTS["answer_llm"],
            top_k=DEFAULTS["top_k"],
            threshold=DEFAULTS["threshold"],
            web_max=DEFAULTS["web_max"],
            patient_info=patient,
            chat_history=chat_history,
            min_kb_hits=DEFAULTS["min_kb_hits"],
            strict_threshold=DEFAULTS["strict_threshold"],
            llm_fn=llm_fn,
        )
        log.info("answer.generated next used_kb=%s used_web=%s best_distance=%s kb_refs=%d", info_next.get("used_kb"), info_next.get("used_web"), info_next.get("best_distance"), len(info_next.get("kb_refs") or []))

        print("\nAssistant:\n")
        print(answer_next)
        # Friendly sources list (if available)
        try:
            refs = info_next.get("user_references", {}) or {}
            kb_refs = refs.get("kb") or []
            web_refs = refs.get("web") or []
            if kb_refs or web_refs:
                print("\nSources:\n")
                for r in kb_refs:
                    print(f"- {r.get('label')}")
                for r in web_refs:
                    label = r.get("label") or r.get("domain") or "web source"
                    print(f"- {label}")
                try:
                    kb_labels = [r.get("label") for r in kb_refs]
                    web_labels = [(r.get("label") or r.get("domain")) for r in web_refs]
                    log.info("sources.list next kb=%s web=%s", kb_labels, web_labels)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            used_kb = info_next.get("used_kb")
            used_web = info_next.get("used_web")
            best = info_next.get("best_distance")
            log.debug("retrieval.debug next used_kb=%s used_web=%s best=%s", used_kb, used_web, best)
            if used_kb or used_web:
                print("\n[debug] retrieval:")
                if best is not None:
                    print(f"- best KB distance: {best:.4f} (threshold {DEFAULTS['threshold']})")
                print(f"- used KB: {used_kb}")
                print(f"- used Web fallback: {used_web}")
                st = info_next.get("strict_threshold")
                mk = info_next.get("min_kb_hits")
                if st is not None:
                    print(f"- strict gating: min_kb_hits={mk}, strict_threshold={st}")
                if used_kb:
                    refs = info_next.get("kb_refs", [])
                    for i, r in enumerate(refs, start=1):
                        src = r.get("source")
                        page = r.get("page")
                        chunk = r.get("chunk")
                        print(f"  KB {i}: {src} (page {page}, chunk {chunk})")
        except Exception:
            pass

    chat_history.append({"role": "assistant", "content": answer_next})
    last_user_norm = norm


if __name__ == "__main__":
    main()
