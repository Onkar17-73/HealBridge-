import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Callable

import streamlit as st

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import logging

from agents.clinicalDbAgent import fetch_by_name
from agents.clinicalAiAgent import generate_answer, call_groq as groq_fallback_call
from agents.receptionistAgent import (
    generate_opening_message,
    generate_handoff_line,
    classify_smalltalk,
)
from agents.clinicalQueryComposerAgent import compose_query

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
    "min_kb_hits": 2,
    "strict_threshold": 0.40,
}


def load_env():
    load_dotenv(override=False)
    try:
        from utils.logging_setup import setup_logging
        setup_logging()
    except Exception:
        pass


@st.cache_resource
def get_llm_fn() -> Optional[Callable[[str], str]]:
    """Build a shared LangChain LLM callable if available (clean, optional)"""
    try:
        from langchain_groq import ChatGroq  # type: ignore
        chat = ChatGroq(model=DEFAULTS["answer_llm"], temperature=0.2)
        rate_state = {"last": 0.0}
        min_interval_s = 1.2

        def _safe_llm(p: str) -> str:
            now = time.time()
            elapsed = now - rate_state["last"]
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            try:
                out = chat.invoke(p, max_tokens=512)
                txt = getattr(out, "content", "")
                txt = txt if isinstance(txt, str) else str(txt)
                return txt.strip()
            except Exception:
                # Fast fallback to Groq SDK
                return groq_fallback_call(p, model=DEFAULTS["answer_llm"], temperature=0.2, max_tokens=512).strip()
            finally:
                rate_state["last"] = time.time()

        return _safe_llm
    except Exception:
        return None


def fetch_patient_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Fetch patient record by name"""
    rows = fetch_by_name(name=name, table="patient_discharges", schema="public", limit=1, match="exact")
    if rows:
        row = rows[0]
        if hasattr(row, "__dataclass_fields__"):
            return asdict(row)
        elif isinstance(row, dict):
            return row
        else:
            try:
                return dict(row)  # type: ignore[arg-type]
            except Exception:
                return {"patient_name": str(row)}
    return None


def display_sources(info: Dict[str, Any]):
    """Display sources in a collapsible dropdown"""
    try:
        refs = info.get("user_references", {}) or {}
        kb_refs = refs.get("kb") or []
        web_refs = refs.get("web") or []
        if kb_refs or web_refs:
            with st.expander("📚 View Sources"):
                for r in kb_refs:
                    st.markdown(f"- {r.get('label')}")
                for r in web_refs:
                    label = r.get("label") or r.get("domain") or "web source"
                    st.markdown(f"- {label}")
    except Exception:
        pass


def display_retrieval_debug(info: Dict[str, Any]):
    """Display retrieval debug info in a collapsible section"""
    try:
        used_kb = info.get("used_kb")
        used_web = info.get("used_web")
        best = info.get("best_distance")
        if used_kb or used_web:
            with st.expander("🔍 Retrieval Details"):
                if best is not None:
                    st.markdown(f"**Best KB distance:** {best:.4f} (threshold {DEFAULTS['threshold']})")
                st.markdown(f"**Used KB:** {used_kb}")
                st.markdown(f"**Used Web Fallback:** {used_web}")
                st = info.get("strict_threshold")
                mk = info.get("min_kb_hits")
                if st is not None:
                    st.markdown(f"**Strict gating:** min_kb_hits={mk}, strict_threshold={st}")
                if used_kb:
                    refs = info.get("kb_refs", [])
                    for i, r in enumerate(refs, start=1):
                        src = r.get("source")
                        page = r.get("page")
                        chunk = r.get("chunk")
                        st.markdown(f"  KB {i}: {src} (page {page}, chunk {chunk})")
    except Exception:
        pass


def main():
    load_env()
    log = logging.getLogger("datasmith.streamlit")

    # Page config
    st.set_page_config(
        page_title="Post-Discharge Care Assistant",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Post-Discharge Care Assistant")

    # Initialize session state
    if "patient" not in st.session_state:
        st.session_state.patient = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_fn" not in st.session_state:
        st.session_state.llm_fn = get_llm_fn()
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = False

    # Main chat area
    if not st.session_state.patient:
        st.info("Welcome! 👋 Let's get started by looking up your patient record.")
        
        # Patient lookup in chat
        col1, col2 = st.columns([3, 1])
        with col1:
            patient_name = st.text_input("Enter your full name as it appears in records:")
        with col2:
            lookup_btn = st.button("Look up Record", type="primary", use_container_width=True)
        
        if lookup_btn and patient_name:
            with st.spinner("Looking up your record..."):
                patient = fetch_patient_by_name(patient_name)
                if patient:
                    st.session_state.patient = patient
                    st.session_state.chat_history = []
                    st.session_state.greeting_shown = False
                    log.info("patient.lookup success name=%s", patient.get("patient_name"))
                    st.success("Record found! ✓")
                    st.rerun()
                else:
                    log.warning("patient.lookup failed name=%s", patient_name)
                    st.error("❌ Record not found. Please check the spelling and try again.")
        return

    # Patient info header with logout
    patient = st.session_state.patient
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Patient:** {patient.get('patient_name', 'Unknown')}")
    with col2:
        if patient.get("primary_diagnosis"):
            st.markdown(f"**Diagnosis:** {patient.get('primary_diagnosis')}")
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.patient = None
            st.session_state.chat_history = []
            st.session_state.greeting_shown = False
            st.rerun()
    st.divider()

    # Show greeting and handoff once
    if not st.session_state.greeting_shown:
        first_name = (patient.get("patient_name") or "there").split()[0]
        log.info("session.start name=%s dx=%s", patient.get("patient_name"), patient.get("primary_diagnosis"))

        opening = generate_opening_message(patient, llm=DEFAULTS["answer_llm"], llm_fn=st.session_state.llm_fn)
        log.info("receptionist.greeting output=%s", opening)

        st.session_state.chat_history.append({"role": "assistant", "content": opening})
        st.session_state.greeting_shown = True

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about your care...", key="chat_input")

    if user_input:
        patient = st.session_state.patient
        first_name = (patient.get("patient_name") or "there").split()[0]

        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        log.info("user.input msg=%s", user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        # For the FIRST question, we need the handoff line before answering
        if len([m for m in st.session_state.chat_history if m["role"] == "user"]) == 1:
            # First user question—generate handoff line
            handoff_msg = generate_handoff_line(first_name, user_input, llm=DEFAULTS["answer_llm"], llm_fn=st.session_state.llm_fn)
            log.info("receptionist.handoff output=%s", handoff_msg)
            
            with st.chat_message("assistant"):
                st.markdown(handoff_msg)
            
            st.session_state.chat_history.append({"role": "assistant", "content": handoff_msg})

        # Check for small-talk
        st_class = classify_smalltalk(user_input)
        assistant_response = None

        if st_class == "gratitude":
            assistant_response = f"You're very welcome, {first_name}. If anything else comes up, I'm here."
            log.info("smalltalk.gratitude reply=%s", assistant_response)
        elif st_class == "goodbye":
            assistant_response = "Wishing you a smooth recovery. Take care!"
            log.info("smalltalk.goodbye")
        elif st_class == "affirm":
            assistant_response = "Noted. Is there anything else you'd like to discuss?"
            log.info("smalltalk.affirm reply=%s", assistant_response)

        # If not small-talk, compose and answer
        if not assistant_response:
            with st.spinner("Composing optimal query..."):
                composed = compose_query(
                    user_input,
                    patient,
                    DEFAULTS["compose_llm"],
                    chat_history=st.session_state.chat_history,
                    llm_fn=st.session_state.llm_fn,
                )
                log.info("composer.query msg=%s", composed)

            with st.spinner("Finding relevant information..."):
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
                    chat_history=st.session_state.chat_history,
                    min_kb_hits=DEFAULTS["min_kb_hits"],
                    strict_threshold=DEFAULTS["strict_threshold"],
                    llm_fn=st.session_state.llm_fn,
                )
                log.info(
                    "answer.generated used_kb=%s used_web=%s best_distance=%s kb_refs=%d",
                    info.get("used_kb"),
                    info.get("used_web"),
                    info.get("best_distance"),
                    len(info.get("kb_refs") or []),
                )
                assistant_response = answer

            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Display sources
            display_sources(info)

            # Display retrieval debug
            display_retrieval_debug(info)
        else:
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        # Add assistant response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
