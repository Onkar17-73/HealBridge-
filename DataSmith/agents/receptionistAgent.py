import os
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import asdict

from agents.clinicalDbAgent import fetch_by_name, load_env
from agents.clinicalAiAgent import generate_answer


def _call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 200) -> str:
    log = logging.getLogger("datasmith.receptionist")
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
            {"role": "system", "content": (
                "You are a compassionate clinical receptionist assisting a recently discharged patient. "
                "Ask only brief, relevant follow-up question(s) based on the provided discharge context. "
                "Do not provide medical advice. Keep it friendly, concise, and non-alarming. "
                "The final output MUST end with a single, clear question phrased like a clinician check-in "
                "(e.g., 'How are you feeling today?')."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    log.debug("groq.receptionist len=%d", len(text))
    return text


def make_opening_message_prompt(name: str, discharge_date: Optional[str], primary_dx: Optional[str], meds: Optional[list], diet: Optional[str], warning_signs: Optional[str], instructions: Optional[str]) -> str:
    # Provide a compact, structured context to the LLM to craft a single natural greeting + check-in.
    meds_list = ", ".join(meds[:5]) if isinstance(meds, list) else (meds or "")
    context = (
        f"Patient: {name}\n"
        f"Discharge date: {discharge_date or 'unknown'}\n"
        f"Primary diagnosis: {primary_dx or 'unknown'}\n"
        f"Medications: {meds_list or 'not listed'}\n"
        f"Dietary restrictions: {diet or 'not specified'}\n"
        f"Warning signs to monitor: {warning_signs or 'not specified'}\n"
        f"Discharge instructions: {instructions or 'not specified'}\n"
    )

    instructions_txt = (
        "Write one short, natural message that combines the greeting and the check-in, and ends with exactly ONE question. "
        "Use the patient's first name. The patient has ALREADY BEEN DISCHARGED; if a date is present, refer to it as a past event (e.g., 'since your discharge on <date>'). "
        "Do NOT say 'scheduled', 'upcoming', or future-tense discharge. Optionally mention 1–2 key meds naturally. "
        "Keep it warm, concise, non-alarming, and conversational (1–2 sentences total). "
        "Do NOT provide medical advice or multiple questions. Do NOT invent dates or appointments. The final character MUST be a question mark."
    )

    return context + "\n" + instructions_txt


def ensure_final_question(text: str, fallback_question: str) -> str:
    t = text.strip()
    if not t:
        return fallback_question if fallback_question.endswith("?") else (fallback_question + "?")
    # If it already ends with a question mark, accept as-is
    if t.endswith("?"):
        return t
    # Otherwise, append a fallback question to guarantee a question ending
    sep = " " if not t.endswith(('.', '!', '?')) else " "
    fq = fallback_question if fallback_question.endswith("?") else (fallback_question + "?")
    return t + sep + fq


def generate_opening_message(patient: Dict[str, Any], llm: str = "llama-3.1-8b-instant", llm_fn: Optional[Callable[[str], str]] = None) -> str:
    log = logging.getLogger("datasmith.receptionist")
    name = patient.get("patient_name") or "there"
    first_name = name.split()[0]
    try:
        prompt = make_opening_message_prompt(
            name=name,
            discharge_date=patient.get("discharge_date"),
            primary_dx=patient.get("primary_diagnosis"),
            meds=patient.get("medications"),
            diet=patient.get("dietary_restrictions"),
            warning_signs=patient.get("warning_signs"),
            instructions=patient.get("discharge_instructions"),
        )
        message = llm_fn(prompt) if llm_fn is not None else _call_groq(prompt, model=llm)
        message = ensure_final_question(message, "How are you feeling today?")
        log.info("greeting.generated name=%s output=%s", first_name, message)
    except Exception:
        date_bit = f" from {patient.get('discharge_date')}" if patient.get("discharge_date") else ""
        dx_bit = f" for {patient.get('primary_diagnosis')}" if patient.get("primary_diagnosis") else ""
        message = f"Hi {first_name}! I see your discharge{date_bit}{dx_bit}. How have you been feeling?"
    message = message.strip().strip('"').strip("'")
    if not message.endswith("?"):
        message = message.rstrip(". ") + "?"
    low = message.lower()
    if "scheduled to be discharged" in low or "scheduled for discharge" in low or "will be discharged" in low:
        message = message.replace("scheduled to be discharged", "were discharged").replace("scheduled for discharge", "were discharged").replace("will be discharged", "were discharged")
    return message


def generate_handoff_line(first_name: str, user_question: str, llm: str = "llama-3.1-8b-instant", llm_fn: Optional[Callable[[str], str]] = None) -> str:
    log = logging.getLogger("datasmith.receptionist")
    prompt = (
        "You are a compassionate clinic receptionist. "
        "Write ONE short handoff line acknowledging the patient's concern and stating you will connect them to the clinical AI assistant. Requirements:\n"
        "- 1 sentence, <= 25 words\n"
        "- Warm, concise, non-alarming\n"
        "- No medical advice\n"
        "- Do NOT ask a question\n"
        f"- Address the patient by first name: {first_name}\n\n"
        f"Patient concern: {user_question}"
    )
    try:
        text = llm_fn(prompt) if llm_fn is not None else _call_groq(prompt, model=llm, temperature=0.2, max_tokens=60)
    except Exception:
        text = (
            f"Thanks for sharing, {first_name}. I’ll connect you with our clinical AI assistant to help based on your records and our knowledge base."
        )
    text = text.strip().strip('"').strip("'")
    if text.endswith("?"):
        text = text.rstrip("?") + "."
    log.info("handoff.generated name=%s len=%d", first_name, len(text))
    return text


def classify_smalltalk(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    gratitude = {"thanks", "thank you", "thank u", "ty", "thx", "much appreciated"}
    goodbye = {"bye", "goodbye", "see ya", "see you", "take care"}
    affirm = {"ok", "okay", "alright", "got it", "noted"}
    if any(kw in t for kw in gratitude):
        return "gratitude"
    if any(t == kw for kw in goodbye):
        return "goodbye"
    if t in affirm:
        return "affirm"
    return None


def run(
    name: str,
    table: str,
    schema: str,
    match: str,
    llm: str,
    persist_dir: str,
    collection: str,
    embed_model: str,
    top_k: int,
    threshold: float,
    web_max: int,
):
    load_env()
    results = fetch_by_name(name=name, table=table, schema=schema, limit=1, match=match)
    if not results:
        print("Thanks! I couldn't find your discharge report. Could you confirm your full name as it appears in our records?")
        return

    row = results[0]

    try:
        message = _call_groq(
            make_opening_message_prompt(
                name=row.patient_name,
                discharge_date=row.discharge_date,
                primary_dx=row.primary_diagnosis,
                meds=row.medications,
                diet=row.dietary_restrictions,
                warning_signs=row.warning_signs,
                instructions=row.discharge_instructions,
            ),
            model=llm,
        )
        # Enforce that the final output ends with a clinician-style question
        fallback_q = "How are you feeling today?"
        message = ensure_final_question(message, fallback_q)
    except Exception:
        # Graceful fallback to a generic question if LLM is unavailable
        first_name = (row.patient_name or "there").split()[0]
        date_bit = f" from {row.discharge_date}" if row.discharge_date else ""
        dx_bit = f" for {row.primary_diagnosis}" if row.primary_diagnosis else ""
        meds_hint = ", ".join((row.medications or [])[:2]) if row.medications else None
        meds_bit = f" — are you able to keep up with {meds_hint}" if meds_hint else ""
        message = f"Hi {first_name}! I see your discharge{date_bit}{dx_bit}. How have you been feeling{meds_bit}?"

    print(message)

    # Capture patient's concern and route to clinical AI agent with RAG + patient context
    try:
        user_query = input("\nPlease tell me what you'd like help with: ").strip()
    except KeyboardInterrupt:
        print("\nGoodbye")
        return
    if not user_query:
        print("Thanks! If you have a question later, just let me know.")
        return

    patient_info = asdict(row)
    try:
        answer, _info = generate_answer(
            query=user_query,
            persist_dir=persist_dir,
            collection=collection,
            emb_model=embed_model,
            llm=llm,
            top_k=top_k,
            threshold=threshold,
            web_max=web_max,
            patient_info=patient_info,
        )
    except Exception as e:
        print(f"\nSorry, I couldn't process that right now: {e}")
        return

    print("\nAssistant:\n")
    print(answer)


"""
Receptionist helper functions. No CLI entrypoint; use app/main.py.
"""
