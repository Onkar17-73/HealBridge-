import argparse
import os
from typing import Optional

from clinicalDbAgent import fetch_by_name, load_env


def _call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 200) -> str:
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
    return resp.choices[0].message.content.strip()


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
        "Use the patient's first name. Acknowledge the discharge date and condition if available. Optionally mention 1–2 key meds naturally. "
        "Keep it warm, concise, non-alarming, and conversational (1–2 sentences total). "
        "Do NOT provide medical advice or multiple questions. The final character MUST be a question mark."
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


def run(name: str, table: str, schema: str, match: str, llm: str):
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


def main():
    parser = argparse.ArgumentParser(description="Receptionist Agent: greet patient and ask LLM-generated follow-up based on discharge record")
    parser.add_argument("--name", type=str, help="Patient's full name (if omitted, will prompt interactively)")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--table", default="patient_discharges")
    parser.add_argument("--match", choices=["exact", "ilike"], default="exact")
    parser.add_argument("--llm", default="llama-3.1-8b-instant", help="Groq model to use for follow-up questions")

    args = parser.parse_args()

    if not args.name:
        try:
            name = input("Hello! I'm your post-discharge care assistant. What's your name? ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye")
            return
    else:
        name = args.name.strip()

    if not name:
        print("I didn't catch your name. Please try again.")
        return

    run(name=name, table=args.table, schema=args.schema, match=args.match, llm=args.llm)


if __name__ == "__main__":
    main()
