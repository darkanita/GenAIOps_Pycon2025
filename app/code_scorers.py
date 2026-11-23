# app/code_scorers.py
"""
Code-based scorers for the HR assistant: deterministic, no LLM, no cost.
Scorers basados en código: deterministas, sin LLM y sin costo.

Used in / Se usan en:
- app/run_eval.py      → offline evaluation, next to the LLM judges
- app/monitoring.py    → online evaluation of chat traces (MLflow 3.18)

MLflow 3.18 rule for online evaluation: the server runs each function's
source code in an isolated container with no network, so every scorer does
its imports inside its body and uses no module-level variables. They work
for answers in Spanish and in English.
"""

from mlflow.genai.scorers import scorer


@scorer
def concise_answer(outputs) -> bool:
    """The answer has at most 150 words. / La respuesta tiene como máximo 150 palabras."""
    return len(str(outputs).split()) <= 150


@scorer
def no_personal_data(outputs) -> bool:
    """No email addresses or phone numbers in the answer. / Sin correos ni teléfonos."""
    import re

    return not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d", str(outputs))


@scorer
def no_errors(trace) -> bool:
    """No step of the trace (retriever, LLM, chain) ended in error. / Ningún paso terminó en error."""
    return all("ERROR" not in str(span.status.status_code) for span in trace.data.spans)


@scorer
def conversation_without_abstentions(session):
    """The assistant never had to say it lacked information. / Nunca dijo que no tenía información."""
    from mlflow.entities import Feedback

    marks = ("no tengo información", "no tengo suficiente información",
             "do not have enough information", "don't have enough information")
    unanswered = [tr.info.trace_id for tr in session
                  if any(m in str(tr.data.response or "").lower() for m in marks)]
    return Feedback(
        value=not unanswered,
        rationale=(f"{len(unanswered)} of {len(session)} turns unanswered: {unanswered}"
                   if unanswered else f"{len(session)} turns answered"),
    )


# Offline (run_eval.py): single traces
EVAL_CODE_SCORERS = [concise_answer, no_personal_data, no_errors]
# Online (monitoring.py): traces + one score per conversation
ONLINE_CODE_SCORERS = EVAL_CODE_SCORERS + [conversation_without_abstentions]
