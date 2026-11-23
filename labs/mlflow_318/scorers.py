# labs/mlflow_318/scorers.py
"""
Code-based scorers (no LLM, no cost, deterministic). They work for answers in
Spanish and in English.

⚠️  MLflow 3.18 rule for online evaluation: the server runs each function's
*source code* in an isolated container with no network. That is why every
scorer does its `import`s inside its body and uses no module-level variables.

    ONLINE_SCORERS   registered and turned on with .start() for new traces
                     (they cannot take `expectations`)
    OFFLINE_SCORERS  used in mlflow.genai.evaluate (they can take `expectations`)
"""

from mlflow.genai.scorers import scorer


# ---------------------------------------------------------------------------
# Quality scorers that work online and offline
# ---------------------------------------------------------------------------
@scorer
def concise_answer(outputs) -> bool:
    """The answer has at most 100 words."""
    return len(str(outputs).split()) <= 100


@scorer
def cites_source(outputs) -> bool:
    """The answer names the document it comes from ('Source: PerksPlus.pdf, p. 3' / 'Fuente: ...')."""
    import re

    text = str(outputs)
    abstains = any(m in text.lower() for m in ("no tengo información", "do not have enough information"))
    return abstains or bool(re.search(r"(Fuente|Source):\s*\S+\.pdf", text))


@scorer
def no_personal_data(outputs) -> bool:
    """The answer does not echo email addresses or phone numbers."""
    import re

    return not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d", str(outputs))


@scorer
def no_tool_errors(trace) -> bool:
    """No step of the trace (retriever, tool, LLM) ended in error."""
    return all("ERROR" not in str(span.status.status_code) for span in trace.data.spans)


@scorer
def answer_length(outputs) -> int:
    """Number of words (lower is better: use it with greater_is_better=False)."""
    return len(str(outputs).split())


# ---------------------------------------------------------------------------
# Session-level scorer: judges the whole conversation (all of its traces)
# ---------------------------------------------------------------------------
@scorer
def session_without_failures(session):
    """The conversation had no failing steps and no apology answers."""
    from mlflow.entities import Feedback

    failed = []
    for tr in session:
        error = any("ERROR" in str(s.status.status_code) for s in tr.data.spans)
        response = str(tr.data.response or "").lower()
        if error or "lo siento" in response or "sorry" in response:
            failed.append(tr.info.trace_id)
    return Feedback(
        value=not failed,
        rationale=(f"{len(session)} turns; with failures: {failed}" if failed
                   else f"{len(session)} turns without failures"),
    )


# ---------------------------------------------------------------------------
# Scorers that compare against the expected answer (offline only)
# ---------------------------------------------------------------------------
@scorer
def covers_key_facts(outputs, expectations) -> float:
    """Fraction of the expected key facts that appear in the answer."""
    import re
    import unicodedata

    def norm(text):
        text = unicodedata.normalize("NFKD", str(text).lower())
        return "".join(c for c in text if not unicodedata.combining(c))

    facts = expectations.get("key_facts") or []
    if not facts:
        return 1.0
    text = norm(outputs)
    found = [f for f in facts if re.search(rf"\b{re.escape(norm(f))}", text)]
    return len(found) / len(facts)


@scorer
def correct_abstention(outputs, expectations) -> bool:
    """Abstains when the question is out of scope, and answers when it is not."""
    text = str(outputs).lower()
    abstains = any(m in text for m in ("no tengo información", "do not have enough information"))
    return abstains == bool(expectations.get("should_abstain"))


ONLINE_SCORERS = [concise_answer, cites_source, no_personal_data,
                  no_tool_errors, session_without_failures]
OFFLINE_SCORERS = [concise_answer, cites_source, no_personal_data,
                   no_tool_errors, answer_length,
                   covers_key_facts, correct_abstention]

# For compare_evaluations: a LOWER value is better for these scorers
GREATER_IS_BETTER = {"answer_length": False}
