# labs/mlflow_318/gate.py
"""
Quality gate on the result of `mlflow.genai.compare_evaluations()`.

Two rules, because an average is not enough:
  1. IMPROVE: the target scorers must improve with statistical significance
     (`ComparisonResult.assert_improved`).
  2. NO REGRESSIONS: on the "guard" scorers no question may get worse by more
     than the tolerance, even if the average goes up (`ComparisonResult.paired_rows`).
"""

import os

import mlflow

from app.i18n import t

IMPROVE_SCORERS = [s for s in os.getenv("GATE_IMPROVE", "cites_source").split(",") if s]
GUARD_SCORERS = [s for s in os.getenv(
    "GATE_GUARD", "covers_key_facts,correct_abstention").split(",") if s]
TOLERANCE = float(os.getenv("GATE_TOLERANCE", "0.0"))
ALPHA = float(os.getenv("GATE_ALPHA", "0.05"))


def _question(trace_id: str) -> str:
    try:
        inputs = mlflow.get_trace(trace_id).data.spans[0].inputs or {}
        return str(inputs.get("question") or inputs)
    except Exception:
        return "?"


def regressed_rows(result, scorers=None, tolerance: float = TOLERANCE) -> list[dict]:
    """Rows (questions) where the candidate did worse than the baseline."""
    rows = []
    for row in result.paired_rows:
        for name, score in row["scores"].items():
            if scorers and name not in scorers:
                continue
            direction = 1 if result.scorers[name].greater_is_better else -1
            if direction * score["delta"] < -tolerance:
                rows.append({
                    "scorer": name,
                    "baseline": score["baseline"],
                    "candidate": score["candidate"],
                    "baseline_trace_id": row["baseline_trace_id"],
                    "candidate_trace_id": row["candidate_trace_id"],
                    "question": _question(row["candidate_trace_id"]),
                })
    return rows


def quality_gate(result, improve=IMPROVE_SCORERS, guard=GUARD_SCORERS,
                 alpha: float = ALPHA, tolerance: float = TOLERANCE) -> list[str]:
    """Returns the list of failures (empty if the candidate passes)."""
    failures = []
    if improve:
        try:
            result.assert_improved(scorers=improve, alpha=alpha)   # ✨ MLflow 3.18
        except AssertionError as e:
            failures.append(str(e).strip())
    for row in regressed_rows(result, scorers=guard, tolerance=tolerance):
        failures.append(t("lab.gate_regression", scorer=row["scorer"], baseline=row["baseline"],
                          candidate=row["candidate"], question=row["question"]))
    return failures
