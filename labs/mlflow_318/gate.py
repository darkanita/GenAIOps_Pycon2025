# labs/mlflow_318/gate.py
"""
Quality gate sobre el resultado de `mlflow.genai.compare_evaluations()`.

Dos reglas, porque un promedio no basta:
  1. MEJORA: los scorers objetivo deben mejorar de forma estadísticamente
     significativa (`ComparisonResult.assert_improved`).
  2. SIN REGRESIONES: en los scorers "guardianes" ninguna pregunta puede empeorar
     más que la tolerancia, aunque el promedio suba (`ComparisonResult.paired_rows`).
"""

import os

import mlflow

IMPROVE_SCORERS = [s for s in os.getenv("GATE_IMPROVE", "cita_fuente").split(",") if s]
GUARD_SCORERS = [s for s in os.getenv(
    "GATE_GUARD", "cubre_palabras_clave,abstencion_correcta").split(",") if s]
TOLERANCE = float(os.getenv("GATE_TOLERANCE", "0.0"))
ALPHA = float(os.getenv("GATE_ALPHA", "0.05"))


def _question(trace_id: str) -> str:
    try:
        inputs = mlflow.get_trace(trace_id).data.spans[0].inputs or {}
        return str(inputs.get("question") or inputs.get("pregunta") or inputs)
    except Exception:
        return "?"


def regressed_rows(result, scorers=None, tolerance: float = TOLERANCE) -> list[dict]:
    """Filas (preguntas) en las que la candidata quedó peor que la línea base."""
    rows = []
    for row in result.paired_rows:
        for name, score in row["scores"].items():
            if scorers and name not in scorers:
                continue
            comparison = result.scorers[name]
            direction = 1 if comparison.greater_is_better else -1
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
    """Devuelve la lista de fallas (vacía si la candidata pasa)."""
    failures = []
    if improve:
        try:
            result.assert_improved(scorers=improve, alpha=alpha)   # ✨ MLflow 3.18
        except AssertionError as e:
            failures.append(str(e).strip())
    for row in regressed_rows(result, scorers=guard, tolerance=tolerance):
        failures.append(f"regresión en '{row['scorer']}': {row['baseline']:.2f} → "
                        f"{row['candidate']:.2f} | {row['question']}")
    return failures
