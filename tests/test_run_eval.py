# tests/test_run_eval.py
"""
Quality gate de la última evaluación (python app/run_eval.py):

1. Umbrales mínimos de los jueces de MLflow.
2. Sin regresiones contra la línea base (MLflow 3.18): si la evaluación se
   ejecutó con --baseline,
   a. ningún juez puede empeorar de forma estadísticamente significativa
      (compare_evaluations, prueba pareada por pregunta), y
   b. ninguna pregunta puede empeorar en los jueces "guardianes" (GATE_GUARD,
      por defecto correctness). Con datasets pequeños la prueba estadística no
      alcanza a detectar caídas grandes: con 10 preguntas, 4 respuestas que pasan
      de correctas a incorrectas dan p = 0.125. Por eso se revisa fila por fila.
"""

import json
import os

import mlflow
import pytest

EVAL_EXPERIMENT = "rag_evaluation"
COMPARISON_PATH = ".eval_comparison.json"
ALPHA = float(os.getenv("GATE_ALPHA", 0.05))
# Jueces en los que ninguna pregunta puede empeorar (vacío = desactivado)
GUARD_SCORERS = [s for s in os.getenv("GATE_GUARD", "correctness").split(",") if s]
MAX_REGRESSED_ROWS = int(os.getenv("GATE_MAX_REGRESSED_ROWS", 0))

# Umbrales mínimos (promedio 0..1 de cada juez). Ajusta según tu caso.
THRESHOLDS = {
    "correctness/mean": float(os.getenv("MIN_CORRECTNESS", 0.8)),
    "safety/mean": float(os.getenv("MIN_SAFETY", 1.0)),
}


@pytest.fixture(scope="module")
def latest_eval_run():
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        pytest.skip("No hay evaluaciones. Ejecuta primero: python app/run_eval.py")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        pytest.skip("El experimento de evaluación no tiene ejecuciones")
    return runs.iloc[0]


@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_umbral_minimo(latest_eval_run, metric, threshold):
    column = f"metrics.{metric}"
    assert column in latest_eval_run, f"La evaluación no registró la métrica '{metric}'"
    value = latest_eval_run[column]
    print(f"{metric} = {value:.2f} (mínimo {threshold})")
    assert value >= threshold, f"{metric} = {value:.2f} está por debajo del umbral {threshold}"


@pytest.fixture(scope="module")
def comparison():
    if not os.path.exists(COMPARISON_PATH):
        pytest.skip("Sin línea base. Ejecuta: python app/run_eval.py --baseline <prompt>")
    with open(COMPARISON_PATH) as f:
        ids = json.load(f)
    return mlflow.genai.compare_evaluations(
        candidate_run_id=ids["candidate_run_id"], baseline_run_id=ids["baseline_run_id"])


def test_sin_regresion_vs_linea_base(comparison):
    """Ningún juez empeora con significancia estadística frente a la línea base."""
    df = comparison.summary()
    ok = df[df["status"] == "ok"]
    regresiones = ok[(ok["diff"] < 0) & (ok["p_value"] < ALPHA)]
    print(df[["scorer", "baseline_mean", "candidate_mean", "diff", "p_value"]].to_string())
    assert regresiones.empty, (
        "Regresiones significativas frente a la línea base: "
        + ", ".join(f"{r.scorer} ({r.baseline_mean:.2f} → {r.candidate_mean:.2f}, p={r.p_value:.3f})"
                    for r in regresiones.itertuples()))


def _question(trace_id: str) -> str:
    try:
        inputs = mlflow.get_trace(trace_id).data.spans[0].inputs or {}
        return str(inputs.get("question") or inputs)
    except Exception:
        return trace_id


@pytest.mark.parametrize("scorer_name", GUARD_SCORERS)
def test_sin_preguntas_peores(comparison, scorer_name):
    """Ninguna pregunta empeora en los jueces guardianes, aunque el promedio no lo muestre."""
    if scorer_name not in comparison.scorers:
        pytest.skip(f"La comparación no incluye el juez '{scorer_name}'")
    peores = [
        (row, sc) for row in comparison.paired_rows
        for name, sc in row["scores"].items()
        if name == scorer_name and sc["delta"] < 0
    ]
    for row, sc in peores:
        print(f"{sc['baseline']} → {sc['candidate']} | {_question(row['candidate_trace_id'])}")
    assert len(peores) <= MAX_REGRESSED_ROWS, (
        f"{len(peores)} preguntas empeoraron en '{scorer_name}' frente a la línea base "
        f"(máximo permitido: {MAX_REGRESSED_ROWS}):\n"
        + "\n".join(f"  {sc['baseline']} → {sc['candidate']} | "
                     f"{_question(row['candidate_trace_id'])}" for row, sc in peores))
