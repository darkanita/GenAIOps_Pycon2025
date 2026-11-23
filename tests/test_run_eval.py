# tests/test_run_eval.py
"""
Quality gate de la última evaluación (python app/run_eval.py):

1. Umbrales mínimos de los jueces de MLflow.
2. Sin regresiones contra la línea base (MLflow 3.18): si la evaluación se
   ejecutó con --baseline, ningún juez puede empeorar de forma estadísticamente
   significativa (compare_evaluations, prueba pareada por pregunta).
"""

import json
import os

import mlflow
import pytest

EVAL_EXPERIMENT = "rag_evaluation"
COMPARISON_PATH = ".eval_comparison.json"
ALPHA = float(os.getenv("GATE_ALPHA", 0.05))

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
