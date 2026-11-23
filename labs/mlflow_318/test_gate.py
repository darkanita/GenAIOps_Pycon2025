# labs/mlflow_318/test_gate.py
"""
Quality gate en pytest: la candidata del último `paso2_comparar.py` debe
mejorar significativamente a la línea base y no empeorar ninguna pregunta.

    python labs/mlflow_318/paso2_comparar.py --candidate v3
    pytest labs/mlflow_318/test_gate.py -v
"""

import json

import mlflow
import pytest

from config import EVAL_EXPERIMENT, LAB_DIR, setup
from gate import GUARD_SCORERS, IMPROVE_SCORERS, quality_gate, regressed_rows
from scorers import GREATER_IS_BETTER

LAST_COMPARISON = LAB_DIR / "ultima_comparacion.json"


@pytest.fixture(scope="module")
def comparison():
    if not LAST_COMPARISON.exists():
        pytest.skip("Ejecuta primero: python labs/mlflow_318/paso2_comparar.py")
    ids = json.loads(LAST_COMPARISON.read_text())
    setup(EVAL_EXPERIMENT)
    return mlflow.genai.compare_evaluations(
        candidate_run_id=ids["candidate_run_id"],
        baseline_run_id=ids["baseline_run_id"],
        greater_is_better=GREATER_IS_BETTER,
    )


@pytest.mark.parametrize("scorer_name", IMPROVE_SCORERS)
def test_mejora_significativa(comparison, scorer_name):
    comparison.assert_improved(scorers=[scorer_name])


@pytest.mark.parametrize("scorer_name", GUARD_SCORERS)
def test_sin_regresiones(comparison, scorer_name):
    rows = regressed_rows(comparison, scorers=[scorer_name])
    assert not rows, "\n".join(
        f"{r['baseline']:.2f} → {r['candidate']:.2f} | {r['question']}" for r in rows)


def test_gate_completo(comparison):
    assert quality_gate(comparison) == []
