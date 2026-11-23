# labs/mlflow_318/test_gate.py
"""
Quality gate in pytest: the candidate from the last `step2_compare.py` run must
improve significantly on the baseline and must not make any question worse.

    python labs/mlflow_318/step2_compare.py --candidate v3
    pytest labs/mlflow_318/test_gate.py -v
"""

import json

import mlflow
import pytest

from config import EVAL_EXPERIMENT, LAB_DIR, setup
from gate import GUARD_SCORERS, IMPROVE_SCORERS, quality_gate, regressed_rows
from scorers import GREATER_IS_BETTER

LAST_COMPARISON = LAB_DIR / "last_comparison.json"


@pytest.fixture(scope="module")
def comparison():
    if not LAST_COMPARISON.exists():
        pytest.skip("Run first: python labs/mlflow_318/step2_compare.py")
    ids = json.loads(LAST_COMPARISON.read_text())
    setup(EVAL_EXPERIMENT)
    return mlflow.genai.compare_evaluations(
        candidate_run_id=ids["candidate_run_id"],
        baseline_run_id=ids["baseline_run_id"],
        greater_is_better=GREATER_IS_BETTER,
    )


@pytest.mark.parametrize("scorer_name", IMPROVE_SCORERS)
def test_significant_improvement(comparison, scorer_name):
    comparison.assert_improved(scorers=[scorer_name])


@pytest.mark.parametrize("scorer_name", GUARD_SCORERS)
def test_no_regressions(comparison, scorer_name):
    rows = regressed_rows(comparison, scorers=[scorer_name])
    assert not rows, "\n".join(
        f"{r['baseline']:.2f} → {r['candidate']:.2f} | {r['question']}" for r in rows)


def test_full_gate(comparison):
    assert quality_gate(comparison) == []
