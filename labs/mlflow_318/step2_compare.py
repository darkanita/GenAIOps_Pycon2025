# labs/mlflow_318/step2_compare.py
"""
Step 2 · Compare evaluations against a baseline (MLflow 3.18).
Paso 2 · Comparar evaluaciones contra una línea base.

A higher average can hide answers that got worse. This script:
  1. evaluates the baseline and the candidate on the same dataset,
  2. compares them with `mlflow.genai.compare_evaluations()` (paired tests,
     confidence intervals and p-values per scorer),
  3. lists the questions that got worse (regressed rows),
  4. optionally applies a quality gate (--gate) for CI.

Usage (no API key, simulated agent):
    python labs/mlflow_318/step2_compare.py                      # v1 (baseline) vs v2
    python labs/mlflow_318/step2_compare.py --candidate v3 --gate

With the repo's real RAG (needs OPENAI_API_KEY or GROQ_API_KEY):
    python labs/mlflow_318/step2_compare.py --agent rag \
        --baseline prompts:/hr_assistant/1 --candidate prompts:/hr_assistant/2 [--judges]
    (with APP_LANG=es the prompt is called asistente_rrhh)

Compare two runs that already exist (e.g. two app/run_eval.py runs):
    python labs/mlflow_318/step2_compare.py --baseline-run-id <ID> --candidate-run-id <ID>
"""

import argparse
import json
import os
import sys

from config import DATASET_PATH, EVAL_EXPERIMENT, LAB_DIR, setup

import mlflow
import pandas as pd

from app.i18n import t
from gate import GUARD_SCORERS, IMPROVE_SCORERS, quality_gate, regressed_rows
from scorers import GREATER_IS_BETTER, OFFLINE_SCORERS

LAST_COMPARISON = LAB_DIR / "last_comparison.json"


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {
                "expected_response": item["expected_response"],
                "key_facts": item["key_facts"],
                "should_abstain": item["should_abstain"],
            },
        }
        for item in raw
    ]


def build_predict_fn(agent: str, version: str):
    """Returns (predict_fn, app version name)."""
    if agent == "mock":
        from mock_agent import answer, configure

        configure(version, environment="evaluation")

        def predict_fn(question: str) -> str:
            return answer(question)

        return predict_fn, f"mock-assistant-{version}"

    # The repo's real RAG: version is a Prompt Registry URI
    from app.rag_pipeline import build_chain, load_vectorstore_from_disk
    from app.tracking import app_version_name, enable_tracing

    enable_tracing()
    k = int(os.getenv("RETRIEVER_K") or 4)
    chain = build_chain(load_vectorstore_from_disk(), prompt_version=version, k=k)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    def predict_fn(question: str) -> str:
        return chain.invoke({"question": question, "chat_history": []})["answer"]

    return predict_fn, app_version_name(prompt_uri, k)


def get_scorers(agent: str, judges: bool):
    scorers = list(OFFLINE_SCORERS)
    if judges:
        from mlflow.genai.scorers import Correctness, RetrievalGroundedness

        from app.llm_provider import get_judge_model_uri

        judge = get_judge_model_uri()
        scorers += [Correctness(model=judge)]
        if agent == "rag":
            scorers += [RetrievalGroundedness(model=judge)]
    return scorers


def evaluate(agent: str, version: str, scorers) -> str:
    predict_fn, app_version = build_predict_fn(agent, version)
    mlflow.set_active_model(name=app_version)
    with mlflow.start_run(run_name=f"{agent}|{version}") as run:
        mlflow.log_params({"agent": agent, "version": version, "dataset": DATASET_PATH.name})
        mlflow.genai.evaluate(data=load_dataset(), predict_fn=predict_fn, scorers=scorers)
    print(t("lab.evaluated", version=version, run_id=run.info.run_id))
    return run.info.run_id


def print_summary(result):
    cols = ["scorer", "status", "n_paired", "baseline_mean", "candidate_mean", "diff",
            "ci_low", "ci_high", "p_value", "method"]
    print("\n" + t("lab.summary_header"))
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(result.summary()[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["mock", "rag"], default="mock")
    parser.add_argument("--baseline", default=None, help="v1|v2|v3 (mock) or prompt URI (rag)")
    parser.add_argument("--candidate", default=None, help="v1|v2|v3 (mock) or prompt URI (rag)")
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--candidate-run-id")
    parser.add_argument("--judges", action="store_true",
                        help="add LLM judges (Correctness...) on top of the code-based scorers")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if the candidate does not pass the quality gate")
    args = parser.parse_args()

    setup(EVAL_EXPERIMENT)

    if args.baseline_run_id and args.candidate_run_id:
        baseline_id, candidate_id = args.baseline_run_id, args.candidate_run_id
    else:
        if args.agent == "mock":
            defaults = ("v1", "v2")
        else:
            from app.prompt_registry import PROMPT_NAME
            defaults = (f"prompts:/{PROMPT_NAME}/1", f"prompts:/{PROMPT_NAME}/2")
        baseline = args.baseline or defaults[0]
        candidate = args.candidate or defaults[1]
        scorers = get_scorers(args.agent, args.judges)
        print(t("lab.scorers", names=[s.name for s in scorers]))
        print(t("lab.evaluating", baseline=baseline, candidate=candidate))
        baseline_id = evaluate(args.agent, baseline, scorers)
        candidate_id = evaluate(args.agent, candidate, scorers)

    # ✨ New in MLflow 3.18
    result = mlflow.genai.compare_evaluations(
        candidate_run_id=candidate_id,
        baseline_run_id=baseline_id,
        greater_is_better=GREATER_IS_BETTER,
        confidence_level=args.confidence_level,
        log_results=True,           # stores the comparison on the candidate run
    )
    print_summary(result)

    print("\n" + t("lab.regressed_header"))
    rows = regressed_rows(result)
    if not rows:
        print(t("lab.none"))
    for row in rows:
        print(f"   • [{row['scorer']}] {row['baseline']:.2f} → {row['candidate']:.2f} | "
              f"{row['question']}")
        print(t("lab.traces", baseline=row["baseline_trace_id"], candidate=row["candidate_trace_id"]))

    LAST_COMPARISON.write_text(json.dumps(
        {"baseline_run_id": baseline_id, "candidate_run_id": candidate_id}, indent=2))
    print("\n" + t("lab.saved_ids", file=LAST_COMPARISON.name))
    print(t("lab.ui_compare", experiment=EVAL_EXPERIMENT))

    if args.gate:
        failures = quality_gate(result, improve=IMPROVE_SCORERS, guard=GUARD_SCORERS)
        if failures:
            print("\n" + t("lab.gate_fail"))
            for f in failures:
                print(f"   - {f}")
            sys.exit(1)
        print("\n" + t("lab.gate_pass", improve=IMPROVE_SCORERS, guard=GUARD_SCORERS))


if __name__ == "__main__":
    main()
