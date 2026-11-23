# labs/mlflow_318/paso2_comparar.py
"""
Paso 2 · Comparar evaluaciones contra una línea base (MLflow 3.18).

Un promedio más alto puede esconder respuestas que empeoraron. Este script:
  1. evalúa la línea base y la candidata sobre el mismo dataset,
  2. las compara con `mlflow.genai.compare_evaluations()` (pruebas pareadas,
     intervalos de confianza y p-values por scorer),
  3. lista las preguntas que empeoraron (filas en regresión),
  4. opcionalmente aplica un quality gate (--gate) para CI.

Uso (sin API key, con el agente simulado):
    python labs/mlflow_318/paso2_comparar.py                      # v1 (base) vs v2
    python labs/mlflow_318/paso2_comparar.py --candidate v3 --gate

Con el RAG real del repositorio (necesita OPENAI_API_KEY o GROQ_API_KEY):
    python labs/mlflow_318/paso2_comparar.py --agent rag \
        --baseline prompts:/asistente_rrhh/1 --candidate prompts:/asistente_rrhh/2 [--judges]

Comparar dos runs que ya existen (p. ej. dos ejecuciones de app/run_eval.py):
    python labs/mlflow_318/paso2_comparar.py --baseline-run-id <ID> --candidate-run-id <ID>
"""

import argparse
import json
import os
import sys

from config import DATASET_PATH, EVAL_EXPERIMENT, LAB_DIR, setup

import mlflow

from gate import GUARD_SCORERS, IMPROVE_SCORERS, quality_gate, regressed_rows
from scorers import GREATER_IS_BETTER, OFFLINE_SCORERS

LAST_COMPARISON = LAB_DIR / "ultima_comparacion.json"


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {
                "expected_response": item["expected_response"],
                "palabras_clave": item["palabras_clave"],
                "debe_abstenerse": item["debe_abstenerse"],
            },
        }
        for item in raw
    ]


def build_predict_fn(agent: str, version: str):
    """Devuelve (predict_fn, nombre de la versión de la app)."""
    if agent == "mock":
        from mock_agent import configurar, responder

        configurar(version, environment="evaluation")

        def predict_fn(question: str) -> str:
            return responder(question)

        return predict_fn, f"asistente-mock-{version}"

    # RAG real del repositorio: version es una URI del Prompt Registry
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
    print(f"   ✅ {version}: run {run.info.run_id}")
    return run.info.run_id


def print_summary(result):
    df = result.summary()
    cols = ["scorer", "status", "n_paired", "baseline_mean", "candidate_mean", "diff",
            "ci_low", "ci_high", "p_value", "method"]
    print("\n📊 Comparación pareada (candidata - línea base):")
    with __import__("pandas").option_context("display.width", 160, "display.max_columns", 20,
                                             "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["mock", "rag"], default="mock")
    parser.add_argument("--baseline", default=None, help="v1|v2|v3 (mock) o URI de prompt (rag)")
    parser.add_argument("--candidate", default=None, help="v1|v2|v3 (mock) o URI de prompt (rag)")
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--candidate-run-id")
    parser.add_argument("--judges", action="store_true",
                        help="agrega jueces LLM (Correctness...) además de los scorers de código")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--gate", action="store_true",
                        help="falla (exit 1) si la candidata no pasa el quality gate")
    args = parser.parse_args()

    setup(EVAL_EXPERIMENT)

    if args.baseline_run_id and args.candidate_run_id:
        baseline_id, candidate_id = args.baseline_run_id, args.candidate_run_id
    else:
        defaults = (("v1", "v2") if args.agent == "mock"
                    else ("prompts:/asistente_rrhh/1", "prompts:/asistente_rrhh/2"))
        baseline = args.baseline or defaults[0]
        candidate = args.candidate or defaults[1]
        scorers = get_scorers(args.agent, args.judges)
        print(f"⚖️  Scorers: {[s.name for s in scorers]}")
        print(f"🧪 Evaluando línea base ({baseline}) y candidata ({candidate})...")
        baseline_id = evaluate(args.agent, baseline, scorers)
        candidate_id = evaluate(args.agent, candidate, scorers)

    # ✨ Nuevo en MLflow 3.18
    result = mlflow.genai.compare_evaluations(
        candidate_run_id=candidate_id,
        baseline_run_id=baseline_id,
        greater_is_better=GREATER_IS_BETTER,
        confidence_level=args.confidence_level,
        log_results=True,           # guarda la comparación en el run candidato
    )
    print_summary(result)

    print("\n🔻 Preguntas que empeoraron (lo que el promedio no muestra):")
    rows = regressed_rows(result)
    if not rows:
        print("   ninguna 🎉")
    for row in rows:
        print(f"   • [{row['scorer']}] {row['baseline']:.2f} → {row['candidate']:.2f} | "
              f"{row['question']}")
        print(f"     trazas: base={row['baseline_trace_id']} candidata={row['candidate_trace_id']}")

    LAST_COMPARISON.write_text(json.dumps(
        {"baseline_run_id": baseline_id, "candidate_run_id": candidate_id}, indent=2))
    print(f"\n💾 IDs guardados en {LAST_COMPARISON.name} (los usa test_gate.py)")
    print("👀 En la UI: Experiments → lab318_evaluaciones → Evaluation runs → selecciona ambos "
          "runs → Compare (fija la línea base y filtra por 'regressed').")

    if args.gate:
        failures = quality_gate(result, improve=IMPROVE_SCORERS, guard=GUARD_SCORERS)
        if failures:
            print("\n🚫 Quality gate: NO pasa")
            for f in failures:
                print(f"   - {f}")
            sys.exit(1)
        print(f"\n✅ Quality gate: pasa (mejora significativa en {IMPROVE_SCORERS}, "
              f"sin regresiones en {GUARD_SCORERS})")


if __name__ == "__main__":
    main()
