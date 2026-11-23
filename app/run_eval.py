# app/run_eval.py
"""
Evaluación del RAG con mlflow.genai.evaluate y jueces LLM (LLM-as-a-judge).

Cada pregunta del dataset genera una traza; los jueces la califican y dejan su
razonamiento (rationale) adjunto a la traza. El resultado agregado queda en un
run del experimento "rag_evaluation", enlazado a la versión de la app
(proveedor + modelo + prompt + k) y a la versión del prompt.

Uso:
    python app/run_eval.py
    PROMPT_VERSION=prompts:/asistente_rrhh@v2_resumido_directo python app/run_eval.py
    RETRIEVER_K=6 python app/run_eval.py
    EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py   # solo algunos jueces

Comparar contra una línea base (MLflow 3.18):
    PROMPT_VERSION=prompts:/asistente_rrhh@v2_resumido_directo \
        python app/run_eval.py --baseline prompts:/asistente_rrhh@production

    Evalúa la línea base y la candidata con el mismo dataset y los mismos jueces,
    y las compara pregunta por pregunta con mlflow.genai.compare_evaluations():
    diferencia, intervalo de confianza y p-value por juez, más la lista de
    preguntas que empeoraron. tests/test_run_eval.py usa esa comparación como
    quality gate (ninguna regresión significativa).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
)

from app.code_scorers import EVAL_CODE_SCORERS
from app.llm_provider import describe, get_judge_model_uri
from app.prompt_registry import DEFAULT_PROMPT
from app.rag_pipeline import build_chain, load_vectorstore_from_disk
from app.tracking import EVAL_EXPERIMENT, app_config, init_tracking

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)        # deben coincidir con el índice construido
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)
K = int(os.getenv("RETRIEVER_K") or 4)
DATASET_PATH = "tests/eval_dataset.json"
COMPARISON_PATH = ".eval_comparison.json"   # IDs de la última comparación (los usa el quality gate)

# Paralelismo de MLflow. Por defecto evalúa ~10 filas x 10 jueces a la vez, lo que
# supera los límites gratuitos de Groq (30 solicitudes y 8K tokens por minuto).
# Con Groq usamos un modo pausado; cualquier valor puesto en .env tiene prioridad.
if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "groq":
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")         # una pregunta a la vez
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")  # 2 jueces en paralelo
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0.2") # ~12 llamadas/min
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "8")         # más reintentos ante 429


def load_eval_dataset(path=DATASET_PATH):
    """Convierte {question, answer} al formato de MLflow: inputs + expectations."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_response": item["answer"]},
        }
        for item in raw
    ]


def get_scorers(judge_model: str):
    """
    Jueces predefinidos de MLflow + guías propias en lenguaje natural.
    👉 Desafío: agrega aquí tus propios scorers (ver README).
    Con EVAL_SCORERS=correctness,safety se ejecutan solo los indicados.
    """
    scorers = [
        Correctness(model=judge_model),            # ¿coincide con la respuesta esperada?
        RelevanceToQuery(model=judge_model),       # ¿responde a lo que se preguntó?
        RetrievalGroundedness(model=judge_model),  # ¿está sustentada en los documentos? (alucinación)
        RetrievalRelevance(model=judge_model),     # ¿los fragmentos recuperados son pertinentes?
        Safety(model=judge_model),                 # ¿contenido dañino u ofensivo?
        Guidelines(
            name="idioma_espanol",
            guidelines="La respuesta debe estar escrita en español.",
            model=judge_model,
        ),
        Guidelines(
            name="tono_profesional",
            guidelines=(
                "La respuesta debe tener un tono formal y profesional, apropiado "
                "para un asistente de Recursos Humanos, y no debe inventar políticas."
            ),
            model=judge_model,
        ),
        # Scorers de código: deterministas, sin LLM y sin costo (app/code_scorers.py)
        *EVAL_CODE_SCORERS,
    ]
    selected = [n.strip() for n in os.getenv("EVAL_SCORERS", "").split(",") if n.strip()]
    if selected:
        unknown = set(selected) - {sc.name for sc in scorers}
        if unknown:
            raise ValueError(f"EVAL_SCORERS desconocidos: {unknown}. "
                             f"Disponibles: {[sc.name for sc in scorers]}")
        scorers = [sc for sc in scorers if sc.name in selected]
    return scorers


def evaluate_version(prompt_version: str, scorers, judge_model: str, role: str | None = None):
    """Evalúa una versión del prompt y devuelve (run_id, results)."""
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=prompt_version, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]
    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    print(f"\n⚙️  App: {describe()} | prompt={prompt_uri} | k={K}" + (f" | {role}" if role else ""))

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    with mlflow.start_run(run_name=run_name, tags={"eval_role": role} if role else None) as run:
        mlflow.log_params({
            **app_config(prompt_uri, K),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": judge_model,
            "dataset": DATASET_PATH,
        })
        results = mlflow.genai.evaluate(
            data=load_eval_dataset(),
            predict_fn=predict_fn,
            scorers=scorers,
            model_id=model_id,
        )

    print("📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    print(f"🔎 Detalle por pregunta: mlflow ui → experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return run.info.run_id, results


def compare_with_baseline(baseline_run_id: str, candidate_run_id: str):
    """Compara pregunta por pregunta la candidata contra la línea base (MLflow 3.18)."""
    comparison = mlflow.genai.compare_evaluations(
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
        log_results=True,           # queda registrada en el run candidato
    )
    df = comparison.summary()
    print("\n📊 Candidata vs línea base (prueba pareada por pregunta):")
    for _, r in df.iterrows():
        if r["status"] != "ok":
            print(f"   {r['scorer']:28s} {r['status']} (n={r['n_paired']})")
            continue
        flecha = "⬆️ " if r["diff"] > 0 else ("⬇️ " if r["diff"] < 0 else "= ")
        print(f"   {r['scorer']:28s} {r['baseline_mean']:.2f} → {r['candidate_mean']:.2f} "
              f"{flecha} p={r['p_value']:.3f}")

    empeoraron = [(row, name, sc) for row in comparison.paired_rows
                  for name, sc in row["scores"].items() if sc["delta"] < 0]
    if empeoraron:
        print("\n🔻 Preguntas que empeoraron (el promedio no las muestra):")
        for row, name, sc in empeoraron:
            trace = mlflow.get_trace(row["candidate_trace_id"])
            question = (trace.data.spans[0].inputs or {}).get("question", "?")
            print(f"   • [{name}] {sc['baseline']} → {sc['candidate']} | {question}")

    with open(COMPARISON_PATH, "w") as f:
        json.dump({"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id}, f)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluación del RAG con jueces de MLflow")
    parser.add_argument("--baseline", metavar="PROMPT",
                        help="evalúa también esta versión y compara contra ella "
                             "(p. ej. prompts:/asistente_rrhh@production)")
    args = parser.parse_args()

    mlflow.set_experiment(EVAL_EXPERIMENT)
    judge_model = get_judge_model_uri()
    scorers = get_scorers(judge_model)
    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚖️  Juez: {judge_model} | {len(scorers)} scorers: {[sc.name for sc in scorers]}")
    if os.getenv("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "1":
        print("🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)")

    if not args.baseline:
        _, results = evaluate_version(PROMPT_VERSION, scorers, judge_model)
        return results

    baseline_id, _ = evaluate_version(args.baseline, scorers, judge_model, role="baseline")
    candidate_id, results = evaluate_version(PROMPT_VERSION, scorers, judge_model, role="candidate")
    compare_with_baseline(baseline_id, candidate_id)
    return results


if __name__ == "__main__":
    main()
