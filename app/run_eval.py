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
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    """
    return [
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
    ]


def main():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    judge_model = get_judge_model_uri()

    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚙️  App: {describe()} | prompt={prompt_uri} | k={K}")
    print(f"⚖️  Juez: {judge_model}")

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    with mlflow.start_run(run_name=run_name):
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
            scorers=get_scorers(judge_model),
            model_id=model_id,
        )

    print("\n📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    print(f"\n🔎 Detalle por pregunta (trazas + razonamiento de cada juez): mlflow ui → "
          f"experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return results


if __name__ == "__main__":
    main()
