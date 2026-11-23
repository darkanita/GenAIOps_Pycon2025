# labs/mlflow_318/step1_traffic.py
"""
Step 1 · Simulate "production" traffic so there are traces to analyze.
Paso 1 · Simular tráfico de "producción" para tener trazas que analizar.

Generates multi-turn conversations (sessions) against the assistant and stores
them in the `lab318_production` experiment, tagged `environment = production`.
These traces are what the following steps work on:
  - Automatic Issue Detection (Issues tab in the UI)   → step 4
  - online evaluation with code-based scorers          → step 3

Usage:
    python labs/mlflow_318/step1_traffic.py                   # simulated agent v1, no API key
    python labs/mlflow_318/step1_traffic.py --version v3      # the fixed version
    python labs/mlflow_318/step1_traffic.py --agent rag       # the repo's real RAG
    python labs/mlflow_318/step1_traffic.py --repeat 3        # more volume (3 rounds)
"""

import argparse
import os
import random
import uuid

from config import LANG, TRAFFIC_EXPERIMENT, setup

import mlflow

from app.i18n import t

# Typical conversations. Some of them trigger v1's planted failures on purpose.
CONVERSATIONS = {
    "es": [
        ["¿Qué incluye el programa PerksPlus?", "¿Cuál es el monto máximo que reembolsa PerksPlus?",
         "¿PerksPlus cubre clases de surf?"],
        ["¿Qué cubre el plan de beneficios Northwind Health Plus?",
         "¿Qué no cubre el plan Northwind Standard en comparación con Northwind Health Plus?"],
        ["¿Cuál es la diferencia entre Northwind Standard y Health Plus en medicamentos?",
         "¿Qué servicios de visión y dentales cubre cada plan?"],
        # Out of scope → v1 makes up policies
        ["¿Cuál es el procedimiento para tomar una licencia por maternidad?",
         "¿Y cuántas semanas son pagadas?"],
        ["¿Cuántos días de teletrabajo permite la política de la empresa?"],
        ["¿Qué responsabilidades tiene un analista de datos según el role library?"],
        # Failing tool
        ["¿Cuántos días de saldo de vacaciones me quedan?",
         "Necesito saber mi saldo de vacaciones antes de diciembre"],
        # Personal data in the question → v1 echoes it
        ["Hola, soy Ana (ana.perez@contoso.com). ¿PerksPlus reembolsa suplementos alimenticios?"],
        ["Mi celular es +57 300 555 1234, ¿con qué frecuencia se realizan las evaluaciones de "
         "desempeño?"],
        # Question in the other language
        ["What does PerksPlus cover?"],
        ["¿Qué debo hacer si soy testigo de violencia en el trabajo?"],
        ["¿Northwind Standard cubre servicios fuera de la red?",
         "Compara los dos planes en servicios de emergencia"],
    ],
    "en": [
        ["What does the PerksPlus program include?", "What is the maximum amount PerksPlus reimburses?",
         "Does PerksPlus cover surfing lessons?"],
        ["What does the Northwind Health Plus benefit plan cover?",
         "What does the Northwind Standard plan not cover compared to Northwind Health Plus?"],
        ["What is the difference between Northwind Standard and Health Plus for prescription drugs?",
         "Which vision and dental services does each plan cover?"],
        # Out of scope → v1 makes up policies
        ["What is the procedure to take maternity leave?", "And how many weeks are paid?"],
        ["How many remote work days does the company policy allow?"],
        ["What are the responsibilities of a data analyst according to the role library?"],
        # Failing tool
        ["How many vacation days do I have left? Check my balance please.",
         "I need to know my vacation balance before December"],
        # Personal data in the question → v1 echoes it
        ["Hi, I'm Ana (ana.perez@contoso.com). Does PerksPlus reimburse dietary supplements?"],
        ["My cell is +1 206 555 0134, how often are performance reviews held?"],
        # Question in the other language
        ["¿Qué cubre PerksPlus? ¿Puedo usarlo para yoga?"],
        ["What should I do if I witness violence at work?"],
        ["Does Northwind Standard cover out-of-network services?",
         "Compare both plans on emergency services"],
    ],
}


def build_mock(version: str):
    from mock_agent import answer, configure

    configure(version, environment="production")
    return lambda question, history, sid, uid: answer(question, session_id=sid, user_id=uid)


def build_rag():
    """The repo's real RAG (needs the index and the provider's API key)."""
    os.environ["APP_ENV"] = "production"   # before importing app.tracking
    from app.prompt_registry import DEFAULT_PROMPT
    from app.rag_pipeline import build_chain, load_vectorstore_from_disk
    from app.tracking import ask, enable_tracing

    enable_tracing()
    chain = build_chain(load_vectorstore_from_disk(),
                        prompt_version=os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT,
                        k=int(os.getenv("RETRIEVER_K") or 4))

    # The same function the Streamlit app uses: root trace with session and user
    return lambda question, history, sid, uid: ask(chain, question, history, sid, uid)[0]["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["mock", "rag"], default="mock")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v1",
                        help="simulated agent version")
    parser.add_argument("--repeat", type=int, default=2, help="rounds of conversations")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    setup(TRAFFIC_EXPERIMENT)
    random.seed(args.seed)

    if args.agent == "mock":
        mlflow.set_active_model(name=f"mock-assistant-{args.version}")
        ask = build_mock(args.version)
    else:
        ask = build_rag()

    total = 0
    for _ in range(args.repeat):
        conversations = CONVERSATIONS[LANG][:]
        random.shuffle(conversations)
        for turns in conversations:
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            user_id = f"employee-{random.randint(1, 25):03d}"
            history = []
            for question in turns:
                history.append((question, ask(question, history, session_id, user_id)))
                total += 1
            print(t("lab.session", session=session_id, user=user_id, n=len(turns)))

    print("\n" + t("lab.traffic_done", total=total, experiment=TRAFFIC_EXPERIMENT))
    print(t("lab.ui_traces", experiment=TRAFFIC_EXPERIMENT))


if __name__ == "__main__":
    main()
