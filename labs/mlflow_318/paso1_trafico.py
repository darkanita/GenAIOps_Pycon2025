# labs/mlflow_318/paso1_trafico.py
"""
Paso 1 · Simular tráfico de "producción" para tener trazas que analizar.

Genera conversaciones (sesiones de varios turnos) contra el asistente y las
deja en el experimento `lab318_produccion`, etiquetadas con
`environment = production`. Sobre estas trazas trabajan:
  - Automatic Issue Detection (pestaña Issues de la UI)  → paso 4
  - la evaluación online con scorers de código            → paso 3

Uso:
    python labs/mlflow_318/paso1_trafico.py                    # agente simulado v1, sin API key
    python labs/mlflow_318/paso1_trafico.py --version v3       # la versión corregida
    python labs/mlflow_318/paso1_trafico.py --agent rag        # el RAG real del repositorio
    python labs/mlflow_318/paso1_trafico.py --repeat 3         # más volumen (3 rondas)
"""

import argparse
import os
import random
import uuid

from config import TRAFFIC_EXPERIMENT, setup

import mlflow

# Conversaciones típicas. Algunas disparan a propósito las fallas del agente v1.
CONVERSACIONES = [
    ["¿Qué incluye el programa PerksPlus?", "¿Cuál es el monto máximo que reembolsa PerksPlus?",
     "¿PerksPlus cubre clases de surf?"],
    ["¿Qué cubre el plan de beneficios Northwind Health Plus?",
     "¿Qué no cubre el plan Northwind Standard en comparación con Northwind Health Plus?"],
    ["¿Cuál es la diferencia entre Northwind Standard y Health Plus en medicamentos?",
     "¿Qué servicios de visión y dentales cubre cada plan?"],
    # Fuera de alcance → v1 inventa políticas
    ["¿Cuál es el procedimiento para tomar una licencia por maternidad?",
     "¿Y cuántas semanas son pagadas?"],
    ["¿Cuántos días de teletrabajo permite la política de la empresa?"],
    ["¿Qué responsabilidades tiene un analista de datos según el role library?"],
    # Herramienta que falla
    ["¿Cuántos días de saldo de vacaciones me quedan?",
     "Necesito saber mi saldo de vacaciones antes de diciembre"],
    # Datos personales en la pregunta → v1 los repite
    ["Hola, soy Ana (ana.perez@contoso.com). ¿PerksPlus reembolsa suplementos alimenticios?"],
    ["Mi celular es +57 300 555 1234, ¿con qué frecuencia se realizan las evaluaciones de "
     "desempeño?"],
    # Pregunta en inglés
    ["What does PerksPlus cover?"],
    ["¿Qué debo hacer si soy testigo de violencia en el trabajo?"],
    ["¿Northwind Standard cubre servicios fuera de la red?",
     "Compara los dos planes en servicios de emergencia"],
]


def build_mock(version: str):
    from mock_agent import configurar, responder

    configurar(version, environment="production")
    return lambda pregunta, historial, sid, uid: responder(pregunta, session_id=sid, user_id=uid)


def build_rag():
    """RAG real del repositorio (necesita el índice y la API key del proveedor)."""
    os.environ["APP_ENV"] = "production"   # antes de importar app.tracking
    from app.prompt_registry import DEFAULT_PROMPT
    from app.rag_pipeline import build_chain, load_vectorstore_from_disk
    from app.tracking import ask, enable_tracing

    enable_tracing()
    chain = build_chain(load_vectorstore_from_disk(),
                        prompt_version=os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT,
                        k=int(os.getenv("RETRIEVER_K") or 4))

    # La misma función que usa la app de Streamlit: traza raíz con sesión y usuario
    return lambda pregunta, historial, sid, uid: ask(chain, pregunta, historial, sid, uid)[0]["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["mock", "rag"], default="mock")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v1",
                        help="versión del agente simulado")
    parser.add_argument("--repeat", type=int, default=2, help="rondas de conversaciones")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    setup(TRAFFIC_EXPERIMENT)
    random.seed(args.seed)

    if args.agent == "mock":
        mlflow.set_active_model(name=f"asistente-mock-{args.version}")
        ask = build_mock(args.version)
    else:
        ask = build_rag()

    total = 0
    for ronda in range(args.repeat):
        conversaciones = CONVERSACIONES[:]
        random.shuffle(conversaciones)
        for turnos in conversaciones:
            session_id = f"sesion-{uuid.uuid4().hex[:8]}"
            user_id = f"empleado-{random.randint(1, 25):03d}"
            historial = []
            for pregunta in turnos:
                respuesta = ask(pregunta, historial, session_id, user_id)
                historial.append((pregunta, respuesta))
                total += 1
            print(f"   💬 {session_id} ({user_id}): {len(turnos)} turnos")

    print(f"\n✅ {total} trazas en '{TRAFFIC_EXPERIMENT}' (environment=production)")
    print("👀 En la UI: Experiments → lab318_produccion → Traces / Sessions")


if __name__ == "__main__":
    main()
