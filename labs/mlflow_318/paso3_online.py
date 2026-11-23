# labs/mlflow_318/paso3_online.py
"""
Paso 3 · Evaluación online con scorers de código (experimental en MLflow 3.18).

Registra los scorers de `scorers.py` en el experimento de producción y los
activa con `.start()`: desde ese momento cada traza nueva (y cada sesión
nueva, para el scorer de sesión) recibe sus evaluaciones automáticamente.
El servidor ejecuta el código de cada scorer en un contenedor Docker aislado.

Requisitos (los cumple ./labs/mlflow_318/start_server.sh):
  - servidor MLflow self-hosted con base de datos (SQLite o PostgreSQL), 1 réplica
  - Docker local en la misma máquina que el servidor
  - MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true en el servidor Y en este cliente

Uso:
    python labs/mlflow_318/paso3_online.py                  # registra y activa (100% del tráfico)
    python labs/mlflow_318/paso3_online.py --sample-rate 0.3
    python labs/mlflow_318/paso3_online.py --status         # ver scorers y su estado
    python labs/mlflow_318/paso3_online.py --stop           # detener la evaluación online

Luego genera tráfico nuevo (python labs/mlflow_318/paso1_trafico.py) y abre la
traza en la UI: aparecen evaluaciones con fuente CODE.
"""

import argparse
import os
import sys

from config import TRAFFIC_EXPERIMENT, setup

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import ScorerSamplingConfig, list_scorers

from scorers import ONLINE_SCORERS

# Debe estar activo también en el cliente; si no, .start() activa y luego falla.
os.environ.setdefault("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")

HINT = """El servidor no está configurado para ejecutar scorers de código. Reinícialo con Docker:
    ./labs/mlflow_318/start_server.sh          # (no uses --no-docker)
que exporta:
    MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true
    MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND=docker"""


def show_status(experiment_id: str):
    registrados = list_scorers(experiment_id=experiment_id)
    if not registrados:
        print("   (no hay scorers registrados)")
    for s in registrados:
        print(f"   • {s.name:24s} estado={s.status.value:8s} sample_rate={s.sample_rate} "
              f"filtro={s.filter_string}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="fracción del tráfico a evaluar (0..1)")
    parser.add_argument("--filter", default="tags.environment = 'production'",
                        help="qué trazas evaluar (sintaxis de search_traces); '' = todas")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    experiment_id = setup(TRAFFIC_EXPERIMENT, require_server=True)

    if args.status:
        show_status(experiment_id)
        return

    if args.stop:
        for s in list_scorers(experiment_id=experiment_id):
            s.stop(experiment_id=experiment_id)
            print(f"   ⏹️  {s.name} detenido")
        return

    config = ScorerSamplingConfig(sample_rate=args.sample_rate,
                                  filter_string=args.filter or None)
    for s in ONLINE_SCORERS:
        try:
            registrado = s.register(name=s.name, experiment_id=experiment_id)
            registrado.start(experiment_id=experiment_id, sampling_config=config)
        except MlflowException as e:
            sys.exit(f"❌ {s.name}: {e.message.splitlines()[0]}\n\n{HINT}")
        nivel = "sesión" if registrado.is_session_level_scorer else "traza"
        print(f"   ▶️  {s.name} activo (nivel {nivel}, sample_rate={args.sample_rate})")

    print("\n📋 Estado:")
    show_status(experiment_id)
    print("\n➡️  Ahora genera tráfico nuevo: python labs/mlflow_318/paso1_trafico.py --repeat 1")
    print("   Las evaluaciones llegan en segundos/minutos (pestaña Assessments de cada traza).")


if __name__ == "__main__":
    main()
