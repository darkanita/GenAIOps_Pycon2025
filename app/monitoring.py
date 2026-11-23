# app/monitoring.py
"""
Evaluación online de las conversaciones del chat (MLflow 3.18, experimental).

Registra los scorers de código de app/code_scorers.py en el experimento del
chat y los activa con .start(): cada traza nueva (y cada conversación nueva,
para el scorer de sesión) recibe sus evaluaciones automáticamente, sin tocar
el código de la app.

Requisitos: un servidor MLflow self-hosted con base de datos y Docker local.
El de labs/mlflow_318/start_server.sh ya viene configurado:
    ./labs/mlflow_318/start_server.sh
    export MLFLOW_TRACKING_URI=http://localhost:5000

Uso:
    python -m app.monitoring                     # activa (100% de las trazas de producción)
    python -m app.monitoring --sample-rate 0.2   # solo el 20%
    python -m app.monitoring --all               # todas las trazas, no solo environment=production
    python -m app.monitoring --status
    python -m app.monitoring --stop
"""

import argparse
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import ScorerSamplingConfig, list_scorers

from app.code_scorers import ONLINE_CODE_SCORERS
from app.i18n import t
from app.tracking import CHAT_EXPERIMENT

load_dotenv()
os.environ.setdefault("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")  # también en el cliente


def main():
    parser = argparse.ArgumentParser(description=t("mon.desc"))
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--all", action="store_true",
                        help=t("mon.arg_all"))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    if not mlflow.get_tracking_uri().startswith("http"):
        sys.exit(t("mon.need_server"))

    experiment_id = mlflow.set_experiment(CHAT_EXPERIMENT).experiment_id

    if args.stop:
        for s in list_scorers(experiment_id=experiment_id):
            s.stop(experiment_id=experiment_id)
            print(t("mon.stopped", name=s.name))
        return

    if not args.status:
        config = ScorerSamplingConfig(
            sample_rate=args.sample_rate,
            filter_string=None if args.all else "tags.environment = 'production'",
        )
        for s in ONLINE_CODE_SCORERS:
            try:
                s.register(name=s.name, experiment_id=experiment_id) \
                 .start(experiment_id=experiment_id, sampling_config=config)
            except MlflowException as e:
                sys.exit(t("mon.start_error", name=s.name, error=e.message.splitlines()[0]))
            print(t("mon.active", name=s.name))
        if not args.all:
            print(t("mon.only_production"))

    for s in list_scorers(experiment_id=experiment_id):
        print(f"   • {s.name:30s} {s.status.value:8s} sample_rate={s.sample_rate}")


if __name__ == "__main__":
    main()
