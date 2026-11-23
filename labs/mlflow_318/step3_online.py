# labs/mlflow_318/step3_online.py
"""
Step 3 · Online evaluation with code-based scorers (experimental in MLflow 3.18).
Paso 3 · Evaluación online con scorers de código.

Registers the scorers in `scorers.py` on the production experiment and turns
them on with `.start()`: from then on every new trace (and every new session,
for the session scorer) gets its assessments automatically. The server runs
each scorer's code in an isolated Docker container.

Requirements (./labs/mlflow_318/start_server.sh meets them):
  - a self-hosted MLflow server with a database (SQLite or PostgreSQL), 1 replica
  - local Docker on the same machine as the server
  - MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true on the server AND on this client

Usage:
    python labs/mlflow_318/step3_online.py                  # register and start (100% of traffic)
    python labs/mlflow_318/step3_online.py --sample-rate 0.3
    python labs/mlflow_318/step3_online.py --status         # see the scorers and their status
    python labs/mlflow_318/step3_online.py --stop           # stop online evaluation

Then generate new traffic (python labs/mlflow_318/step1_traffic.py) and open a
trace in the UI: assessments with source CODE show up.
"""

import argparse
import os
import sys

from config import TRAFFIC_EXPERIMENT, setup

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import ScorerSamplingConfig, list_scorers

from app.i18n import t
from scorers import ONLINE_SCORERS

# Must also be set on the client; otherwise .start() fails
os.environ.setdefault("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")


def show_status(experiment_id: str):
    registered = list_scorers(experiment_id=experiment_id)
    if not registered:
        print(t("lab.no_scorers"))
    for s in registered:
        print(f"   • {s.name:26s} status={s.status.value:8s} sample_rate={s.sample_rate} "
              f"filter={s.filter_string}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="fraction of traffic to score (0..1)")
    parser.add_argument("--filter", default="tags.environment = 'production'",
                        help="which traces to score (search_traces syntax); '' = all")
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
            print(t("mon.stopped", name=s.name))
        return

    config = ScorerSamplingConfig(sample_rate=args.sample_rate,
                                  filter_string=args.filter or None)
    for s in ONLINE_SCORERS:
        try:
            registered = s.register(name=s.name, experiment_id=experiment_id)
            registered.start(experiment_id=experiment_id, sampling_config=config)
        except MlflowException as e:
            sys.exit(f"❌ {s.name}: {e.message.splitlines()[0]}\n\n" + t("lab.online_hint"))
        level = "session" if registered.is_session_level_scorer else "trace"
        print(t("lab.scorer_started", name=s.name, level=level, rate=args.sample_rate))

    print("\n" + t("lab.status_header"))
    show_status(experiment_id)
    print("\n" + t("lab.online_next"))


if __name__ == "__main__":
    main()
