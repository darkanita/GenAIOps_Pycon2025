# labs/mlflow_318/config.py
"""Shared settings for the MLflow 3.18 lab (language comes from APP_LANG=es|en)."""

import os
import sys
import urllib.request
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))   # to import app.* (i18n, and the real RAG in --agent rag)
sys.path.insert(0, str(LAB_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import mlflow  # noqa: E402

from app.i18n import LANG, t  # noqa: E402
from texts import register as register_texts  # noqa: E402

register_texts()   # the lab's messages in Spanish and English

# Lab experiments (separate from the original chatbot's)
TRAFFIC_EXPERIMENT = "lab318_production"    # "production" traces: issues + online evaluation
EVAL_EXPERIMENT = "lab318_evaluations"      # offline evaluations + comparisons
DATASET_PATH = LAB_DIR / "data" / f"lab_questions.{LANG}.json"

# The new features (issues, online evaluation) need an MLflow server with a
# database: by default the one ./start_server.sh starts
DEFAULT_TRACKING_URI = "http://localhost:5000"


def setup(experiment: str, require_server: bool = False) -> str:
    """Sets the tracking URI and the experiment. Returns the experiment_id."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    if uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3)
        except Exception:
            sys.exit(t("lab.no_server", uri=uri))
    elif require_server:
        sys.exit(t("lab.need_server", uri=uri, default=DEFAULT_TRACKING_URI))

    exp = mlflow.set_experiment(experiment)
    print(t("lab.connected", uri=uri, experiment=experiment, lang=LANG))
    return exp.experiment_id
