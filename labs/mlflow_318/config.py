# labs/mlflow_318/config.py
"""Configuración compartida del laboratorio de MLflow 3.18."""

import os
import sys
import urllib.request
from pathlib import Path

import mlflow

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))   # para poder importar app.* en modo --agent rag
sys.path.insert(0, str(LAB_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Experimentos del laboratorio (separados de los del chatbot original)
TRAFFIC_EXPERIMENT = "lab318_produccion"   # trazas "en producción": issues + evaluación online
EVAL_EXPERIMENT = "lab318_evaluaciones"    # evaluaciones offline + comparaciones
DATASET_PATH = LAB_DIR / "data" / "preguntas_lab.json"

# Las funciones nuevas (issues, evaluación online) necesitan un servidor MLflow con
# base de datos: por defecto el que levanta ./start_server.sh
DEFAULT_TRACKING_URI = "http://localhost:5000"


def setup(experiment: str, require_server: bool = False) -> str:
    """Configura el tracking URI y el experimento. Devuelve el experiment_id."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    if uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3)
        except Exception:
            sys.exit(f"❌ No hay un servidor MLflow en {uri}.\n"
                     f"   Levántalo con: ./labs/mlflow_318/start_server.sh")
    elif require_server:
        sys.exit(f"❌ Este paso necesita un servidor MLflow (MLFLOW_TRACKING_URI={uri}).\n"
                 f"   Levántalo con ./labs/mlflow_318/start_server.sh y usa "
                 f"MLFLOW_TRACKING_URI={DEFAULT_TRACKING_URI}")

    exp = mlflow.set_experiment(experiment)
    print(f"🔗 MLflow: {uri} | experimento: {experiment}")
    return exp.experiment_id
