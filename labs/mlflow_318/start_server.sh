#!/usr/bin/env bash
# labs/mlflow_318/start_server.sh
# Self-hosted MLflow 3.18 server for the lab / Servidor MLflow 3.18 para el laboratorio:
#   - SQLite backend (Issue Detection and online evaluation need a database)
#   - background job execution + periodic tasks (scheduled detection)
#   - code-based scorers in Docker containers (online evaluation)
#
# Usage / Uso:
#   ./labs/mlflow_318/start_server.sh             # with Docker (the whole lab)
#   ./labs/mlflow_318/start_server.sh --no-docker # without Docker (everything but step 3)
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root / raíz del repositorio
# Language from the environment or .env (es | en) / Idioma desde el entorno o .env
LANG_APP="${APP_LANG:-$(grep -E '^APP_LANG=' .env 2>/dev/null | cut -d= -f2)}"
msg() { if [[ "${LANG_APP:-es}" == en* ]]; then echo "$2"; else echo "$1"; fi; }
PORT="${PORT:-5000}"
DB="${MLFLOW_LAB_DB:-sqlite:///mlflow_lab318.db}"

export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true        # background jobs (issue detection)
export MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=true   # scheduled detection (daily/weekly)

if [[ "${1:-}" != "--no-docker" ]]; then
  if ! docker info >/dev/null 2>&1; then
    msg "❌ Docker no está disponible. Inícialo o usa: $0 --no-docker" \
        "❌ Docker is not available. Start it or use: $0 --no-docker"; exit 1
  fi
  export MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true
  export MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor
  export MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND=docker
  # Without MLFLOW_JOB_IMAGE, MLflow builds a minimal image with your Python and MLflow
  # versions at startup (it pulls python:<version>-slim). If you set MLFLOW_JOB_IMAGE, the
  # image must exist locally or be pullable (docker pull).
  # macOS + Colima: Docker only sees your home, so the server's TMPDIR must live there
  if [[ "$(uname)" == "Darwin" ]]; then
    mkdir -p "$HOME/mlflow-tmp"; export TMPDIR="$HOME/mlflow-tmp"
  fi
  msg "🐳 Scorers de código en contenedores (imagen: ${MLFLOW_JOB_IMAGE:-construida por MLflow})" \
      "🐳 Code-based scorers in containers (image: ${MLFLOW_JOB_IMAGE:-built by MLflow})"
else
  msg "⚠️  Sin Docker: la evaluación online con scorers de código (paso 3) queda desactivada" \
      "⚠️  No Docker: online evaluation with code-based scorers (step 3) is disabled"
fi

msg "🚀 MLflow en http://localhost:${PORT}  (backend: ${DB})" \
    "🚀 MLflow at http://localhost:${PORT}  (backend: ${DB})"
exec mlflow server \
  --backend-store-uri "$DB" \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port "$PORT" \
  --workers 1
