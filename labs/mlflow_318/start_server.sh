#!/usr/bin/env bash
# labs/mlflow_318/start_server.sh
# Servidor MLflow 3.18 self-hosted para el laboratorio:
#   - backend SQLite (Issue Detection y la evaluación online necesitan base de datos)
#   - ejecución de jobs en segundo plano + tareas periódicas (detección programada)
#   - scorers de código en contenedores Docker (evaluación online)
#
# Uso:
#   ./labs/mlflow_318/start_server.sh             # con Docker (todo el laboratorio)
#   ./labs/mlflow_318/start_server.sh --no-docker # sin Docker (todo menos el paso 3)
set -euo pipefail

cd "$(dirname "$0")/../.."          # raíz del repositorio
PORT="${PORT:-5000}"
DB="${MLFLOW_LAB_DB:-sqlite:///mlflow_lab318.db}"

export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true        # jobs en segundo plano (issue detection)
export MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=true   # detección programada (diaria/semanal)

if [[ "${1:-}" != "--no-docker" ]]; then
  if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker no está disponible. Inícialo o usa: $0 --no-docker"; exit 1
  fi
  export MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true
  export MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor
  export MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND=docker
  # Sin MLFLOW_JOB_IMAGE, MLflow construye al arrancar una imagen mínima con tu versión de
  # Python y de MLflow (necesita descargar python:<versión>-slim). Si defines MLFLOW_JOB_IMAGE,
  # la imagen debe existir localmente o poder descargarse (docker pull).
  # macOS + Colima: Docker solo ve tu home, así que el TMPDIR del servidor debe estar ahí
  if [[ "$(uname)" == "Darwin" ]]; then
    mkdir -p "$HOME/mlflow-tmp"; export TMPDIR="$HOME/mlflow-tmp"
  fi
  echo "🐳 Scorers de código en contenedores (imagen: ${MLFLOW_JOB_IMAGE:-construida por MLflow})"
else
  echo "⚠️  Sin Docker: la evaluación online con scorers de código (paso 3) queda desactivada"
fi

echo "🚀 MLflow en http://localhost:${PORT}  (backend: ${DB})"
exec mlflow server \
  --backend-store-uri "$DB" \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port "$PORT" \
  --workers 1
