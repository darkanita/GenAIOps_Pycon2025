# labs/mlflow_318/paso5_skills.py
"""
Paso 5 · Agent Skill Registry (experimental en MLflow 3.18).

Un "skill" es una carpeta con un SKILL.md (instrucciones + frontmatter con
`name` y `description`) y archivos de apoyo. El registro de MLflow le da un
hogar compartido: versiones, alias (p. ej. `production`) y búsqueda, para que
todo el equipo instale la MISMA versión en su agente de código.

Este script:
  1. registra skills/genaiops-rrhh-triage como nueva versión,
  2. le asigna el alias `production`,
  3. busca los skills activos del registro,
  4. instala la versión fijada en .claude/skills/ (Claude Code),
     o en otra carpeta con --destino (Codex, Cursor...).

Uso:
    python labs/mlflow_318/paso5_skills.py
    python labs/mlflow_318/paso5_skills.py --destino .cursor/skills/genaiops-rrhh-triage

Equivalente con la CLI:
    mlflow skills register --source labs/mlflow_318/skills/genaiops-rrhh-triage
    mlflow skills set-alias skills:/genaiops-rrhh-triage --alias production --version 1
    mlflow skills search --filter-string "status = 'active'"
    mlflow skills pull skills:/genaiops-rrhh-triage@production -d .claude/skills/genaiops-rrhh-triage
"""

import argparse
import shutil

from config import LAB_DIR, REPO_ROOT, TRAFFIC_EXPERIMENT, setup

import mlflow
from mlflow import MlflowClient

SKILL_DIR = LAB_DIR / "skills" / "genaiops-rrhh-triage"
SKILL_NAME = "genaiops-rrhh-triage"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", default="production")
    parser.add_argument("--destino", default=str(REPO_ROOT / ".claude" / "skills" / SKILL_NAME),
                        help="carpeta donde instalar el skill (debe no existir o estar vacía)")
    parser.add_argument("--sin-instalar", action="store_true")
    args = parser.parse_args()

    setup(TRAFFIC_EXPERIMENT)
    client = MlflowClient()

    # 1. Registrar (crea el skill si no existe y agrega una versión nueva)
    version = mlflow.genai.register_skill(source=str(SKILL_DIR))
    print(f"📚 Registrado: {version.name} v{version.version}")

    # 2. Alias: el equipo instala "production", no un número de versión
    client.set_skill_alias(name=SKILL_NAME, alias=args.alias, version=int(version.version))
    print(f"🏷️  Alias @{args.alias} → v{version.version}")

    # 3. Descubrir lo que hay en el registro
    print("\n🔎 Skills activos en el registro:")
    for skill in mlflow.genai.search_skills(filter_string="status = 'active'"):
        aliases = ", ".join(f"@{a}→v{v}" for a, v in (skill.aliases or {}).items())
        print(f"   • {skill.name} (última v{skill.latest_version}) {aliases}")

    # 4. Instalar la versión fijada en el agente de código
    if args.sin_instalar:
        return
    uri = f"skills:/{SKILL_NAME}@{args.alias}"
    destino = args.destino
    shutil.rmtree(destino, ignore_errors=True)   # pull exige una carpeta vacía o inexistente
    path = mlflow.genai.pull(uri, destination=destino)
    print(f"\n⬇️  {uri} instalado en {path}")
    print("   Abre Claude Code en este repo y pide: \"usa el skill genaiops-rrhh-triage para "
          "resolver los issues de issues_para_agente.md\"")


if __name__ == "__main__":
    main()
