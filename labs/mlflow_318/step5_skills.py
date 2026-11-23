# labs/mlflow_318/step5_skills.py
"""
Step 5 · Agent Skill Registry (experimental in MLflow 3.18).
Paso 5 · Agent Skill Registry.

A "skill" is a folder with a SKILL.md (instructions + frontmatter with `name`
and `description`) and supporting files. MLflow's registry gives it a shared
home: versions, aliases (e.g. `production`) and search, so the whole team
installs the SAME version into their coding agent.

This script:
  1. registers skills/genaiops-hr-triage as a new version,
  2. points the `production` alias at it,
  3. searches the registry for active skills,
  4. installs the pinned version into .claude/skills/ (Claude Code),
     or another folder with --dest (Codex, Cursor...).

Usage:
    python labs/mlflow_318/step5_skills.py
    python labs/mlflow_318/step5_skills.py --dest .cursor/skills/genaiops-hr-triage

Same thing with the CLI:
    mlflow skills register --source labs/mlflow_318/skills/genaiops-hr-triage
    mlflow skills set-alias skills:/genaiops-hr-triage --alias production --version 1
    mlflow skills search --filter-string "status = 'active'"
    mlflow skills pull skills:/genaiops-hr-triage@production -d .claude/skills/genaiops-hr-triage
"""

import argparse
import shutil

from config import LAB_DIR, REPO_ROOT, TRAFFIC_EXPERIMENT, setup

import mlflow
from mlflow import MlflowClient

from app.i18n import t

SKILL_NAME = "genaiops-hr-triage"
SKILL_DIR = LAB_DIR / "skills" / SKILL_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", default="production")
    parser.add_argument("--dest", default=str(REPO_ROOT / ".claude" / "skills" / SKILL_NAME),
                        help="folder to install the skill into (must not exist or be empty)")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()

    setup(TRAFFIC_EXPERIMENT)
    client = MlflowClient()

    # 1. Register (creates the skill if needed and adds a new version)
    version = mlflow.genai.register_skill(source=str(SKILL_DIR))
    print(t("lab.skill_registered", name=version.name, version=version.version))

    # 2. Alias: the team installs "production", not a version number
    client.set_skill_alias(name=SKILL_NAME, alias=args.alias, version=int(version.version))
    print(t("lab.skill_alias", alias=args.alias, version=version.version))

    # 3. Discover what is in the registry
    print("\n" + t("lab.skill_search"))
    for skill in mlflow.genai.search_skills(filter_string="status = 'active'"):
        aliases = ", ".join(f"@{a}→v{v}" for a, v in (skill.aliases or {}).items())
        print(f"   • {skill.name} (latest v{skill.latest_version}) {aliases}")

    # 4. Install the pinned version into the coding agent
    if args.no_install:
        return
    uri = f"skills:/{SKILL_NAME}@{args.alias}"
    shutil.rmtree(args.dest, ignore_errors=True)   # pull requires an empty or missing folder
    path = mlflow.genai.pull(uri, destination=args.dest)
    print("\n" + t("lab.skill_installed", uri=uri, path=path))
    print(t("lab.skill_try"))


if __name__ == "__main__":
    main()
