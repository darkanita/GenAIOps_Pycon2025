# app/prompt_registry.py
"""
Versionamiento de prompts con el Prompt Registry de MLflow.

Los archivos de app/prompts/*.txt son la fuente: cada archivo se registra
como una VERSIÓN del prompt "asistente_rrhh" (PROMPT_NAME). Además:
  - cada versión recibe un alias con el nombre de su archivo (p. ej. @v1_asistente_rrhh)
  - el alias @production apunta a la versión que usa la app por defecto

Uso:
    python -m app.prompt_registry                    # registra/actualiza versiones
    python -m app.prompt_registry --promote 2        # mueve @production a la versión 2

En el código, un prompt se referencia con una URI:
    prompts:/asistente_rrhh@production   (alias)
    prompts:/asistente_rrhh/2            (versión fija)
    prompts:/asistente_rrhh@latest       (última versión)
También se acepta el nombre de un archivo local (v1_asistente_rrhh) sin usar el registro.
"""

import argparse
import os
import re

import mlflow
from mlflow import MlflowClient

PROMPT_DIR = "app/prompts"
PROMPT_NAME = os.getenv("PROMPT_NAME", "asistente_rrhh")
PRODUCTION_ALIAS = "production"
DEFAULT_PROMPT = f"prompts:/{PROMPT_NAME}@{PRODUCTION_ALIAS}"


def _to_mlflow_template(text: str) -> str:
    """LangChain usa {var}; el registro de MLflow usa {{var}}."""
    return re.sub(r"(?<!\{)\{(\w+)\}(?!\})", r"{{\1}}", text)


def _local_files(path=PROMPT_DIR):
    return sorted(f for f in os.listdir(path) if f.endswith(".txt"))


def _existing_versions(name=PROMPT_NAME):
    try:
        return list(MlflowClient().search_prompt_versions(name))
    except Exception:
        return []


def register_prompts(name=PROMPT_NAME, path=PROMPT_DIR, verbose=True):
    """
    Registra cada archivo .txt como versión del prompt, solo si su contenido
    cambió respecto a la versión que ya tiene ese alias. Es idempotente.
    """
    client = MlflowClient()
    registered = []
    for filename in _local_files(path):
        alias = filename[:-4]
        with open(os.path.join(path, filename), encoding="utf-8") as f:
            template = _to_mlflow_template(f.read())

        current = mlflow.genai.load_prompt(f"prompts:/{name}@{alias}", allow_missing=True) \
            if _existing_versions(name) else None
        if current is not None and current.template == template:
            continue  # sin cambios

        pv = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=f"Desde {filename}" + (" (actualizado)" if current else ""),
            tags={"source_file": filename},
        )
        client.set_prompt_alias(name, alias, pv.version)
        registered.append((filename, pv.version))
        if verbose:
            print(f"📝 {filename} → {name} v{pv.version} (@{alias})")

    # La primera vez, @production apunta a la primera versión registrada
    versions = _existing_versions(name)
    if versions and mlflow.genai.load_prompt(
            f"prompts:/{name}@{PRODUCTION_ALIAS}", allow_missing=True) is None:
        first = min(int(v.version) for v in versions)
        client.set_prompt_alias(name, PRODUCTION_ALIAS, first)
        if verbose:
            print(f"🚀 @{PRODUCTION_ALIAS} → v{first}")

    if verbose and not registered:
        print("✔️  Prompts al día, no hay versiones nuevas.")
    return registered


def promote(version: int, alias=PRODUCTION_ALIAS, name=PROMPT_NAME):
    MlflowClient().set_prompt_alias(name, alias, int(version))
    print(f"🚀 {name}@{alias} → v{version}")


def load_prompt_text(ref: str) -> tuple[str, str]:
    """
    Devuelve (template_en_formato_langchain, uri_resuelta).
    - ref = "prompts:/..."  → se carga del registro (y se enlaza a la traza/modelo activo)
    - ref = nombre de archivo local → se lee de app/prompts/
    """
    if ref.startswith("prompts:/"):
        pv = mlflow.genai.load_prompt(ref, allow_missing=True)
        if pv is None:  # primera ejecución: registrar los archivos locales
            register_prompts(verbose=True)
            pv = mlflow.genai.load_prompt(ref)
        return pv.to_single_brace_format(), f"prompts:/{pv.name}/{pv.version}"

    path = os.path.join(PROMPT_DIR, f"{ref}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read(), path


def list_prompt_options(name=PROMPT_NAME) -> list[str]:
    """Opciones para la UI: alias primero, luego versiones numeradas."""
    versions = _existing_versions(name)
    if not versions:
        try:
            register_prompts(verbose=False)
            versions = _existing_versions(name)
        except Exception:
            return [f[:-4] for f in _local_files()]  # sin registro disponible

    aliases, numbered = [], []
    for v in sorted(versions, key=lambda v: int(v.version), reverse=True):
        numbered.append(f"prompts:/{name}/{v.version}")
        pv = mlflow.genai.load_prompt(f"prompts:/{name}/{v.version}")
        aliases += [f"prompts:/{name}@{a}" for a in pv.aliases]
    aliases.sort(key=lambda a: (not a.endswith(f"@{PRODUCTION_ALIAS}"), a))
    return aliases + numbered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registro de prompts en MLflow")
    parser.add_argument("--promote", type=int, metavar="VERSION",
                        help="Mueve el alias @production a esta versión")
    args = parser.parse_args()
    if args.promote:
        promote(args.promote)
    else:
        register_prompts()
