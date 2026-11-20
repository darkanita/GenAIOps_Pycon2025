# labs/mlflow_318/paso4_issues.py
"""
Paso 4 · Trabajar con los issues detectados (Automatic Issue Detection, MLflow 3.18).

La detección se lanza desde la UI (no tiene API de Python):
    Experiments → lab318_produccion → pestaña Issues → "Run issue detection"
    (elige OpenAI, Anthropic, Gemini o un endpoint del AI Gateway).
    También puedes programarla: "Schedule detection" → diaria o semanal,
    con filtro opcional `environment = production`.

Este script usa la API de gestión de issues para:
  - listar los issues con su evidencia (trazas que los respaldan),
  - exportar un reporte en Markdown listo para pegar en tu agente de código
    (Claude Code, Codex, Cursor) junto con el skill `genaiops-rrhh-triage`,
  - crear un issue a partir de un scorer online que está fallando,
  - marcar un issue como resuelto o como "no es un issue".

Uso:
    python labs/mlflow_318/paso4_issues.py                          # listar + exportar reporte
    python labs/mlflow_318/paso4_issues.py --desde-scorer cita_fuente
    python labs/mlflow_318/paso4_issues.py --resolver <ISSUE_ID>
    python labs/mlflow_318/paso4_issues.py --rechazar <ISSUE_ID>
"""

import argparse
import time

from config import LAB_DIR, TRAFFIC_EXPERIMENT, setup

import mlflow
from mlflow.entities.assessment import IssueReference
from mlflow.entities.issue import IssueSeverity, IssueStatus

REPORT = LAB_DIR / "issues_para_agente.md"


def traces_for_issue(experiment_id: str, issue_id: str, n: int = 500):
    """Trazas enlazadas al issue (la detección automática las enlaza como evidencia)."""
    return mlflow.search_traces(locations=[experiment_id], filter_string=f"issue.id = '{issue_id}'",
                                max_results=n, return_type="list")


def summarize_trace(trace) -> str:
    inputs = trace.data.spans[0].inputs if trace.data.spans else {}
    pregunta = (inputs or {}).get("pregunta") or (inputs or {}).get("question") or inputs
    errores = [s.name for s in trace.data.spans if "ERROR" in str(s.status.status_code)]
    detalle = f" — spans con error: {errores}" if errores else ""
    return f"`{trace.info.trace_id}` ({trace.info.execution_duration} ms): {pregunta}{detalle}"


def list_and_export(experiment_id: str):
    issues = list(mlflow.genai.search_issues(experiment_id=experiment_id,
                                             filter_string="status = 'pending'"))
    if not issues:
        print("   No hay issues abiertos. Lanza la detección desde la pestaña Issues de la UI\n"
              "   (o crea uno con --desde-scorer <nombre>).")
        return

    lines = [
        "# Issues abiertos del asistente de RR.HH.",
        "",
        f"Servidor MLflow: `{mlflow.get_tracking_uri()}` · experimento: `{TRAFFIC_EXPERIMENT}` "
        f"(id `{experiment_id}`)",
        "",
        "Usa el skill `genaiops-rrhh-triage` para investigar y corregir cada issue.",
        "",
    ]
    for issue in issues:
        sev = issue.severity.value if issue.severity else "?"
        cats = ", ".join(str(getattr(c, "value", c)) for c in (issue.categories or []))
        evidencia = traces_for_issue(experiment_id, issue.issue_id)
        n_trazas = issue.trace_count or len(evidencia)
        print(f"\n🐞 [{sev}] {issue.name}  ({issue.issue_id})")
        print(f"   categorías: {cats or '-'} | trazas: {n_trazas}")
        print(f"   {issue.description[:300]}")

        lines += [f"## [{sev}] {issue.name}", "",
                  f"- ID: `{issue.issue_id}`",
                  f"- Categorías: {cats or '-'}",
                  f"- Trazas afectadas: {n_trazas}", "",
                  issue.description, ""]
        if issue.root_causes:
            lines += ["**Causas probables:**", ""] + [f"- {c}" for c in issue.root_causes] + [""]
        if evidencia:
            lines += ["**Trazas de ejemplo:**", ""]
            for t in evidencia[:3]:
                lines.append(f"- {summarize_trace(t)}")
                print(f"   • {summarize_trace(t)[:160]}")
            lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📝 Reporte para tu agente de código: {REPORT.relative_to(LAB_DIR.parents[1])}")
    print("   Tip: en la UI, cada issue tiene 'Fix this issue' → Copy con un prompt listo.")


def issue_from_scorer(experiment_id: str, scorer_name: str):
    """Convierte las fallas de un scorer online en un issue (creación manual)."""
    failing = mlflow.search_traces(locations=[experiment_id],
                                   filter_string=f"feedback.{scorer_name} = 'false'",
                                   max_results=200, return_type="list")
    if not failing:
        print(f"   ✅ Ninguna traza falla '{scorer_name}'. ¿Ya activaste el paso 3 y generaste "
              "tráfico nuevo?")
        return
    ejemplos = [t.info.trace_id for t in failing[:5]]
    issue = mlflow.genai.create_issue(
        name=f"Falla recurrente del scorer '{scorer_name}'",
        severity=IssueSeverity.MEDIUM,
        description=(f"{len(failing)} trazas de producción reprobaron el scorer de código "
                     f"'{scorer_name}'. Ejemplos: {', '.join(ejemplos)}"),
        categories=["adherence"],
        last_seen_timestamp=int(time.time() * 1000),
        metadata={"scorer": scorer_name, "ejemplos": ",".join(ejemplos)},
        experiment_id=experiment_id,
    )
    # Enlaza cada traza como evidencia (así aparecen en la pestaña Issues y en `issue.id = ...`)
    for t in failing:
        mlflow.log_assessment(trace_id=t.info.trace_id, assessment=IssueReference(
            issue_id=issue.issue_id, issue_name=issue.name,
            rationale=f"La traza reprobó el scorer '{scorer_name}'"))
    print(f"   🐞 Issue creado: {issue.issue_id} ({len(failing)} trazas enlazadas)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde-scorer", metavar="SCORER")
    parser.add_argument("--resolver", metavar="ISSUE_ID")
    parser.add_argument("--rechazar", metavar="ISSUE_ID", help="marcar como 'Not an issue'")
    args = parser.parse_args()

    experiment_id = setup(TRAFFIC_EXPERIMENT)

    if args.resolver:
        mlflow.genai.update_issue(args.resolver, status=IssueStatus.RESOLVED,
                                  experiment_id=experiment_id)
        print(f"   ✅ {args.resolver} → Resolved")
    elif args.rechazar:
        mlflow.genai.update_issue(args.rechazar, status=IssueStatus.REJECTED,
                                  experiment_id=experiment_id)
        print(f"   🙅 {args.rechazar} → Not an issue")
    elif args.desde_scorer:
        issue_from_scorer(experiment_id, args.desde_scorer)
    else:
        list_and_export(experiment_id)


if __name__ == "__main__":
    main()
