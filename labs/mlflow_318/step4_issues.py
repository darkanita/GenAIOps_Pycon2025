# labs/mlflow_318/step4_issues.py
"""
Step 4 · Work with the detected issues (Automatic Issue Detection, MLflow 3.18).
Paso 4 · Trabajar con los issues detectados.

Detection is launched from the UI (it has no Python API):
    Experiments → lab318_production → Issues tab → "Run issue detection"
    (pick OpenAI, Anthropic, Gemini or an AI Gateway endpoint).
    You can also schedule it: "Schedule detection" → daily or weekly,
    with an optional `environment = production` filter.

This script uses the issue management API to:
  - list the issues with their evidence (the traces that support them),
  - export a Markdown report ready to paste into your coding agent
    (Claude Code, Codex, Cursor) together with the `genaiops-hr-triage` skill,
  - create an issue from an online scorer that keeps failing,
  - mark an issue as resolved or as "not an issue".

Usage:
    python labs/mlflow_318/step4_issues.py                          # list + export report
    python labs/mlflow_318/step4_issues.py --from-scorer cites_source
    python labs/mlflow_318/step4_issues.py --resolve <ISSUE_ID>
    python labs/mlflow_318/step4_issues.py --reject <ISSUE_ID>
"""

import argparse
import time

from config import LAB_DIR, TRAFFIC_EXPERIMENT, setup

import mlflow
from mlflow.entities.assessment import IssueReference
from mlflow.entities.issue import IssueSeverity, IssueStatus

from app.i18n import t

REPORT = LAB_DIR / "issues_for_agent.md"


def traces_for_issue(experiment_id: str, issue_id: str, n: int = 500):
    """Traces linked to the issue (automatic detection links them as evidence)."""
    return mlflow.search_traces(locations=[experiment_id], filter_string=f"issue.id = '{issue_id}'",
                                max_results=n, return_type="list")


def summarize_trace(trace) -> str:
    inputs = (trace.data.spans[0].inputs if trace.data.spans else {}) or {}
    question = inputs.get("question") or inputs
    errors = [s.name for s in trace.data.spans if "ERROR" in str(s.status.status_code)]
    detail = t("lab.error_spans", spans=errors) if errors else ""
    return f"`{trace.info.trace_id}` ({trace.info.execution_duration} ms): {question}{detail}"


def list_and_export(experiment_id: str):
    issues = list(mlflow.genai.search_issues(experiment_id=experiment_id,
                                             filter_string="status = 'pending'"))
    if not issues:
        print(t("lab.no_issues"))
        return

    lines = [
        t("lab.report_title"), "",
        t("lab.report_server", uri=mlflow.get_tracking_uri(), experiment=TRAFFIC_EXPERIMENT,
          id=experiment_id), "",
        t("lab.report_skill"), "",
    ]
    for issue in issues:
        sev = issue.severity.value if issue.severity else "?"
        cats = ", ".join(str(getattr(c, "value", c)) for c in (issue.categories or []))
        evidence = traces_for_issue(experiment_id, issue.issue_id)
        n_traces = issue.trace_count or len(evidence)
        print(f"\n🐞 [{sev}] {issue.name}  ({issue.issue_id})")
        print(t("lab.issue_meta", categories=cats or "-", n=n_traces))
        print(f"   {issue.description[:300]}")

        lines += [f"## [{sev}] {issue.name}", "",
                  f"- ID: `{issue.issue_id}`",
                  t("lab.report_categories", categories=cats or "-"),
                  t("lab.report_traces", n=n_traces), "",
                  issue.description, ""]
        if issue.root_causes:
            lines += [t("lab.report_causes"), ""] + [f"- {c}" for c in issue.root_causes] + [""]
        if evidence:
            lines += [t("lab.report_examples"), ""]
            for tr in evidence[:3]:
                lines.append(f"- {summarize_trace(tr)}")
                print(f"   • {summarize_trace(tr)[:160]}")
            lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + t("lab.report_saved", path=REPORT.relative_to(LAB_DIR.parents[1])))
    print(t("lab.report_tip"))


def issue_from_scorer(experiment_id: str, scorer_name: str):
    """Turns the failures of an online scorer into an issue (manual creation)."""
    failing = mlflow.search_traces(locations=[experiment_id],
                                   filter_string=f"feedback.{scorer_name} = 'false'",
                                   max_results=200, return_type="list")
    if not failing:
        print(t("lab.no_failing", scorer=scorer_name))
        return
    examples = [tr.info.trace_id for tr in failing[:5]]
    issue = mlflow.genai.create_issue(
        name=t("lab.issue_name", scorer=scorer_name),
        severity=IssueSeverity.MEDIUM,
        description=t("lab.issue_description", n=len(failing), scorer=scorer_name,
                      examples=", ".join(examples)),
        categories=["adherence"],
        last_seen_timestamp=int(time.time() * 1000),
        metadata={"scorer": scorer_name, "examples": ",".join(examples)},
        experiment_id=experiment_id,
    )
    # Link every trace as evidence (so they show in the Issues tab and in `issue.id = ...`)
    for tr in failing:
        mlflow.log_assessment(trace_id=tr.info.trace_id, assessment=IssueReference(
            issue_id=issue.issue_id, issue_name=issue.name,
            rationale=t("lab.issue_rationale", scorer=scorer_name)))
    print(t("lab.issue_created", id=issue.issue_id, n=len(failing)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-scorer", metavar="SCORER")
    parser.add_argument("--resolve", metavar="ISSUE_ID")
    parser.add_argument("--reject", metavar="ISSUE_ID", help="mark as 'Not an issue'")
    args = parser.parse_args()

    experiment_id = setup(TRAFFIC_EXPERIMENT)

    if args.resolve:
        mlflow.genai.update_issue(args.resolve, status=IssueStatus.RESOLVED,
                                  experiment_id=experiment_id)
        print(f"   ✅ {args.resolve} → Resolved")
    elif args.reject:
        mlflow.genai.update_issue(args.reject, status=IssueStatus.REJECTED,
                                  experiment_id=experiment_id)
        print(f"   🙅 {args.reject} → Not an issue")
    elif args.from_scorer:
        issue_from_scorer(experiment_id, args.from_scorer)
    else:
        list_and_export(experiment_id)


if __name__ == "__main__":
    main()
