# labs/mlflow_318/texts.py
"""Lab messages in Spanish and English (picked with APP_LANG=es|en)."""

from app.i18n import add_messages

LAB_MESSAGES = {
    # config.py
    "lab.connected": {"es": "🔗 MLflow: {uri} | experimento: {experiment} | idioma: {lang}",
                      "en": "🔗 MLflow: {uri} | experiment: {experiment} | language: {lang}"},
    "lab.no_server": {"es": "❌ No hay un servidor MLflow en {uri}.\n   Levántalo con: ./labs/mlflow_318/start_server.sh",
                      "en": "❌ No MLflow server at {uri}.\n   Start it with: ./labs/mlflow_318/start_server.sh"},
    "lab.need_server": {"es": "❌ Este paso necesita un servidor MLflow (MLFLOW_TRACKING_URI={uri}).\n"
                              "   Levántalo con ./labs/mlflow_318/start_server.sh y usa MLFLOW_TRACKING_URI={default}",
                        "en": "❌ This step needs an MLflow server (MLFLOW_TRACKING_URI={uri}).\n"
                              "   Start it with ./labs/mlflow_318/start_server.sh and use MLFLOW_TRACKING_URI={default}"},

    # step1_traffic.py
    "lab.session": {"es": "   💬 {session} ({user}): {n} turnos", "en": "   💬 {session} ({user}): {n} turns"},
    "lab.traffic_done": {"es": "✅ {total} trazas en '{experiment}' (environment=production)",
                         "en": "✅ {total} traces in '{experiment}' (environment=production)"},
    "lab.ui_traces": {"es": "👀 En la UI: Experiments → {experiment} → Traces / Sessions",
                      "en": "👀 In the UI: Experiments → {experiment} → Traces / Sessions"},

    # step2_compare.py
    "lab.scorers": {"es": "⚖️  Scorers: {names}", "en": "⚖️  Scorers: {names}"},
    "lab.evaluating": {"es": "🧪 Evaluando línea base ({baseline}) y candidata ({candidate})...",
                       "en": "🧪 Evaluating baseline ({baseline}) and candidate ({candidate})..."},
    "lab.evaluated": {"es": "   ✅ {version}: run {run_id}", "en": "   ✅ {version}: run {run_id}"},
    "lab.summary_header": {"es": "📊 Comparación pareada (candidata - línea base):",
                           "en": "📊 Paired comparison (candidate - baseline):"},
    "lab.regressed_header": {"es": "🔻 Preguntas que empeoraron (lo que el promedio no muestra):",
                             "en": "🔻 Questions that got worse (what the average does not show):"},
    "lab.none": {"es": "   ninguna 🎉", "en": "   none 🎉"},
    "lab.traces": {"es": "     trazas: base={baseline} candidata={candidate}",
                   "en": "     traces: baseline={baseline} candidate={candidate}"},
    "lab.saved_ids": {"es": "💾 IDs guardados en {file} (los usa test_gate.py)",
                      "en": "💾 IDs saved to {file} (used by test_gate.py)"},
    "lab.ui_compare": {"es": "👀 En la UI: Experiments → {experiment} → Evaluation runs → selecciona ambos "
                             "runs → Compare (fija la línea base y filtra los ejemplos en regresión).",
                       "en": "👀 In the UI: Experiments → {experiment} → Evaluation runs → select both "
                             "runs → Compare (pin the baseline and filter to regressed examples)."},
    "lab.gate_fail": {"es": "🚫 Quality gate: NO pasa", "en": "🚫 Quality gate: FAILED"},
    "lab.gate_pass": {"es": "✅ Quality gate: pasa (mejora significativa en {improve}, sin regresiones en {guard})",
                      "en": "✅ Quality gate: passed (significant improvement on {improve}, no regressions on {guard})"},
    "lab.gate_regression": {"es": "regresión en '{scorer}': {baseline:.2f} → {candidate:.2f} | {question}",
                            "en": "regression on '{scorer}': {baseline:.2f} → {candidate:.2f} | {question}"},

    # step3_online.py
    "lab.no_scorers": {"es": "   (no hay scorers registrados)", "en": "   (no registered scorers)"},
    "lab.scorer_started": {"es": "   ▶️  {name} activo (nivel {level}, sample_rate={rate})",
                           "en": "   ▶️  {name} active ({level} level, sample_rate={rate})"},
    "lab.status_header": {"es": "📋 Estado:", "en": "📋 Status:"},
    "lab.online_next": {"es": "➡️  Ahora genera tráfico nuevo: python labs/mlflow_318/step1_traffic.py --repeat 1\n"
                              "   Las evaluaciones llegan en segundos o minutos (pestaña Assessments de cada traza).",
                        "en": "➡️  Now generate new traffic: python labs/mlflow_318/step1_traffic.py --repeat 1\n"
                              "   Assessments arrive within seconds or minutes (Assessments tab of each trace)."},
    "lab.online_hint": {"es": "El servidor no está configurado para ejecutar scorers de código. Reinícialo con Docker:\n"
                              "    ./labs/mlflow_318/start_server.sh          # (sin --no-docker)\n"
                              "que exporta:\n"
                              "    MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true\n"
                              "    MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor\n"
                              "    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND=docker",
                        "en": "The server is not set up to run code-based scorers. Restart it with Docker:\n"
                              "    ./labs/mlflow_318/start_server.sh          # (without --no-docker)\n"
                              "which exports:\n"
                              "    MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS=true\n"
                              "    MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor\n"
                              "    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND=docker"},

    # step4_issues.py
    "lab.no_issues": {"es": "   No hay issues abiertos. Lanza la detección desde la pestaña Issues de la UI\n"
                            "   (o crea uno con --from-scorer <nombre>).",
                      "en": "   No open issues. Run detection from the Issues tab in the UI\n"
                            "   (or create one with --from-scorer <name>)."},
    "lab.issue_meta": {"es": "   categorías: {categories} | trazas: {n}", "en": "   categories: {categories} | traces: {n}"},
    "lab.error_spans": {"es": " — spans con error: {spans}", "en": " — failing spans: {spans}"},
    "lab.report_title": {"es": "# Issues abiertos del asistente de RR.HH.", "en": "# Open issues of the HR assistant"},
    "lab.report_server": {"es": "Servidor MLflow: `{uri}` · experimento: `{experiment}` (id `{id}`)",
                          "en": "MLflow server: `{uri}` · experiment: `{experiment}` (id `{id}`)"},
    "lab.report_skill": {"es": "Usa el skill `genaiops-hr-triage` para investigar y corregir cada issue.",
                         "en": "Use the `genaiops-hr-triage` skill to investigate and fix each issue."},
    "lab.report_categories": {"es": "- Categorías: {categories}", "en": "- Categories: {categories}"},
    "lab.report_traces": {"es": "- Trazas afectadas: {n}", "en": "- Affected traces: {n}"},
    "lab.report_causes": {"es": "**Causas probables:**", "en": "**Likely root causes:**"},
    "lab.report_examples": {"es": "**Trazas de ejemplo:**", "en": "**Sample traces:**"},
    "lab.report_saved": {"es": "📝 Reporte para tu agente de código: {path}", "en": "📝 Report for your coding agent: {path}"},
    "lab.report_tip": {"es": "   Tip: en la UI, cada issue tiene 'Fix this issue' → Copy con un prompt listo.",
                       "en": "   Tip: in the UI, every issue has 'Fix this issue' → Copy with a ready-made prompt."},
    "lab.no_failing": {"es": "   ✅ Ninguna traza falla '{scorer}'. ¿Ya activaste el paso 3 y generaste tráfico nuevo?",
                       "en": "   ✅ No trace fails '{scorer}'. Did you turn on step 3 and generate new traffic?"},
    "lab.issue_name": {"es": "Falla recurrente del scorer '{scorer}'", "en": "Recurring failure of scorer '{scorer}'"},
    "lab.issue_description": {"es": "{n} trazas de producción reprobaron el scorer de código '{scorer}'. Ejemplos: {examples}",
                              "en": "{n} production traces failed the code-based scorer '{scorer}'. Examples: {examples}"},
    "lab.issue_rationale": {"es": "La traza reprobó el scorer '{scorer}'", "en": "The trace failed the scorer '{scorer}'"},
    "lab.issue_created": {"es": "   🐞 Issue creado: {id} ({n} trazas enlazadas)", "en": "   🐞 Issue created: {id} ({n} linked traces)"},

    # step5_skills.py
    "lab.skill_registered": {"es": "📚 Registrado: {name} v{version}", "en": "📚 Registered: {name} v{version}"},
    "lab.skill_alias": {"es": "🏷️  Alias @{alias} → v{version}", "en": "🏷️  Alias @{alias} → v{version}"},
    "lab.skill_search": {"es": "🔎 Skills activos en el registro:", "en": "🔎 Active skills in the registry:"},
    "lab.skill_installed": {"es": "⬇️  {uri} instalado en {path}", "en": "⬇️  {uri} installed at {path}"},
    "lab.skill_try": {"es": "   Abre Claude Code en este repo y pide: \"usa el skill genaiops-hr-triage para "
                            "resolver los issues de labs/mlflow_318/issues_for_agent.md\"",
                      "en": "   Open Claude Code in this repo and ask: \"use the genaiops-hr-triage skill to "
                            "fix the issues in labs/mlflow_318/issues_for_agent.md\""},
}


def register() -> None:
    add_messages(LAB_MESSAGES)
