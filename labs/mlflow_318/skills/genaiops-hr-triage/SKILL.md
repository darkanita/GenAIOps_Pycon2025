---
name: genaiops-hr-triage
description: Investigates and fixes quality issues of the HR assistant (GenAIOps_Pycon2025 repo) from MLflow traces, and proves the fix with compare_evaluations before closing the issue. Use it when asked to fix an MLflow issue, an evaluation regression or a failure seen in production.
---

# HR assistant issue triage

Flow: **issue → evidence in traces → cause → smallest change → compared evaluation → close the issue**.
Never call an issue resolved without a comparison that proves it.
Reply to the user in the language they write in (the repo works in Spanish and English: `APP_LANG=es|en`).

## 0. Connection

```bash
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
```
Experiments: `lab318_production` (traces and issues), `lab318_evaluations` (evaluations).
For the real chatbot: `chatbot_rrhh` (traces) and `rag_evaluation` (evaluations).

## 1. Read the issue and its evidence

```bash
python labs/mlflow_318/step4_issues.py          # lists open issues and writes issues_for_agent.md
```
For a single trace:
```python
import mlflow
t = mlflow.get_trace("<trace_id>")
for s in t.data.spans:
    print(s.span_type, s.name, s.status.status_code, str(s.outputs)[:200])
```

## 2. Locate the cause from the failing span

| Symptom in the trace | Where to look |
|---|---|
| `RETRIEVER` with no or irrelevant documents | `app/rag_pipeline.py` (k, chunking) / `mock_agent.search_documents` |
| `TOOL` with an error (timeout, exception) | the tool; add error handling and a message for the user |
| Made-up answer when there is no context | prompt in `app/prompts/` (es) or `app/prompts/en/` (en) / `mock_agent._compose` |
| Long answer, no source, or wrong language | prompt of the active version |
| Many input tokens (cost) | context sent to the LLM |
| Personal data echoed back | answer post-processing / prompt instructions |

## 3. Make the smallest change

- Prompts: create a **new** version file and register it with `python -m app.prompt_registry`
  (never edit the version in production).
- Code: change only the component the trace points to.

## 4. Prove the fix (mandatory)

```bash
# simulated agent
python labs/mlflow_318/step2_compare.py --baseline v1 --candidate <new> --gate
# real RAG (prompt name: hr_assistant with APP_LANG=en, asistente_rrhh with APP_LANG=es)
python labs/mlflow_318/step2_compare.py --agent rag \
  --baseline prompts:/hr_assistant@production --candidate prompts:/hr_assistant/<N> --gate
```
The gate requires a significant improvement (`assert_improved`) **and** no question worse on
`covers_key_facts` and `correct_abstention`. If any row got worse, review those traces before
moving on: a higher average is not enough.

## 5. Close the loop

```bash
python labs/mlflow_318/step4_issues.py --resolve <ISSUE_ID>
python -m app.prompt_registry --promote <N>     # only if the gate passed and a human approves
```

Report to the user: the issue, the cause (with trace IDs), the change, the
`compare_evaluations` table and the rows that changed. Ask for confirmation before promoting to production.
