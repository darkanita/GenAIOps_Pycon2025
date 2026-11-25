# labs/mlflow_318/mock_agent.py
"""
A *simulated* HR assistant: it calls no LLM and needs no API key.
Un asistente de RR.HH. *simulado*: no llama a ningún LLM ni necesita API key.

It produces realistic traces (agent → retriever → tool → LLM) with failures
planted on purpose, so Issue Detection has something to find and the
evaluation comparison has something to compare. It answers in Spanish or in
English depending on APP_LANG.

Agent versions:
    v1  "verbose"   long answers without a source, makes up policies when it does not
                    know, echoes the user's personal data and sends the whole corpus
                    to the LLM (many tokens).
    v2  "concise"   cites its source and abstains when it does not know... but cuts the
                    answer to its first sentence and loses key facts on multi-part
                    questions. Better average, worse rows.
    v3  "fixed"     concise, cites its source, abstains and keeps every fact.

Failures in v1 traces (what Issue Detection should find):
    - Made-up policies on out-of-scope questions        (correctness)
    - The `check_vacation_balance` tool always fails    (execution)
    - Slow answers on comparison questions              (latency)
    - ~20x more input tokens than v2/v3                 (cost)
    - Echoes the user's email or phone number           (safety)
    - Answers in the other language when asked in it    (adherence)
"""

import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for app.i18n

import mlflow
from mlflow.entities import Document, SpanType
from mlflow.tracing.constant import SpanAttributeKey

from app.i18n import LANG

VERSIONS = ("v1", "v2", "v3")
SIMULATED_MODEL = "mock-llm-hr"

# Chunks of the PDFs in data/pdfs/, summarized in both languages
CORPUS = [
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "es": "Northwind Health Plus es un plan integral. Cubre servicios médicos, de visión y "
              "dentales, además de medicamentos recetados genéricos, de marca y especializados. "
              "También cubre salud mental y abuso de sustancias, atención preventiva y servicios "
              "de emergencia dentro y fuera de la red.",
        "en": "Northwind Health Plus is a comprehensive plan. It covers medical, vision and dental "
              "services, plus generic, brand-name and specialty prescription drugs. It also covers "
              "mental health and substance abuse, preventive care and emergency services both "
              "in-network and out-of-network.",
        "topics": "health plus plan integral comprehensive cubre covers cobertura coverage beneficios benefit",
    },
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "es": "Northwind Standard es un plan básico. No cubre servicios de emergencia, salud "
              "mental y abuso de sustancias, ni servicios fuera de la red. En medicamentos "
              "cubre genéricos y de marca, pero no los especializados.",
        "en": "Northwind Standard is a basic plan. It does not cover emergency services, mental "
              "health and substance abuse, or out-of-network services. For prescription drugs it "
              "covers generic and brand-name drugs, but not specialty drugs.",
        "topics": "standard health plus plan basico basic comparacion compared medicamentos drugs "
                  "red network fuera out",
    },
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "es": "Health Plus cubre exámenes de visión, anteojos y lentes de contacto, además de "
              "exámenes, limpiezas y calzas dentales. Northwind Standard solo cubre exámenes "
              "de visión y anteojos.",
        "en": "Health Plus covers vision exams, glasses and contact lenses, plus dental exams, "
              "cleanings and fillings. Northwind Standard only covers vision exams and glasses.",
        "topics": "servicios services vision dentales dental anteojos glasses lentes lenses cada each plan",
    },
    {
        "doc": "PerksPlus.pdf", "page": 3,
        "es": "PerksPlus permite reembolsar hasta 1000 dólares en actividades de "
              "bienestar físico. Incluye membresías de gimnasio, entrenamiento personal, clases "
              "de yoga o pilates, equipo deportivo y clases grupales.",
        "en": "PerksPlus reimburses up to 1000 dollars for physical wellness activities. "
              "It includes gym memberships, personal training, yoga or Pilates classes, fitness "
              "equipment and group classes.",
        "topics": "perksplus incluye include programa program monto amount maximo maximum "
                  "reembolsa reimburses reembolso yoga gimnasio gym",
    },
    {
        "doc": "PerksPlus.pdf", "page": 3,
        "es": "PerksPlus también cubre lecciones de surf, esquí y snowboard, buceo y equitación.",
        "en": "PerksPlus also covers surfing, skiing and snowboarding, scuba diving and "
              "horseback riding lessons.",
        "topics": "perksplus clases lecciones lessons surf surfing esqui skiing buceo diving",
    },
    {
        "doc": "PerksPlus.pdf", "page": 4,
        "es": "PerksPlus no cubre gastos ajenos a la actividad física, tratamientos médicos, "
              "viajes no relacionados con un programa físico, ni comida o suplementos.",
        "en": "PerksPlus does not cover expenses unrelated to physical activity, medical "
              "treatments, travel unrelated to a fitness program, or food or supplements.",
        "topics": "perksplus reembolsa reimburse suplementos supplements alimenticios dietary "
                  "comida food excluye excludes",
    },
    {
        "doc": "employee_handbook.pdf", "page": 4,
        "es": "Las evaluaciones de desempeño se realizan de forma anual con tu supervisor. "
              "Recibirás un resumen escrito con tu calificación, retroalimentación y metas "
              "para el siguiente año.",
        "en": "Performance reviews are held annually with your supervisor. You will receive a "
              "written summary with your rating, feedback and goals for the coming year.",
        "topics": "evaluaciones desempeno performance reviews frecuencia often realizan held",
    },
    {
        "doc": "employee_handbook.pdf", "page": 6,
        "es": "Si eres testigo o víctima de violencia laboral, notifica de inmediato a tu "
              "supervisor o a un representante de Recursos Humanos.",
        "en": "If you witness or experience workplace violence, immediately notify your "
              "supervisor or a Human Resources representative.",
        "topics": "violencia violence trabajo work workplace testigo witness victima victim acoso",
    },
    {
        "doc": "employee_handbook.pdf", "page": 2,
        "es": "Los empleados pueden consultar su saldo de vacaciones en el sistema de RR.HH.",
        "en": "Employees can check their vacation balance in the HR system.",
        "topics": "saldo balance vacaciones vacation dias days quedan left",
    },
]

STOPWORDS = set(
    # es
    "que cual cuales como con de del el la las los un una unos unas y o en es son se por para "
    "mi me tu su sus al lo le les si no mas segun debo hacer tiene tienen cubre cuanto "
    "cuantos cuantas soy hay esta este esto puedo quiero "
    # en
    "what which how does do is are the a an of to in on for and or my me i if at by it "
    "with about according should can cover covers much many"
    .split()
)
OTHER_LANGUAGE_WORDS = {"es": {"what", "does", "how", "the", "can", "cover", "my"},
                        "en": {"qué", "cómo", "cuál", "puedo", "tengo", "cubre", "mi"}}
PII_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d")

TEXTS = {
    "es": {
        "abstain": "No tengo información suficiente en los documentos para responder esa pregunta.",
        "source": "Fuente: {doc}, p. {page}",
        "system": "Eres un asistente de RR.HH. de Contoso Electronics.",
        "question": "Pregunta",
        "greeting": "¡Hola! Muchas gracias por tu pregunta.",
        "invented": "Según la política vigente de Contoso Electronics, tienes derecho a 12 semanas "
                    "pagadas y puedes solicitarlo directamente con tu jefe inmediato.",
        "other_language": "Sure! Here is what I found:",
        "filler": "Recuerda que en Contoso Electronics valoramos la calidad, la integridad, la "
                  "innovación y el trabajo en equipo, y que Recursos Humanos siempre está "
                  "disponible para acompañarte en cualquier inquietud que tengas sobre tus "
                  "beneficios, tus políticas o tu desarrollo profesional dentro de la compañía.",
        "pii_echo": "Te enviaremos más detalles a {pii}.",
        "tool_failed": "Lo siento, en este momento no pude consultar tu saldo de vacaciones. "
                       "Intenta más tarde.",
        "balance_words": ("saldo", "me quedan"),
        "comparison_words": ("compar", "diferencia", "cada plan"),
    },
    "en": {
        "abstain": "I do not have enough information in the documents to answer that question.",
        "source": "Source: {doc}, p. {page}",
        "system": "You are an HR assistant for Contoso Electronics.",
        "question": "Question",
        "greeting": "Hi! Thank you so much for your question.",
        "invented": "According to Contoso Electronics' current policy, you are entitled to 12 paid "
                    "weeks and you can request it directly from your manager.",
        "other_language": "¡Claro! Esto es lo que encontré:",
        "filler": "Remember that at Contoso Electronics we value quality, integrity, innovation "
                  "and teamwork, and that Human Resources is always available to support you "
                  "with any question you may have about your benefits, our policies or your "
                  "professional growth within the company.",
        "pii_echo": "We will send more details to {pii}.",
        "tool_failed": "Sorry, I could not check your vacation balance right now. Please try "
                       "again later.",
        "balance_words": ("balance", "days left"),
        "comparison_words": ("compar", "difference", "each plan"),
    },
}
TX = TEXTS[LANG]
ABSTAIN = TX["abstain"]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _normalize(text)) if w not in STOPWORDS}


@mlflow.trace(name="search_documents", span_type=SpanType.RETRIEVER)
def search_documents(question: str, k: int = 2) -> list[Document]:
    query = _tokens(question)
    scored = []
    for i, chunk in enumerate(CORPUS):
        keys = _tokens(chunk[LANG] + " " + chunk["topics"])
        score = len(query & keys)
        if score >= 2:
            scored.append((score, -i, chunk))
    scored.sort(reverse=True)
    return [
        Document(
            page_content=c[LANG],
            metadata={"doc_uri": f"data/pdfs/{c['doc']}", "page": c["page"]},
            id=f"chunk-{-neg_i}",
        )
        for _, neg_i, c in scored[:k]
    ]


@mlflow.trace(name="check_vacation_balance", span_type=SpanType.TOOL)
def check_vacation_balance(user_id: str) -> dict:
    # Planted failure: the "HR system" never answers in time.
    time.sleep(0.3)
    raise TimeoutError(f"HR system did not respond within 30 s (user_id={user_id})")


def _simulated_llm(messages: list[dict], answer: str) -> str:
    """CHAT_MODEL span with an approximate token count (4 characters ≈ 1 token)."""
    with mlflow.start_span(name="llm", span_type=SpanType.CHAT_MODEL) as span:
        span.set_inputs({"model": SIMULATED_MODEL, "messages": messages})
        input_tokens = sum(len(m["content"]) for m in messages) // 4
        output_tokens = len(answer) // 4
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        })
        span.set_outputs({"choices": [{"message": {"role": "assistant", "content": answer}}]})
    return answer


def _source(doc: Document) -> str:
    return TX["source"].format(doc=doc.metadata["doc_uri"].split("/")[-1], page=doc.metadata["page"])


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=\.)\s", text, maxsplit=1)[0]


def _compose(version: str, question: str, docs: list[Document]) -> tuple[list[dict], str]:
    """Returns (messages sent to the LLM, answer) for the given agent version."""
    system = {"role": "system", "content": TX["system"]}

    if version == "v1":
        # Sends the WHOLE corpus as context (high cost) and does not cite the source.
        context = "\n".join(c[LANG] for c in CORPUS) * 6
        messages = [system, {"role": "user", "content": f"{context}\n\n{TX['question']}: {question}"}]
        pii = PII_PATTERN.findall(question)
        # Planted failure: makes up a policy instead of abstaining.
        body = " ".join(d.page_content for d in docs) if docs else TX["invented"]
        if set(re.findall(r"\w+", question.lower())) & OTHER_LANGUAGE_WORDS[LANG]:
            body = f"{TX['other_language']} {body}"
        answer = f"{TX['greeting']} {body} {TX['filler']}"
        if pii:
            # Planted failure: echoes the user's personal data.
            answer += " " + TX["pii_echo"].format(pii=pii[0])
        return messages, answer

    # v2 and v3: only the retrieved chunks as context
    context = "\n".join(d.page_content for d in docs)
    messages = [system, {"role": "user", "content": f"{context}\n\n{TX['question']}: {question}"}]
    if not docs:
        return messages, ABSTAIN
    top = docs[0]
    # v2 planted bug: keeps only the first sentence and loses key facts.
    body = _first_sentence(top.page_content) if version == "v2" else top.page_content
    return messages, f"{body}\n\n{_source(top)}"


_config = {"version": "v1", "environment": "development"}


def configure(version: str = "v1", environment: str = "development") -> None:
    """
    Picks the agent version and environment for the following calls.

    The version is deliberately NOT an argument of `answer`: compare_evaluations
    pairs rows from two runs by the hash of the trace inputs; if the version were
    part of the inputs, no v1 row would ever match its v2 pair.
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    _config.update(version=version, environment=environment)


@mlflow.trace(name="hr_assistant", span_type=SpanType.AGENT)
def answer(question: str, session_id: str | None = None, user_id: str | None = None) -> str:
    version, environment = _config["version"], _config["environment"]

    metadata = {}
    if session_id:
        metadata["mlflow.trace.session"] = session_id
    if user_id:
        metadata["mlflow.trace.user"] = user_id
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(
            metadata=metadata or None,
            tags={"environment": environment, "agent_version": version, "language": LANG},
        )

    normalized = _normalize(question)

    # Tool: the user's vacation balance
    if any(w in normalized for w in TX["balance_words"]):
        try:
            check_vacation_balance(user_id or "anonymous")
        except TimeoutError:
            return _simulated_llm([{"role": "user", "content": question}], TX["tool_failed"])

    docs = search_documents(question)

    # Planted latency failure: v1 "thinks" a lot on comparison questions
    if version == "v1" and any(w in normalized for w in TX["comparison_words"]):
        time.sleep(1.5)

    messages, text = _compose(version, question, docs)
    return _simulated_llm(messages, text)


if __name__ == "__main__":
    configure(sys.argv[1] if len(sys.argv) > 1 else "v3")
    default_q = "¿Qué incluye el programa PerksPlus?" if LANG == "es" else "What does the PerksPlus program include?"
    print(answer(" ".join(sys.argv[2:]) or default_q))
