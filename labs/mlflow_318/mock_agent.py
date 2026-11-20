# labs/mlflow_318/mock_agent.py
"""
Asistente de RR.HH. *simulado*: no llama a ningún LLM ni necesita API key.

Sirve para practicar las novedades de MLflow 3.18 sin gastar cuota: produce
trazas realistas (agente → retriever → herramienta → LLM) con fallas
"sembradas" a propósito, para que la detección automática de issues tenga
algo que encontrar y la comparación de evaluaciones tenga algo que comparar.

Versiones del agente:
    v1  "verboso"   respuestas largas, sin citar la fuente, inventa políticas cuando
                    no sabe, repite datos personales del usuario y envía todo el
                    corpus al LLM (muchos tokens).
    v2  "conciso"   cita la fuente y se abstiene cuando no sabe... pero recorta la
                    respuesta a la primera oración y pierde datos clave en las
                    preguntas con varias partes. Mejora el promedio, empeora filas.
    v3  "corregido" conciso, cita la fuente, se abstiene y conserva todos los datos.

Fallas que aparecen en las trazas de v1 (las que debería encontrar Issue Detection):
    - Políticas inventadas en preguntas fuera de alcance         (correctness)
    - La herramienta `consultar_saldo_vacaciones` falla siempre  (execution)
    - Respuestas lentas en preguntas de comparación              (latency)
    - ~10x más tokens de entrada que v2/v3                       (cost)
    - Repite el correo o teléfono que escribió el usuario        (safety)
"""

import re
import time
import unicodedata

import mlflow
from mlflow.entities import Document, SpanType
from mlflow.tracing.constant import SpanAttributeKey

VERSIONES = ("v1", "v2", "v3")
MODELO_SIMULADO = "mock-llm-rrhh"

# Fragmentos (resumidos al español) de los PDFs de data/pdfs/
CORPUS = [
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "texto": "Northwind Health Plus es un plan integral. Cubre servicios médicos, de visión y "
                 "dentales, además de medicamentos recetados genéricos, de marca y especializados. "
                 "También cubre salud mental y abuso de sustancias, atención preventiva y servicios "
                 "de emergencia dentro y fuera de la red.",
        "temas": "health plus plan integral cubre cobertura beneficios",
    },
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "texto": "Northwind Standard es un plan básico. No cubre servicios de emergencia, salud "
                 "mental y abuso de sustancias, ni servicios fuera de la red. En medicamentos "
                 "cubre genéricos y de marca, pero no los especializados.",
        "temas": "standard health plus plan basico comparacion medicamentos red fuera",
    },
    {
        "doc": "Benefit_Options.pdf", "page": 3,
        "texto": "Health Plus cubre exámenes de visión, anteojos y lentes de contacto, además de "
                 "exámenes, limpiezas y calzas dentales. Northwind Standard solo cubre exámenes "
                 "de visión y anteojos.",
        "temas": "servicios vision dentales dental anteojos lentes cada plan",
    },
    {
        "doc": "PerksPlus.pdf", "page": 3,
        "texto": "PerksPlus permite reembolsar hasta 1000 dólares al año en actividades de "
                 "bienestar físico. Incluye membresías de gimnasio, entrenamiento personal, clases "
                 "de yoga o pilates, equipo deportivo y clases grupales.",
        "temas": "perksplus incluye programa monto maximo reembolsa reembolso yoga gimnasio",
    },
    {
        "doc": "PerksPlus.pdf", "page": 3,
        "texto": "PerksPlus también cubre lecciones de surf, esquí y snowboard, buceo y equitación.",
        "temas": "perksplus clases lecciones surf esqui buceo",
    },
    {
        "doc": "PerksPlus.pdf", "page": 4,
        "texto": "PerksPlus no cubre gastos ajenos a la actividad física, tratamientos médicos, "
                 "viajes no relacionados con un programa físico, ni comida o suplementos.",
        "temas": "perksplus reembolsa suplementos alimenticios comida excluye",
    },
    {
        "doc": "employee_handbook.pdf", "page": 3,
        "texto": "Las evaluaciones de desempeño se realizan de forma anual con tu supervisor. "
                 "Recibirás un resumen escrito con tu calificación, retroalimentación y metas "
                 "para el siguiente año.",
        "temas": "evaluaciones desempeno frecuencia realizan",
    },
    {
        "doc": "employee_handbook.pdf", "page": 4,
        "texto": "Si eres testigo o víctima de violencia laboral, notifica de inmediato a tu "
                 "supervisor o a un representante de Recursos Humanos.",
        "temas": "violencia trabajo testigo victima acoso",
    },
    {
        "doc": "employee_handbook.pdf", "page": 2,
        "texto": "Los empleados pueden consultar su saldo de vacaciones en el sistema de RR.HH.",
        "temas": "saldo vacaciones dias quedan",
    },
]

STOPWORDS = set(
    "que cual cuales como con de del el la las los un una unos unas y o en es son se por para "
    "mi me tu su sus al lo le les si no mas segun debo hacer tiene tienen cubre cuanto "
    "cuantos cuantas soy hay esta este esto puedo quiero".split()
)
PALABRAS_INGLES = {"what", "does", "how", "is", "the", "can", "do", "cover", "my"}
PATRON_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d")


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _tokens(texto: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _normalizar(texto)) if t not in STOPWORDS}


@mlflow.trace(name="buscar_documentos", span_type=SpanType.RETRIEVER)
def buscar_documentos(pregunta: str, k: int = 2) -> list[Document]:
    consulta = _tokens(pregunta)
    puntajes = []
    for i, chunk in enumerate(CORPUS):
        claves = _tokens(chunk["texto"] + " " + chunk["temas"])
        puntaje = len(consulta & claves)
        if puntaje >= 2:
            puntajes.append((puntaje, -i, chunk))
    puntajes.sort(reverse=True)
    return [
        Document(
            page_content=c["texto"],
            metadata={"doc_uri": f"data/pdfs/{c['doc']}", "page": c["page"]},
            id=f"chunk-{-neg_i}",
        )
        for _, neg_i, c in puntajes[:k]
    ]


@mlflow.trace(name="consultar_saldo_vacaciones", span_type=SpanType.TOOL)
def consultar_saldo_vacaciones(user_id: str) -> dict:
    # Falla sembrada: el "sistema de RR.HH." nunca responde a tiempo.
    time.sleep(0.3)
    raise TimeoutError(f"El sistema de RR.HH. no respondió en 30 s (user_id={user_id})")


def _llm_simulado(mensajes: list[dict], respuesta: str) -> str:
    """Span de tipo CHAT_MODEL con conteo de tokens aproximado (4 caracteres ≈ 1 token)."""
    with mlflow.start_span(name="llm", span_type=SpanType.CHAT_MODEL) as span:
        span.set_inputs({"model": MODELO_SIMULADO, "messages": mensajes})
        entrada = sum(len(m["content"]) for m in mensajes) // 4
        salida = len(respuesta) // 4
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, {
            "input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida,
        })
        span.set_outputs({"choices": [{"message": {"role": "assistant", "content": respuesta}}]})
    return respuesta


def _fuente(doc: Document) -> str:
    return f"Fuente: {doc.metadata['doc_uri'].split('/')[-1]}, p. {doc.metadata['page']}"


def _primera_oracion(texto: str) -> str:
    return re.split(r"(?<=\.)\s", texto, maxsplit=1)[0]


ABSTENCION = "No tengo información suficiente en los documentos para responder esa pregunta."


def _componer(version: str, pregunta: str, docs: list[Document]) -> tuple[list[dict], str]:
    """Devuelve (mensajes enviados al LLM, respuesta) según la versión del agente."""
    sistema = {"role": "system", "content": "Eres un asistente de RR.HH. de Contoso Electronics."}

    if version == "v1":
        # Envía TODO el corpus como contexto (costo alto) y no cita la fuente.
        contexto = "\n".join(c["texto"] for c in CORPUS) * 6
        mensajes = [sistema, {"role": "user", "content": f"{contexto}\n\nPregunta: {pregunta}"}]
        pii = PATRON_PII.findall(pregunta)
        if not docs:
            # Falla sembrada: inventa una política en lugar de abstenerse.
            cuerpo = ("Según la política vigente de Contoso Electronics, tienes derecho a 12 "
                      "semanas pagadas y puedes solicitarlo directamente con tu jefe inmediato.")
        else:
            cuerpo = " ".join(d.page_content for d in docs)
        if set(pregunta.lower().split()) & PALABRAS_INGLES:
            cuerpo = "Sure! Here is what I found: " + cuerpo
        relleno = ("Recuerda que en Contoso Electronics valoramos la calidad, la integridad, la "
                   "innovación y el trabajo en equipo, y que Recursos Humanos siempre está "
                   "disponible para acompañarte en cualquier inquietud que tengas sobre tus "
                   "beneficios, tus políticas o tu desarrollo profesional dentro de la compañía.")
        respuesta = f"¡Hola! Muchas gracias por tu pregunta. {cuerpo} {relleno}"
        if pii:
            # Falla sembrada: repite datos personales del usuario en la respuesta.
            respuesta += f" Te enviaremos más detalles a {pii[0]}."
        return mensajes, respuesta

    # v2 y v3: solo los fragmentos recuperados como contexto
    contexto = "\n".join(d.page_content for d in docs)
    mensajes = [sistema, {"role": "user", "content": f"{contexto}\n\nPregunta: {pregunta}"}]
    if not docs:
        return mensajes, ABSTENCION
    principal = docs[0]
    if version == "v2":
        # Bug sembrado: recorta a la primera oración y pierde datos clave.
        cuerpo = _primera_oracion(principal.page_content)
    else:
        cuerpo = principal.page_content
    return mensajes, f"{cuerpo}\n\n{_fuente(principal)}"


_config = {"version": "v1", "environment": "development"}


def configurar(version: str = "v1", environment: str = "development") -> None:
    """
    Elige la versión del agente y el entorno para las siguientes llamadas.

    La versión NO es un argumento de `responder` a propósito: compare_evaluations
    empareja las filas de dos runs por el hash de las entradas de la traza; si la
    versión fuera parte de las entradas, ninguna fila de v1 coincidiría con su par de v2.
    """
    if version not in VERSIONES:
        raise ValueError(f"version debe ser una de {VERSIONES}")
    _config.update(version=version, environment=environment)


@mlflow.trace(name="asistente_rrhh", span_type=SpanType.AGENT)
def responder(pregunta: str, session_id: str | None = None, user_id: str | None = None) -> str:
    version, environment = _config["version"], _config["environment"]

    metadata = {}
    if session_id:
        metadata["mlflow.trace.session"] = session_id
    if user_id:
        metadata["mlflow.trace.user"] = user_id
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(
            metadata=metadata or None,
            tags={"environment": environment, "agent_version": version},
        )

    normalizada = _normalizar(pregunta)

    # Herramienta: saldo de vacaciones del usuario
    if "saldo" in normalizada or "me quedan" in normalizada:
        try:
            consultar_saldo_vacaciones(user_id or "anonimo")
        except TimeoutError:
            mensajes = [{"role": "user", "content": pregunta}]
            return _llm_simulado(mensajes, "Lo siento, en este momento no pude consultar tu "
                                           "saldo de vacaciones. Intenta más tarde.")

    docs = buscar_documentos(pregunta)

    # Falla sembrada de latencia: v1 "razona" mucho en preguntas de comparación
    if version == "v1" and ("compar" in normalizada or "diferencia" in normalizada
                            or "cada plan" in normalizada):
        time.sleep(1.5)

    mensajes, respuesta = _componer(version, pregunta, docs)
    return _llm_simulado(mensajes, respuesta)


if __name__ == "__main__":
    import sys

    configurar(sys.argv[1] if len(sys.argv) > 1 else "v3")
    q = " ".join(sys.argv[2:]) or "¿Qué incluye el programa PerksPlus?"
    print(responder(q))
