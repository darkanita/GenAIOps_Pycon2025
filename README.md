# 🤖 Chatbot GenAI - Caso de Estudio Recursos Humanos

Este proyecto demuestra cómo construir, evaluar y automatizar un chatbot de tipo RAG (Retrieval Augmented Generation) con buenas prácticas de **GenAIOps**.

**¿Primera vez aquí?** Sigue el [🧭 Recorrido guiado](#-recorrido-guiado-empieza-aquí): en unos 60–90 minutos recorres el repositorio de punta a punta.

| Sección | Para qué |
|---|---|
| [⚡ Inicio rápido](#-inicio-rápido) | Levantar todo con 5 comandos |
| [🧭 Recorrido guiado](#-recorrido-guiado-empieza-aquí) | Aprender el repositorio paso a paso, con ejercicios |
| [🚦 Ciclo de vida GenAIOps](#-ciclo-de-vida-genaiops-aplicado) | Referencia de cada etapa y sus comandos |
| [🔬 Cómo funciona MLflow](#-cómo-funciona-mlflow-en-este-proyecto) | Conceptos, qué se guarda, ciclo de mejora y guía de la UI |
| [🛠️ Solución de problemas](#️-solución-de-problemas) | Errores frecuentes |
| [🎓 Desafío para estudiantes](#-desafío-para-estudiantes) | Retos para practicar |

---

## 🧠 Caso de Estudio

El chatbot responde preguntas sobre beneficios, políticas internas y roles de una empresa ficticia (**Contoso Electronics**), usando como base una colección de documentos PDF internos.

### ✨ Qué incluye

| Capacidad | Cómo se implementa |
|---|---|
| RAG sobre PDFs | LangChain + FAISS, con referencias (documento y página) en cada respuesta |
| Proveedor configurable | **OpenAI** o **Groq** para el LLM; embeddings de OpenAI o locales con FastEmbed |
| Versionamiento de prompts | **MLflow Prompt Registry** con alias (`@production`) |
| Observabilidad | **MLflow Tracing**: cada respuesta es una traza con todos sus pasos |
| Evaluación | `mlflow.genai.evaluate` con **jueces LLM** (corrección, alucinación, relevancia, seguridad, guías propias) |
| Feedback humano | 👍/👎 en la app, guardado sobre la traza de cada respuesta |
| Comparación de versiones | Cada combinación proveedor + modelo + prompt + `k` es una versión de la app en MLflow |
| CI/CD | GitHub Actions evalúa y aplica un *quality gate* en cada cambio |

### 🗺️ Cómo encajan las piezas

```mermaid
flowchart LR
    PDF[📄 PDFs] -->|"chunks + embeddings"| IDX[(FAISS)]
    TXT[📝 app/prompts/*.txt] -->|"prompt_registry"| PR[(MLflow<br/>Prompt Registry)]
    IDX --> RAG[🧠 Cadena RAG]
    PR -->|"@production"| RAG
    LLM[🤖 OpenAI / Groq] --> RAG
    RAG --> UI[💬 Streamlit]
    RAG --> EV[⚖️ run_eval.py<br/>jueces LLM]
    UI -->|"trazas + 👍/👎"| ML[(MLflow<br/>mlflow.db)]
    EV -->|"métricas + razonamientos"| ML
    ML --> DASH[📈 Dashboard / mlflow ui]
    EV --> CI[🔁 GitHub Actions<br/>quality gate]
```

### ⚡ Inicio rápido

```bash
pip install -r requirements.txt
cp .env.example .env                    # pon tu OPENAI_API_KEY, o LLM_PROVIDER=groq + GROQ_API_KEY
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"   # índice
python -m app.prompt_registry           # registra los prompts en MLflow
streamlit run app/main_interface.py     # chatbot + métricas
```

Luego, para evaluar y explorar los resultados:

```bash
python app/run_eval.py                  # evaluación con jueces LLM
mlflow ui --port 5000                   # trazas, prompts, evaluaciones y versiones
```

> 🔄 Para entender qué hace MLflow en cada paso y cómo iterar sobre el prompt, ve a [Cómo funciona MLflow en este proyecto](#-cómo-funciona-mlflow-en-este-proyecto).

> 💻 **GitHub Codespaces**: abre el repo en **Code → Codespaces → Create codespace**. Las dependencias se instalan solas y los puertos `8501` (Streamlit) y `5000` (MLflow) quedan reenviados. Puedes guardar `OPENAI_API_KEY` / `GROQ_API_KEY` como [secrets de Codespaces](https://github.com/settings/codespaces) en lugar de usar `.env`.

---

## 🧭 Recorrido guiado (empieza aquí)

Este recorrido te lleva por todo el repositorio en orden. En cada paso: **haz** algo, **observa** el resultado, **entiende** el concepto y **lee** el archivo que lo implementa. Necesitas dos terminales: una para la app y otra para MLflow.

> Antes de empezar: completa la [preparación del entorno](#1--preparación-del-entorno) (instalar dependencias y crear `.env`). Si usas Groq, revisa [Evaluar con el plan gratuito de Groq](#-evaluar-con-el-plan-gratuito-de-groq).

### Paso 1 · Construir el índice de documentos
**Haz:**
```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```
**Observa:** aparece la carpeta `vectorstore/openai/` o `vectorstore/fastembed/`.
**Entiende:** los PDFs se dividen en fragmentos (*chunks*), cada fragmento se convierte en un vector (*embedding*) y se guarda en un índice FAISS para buscarlos por similitud.
**Lee:** `app/rag_pipeline.py` → `save_vectorstore()` y `app/llm_provider.py` → `get_embeddings()`.

### Paso 2 · Registrar los prompts y abrir MLflow
**Haz:**
```bash
python -m app.prompt_registry
mlflow ui --port 5000        # en otra terminal; abre http://localhost:5000
```
**Observa:** la consola muestra `asistente_rrhh v1`, `v2` y `@production → v1`. En MLflow, menú **Prompts** → `asistente_rrhh`: las dos versiones con sus alias.
**Entiende:** cada archivo de `app/prompts/` es una **versión** del mismo prompt; el alias `@production` dice cuál usa la app.
**Lee:** `app/prompts/*.txt` y `app/prompt_registry.py`.

### Paso 3 · Conversar con el chatbot
**Haz:**
```bash
streamlit run app/main_interface.py
```
Haz clic en una pregunta de ejemplo, abre **📚 Referencias** y califica la respuesta con 👍 o 👎.
**Observa:** la respuesta, los documentos y páginas usados, el tiempo de respuesta y, en la barra lateral, el proveedor, el modelo y el **Prompt resuelto** (`asistente_rrhh/1`).
**Entiende:** es un RAG: busca los fragmentos más parecidos a la pregunta y se los entrega al LLM junto con el prompt.
**Lee:** `app/chat_ui.py` → `render_chat()` y `app/rag_pipeline.py` → `build_chain()`.

### Paso 4 · Inspeccionar la traza de esa respuesta
**Haz:** en MLflow → **Experiments** → `chatbot_rrhh` → **Traces** → abre la traza más reciente.
**Observa:** el árbol de pasos (*spans*): reformulación de la pregunta, el **retriever** con los fragmentos que encontró, el prompt final enviado al LLM y su respuesta, con tokens y latencia. En *Assessments* está tu 👍/👎 como `user_feedback`.
**Entiende:** el **tracing** permite saber *por qué* el bot respondió lo que respondió. Si la respuesta es mala, la traza muestra si falló la búsqueda o el LLM.
**Lee:** `app/tracking.py`.

### Paso 5 · Experimentar con la recuperación
**Haz:** en la barra lateral cambia **Fragmentos a recuperar (k)** a `1` y repite la misma pregunta; luego prueba con `8`.
**Observa:** cambian las referencias y la respuesta. En **Traces**, cada respuesta queda enlazada a una versión distinta de la app (`...-k1`, `...-k8`), visible en **Agent versions**.
**Entiende:** la calidad de un RAG depende tanto de *qué se recupera* como del LLM. Cada combinación proveedor + modelo + prompt + k es una **versión de la app** que se puede comparar.

### Paso 6 · Evaluar con jueces LLM
**Haz:**
```bash
python app/run_eval.py
# Con Groq, para ir más rápido: EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py
```
**Observa:** la consola muestra el promedio de cada juez (`correctness/mean`, `safety/mean`, …). En MLflow → `rag_evaluation` → **Evaluation runs**, abre el run: cada pregunta con el veredicto y el **razonamiento** de cada juez.
**Entiende:** en lugar de revisar respuestas a mano, otros LLM las califican contra la respuesta esperada (`tests/eval_dataset.json`) y contra criterios como "no inventar información".
**Lee:** `app/run_eval.py` → `get_scorers()` y `tests/eval_dataset.json`.

### Paso 7 · Comparar dos versiones del prompt
**Haz:**
```bash
PROMPT_VERSION=prompts:/asistente_rrhh@v2_resumido_directo python app/run_eval.py
streamlit run app/dashboard.py
```
**Observa:** en el dashboard, **Comparación de configuraciones** muestra las dos evaluaciones lado a lado; en **Detalle por pregunta** ves en qué preguntas difieren.
**Entiende:** así se decide con evidencia qué prompt es mejor, en vez de por intuición.
**Lee:** `app/metrics_ui.py`.

### Paso 8 · Promover la mejor versión
**Haz:** si la v2 fue mejor:
```bash
python -m app.prompt_registry --promote 2
```
**Observa:** en tu siguiente pregunta (puede tardar hasta 30 s en notarse) la barra lateral del chat muestra **Prompt resuelto: `asistente_rrhh/2`**, sin reiniciar ni tocar código. En MLflow → **Prompts**, el alias `@production` ahora está en la v2.
**Entiende:** separar *qué versión existe* (registro) de *cuál está en producción* (alias) permite cambiar y revertir de forma segura.

### Paso 9 · Crear tu propia versión
**Haz:** crea `app/prompts/v3_mi_prompt.txt` (debe incluir `{question}` y `{context}`), luego:
```bash
python -m app.prompt_registry          # → asistente_rrhh v3 (@v3_mi_prompt)
PROMPT_VERSION=prompts:/asistente_rrhh/3 python app/run_eval.py
```
**Observa:** la v3 aparece en el registro, en el selector de prompts del chat y en el dashboard junto a las demás.
**Entiende:** este es el [ciclo de mejora](#-el-ciclo-de-mejora) completo: editar → registrar → evaluar → comparar → promover.

### Paso 10 · El quality gate y la automatización
**Haz:**
```bash
pytest tests/test_run_eval.py -v
```
**Observa:** el test pasa o falla según la **última** evaluación (`correctness ≥ 0.8`, `safety = 1.0`).
**Entiende:** en GitHub Actions estos mismos pasos (registrar, indexar, evaluar, verificar) corren en cada push y bloquean cambios que empeoren la calidad.
**Lee:** `tests/test_run_eval.py` y `.github/workflows/eval.yml`.

### 🗺️ Mapa de archivos: qué leer y en qué orden

| Orden | Archivo | Qué aprendes |
|---|---|---|
| 1 | `app/llm_provider.py` | Cómo se elige el LLM, los embeddings y el juez según `.env` |
| 2 | `app/rag_pipeline.py` | Indexación de PDFs y construcción de la cadena RAG |
| 3 | `app/prompt_registry.py` | Registro, alias y carga de prompts en MLflow |
| 4 | `app/tracking.py` | Tracing, versiones de la app y feedback |
| 5 | `app/chat_ui.py` | La interfaz de chat y cómo usa todo lo anterior |
| 6 | `app/run_eval.py` | Evaluación con jueces LLM |
| 7 | `app/metrics_ui.py` | Cómo se leen los resultados de MLflow para el dashboard |
| 8 | `tests/test_run_eval.py` y `.github/workflows/eval.yml` | Quality gate y CI |

`ui_streamlit.py`, `main_interface.py` y `dashboard.py` son solo puntos de entrada que llaman a `chat_ui.py` y `metrics_ui.py`.

---

## 📂 Estructura del Proyecto

```
├── app/
│   ├── ui_streamlit.py           ← interfaz simple del chatbot
│   ├── main_interface.py         ← interfaz combinada con métricas
│   ├── run_eval.py               ← evaluación automática
│   ├── rag_pipeline.py           ← lógica de ingestión y RAG
│   ├── llm_provider.py           ← selección de proveedor (OpenAI / Groq)
│   ├── chat_ui.py                ← componente de chat compartido por las UIs
│   ├── metrics_ui.py             ← vista de métricas (evaluación + feedback)
│   ├── prompt_registry.py        ← versionamiento de prompts en MLflow
│   ├── tracking.py               ← tracing, versión de la app y feedback en MLflow
│   └── prompts/                  ← fuente de las versiones del prompt
│       ├── v1_asistente_rrhh.txt
│       └── v2_resumido_directo.txt
├── data/pdfs/                    ← documentos fuente
├── tests/
│   ├── test_run_eval.py
│   ├── eval_dataset.json         ← dataset de evaluación
│   └── eval_dataset.csv
├── .env.example
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
├── .github/workflows/
│   ├── eval.yml
│   └── test.yml
├── vectorstore/                  ← (generado) índices FAISS por proveedor de embeddings
└── mlflow.db                     ← (generado) base de datos de MLflow: prompts, trazas, evaluaciones
```

---

## 🚦 Ciclo de vida GenAIOps aplicado

### 1. 🧱 Preparación del entorno

```bash
git clone https://github.com/darkanita/GenAIOps_Pycon2025 chatbot-genaiops
cd chatbot-genaiops
conda create -n chatbot-genaiops python=3.10 -y
conda activate chatbot-genaiops
pip install -r requirements.txt
cp .env.example .env  # Agrega tu API KEY de OpenAI o de Groq
```

#### 🔀 Elegir proveedor: OpenAI o Groq

Todo se configura en `.env`, sin tocar código:

| Variable | Para qué sirve | Default |
|---|---|---|
| `LLM_PROVIDER` | proveedor del LLM: `openai` \| `groq` | `openai` |
| `LLM_MODEL` | modelo de chat | `gpt-4o` (OpenAI) / `openai/gpt-oss-120b` (Groq) |
| `OPENAI_API_KEY` / `GROQ_API_KEY` | key del proveedor que uses | — |
| `EMBEDDINGS_PROVIDER` | `openai` \| `fastembed` (local) | `openai` con OpenAI, `fastembed` con Groq |
| `EMBEDDINGS_MODEL` | modelo de embeddings | `text-embedding-ada-002` / `paraphrase-multilingual-MiniLM-L12-v2` |
| `EVAL_MODEL` | juez de `run_eval.py`, formato `proveedor:/modelo` | `openai:/gpt-4o-mini` / `groq:/openai/gpt-oss-20b` |
| `EVAL_SCORERS` | ejecutar solo algunos jueces, separados por coma | todos |
| `PROMPT_VERSION` | prompt a usar (ver paso 3) | `prompts:/asistente_rrhh@production` |
| `RETRIEVER_K` | fragmentos que se envían al LLM en `run_eval.py` | `4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | tamaño y solape de los fragmentos al construir el índice (paso 2) | `512` / `50` |
| `MLFLOW_TRACKING_URI` | servidor de MLflow (opcional) | `./mlflow.db` (SQLite) |

Las variables que dejes vacías o comentadas toman su valor por defecto.

**Usar Groq** (obtén tu key en [console.groq.com](https://console.groq.com/keys)):

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx
```

> ℹ️ Groq no ofrece API de embeddings, así que con Groq los embeddings se calculan **localmente** con [FastEmbed](https://github.com/qdrant/fastembed) (CPU, sin API key; el modelo se descarga la primera vez). Si tienes key de OpenAI también puedes combinar `LLM_PROVIDER=groq` con `EMBEDDINGS_PROVIDER=openai`.
>
> El índice FAISS se guarda en `vectorstore/<proveedor_embeddings>/`, así que **al cambiar de proveedor de embeddings debes regenerarlo** (paso 2).

---

### 2. 🔍 Ingesta y vectorización de documentos

Procesa los PDFs y genera el índice vectorial:

```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```

Esto:
- Divide los documentos en chunks (`CHUNK_SIZE` y `CHUNK_OVERLAP` del `.env`; por defecto 512 y 50)
- Genera embeddings con OpenAI o FastEmbed (según `EMBEDDINGS_PROVIDER`)
- Guarda el índice vectorial en `vectorstore/openai/` o `vectorstore/fastembed/`
- Registra los parámetros en **MLflow**

🔧 Para probar otro chunking, cambia `CHUNK_SIZE`/`CHUNK_OVERLAP` en `.env` y vuelve a construir el índice (también puedes pasarlos directamente):
```python
save_vectorstore(chunk_size=1024, chunk_overlap=100)
```
> Si cambias el chunking, mantén los mismos valores en `.env` al correr `run_eval.py` para que la evaluación registre la configuración correcta.

♻️ Para reutilizarlo directamente:
```python
vectordb = load_vectorstore_from_disk()
```

---

### 3. 📝 Versionamiento de prompts (MLflow Prompt Registry)

Los archivos de `app/prompts/` se registran como **versiones** del prompt `asistente_rrhh` en MLflow:

```bash
python -m app.prompt_registry
# 📝 v1_asistente_rrhh.txt → asistente_rrhh v1 (@v1_asistente_rrhh)
# 📝 v2_resumido_directo.txt → asistente_rrhh v2 (@v2_resumido_directo)
# 🚀 @production → v1
```

- Cada versión guarda su plantilla, un mensaje de commit y la etiqueta `source_file`.
- Si editas un archivo y vuelves a ejecutar el comando, se crea una **versión nueva** (solo si el contenido cambió).
- El alias `@production` indica qué versión usa la app. Para promover otra versión **sin tocar código**:

```bash
python -m app.prompt_registry --promote 2
```

La app referencia el prompt con una URI (variable `PROMPT_VERSION` en `.env`):

| URI | Significado |
|---|---|
| `prompts:/asistente_rrhh@production` | la versión que tenga el alias (default) |
| `prompts:/asistente_rrhh/2` | una versión fija |
| `prompts:/asistente_rrhh@latest` | la última versión registrada |
| `v1_asistente_rrhh` | archivo local, sin usar el registro |

> ⚠️ **¿Cuándo se registra un prompt?**
> - **Automático** solo la primera vez (si el registro está vacío) y en el CI.
> - **Manual** cada vez que edites o agregues un `.txt`: ejecuta `python -m app.prompt_registry`. Hasta entonces, la app sigue usando la versión anterior.
> - **Promover** a producción siempre es manual (`--promote N`), para que ningún cambio llegue a producción sin decidirlo.

En la UI de MLflow (`mlflow ui`) → menú **Prompts** → `asistente_rrhh` puedes ver las versiones, compararlas (diff) y mover alias.

---

### 4. 🧠 Construcción del pipeline RAG

```python
from app.rag_pipeline import build_chain
chain = build_chain(vectordb, prompt_version="prompts:/asistente_rrhh@production", k=4)
```

- Usa `ConversationalRetrievalChain` con `LangChain` + `OpenAI` o `Groq` (según `LLM_PROVIDER`)
- Devuelve también los documentos fuente (`source_documents`) para mostrar referencias

---

### 5. 💬 Interacción vía Streamlit

Versión básica:
```bash
streamlit run app/ui_streamlit.py
```

Versión combinada con métricas:
```bash
streamlit run app/main_interface.py
```

La interfaz de chat incluye:
- 💡 **Preguntas de ejemplo** (tomadas de `tests/eval_dataset.json`: si cambias de dominio, cambian solas)
- 📚 **Referencias**: documento, página y fragmento usados para cada respuesta
- ⚙️ **Panel lateral**: proveedor y modelo activos, selector de prompt del **Prompt Registry** y nº de fragmentos (`k`)
- 🔭 **Tracing**: cada respuesta queda como una traza en MLflow (experimento `chatbot_rrhh`)
- 👍/👎 **Feedback humano** adjunto a la traza de la respuesta (`mlflow.log_feedback`)
- ⏱️ **Tiempo de respuesta** por mensaje

---

### 6. 🔭 Observabilidad con MLflow Tracing

Con `mlflow.langchain.autolog()` cada pregunta genera una **traza** con todos sus pasos: reformulación de la pregunta, fragmentos recuperados por el retriever, prompt final, respuesta del LLM, tokens y latencias.

Además, cada combinación **proveedor + modelo + prompt + k** se registra como una **versión de la app** (`LoggedModel`). Las trazas, el feedback y las evaluaciones quedan enlazadas a ella, así puedes compararlas en la UI de MLflow.

```bash
mlflow ui --port 5000
```

Ver [Dónde mirar en la UI de MLflow](#-dónde-mirar-en-la-ui-de-mlflow) para saber qué encontrar en cada pantalla.

---

### 7. ⚖️ Evaluación automática con jueces LLM

```bash
python app/run_eval.py

# Comparar otra configuración
PROMPT_VERSION=prompts:/asistente_rrhh@v2_resumido_directo RETRIEVER_K=6 python app/run_eval.py
```

Usa `mlflow.genai.evaluate` con `tests/eval_dataset.json` como ground truth y estos **jueces** (LLM-as-a-judge):

| Juez | Qué evalúa |
|---|---|
| `Correctness` | ¿La respuesta coincide con la respuesta esperada? |
| `RelevanceToQuery` | ¿Responde a lo que se preguntó? |
| `RetrievalGroundedness` | ¿Está sustentada en los documentos recuperados? (detecta alucinaciones) |
| `RetrievalRelevance` | ¿Los fragmentos recuperados son pertinentes? |
| `Safety` | ¿Contiene contenido dañino u ofensivo? |
| `Guidelines` `idioma_espanol` | ¿Está escrita en español? |
| `Guidelines` `tono_profesional` | ¿Tono formal y sin inventar políticas? |

- Cada juez deja un **veredicto y su razonamiento** en la traza de cada pregunta.
- Las métricas agregadas (`correctness/mean`, `safety/mean`, …) quedan en un run del experimento `rag_evaluation`, enlazado a la versión de la app y del prompt.
- El juez se configura con `EVAL_MODEL` (default: `openai:/gpt-4o-mini` o `groq:/openai/gpt-oss-20b`).
- Para ahorrar cuota mientras iteras, ejecuta solo algunos jueces:
  ```bash
  EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py
  ```

#### 🐢 Evaluar con el plan gratuito de Groq

El plan gratuito de Groq limita cada modelo a **30 solicitudes y 8K tokens por minuto, y 200K tokens por día**. Una evaluación completa (5 preguntas × 7 jueces) hace unas 50 llamadas al juez, así que:

- Con `LLM_PROVIDER=groq`, `run_eval.py` activa un **modo pausado** automático (1 pregunta y 2 jueces a la vez, ~12 llamadas/min, 8 reintentos ante errores 429). Una evaluación completa tarda **varios minutos**.
- El juez por defecto (`gpt-oss-20b`) es **distinto del modelo del chat** (`gpt-oss-120b`): en Groq los límites son por modelo, así no compiten por la misma cuota.
- Los mensajes `Rate-limited (attempt 1/4), retrying...` son normales: MLflow reintenta solo. Solo es un problema si al final aparecen jueces con ⚠️.
- Si se agota la cuota diaria, usa `EVAL_SCORERS` con menos jueces, prueba otro modelo juez (p. ej. `EVAL_MODEL=groq:/qwen/qwen3.6-27b`) o usa OpenAI como juez (`EVAL_MODEL=openai:/gpt-4o-mini`).

---

### 8. 📈 Visualización de resultados

```bash
streamlit run app/dashboard.py
```

- **Comparación de configuraciones**: promedio de cada juez por evaluación, con gráfico seleccionable
- **Detalle por pregunta**: veredicto (✅/❌) de cada juez y su razonamiento
- **Feedback de usuarios**: satisfacción por versión de la app y respuestas con 👎 (candidatas para ampliar el dataset de evaluación)

---

### 9. 🔁 Automatización con GitHub Actions

- CI de evaluación: `.github/workflows/eval.yml` → registra prompts, construye el índice, evalúa, aplica el *quality gate* y guarda `mlflow.db` como artefacto
  - Secrets: `OPENAI_API_KEY` y/o `GROQ_API_KEY`
  - Variables opcionales: `LLM_PROVIDER`, `EVAL_MODEL`, `PROMPT_VERSION` (Settings → Secrets and variables → Actions)
- Test unitarios: `.github/workflows/test.yml`

---

### 10. 🧪 Validación automatizada (quality gate)

```bash
pytest tests/test_run_eval.py -v
```

- Verifica que la última evaluación supere los umbrales: `correctness ≥ 0.8` y `safety = 1.0`
- Ajustables con `MIN_CORRECTNESS` y `MIN_SAFETY`

---

## 🔬 Cómo funciona MLflow en este proyecto

### 📖 Conceptos clave

| Concepto | Qué es aquí |
|---|---|
| **Experimento** | Carpeta lógica que agrupa resultados. Hay uno para el chat (`chatbot_rrhh`) y otro para las evaluaciones (`rag_evaluation`). |
| **Run** | Una ejecución con parámetros y métricas. Cada `python app/run_eval.py` crea un run de evaluación. |
| **Traza** (*trace*) | El registro de **una** pregunta respondida, con sus pasos (*spans*): reformulación, retriever, prompt, LLM, tokens y latencia. |
| **Assessment / Feedback** | Una calificación adjunta a una traza: el veredicto de un juez (con su razonamiento) o el 👍/👎 de un usuario. |
| **Juez** (*scorer*) | Función que califica una traza. Puede ser un LLM (`Correctness`, `Guidelines`…) o código Python (`@scorer`). |
| **Prompt Registry** | Registro de prompts con versiones inmutables (1, 2, 3…) y **alias** movibles (`@production`). |
| **Versión de la app** (*Agent version* / `LoggedModel`) | Una combinación concreta de proveedor + modelo + prompt + `k`. Trazas y evaluaciones quedan enlazadas a ella para poder comparar. |

### 🗄️ Qué guarda MLflow y dónde

Todo vive en `./mlflow.db` (SQLite), en la raíz del repo.

| Dónde | Qué contiene | Lo genera |
|---|---|---|
| Prompt Registry → `asistente_rrhh` | Versiones del prompt y sus alias | `python -m app.prompt_registry` |
| Experimento `chatbot_rrhh` | Una traza por cada pregunta hecha en Streamlit + el 👍/👎 del usuario | La app de Streamlit |
| Experimento `rag_evaluation` | Un run por evaluación (métricas promedio por juez) + una traza por pregunta con el veredicto y razonamiento de cada juez | `python app/run_eval.py` |
| Experimento `vectorstore_tracking` | Parámetros de cada construcción del índice (chunks, documentos) | `save_vectorstore()` |

### 🔄 El ciclo de mejora

Así se usa MLflow para mejorar el chatbot de forma controlada:

```mermaid
flowchart LR
    A["1️⃣ Editar prompt<br/>app/prompts/*.txt"] --> B["2️⃣ Registrar versión<br/>prompt_registry"]
    B --> C["3️⃣ Evaluar<br/>run_eval.py"]
    C --> D["4️⃣ Comparar<br/>dashboard / mlflow ui"]
    D -->|mejor| E["5️⃣ Promover<br/>--promote N"]
    D -->|peor| A
    E --> F["6️⃣ Usuarios<br/>trazas + 👍/👎"]
    F --> G["7️⃣ Respuestas con 👎<br/>al dataset de evaluación"]
    G --> A
```

| Paso | Comando | Qué pasa en MLflow |
|---|---|---|
| 1. Editar | Modifica o crea un `.txt` en `app/prompts/` | Nada todavía |
| 2. Registrar | `python -m app.prompt_registry` | Nueva versión del prompt (p. ej. v3) con alias por nombre de archivo |
| 3. Evaluar | `PROMPT_VERSION=prompts:/asistente_rrhh/3 python app/run_eval.py` | Run en `rag_evaluation` con métricas por juez y una traza por pregunta |
| 4. Comparar | `streamlit run app/dashboard.py` o `mlflow ui` | Ves la v3 junto a las versiones anteriores |
| 5. Promover | `python -m app.prompt_registry --promote 3` | El alias `@production` apunta a v3; la app la usa sin cambiar código (se refresca en ~30 s) |
| 6. Monitorear | Usuarios usan el chat | Trazas en `chatbot_rrhh` enlazadas a la versión de la app, con su 👍/👎 |
| 7. Ampliar el dataset | Revisa las respuestas con 👎 en el dashboard y agrégalas a `tests/eval_dataset.json` | La siguiente evaluación cubre esos casos |

El CI (`eval.yml`) repite los pasos 2 y 3 en cada push y bloquea el cambio si no supera el *quality gate*.

### 🧭 Dónde mirar en la UI de MLflow

```bash
mlflow ui --port 5000      # desde la raíz del repo (donde está mlflow.db)
```

En el menú principal están **Experiments** y **Prompts**. Al abrir un experimento aparece una barra lateral con estas pantallas:

| Quiero ver… | Ve a |
|---|---|
| Las versiones del prompt, su diff y alias | Menú **Prompts** → `asistente_rrhh` |
| Las conversaciones reales y su 👍/👎 | **Experiments** → `chatbot_rrhh` → **Traces** |
| Los pasos de una respuesta (qué recuperó, qué prompt usó) | Cualquier traza → abrir → árbol de *spans* |
| Las evaluaciones y sus métricas promedio | **Experiments** → `rag_evaluation` → **Evaluation runs** |
| El veredicto y razonamiento de cada juez por pregunta | **Evaluation runs** → abrir un run, o **Traces** → abrir una traza → *Assessments* |
| Las versiones de la app (proveedor + modelo + prompt + k) | Dentro de cada experimento → **Agent versions** |

> Las pantallas **Judges**, **Review**, **Datasets** y **Playground** también existen en MLflow 3, pero este proyecto aún no las usa: son buenas candidatas para el desafío.

---

## 🔍 ¿Qué puedes hacer?

- 💬 Hacer preguntas al chatbot y ver de dónde sale cada respuesta
- 📝 Versionar prompts y promoverlos con alias
- 🔁 Evaluar diferentes prompts, proveedores, `k` y estrategias de chunking
- ⚖️ Calificar con jueces LLM y revisar su razonamiento
- 🔭 Inspeccionar cada paso del RAG con trazas
- 👥 Recoger feedback humano y llevarlo al dataset de evaluación
- 🔄 Adaptar a otros dominios (legal, salud, educación…)

---

## ⚙️ Stack Tecnológico

- **OpenAI / Groq + LangChain** – LLM + RAG (proveedor configurable)
- **FastEmbed** – Embeddings locales (cuando se usa Groq)
- **FAISS** – Vectorstore
- **Streamlit** – UI
- **MLflow 3** – Prompt Registry, Tracing, evaluación con jueces LLM, feedback y versiones de la app
- **GitHub Actions** – CI/CD
- **DevContainer** – Desarrollo portable

---

## 🛠️ Solución de problemas

| Síntoma | Causa / solución |
|---|---|
| `No existe el índice en 'vectorstore/...'` | Falta construir el índice, o cambiaste `EMBEDDINGS_PROVIDER`: ejecuta el paso 2. |
| La primera consulta con Groq tarda mucho | FastEmbed descarga el modelo de embeddings (~220 MB) la primera vez. |
| Muchos `Rate-limited ... retrying` durante `run_eval.py` | Normal con el plan gratuito de Groq: MLflow reintenta solo. Ver [Evaluar con el plan gratuito de Groq](#-evaluar-con-el-plan-gratuito-de-groq). |
| Jueces con ⚠️ al final de la evaluación | Se agotaron los reintentos o la cuota diaria. Reduce jueces con `EVAL_SCORERS`, baja el ritmo con `MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT=0.1` o cambia `EVAL_MODEL`. |
| El dashboard no muestra evaluaciones anteriores | Las versiones previas del repo guardaban en `mlruns/` con otro formato; ahora todo vive en `mlflow.db`. Vuelve a ejecutar `run_eval.py`. |
| `mlflow ui` no muestra nada | Ejecútalo desde la raíz del repo (donde está `mlflow.db`) o usa `mlflow ui --backend-store-uri sqlite:///mlflow.db`. |
| El prompt editado no cambia en la app | Registra la nueva versión (`python -m app.prompt_registry`) y, si la quieres en producción, promuévela con `--promote N`. La lista de prompts en la app se refresca cada 30 s. |
| `test.yml` aparece como *skipped* en GitHub | Es esperado: el *quality gate* real corre en `eval.yml`, después de la evaluación. |

---

## 🎓 Desafío para estudiantes

🧩 Parte 1: Personalización

1. Elige un nuevo dominio
Ejemplos: salud, educación, legal, bancario, etc.

2. Reemplaza los documentos PDF
Ubícalos en data/pdfs/.

3. Crea tus prompts y versiónalos
Edita o agrega archivos en app/prompts/ y regístralos con `python -m app.prompt_registry`.

4. Crea un conjunto de pruebas
En tests/eval_dataset.json, define preguntas y respuestas esperadas para evaluar a tu chatbot.

✅ Parte 2: Evaluación Automática

1. Ejecuta `run_eval.py` con al menos dos versiones de prompt y dos valores de `k`.
2. Abre `mlflow ui` y revisa, para las preguntas que fallan, el razonamiento de los jueces y la traza: ¿falló la recuperación (`retrieval_relevance`) o la generación (`correctness`, `retrieval_groundedness`)?

🔧 Parte 3: ¡Tu reto! (👨‍🔬 nivel investigador)

1. Agrega tus propios evaluadores en `get_scorers()` de `app/run_eval.py`:

    * Un **scorer de código** con `@scorer`, por ejemplo: ¿la respuesta menciona el documento fuente? ¿supera cierta longitud?
    * Un **juez LLM personalizado** con `mlflow.genai.judges.make_judge`, por ejemplo "claridad" o "empatía", con su propia rúbrica.
    * Al menos un criterio nuevo con `Guidelines` propio de tu dominio (p. ej. "no da diagnósticos médicos").

    📚 Revisa la documentación de [scorers personalizados](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/) y [jueces con make_judge](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge).

2. Compara dos modelos juez (por ejemplo OpenAI vs Groq con `EVAL_MODEL`). ¿Coinciden sus veredictos?

📊 Parte 4: Mejora el dashboard

1. Extiende `app/metrics_ui.py` para visualizar tus nuevos evaluadores.
2. Agrega una vista que compare **dos versiones del prompt** lado a lado (plantilla + métricas).
3. (Opcional) Agrega un botón para llevar las respuestas con 👎 al dataset de evaluación.

🧪 Parte 5: Presenta y reflexiona
1. Compara configuraciones distintas (prompt, `k`, chunk size, proveedor) y justifica tu selección.
    * ¿Cuál configuración genera mejores respuestas? ¿Cuánto cuesta en latencia?
    * ¿En qué fallan los modelos? ¿Alucinan, recuperan mal o no siguen las guías?
    * Promueve tu mejor versión a `@production` y explica por qué. Usa evidencias de MLflow.

🚀 Bonus

- Prueba la optimización automática de prompts con `mlflow.genai.optimize_prompts` y compara el prompt optimizado contra el tuyo.

---

¡Listo para ser usado en clase, investigación o producción educativa! 🚀
