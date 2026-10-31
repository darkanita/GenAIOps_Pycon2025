# 🤖 Chatbot GenAI - Caso de Estudio Recursos Humanos

Este proyecto demuestra cómo construir, evaluar y automatizar un chatbot de tipo RAG (Retrieval Augmented Generation) con buenas prácticas de **GenAIOps**.

---

## 🧠 Caso de Estudio

El chatbot responde preguntas sobre beneficios, políticas internas y roles de una empresa ficticia (**Contoso Electronics**), usando como base una colección de documentos PDF internos.

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
│   └── prompts/
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

El proveedor se elige en `.env`, sin tocar código (ver `app/llm_provider.py`):

| Variable | Valores | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai` \| `groq` | `openai` |
| `LLM_MODEL` | cualquier modelo del proveedor | `gpt-4o` / `openai/gpt-oss-120b` |
| `EVAL_MODEL` | modelo juez para `run_eval.py` | igual a `LLM_MODEL` |
| `EMBEDDINGS_PROVIDER` | `openai` \| `fastembed` | `openai` con OpenAI, `fastembed` con Groq |
| `EMBEDDINGS_MODEL` | modelo de embeddings | `text-embedding-ada-002` / `paraphrase-multilingual-MiniLM-L12-v2` |

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
- Divide los documentos en chunks (por defecto `chunk_size=512`, `chunk_overlap=50`)
- Genera embeddings con OpenAI o FastEmbed (según `EMBEDDINGS_PROVIDER`)
- Guarda el índice vectorial en `vectorstore/openai/` o `vectorstore/fastembed/`
- Registra los parámetros en **MLflow**

🔧 Para personalizar:
```python
save_vectorstore(chunk_size=1024, chunk_overlap=100)
```

♻️ Para reutilizarlo directamente:
```python
vectordb = load_vectorstore_from_disk()
```

---

### 3. 🧠 Construcción del pipeline RAG

```python
from app.rag_pipeline import build_chain
chain = build_chain(vectordb, prompt_version="v1_asistente_rrhh")
```

- Soporta múltiples versiones de prompt
- Usa `ConversationalRetrievalChain` con `LangChain` + `OpenAI` o `Groq` (según `LLM_PROVIDER`)

---

### 4. 💬 Interacción vía Streamlit

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
- ⚙️ **Panel lateral**: proveedor y modelo activos, selector de versión de prompt y nº de fragmentos (`k`)
- ⏱️ **Tiempo de respuesta** por mensaje
- 👍/👎 **Feedback** registrado en MLflow (experimento `chat_feedback`) con pregunta, respuesta, fuentes, latencia y configuración
- El índice y la cadena se cargan una sola vez (`st.cache_resource`)

---

### 5. 🧪 Evaluación automática de calidad

Ejecuta:

```bash
python app/run_eval.py
```

Esto:
- Usa `tests/eval_dataset.json` como ground truth
- Genera respuestas usando el RAG actual
- Evalúa con `LangChain Eval (QAEvalChain)`
- Registra resultados en **MLflow**, incluyendo `llm_provider`, `llm_model` y `embeddings_*`, para comparar proveedores

---

### 6. 📈 Visualización de resultados

Dashboard completo:

```bash
streamlit run app/dashboard.py
```

- Tabla con todas las preguntas evaluadas
- Gráficos de precisión por configuración (`prompt + chunk_size`)
- Filtrado por experimento MLflow

---

### 7. 🔁 Automatización con GitHub Actions

- CI de evaluación: `.github/workflows/eval.yml`
  - Para usar Groq en CI: crea el secret `GROQ_API_KEY` y la variable de repositorio `LLM_PROVIDER=groq` (Settings → Secrets and variables → Actions)
- Test unitarios: `.github/workflows/test.yml`

---

### 8. 🧪 Validación automatizada

```bash
pytest tests/test_run_eval.py
```

- Evalúa que el sistema tenga al menos 80% de precisión con el dataset base

---

## 🔍 ¿Qué puedes hacer?

- 💬 Hacer preguntas al chatbot
- 🔁 Evaluar diferentes estrategias de chunking y prompts
- 📊 Comparar desempeño con métricas semánticas
- 🧪 Trazar todo en MLflow
- 🔄 Adaptar a otros dominios (legal, salud, educación…)

---

## ⚙️ Stack Tecnológico

- **OpenAI / Groq + LangChain** – LLM + RAG (proveedor configurable)
- **FastEmbed** – Embeddings locales (cuando se usa Groq)
- **FAISS** – Vectorstore
- **Streamlit** – UI
- **MLflow** – Registro de experimentos
- **LangChain Eval** – Evaluación semántica
- **GitHub Actions** – CI/CD
- **DevContainer** – Desarrollo portable

---

## 🎓 Desafío para estudiantes

🧩 Parte 1: Personalización

1. Elige un nuevo dominio
Ejemplos: salud, educación, legal, bancario, etc.

2. Reemplaza los documentos PDF
Ubícalos en data/pdfs/.

3. Modifica o crea tus prompts
Edita los archivos en app/prompts/.

4. Crea un conjunto de pruebas
En tests/eval_dataset.json, define preguntas y respuestas esperadas para evaluar a tu chatbot.

✅ Parte 2: Evaluación Automática

1. Ejecuta run_eval.py para probar tu sistema actual.
Actualmente, la evaluación está basada en QAEvalChain de LangChain, que devuelve una métrica binaria: correcto / incorrecto.

🔧 Parte 3: ¡Tu reto! (👨‍🔬 nivel investigador)

1. Mejora el sistema de evaluación:

    * Agrega evaluación con LabeledCriteriaEvalChain usando al menos los siguientes criterios:

        * "correctness" – ¿Es correcta la respuesta?
        * "relevance" – ¿Es relevante respecto a la pregunta?
        * "coherence" – ¿Está bien estructurada la respuesta?
        * "toxicity" – ¿Contiene lenguaje ofensivo o riesgoso?
        * "harmfulness" – ¿Podría causar daño la información?

    * Cada criterio debe registrar:

        * Una métrica en MLflow (score)

    * Y opcionalmente, un razonamiento como artefacto (reasoning)

    📚 Revisa la [documentación de LabeledCriteriaEvalChain](https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.criteria.eval_chain.LabeledCriteriaEvalChain.html) para implementarlo.

📊 Parte 4: Mejora el dashboard

1. Extiende dashboard.py o main_interface.py para visualizar:

    * Las métricas por criterio (correctness_score, toxicity_score, etc.).
    * Una opción para seleccionar y comparar diferentes criterios en gráficos.
    * (Opcional) Razonamientos del modelo como texto.    

🧪 Parte 5: Presenta y reflexiona
1. Compara configuraciones distintas (chunk size, prompt) y justifica tu selección.
    * ¿Cuál configuración genera mejores respuestas?
    * ¿En qué fallan los modelos? ¿Fueron tóxicos o incoherentes?
    * Usa evidencias desde MLflow y capturas del dashboard.

🚀 Bonus

- ¿Te animas a crear un nuevo criterio como "claridad" o "creatividad"? Puedes definirlo tú mismo y usarlo con LabeledCriteriaEvalChain.

---

¡Listo para ser usado en clase, investigación o producción educativa! 🚀
