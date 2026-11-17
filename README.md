# Chatbot RAG de investigación en trading intradía

Proyecto final del curso de GenAIOps. Partí del repositorio de clase ([darkanita/GenAIOps_Pycon2025](https://github.com/darkanita/GenAIOps_Pycon2025)), que respondía preguntas de Recursos Humanos sobre una empresa ficticia, y lo llevé a un dominio que uso: la investigación detrás de mi estrategia intradía en futuros MNQ (VWAP con bandas de desviación).

Mantengo la arquitectura de clase (LangChain, FAISS, MLflow, Streamlit y GitHub Actions); cambié los documentos, los prompts, el dataset de evaluación y los jueces.

## Por qué este dominio

Cuando diseño una regla de entrada, un stop o un filtro horario, necesito saber qué se ha probado, con qué método y con qué resultado. Esa información está repartida en papers largos y con calidad de evidencia muy distinta: algunos son backtests de practicantes y otros son pruebas estadísticas formales. El bot me sirve para consultar esa evidencia rápido y sin que me la mezcle.

## Documentos

| Tema | Paper | Tipo de evidencia |
|---|---|---|
| VWAP | Zarattini y Aziz (2023), *VWAP: The Holy Grail for Day Trading Systems* | Backtest de practicantes |
| Opening range breakout | Zarattini, Barbon y Aziz (2024), *A Profitable Day Trading Strategy for the U.S. Equity Market* | Backtest de practicantes |
| Momentum intradía | Gao, Han, Li y Zhou (2015), *Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return* | Regresiones predictivas |
| Patrones técnicos | Lo, Mamaysky y Wang (2000), *Foundations of Technical Analysis* | Regresión kernel, bondad de ajuste, Kolmogorov-Smirnov |
| Tendencias y retrocesos | Kempen y Maier-Paape (2016), *Survey on log-normally distributed market-technical trend data* | Ajuste log-normal, Anderson-Darling |
| Fibonacci | Batchelor y Ramyar (2006), *Magic Numbers in the Dow* | Bootstrap estacionario por bloques |
| Fibonacci | Tsinaslanidis, Guijarro y Voukelatos (2022), *Automatic identification and evaluation of Fibonacci retracements* | Regresión logística con bootstrap |
| Perfil de volumen | Howard (2026), *Intraday Value Area Breakouts: Evidence from E-mini S&P 500 Futures* | Pruebas t con errores agrupados por día, prueba de signos |

Los PDFs están en `data/pdfs/`. Suman 273 páginas y 1.234 fragmentos con `chunk_size=512`.

Elegí a propósito papers que se contradicen o que dan evidencia negativa (los tres de Fibonacci concluyen que los niveles no se distinguen del azar), porque eso permite evaluar si el bot reporta lo que dice el documento y no lo que el usuario quiere oír.

## Cómo ejecutarlo

Desde la raíz del repositorio, en GitHub Codespaces o en local con Python 3.10+:

```bash
pip install -r requirements.txt
cp .env.example .env        # LLM_PROVIDER=groq y GROQ_API_KEY
python -m app.prompt_registry
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
streamlit run app/main_interface.py
```

Para evaluar y revisar resultados:

```bash
python app/run_eval.py
mlflow ui --port 5000
pytest tests/test_run_eval.py -v
```

En un Codespace de 2 núcleos conviene correr Streamlit, MLflow y la evaluación por separado; juntos agotan la memoria.

## Estructura

| Archivo | Función |
|---|---|
| `app/rag_pipeline.py` | Carga los PDFs, crea el índice FAISS y arma la cadena RAG |
| `app/llm_provider.py` | Selección del LLM y de los embeddings según `.env` |
| `app/prompts/` | Versiones del prompt (`v1_asistente_trading`, `v2_resumido_directo`) |
| `app/prompt_registry.py` | Registro de prompts en MLflow y promoción del alias `@production` |
| `app/main_interface.py`, `app/chat_ui.py` | Interfaz de chat en Streamlit |
| `app/run_eval.py` | Evaluación con jueces LLM y scorers propios |
| `app/dashboard.py` | Comparación de configuraciones evaluadas |
| `tests/eval_dataset.json` | Preguntas de evaluación con respuesta esperada, tipo y fuente |
| `tests/test_run_eval.py` | Umbrales mínimos de calidad |
| `.github/workflows/eval.yml` | Indexa, evalúa y aplica el umbral en cada push |

## Estado

| Parte del desafío | Estado |
|---|---|
| 1. Cambio de dominio | Listo |
| 2. Dataset de evaluación | 19 preguntas listas (9 directas, 4 multi-documento, 3 sin respuesta, 3 adversarias); falta la primera evaluación |
| 3. Experimentos | Pendiente |
| 4. Evaluadores propios | Pendiente |
| 5. Confianza en el juez | Pendiente |
| 6. Producción y CI | Pendiente |
| 7. Informe | Pendiente |

Los resultados y las decisiones van en `INFORME.md`.

## Limitaciones que ya identifiqué

- Las tablas de Lo, Mamaysky y Wang salen ilegibles porque el PDF es un escaneo, y las tablas de Kempen y Maier-Paape son imágenes sin texto.
- El LLM recibe el texto de cada fragmento pero no el nombre del paper, así que en preguntas que comparan autores puede atribuir mal.
- Ningún paper usa MNQ en barras de ticks; el bot informa la evidencia, no la valida para mi estrategia.

Juan Felipe Botero · Finanzas, EAFIT
