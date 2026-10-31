# app/rag_pipeline.py
"""
Pipeline RAG (Retrieval Augmented Generation): de los PDFs a la cadena que responde.

Hay dos momentos distintos:

1. INDEXACIÓN (una vez, o cuando cambian los PDFs o el chunking)
       PDFs ──load_documents()──▶ páginas ──_split()──▶ fragmentos (chunks)
            ──embeddings──▶ vectores ──FAISS──▶ índice guardado en vectorstore/<proveedor>/

   python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"

2. CONSULTA (en cada pregunta)
       pregunta ──retriever──▶ los k fragmentos más parecidos
                ──prompt──▶ "Pregunta: ... Contexto: ..." ──LLM──▶ respuesta

   build_chain() arma esta cadena con ConversationalRetrievalChain de LangChain.
   Si hay historial, la cadena primero reformula la pregunta para que sea
   autocontenida (p. ej. "¿y el dental?" → "¿Qué cubre el plan dental?").

El proveedor del LLM y de los embeddings se elige en .env (ver app/llm_provider.py).
El prompt viene del MLflow Prompt Registry (ver app/prompt_registry.py).
"""

import os

import mlflow
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_verbose
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm_provider import describe, get_embeddings, get_llm, get_vectorstore_dir
from app.prompt_registry import DEFAULT_PROMPT, list_prompt_options, load_prompt_text

load_dotenv()

# Muestra en consola "> Entering new ConversationalRetrievalChain chain..."
# en cada pregunta. Útil para aprender; cámbialo a False para una consola limpia.
set_verbose(True)

DATA_DIR = "data/pdfs"
VECTOR_DIR = get_vectorstore_dir()  # vectorstore/openai o vectorstore/fastembed

# Tamaño y solape de los fragmentos (en caracteres). Se leen de .env.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)


# ============================================================
# 1. Indexación
# ============================================================

def load_documents(path=DATA_DIR):
    """Lee todos los PDFs de la carpeta. Devuelve un Document por página (con source y page)."""
    docs = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(path, file)).load())
    return docs


def _split(docs, chunk_size, chunk_overlap):
    """
    Divide las páginas en fragmentos. El solape evita que una idea quede
    cortada justo en el borde entre dos fragmentos.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def save_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, persist_path=VECTOR_DIR):
    """
    Construye el índice FAISS a partir de los PDFs y lo guarda en disco.
    Registra la configuración en el experimento "vectorstore_tracking" de MLflow.
    """
    docs = load_documents()
    chunks = _split(docs, chunk_size, chunk_overlap)
    vectordb = FAISS.from_documents(chunks, embedding=get_embeddings())
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_params({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "n_chunks": len(chunks),
            "n_docs": len(docs),
            **describe(),
        })
        mlflow.set_tag("vectorstore", persist_path)


def load_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Construye el índice en memoria, sin guardarlo (útil para experimentos rápidos)."""
    return FAISS.from_documents(_split(load_documents(), chunk_size, chunk_overlap),
                                embedding=get_embeddings())


def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    """Carga el índice guardado por save_vectorstore()."""
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"No existe el índice en '{persist_path}'. Genéralo primero con: "
            'python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"'
        )
    # allow_dangerous_deserialization: FAISS guarda metadatos con pickle.
    # Es seguro porque el índice lo generamos nosotros mismos.
    return FAISS.load_local(persist_path, get_embeddings(), allow_dangerous_deserialization=True)


# ============================================================
# 2. Consulta
# ============================================================

def load_prompt(version=DEFAULT_PROMPT):
    """
    Carga el prompt desde el MLflow Prompt Registry (prompts:/nombre@alias o /versión)
    o desde un archivo local de app/prompts/ (nombre sin .txt).
    La URI resuelta (p. ej. prompts:/asistente_rrhh/2) queda en prompt.metadata["prompt_uri"],
    para registrar en MLflow qué versión exacta respondió.
    """
    prompt_text, uri = load_prompt_text(version)
    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text,
        metadata={"prompt_uri": uri},
    )


def list_prompt_versions():
    """Alias y versiones disponibles en el registro (o archivos locales si no hay registro)."""
    return list_prompt_options()


def build_chain(vectordb, prompt_version=DEFAULT_PROMPT, k=4):
    """
    Arma la cadena RAG:
      - retriever: busca los k fragmentos más parecidos a la pregunta en el índice
      - prompt:    plantilla con {question} y {context} (los fragmentos recuperados)
      - LLM:       genera la respuesta (OpenAI o Groq según LLM_PROVIDER)

    Uso:
        result = chain.invoke({"question": "...", "chat_history": [(pregunta, respuesta), ...]})
        result["answer"], result["source_documents"]
    """
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(temperature=0),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,  # para mostrar las referencias en la UI
    )
