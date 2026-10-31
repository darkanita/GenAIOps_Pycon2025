# app/rag_pipeline.py

import os
from langchain.globals import set_verbose, get_verbose

set_verbose(True)  # Si quieres ver logs detallados

from app.llm_provider import get_llm, get_embeddings, get_vectorstore_dir, describe
from app.prompt_registry import DEFAULT_PROMPT, load_prompt_text, list_prompt_options

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
import mlflow

load_dotenv()

DATA_DIR = "data/pdfs"
VECTOR_DIR = get_vectorstore_dir()  # vectorstore/openai o vectorstore/fastembed

def load_documents(path=DATA_DIR):
    docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)

def save_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, persist_path=VECTOR_DIR):
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.log_params(describe())
        mlflow.set_tag("vectorstore", persist_path)

def load_vectorstore(chunk_size=512, chunk_overlap=50):
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"No existe el índice en '{persist_path}'. Genéralo primero con: "
            'python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"'
        )
    embeddings = get_embeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

def load_prompt(version=DEFAULT_PROMPT):
    """
    Carga el prompt desde el Prompt Registry de MLflow (prompts:/nombre@alias)
    o desde un archivo local de app/prompts/ (nombre sin .txt).
    La URI resuelta queda en prompt.metadata["prompt_uri"].
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
    prompt = load_prompt(prompt_version)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(temperature=0),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True  # para mostrar las referencias en la UI
    )
