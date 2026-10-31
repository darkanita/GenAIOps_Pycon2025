# app/llm_provider.py
"""
Fábrica de modelos: permite elegir el proveedor del LLM y de los embeddings
mediante variables de entorno, sin tocar el resto del código.

Variables de entorno:
    LLM_PROVIDER          openai | groq                  (default: openai)
    LLM_MODEL             modelo de chat a usar          (default según proveedor)
    EVAL_MODEL            modelo para el evaluador       (default: LLM_MODEL)
    EMBEDDINGS_PROVIDER   openai | fastembed             (default: openai si LLM_PROVIDER=openai,
                                                                   fastembed si LLM_PROVIDER=groq)
    EMBEDDINGS_MODEL      modelo de embeddings           (default según proveedor)

Groq no ofrece un endpoint de embeddings, por eso con Groq se usan
embeddings locales con FastEmbed (ONNX, no requiere GPU ni API key).
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o",
    "groq": "openai/gpt-oss-120b",
}

DEFAULT_EMBEDDINGS_MODELS = {
    "openai": "text-embedding-ada-002",
    # Multilingüe: los PDFs están en inglés y las preguntas en español
    "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(
            f"LLM_PROVIDER no soportado: '{provider}'. Usa uno de: {list(DEFAULT_LLM_MODELS)}"
        )
    return provider


def get_embeddings_provider() -> str:
    default = "openai" if get_llm_provider() == "openai" else "fastembed"
    provider = os.getenv("EMBEDDINGS_PROVIDER", default).strip().lower()
    if provider not in DEFAULT_EMBEDDINGS_MODELS:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER no soportado: '{provider}'. "
            f"Usa uno de: {list(DEFAULT_EMBEDDINGS_MODELS)}"
        )
    return provider


def get_llm_model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_LLM_MODELS[get_llm_provider()]


def get_embeddings_model_name() -> str:
    return os.getenv("EMBEDDINGS_MODEL") or DEFAULT_EMBEDDINGS_MODELS[get_embeddings_provider()]


def get_llm(temperature: float = 0, model: str | None = None):
    """Devuelve un chat model de LangChain según LLM_PROVIDER."""
    provider = get_llm_provider()
    model = model or get_llm_model_name()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("Falta GROQ_API_KEY en el entorno o en .env")
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)


def get_eval_llm(temperature: float = 0):
    """LLM usado como juez en la evaluación (puede ser distinto al del chatbot)."""
    return get_llm(temperature=temperature, model=os.getenv("EVAL_MODEL") or None)


def get_embeddings():
    """Devuelve un modelo de embeddings de LangChain según EMBEDDINGS_PROVIDER."""
    provider = get_embeddings_provider()
    model = get_embeddings_model_name()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)

    if provider == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=model)


def get_vectorstore_dir(base_dir: str = "vectorstore") -> str:
    """
    Cada proveedor de embeddings genera vectores de distinta dimensión, así que
    el índice FAISS se guarda en una carpeta por proveedor (vectorstore/openai,
    vectorstore/fastembed) para que no se mezclen.
    """
    return os.getenv("VECTOR_DIR") or os.path.join(base_dir, get_embeddings_provider())


def describe() -> dict:
    """Resumen de la configuración activa (útil para loguear en MLflow)."""
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": get_llm_model_name(),
        "embeddings_provider": get_embeddings_provider(),
        "embeddings_model": get_embeddings_model_name(),
    }
