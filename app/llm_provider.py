# app/llm_provider.py
"""
Fábrica de modelos: permite elegir el proveedor del LLM y de los embeddings
mediante variables de entorno, sin tocar el resto del código.

Variables de entorno:
    LLM_PROVIDER          openai | groq                  (default: openai)
    LLM_MODEL             modelo de chat a usar          (default según proveedor)
    EVAL_MODEL            juez de run_eval.py, formato proveedor:/modelo
                          (default: openai:/gpt-4o-mini o groq:/openai/gpt-oss-20b)
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

# Jueces (LLM-as-a-judge) de mlflow.genai.evaluate. Formato "<proveedor>:/<modelo>";
# los proveedores distintos de OpenAI/Anthropic se resuelven vía LiteLLM.
DEFAULT_JUDGE_MODELS = {
    "openai": "openai:/gpt-4o-mini",
    # Modelo distinto al del chat: en Groq los límites son por modelo,
    # así el juez no consume la cuota del chatbot.
    "groq": "groq:/openai/gpt-oss-20b",
}

DEFAULT_EMBEDDINGS_MODELS = {
    "openai": "text-embedding-ada-002",
    # Multilingüe: los PDFs están en inglés y las preguntas en español
    "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def get_llm_provider() -> str:
    """Proveedor del LLM según LLM_PROVIDER ("openai" por defecto)."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(
            f"LLM_PROVIDER no soportado: '{provider}'. Usa uno de: {list(DEFAULT_LLM_MODELS)}"
        )
    return provider


def get_embeddings_provider() -> str:
    """Proveedor de embeddings: EMBEDDINGS_PROVIDER, o el que corresponde al LLM (Groq → fastembed)."""
    default = "openai" if get_llm_provider() == "openai" else "fastembed"
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or default).strip().lower()
    if provider not in DEFAULT_EMBEDDINGS_MODELS:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER no soportado: '{provider}'. "
            f"Usa uno de: {list(DEFAULT_EMBEDDINGS_MODELS)}"
        )
    return provider


def get_llm_model_name() -> str:
    """Modelo de chat: LLM_MODEL o el default del proveedor."""
    return os.getenv("LLM_MODEL") or DEFAULT_LLM_MODELS[get_llm_provider()]


def get_embeddings_model_name() -> str:
    """Modelo de embeddings: EMBEDDINGS_MODEL o el default del proveedor."""
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


def get_judge_model_uri() -> str:
    """
    Modelo juez para mlflow.genai.evaluate. EVAL_MODEL acepta "proveedor:/modelo"
    o solo el nombre del modelo (se usa el proveedor de LLM_PROVIDER).
    """
    provider = get_llm_provider()
    model = os.getenv("EVAL_MODEL")
    if not model:
        return DEFAULT_JUDGE_MODELS[provider]
    return model if ":/" in model else f"{provider}:/{model}"


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
