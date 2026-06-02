"""Embedding helpers backed by sentence-transformers."""

from importlib import import_module
from functools import lru_cache


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence-transformers embedding model."""
    try:
        sentence_transformers = import_module("sentence_transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from error

    return sentence_transformers.SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with all-MiniLM-L6-v2.

    TODO: Add batching configuration for large document collections.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """Embed a single text string with all-MiniLM-L6-v2."""
    return embed_texts([text])[0]
