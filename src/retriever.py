"""ChromaDB storage and retrieval helpers for RAG chunks."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from embeddings import embed_text, embed_texts


COLLECTION_NAME = "career_copilot_documents"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from the vector store with citation metadata."""

    text: str
    source: str
    chunk_id: int
    distance: float | None = None

    @property
    def citation(self) -> str:
        """Return a simple citation label for display."""
        return f"{self.source}#chunk-{self.chunk_id}"


def get_chroma_collection(
    persist_directory: str | Path = "vector_store",
    collection_name: str = COLLECTION_NAME,
):
    """Create or load the ChromaDB collection used for document chunks."""
    try:
        chromadb = import_module("chromadb")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency: chromadb. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    client = chromadb.PersistentClient(path=str(persist_directory))
    return client.get_or_create_collection(name=collection_name)


def store_chunks(
    chunks: list[str],
    source: str,
    persist_directory: str | Path = "vector_store",
) -> None:
    """Embed and upsert chunks into ChromaDB."""
    if not chunks:
        return

    collection = get_chroma_collection(persist_directory)

    existing = collection.get(where={"source": source})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    embeddings = embed_texts(chunks)
    ids = [f"{source}:chunk-{index}" for index in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_id": index}
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def retrieve_chunks(
    query: str,
    top_k: int = 3,
    persist_directory: str | Path = "vector_store",
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query from ChromaDB."""
    if top_k <= 0:
        return []

    collection = get_chroma_collection(persist_directory)
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        retrieved.append(
            RetrievedChunk(
                text=text,
                source=str(metadata.get("source", "unknown")),
                chunk_id=int(metadata.get("chunk_id", 0)),
                distance=float(distance) if distance is not None else None,
            )
        )

    return retrieved
