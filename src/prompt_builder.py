"""Prompt templates for retrieval-augmented generation (RAG)."""

from retriever import RetrievedChunk


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build a basic RAG prompt from retrieved chunks and a user question.

    TODO: Add stricter citation formatting and structured output later.
    """
    context = "\n\n".join(
        f"Citation: [{index}] {chunk.citation}\n{chunk.text}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )

    return f"""You are AI Career Copilot, a helpful career assistant.
Answer the question using only the context below. If the answer is not in the
context, say you do not know based on the provided document.

Context:
{context or "No retrieved context was provided."}

Question:
{question}

Answer with a concise response and mention citation numbers when useful:
"""
