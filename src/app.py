"""Minimal app for PDF RAG and resume/job-description matching."""

from pathlib import Path

from ats_engine import calculate_ats_score
from chunker import chunk_text
from config import settings
from document_loader import load_document
from jd_analyzer import (
    DEFAULT_JOB_DESCRIPTION_PATH,
    analyze_job_description,
    load_job_description_text,
)
from llm import chat_with_openai
from prompt_builder import build_rag_prompt
from resume_analyzer import DEFAULT_RESUME_PATH, analyze_resume, load_resume_text
from retriever import RetrievedChunk, retrieve_chunks, store_chunks


SAMPLE_PDF_PATH = Path("data/documents/sample.pdf")
DEFAULT_QUESTION = "What are the key points in this document?"
TOP_K = 3


def format_list(items: list[str]) -> str:
    """Format a list for readable terminal output."""
    return ", ".join(items) if items else "None found"


def format_citations(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as simple citations for terminal output."""
    lines = ["Top retrieved chunks:"]
    for index, chunk in enumerate(chunks, start=1):
        preview = chunk.text[:500]
        if len(chunk.text) > 500:
            preview += "..."
        lines.append(f"[{index}] {chunk.citation}\n{preview}")
    return "\n\n".join(lines)


def run_resume_job_match() -> None:
    """Load resume.txt and job_description.txt, then print a simple ATS report."""
    print("Resume ↔ Job Description Match")
    print("-" * 36)

    if not DEFAULT_RESUME_PATH.exists() or not DEFAULT_JOB_DESCRIPTION_PATH.exists():
        print("Missing resume or job description file:")
        print(f"- {DEFAULT_RESUME_PATH}")
        print(f"- {DEFAULT_JOB_DESCRIPTION_PATH}")
        print("Add both files to run the simple ATS analyzer.\n")
        return

    try:
        resume_text = load_resume_text(DEFAULT_RESUME_PATH)
        job_description = load_job_description_text(DEFAULT_JOB_DESCRIPTION_PATH)
    except OSError as error:
        print(f"Could not read resume or job description: {error}")
        return

    resume_analysis = analyze_resume(resume_text)
    job_analysis = analyze_job_description(job_description)
    ats_report = calculate_ats_score(resume_text, job_description)

    print(f"Resume skills: {format_list(resume_analysis['skills'])}")
    print(f"Required job skills: {format_list(job_analysis['required_skills'])}")
    print(f"ATS match score: {ats_report['score']}%")
    print(f"Matched skills: {format_list(ats_report['matched_skills'])}")
    print(f"Missing skills: {format_list(ats_report['missing_skills'])}")
    print("Improvement suggestions:")
    for index, suggestion in enumerate(ats_report["suggestions"], start=1):
        print(f"{index}. {suggestion}")
    print()


def run_pdf_rag() -> None:
    """Ingest data/documents/sample.pdf, ask one question, and print citations."""
    print("PDF RAG Q&A")
    print("-" * 11)

    if not SAMPLE_PDF_PATH.exists():
        print(f"Missing sample PDF: {SAMPLE_PDF_PATH}")
        print("Add a PDF at data/documents/sample.pdf to run PDF Q&A.")
        return

    try:
        question = input(f"Question [{DEFAULT_QUESTION}]: ").strip() or DEFAULT_QUESTION
    except EOFError:
        question = DEFAULT_QUESTION

    try:
        document = load_document(SAMPLE_PDF_PATH)
        chunks = chunk_text(document["text"])
        if not chunks:
            print(f"No text could be extracted from {SAMPLE_PDF_PATH}.")
            return

        store_chunks(
            chunks,
            source=document["source"],
            persist_directory=settings.vector_store_dir,
        )
        retrieved = retrieve_chunks(
            question,
            top_k=TOP_K,
            persist_directory=settings.vector_store_dir,
        )
        prompt = build_rag_prompt(question=question, retrieved_chunks=retrieved)
        answer = chat_with_openai(prompt)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PDF RAG failed: {error}")
        print("Check setup, dependencies, .env values, and the sample PDF, then rerun.")
        return

    print("\nAnswer:")
    print(answer)
    print("\n" + format_citations(retrieved))


def main() -> None:
    """Run the simple resume/JD analyzer, then the optional PDF RAG demo."""
    run_resume_job_match()
    run_pdf_rag()


if __name__ == "__main__":
    main()
