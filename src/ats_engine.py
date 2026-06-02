"""Simple ATS-style skill matching helpers.

This module keeps the scoring intentionally explainable: it looks for skills
from one predefined list, compares resume skills against job-description skills,
and calculates `matched skills / required skills * 100`.
"""

from __future__ import annotations

import re


PREDEFINED_SKILLS = [
    "python",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "machine learning",
    "deep learning",
    "nlp",
    "rag",
    "langchain",
    "llamaindex",
    "openai",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "fastapi",
    "flask",
    "streamlit",
    "react",
    "javascript",
    "typescript",
    "html",
    "css",
    "data analysis",
    "data visualization",
    "project management",
    "communication",
    "leadership",
]


def _skill_pattern(skill: str) -> re.Pattern[str]:
    """Build a case-insensitive pattern for one skill phrase."""
    escaped = re.escape(skill).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def extract_skills(text: str, skill_list: list[str] | None = None) -> list[str]:
    """Extract skills that appear in text using a predefined skill list.

    TODO: Replace this exact keyword approach with configurable taxonomies or
    model-assisted extraction once the project needs deeper analysis.
    """
    skills = skill_list or PREDEFINED_SKILLS
    found = [skill for skill in skills if _skill_pattern(skill).search(text)]
    return sorted(found)


def build_improvement_suggestions(missing_skills: list[str]) -> list[str]:
    """Return exactly three simple, actionable improvement suggestions."""
    if missing_skills:
        top_missing = ", ".join(missing_skills[:5])
        first = f"Add truthful resume evidence for these job skills: {top_missing}."
    else:
        first = "Keep the resume aligned to the job description by preserving relevant skill keywords."

    return [
        first,
        "Add 2-3 measurable achievements that show business impact, scope, or results.",
        "Tailor the summary and project bullets so they mirror the job's most important requirements.",
    ]


def calculate_ats_score(resume_text: str, job_description: str) -> dict[str, object]:
    """Calculate a basic ATS match score from extracted skills.

    Score formula: matched skills / required skills * 100.
    """
    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_description)
    matched_skills = sorted(set(resume_skills).intersection(required_skills))
    missing_skills = sorted(set(required_skills).difference(resume_skills))

    if not required_skills:
        score = 0.0
    else:
        score = round((len(matched_skills) / len(required_skills)) * 100, 2)

    return {
        "score": score,
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": build_improvement_suggestions(missing_skills),
    }
