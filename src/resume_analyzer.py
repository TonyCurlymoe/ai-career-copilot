"""Simple resume analyzer based on predefined skill keywords."""

from pathlib import Path

from ats_engine import extract_skills


DEFAULT_RESUME_PATH = Path("data/resumes/resume.txt")


def load_resume_text(file_path: str | Path = DEFAULT_RESUME_PATH) -> str:
    """Load resume text from data/resumes/resume.txt by default."""
    return Path(file_path).read_text(encoding="utf-8")


def analyze_resume(resume_text: str) -> dict[str, object]:
    """Extract basic resume details in a simple, explainable way.

    TODO: Add section parsing for experience, education, projects, and metrics.
    """
    skills = extract_skills(resume_text)
    return {
        "word_count": len(resume_text.split()),
        "skills": skills,
        "skill_count": len(skills),
    }
