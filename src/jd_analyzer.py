"""Simple job description analyzer based on predefined skill keywords."""

from pathlib import Path

from ats_engine import extract_skills


DEFAULT_JOB_DESCRIPTION_PATH = Path("data/jobs/job_description.txt")


def load_job_description_text(
    file_path: str | Path = DEFAULT_JOB_DESCRIPTION_PATH,
) -> str:
    """Load job description text from data/jobs/job_description.txt by default."""
    return Path(file_path).read_text(encoding="utf-8")


def analyze_job_description(job_description: str) -> dict[str, object]:
    """Extract required skills from a job description using keyword matching.

    TODO: Separate required skills from preferred skills when richer parsing is
    added later.
    """
    required_skills = extract_skills(job_description)
    return {
        "word_count": len(job_description.split()),
        "required_skills": required_skills,
        "required_skill_count": len(required_skills),
    }
