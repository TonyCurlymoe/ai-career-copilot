"""Configuration for AI Career Copilot.

The app reads values from a local `.env` file first, then falls back to normal
environment variables. Keep real API keys in `.env` and out of source control.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_dotenv_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from a `.env` file into the environment.

    TODO: Replace this tiny parser with richer settings validation as the app
    grows. It is intentionally small so beginners can understand the flow.
    """
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv_file()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "vector_store")


settings = Settings()
