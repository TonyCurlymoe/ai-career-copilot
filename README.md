# AI Career Copilot

AI Career Copilot is a Python starter project for a RAG-based career assistant.
This version includes two beginner-friendly workflows:

1. A simple resume-to-job-description analyzer that reads `data/resumes/resume.txt` and `data/jobs/job_description.txt`, extracts skills from a predefined list, calculates an ATS-style match score, and prints matched skills, missing skills, and three improvement suggestions.
2. A basic PDF RAG pipeline that loads `data/documents/sample.pdf`, chunks it, embeds chunks with `sentence-transformers` model `all-MiniLM-L6-v2`, stores/retrieves with ChromaDB, and sends retrieved context to an OpenAI chat model.

The long-term goal is to support document Q&A, resume-to-job-description matching,
ATS scoring, skill gap analysis, and local/cloud LLM support.

## Project structure

```text
.
├── data/
│   ├── documents/        # Put sample.pdf here
│   ├── jobs/             # Put job_description.txt here
│   └── resumes/          # Put resume.txt here
├── logs/                 # Future application logs
├── src/
│   ├── app.py            # Runs resume/JD matching and optional PDF RAG
│   ├── chunker.py        # 800-character chunks with 100 overlap
│   ├── config.py         # Loads .env settings
│   ├── document_loader.py # PDF loading with pypdf
│   ├── embeddings.py     # sentence-transformers embeddings
│   ├── llm.py            # Simple OpenAI chat function
│   ├── prompt_builder.py # Basic RAG prompt template
│   ├── retriever.py      # ChromaDB storage and retrieval
│   ├── resume_analyzer.py # Resume skill extraction
│   ├── jd_analyzer.py    # Job-description skill extraction
│   └── ats_engine.py     # Simple ATS score and suggestions
├── vector_store/         # ChromaDB persistence directory
├── .env.example
└── requirements.txt
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may download the `all-MiniLM-L6-v2` embedding model.

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
OPENAI_API_KEY="your_openai_api_key_here"
LLM_MODEL="gpt-4o-mini"
```

### 4. Add resume and job description text files

Create these two plain-text files:

```text
data/resumes/resume.txt
data/jobs/job_description.txt
```

The analyzer uses a predefined skill list, so mention skills plainly, such as `Python`, `SQL`, `AWS`, or `machine learning`.

### 5. Optional: add a sample PDF

To also run PDF Q&A, place one PDF at:

```text
data/documents/sample.pdf
```

## Run

Start the app:

```bash
python src/app.py
```

The app will first run the resume/job-description analyzer. It prints:

- resume skills found from the predefined list,
- required job skills found from the predefined list,
- ATS match score using `matched skills / required skills * 100`,
- matched skills,
- missing skills,
- three improvement suggestions.

If `data/documents/sample.pdf` exists, the app also runs PDF Q&A. It will ask for one question, retrieve the top matching chunks from ChromaDB, send the RAG prompt to OpenAI, and print the answer plus chunk citations.

For PDF Q&A, if you press Enter without typing a question, the app uses:

```text
What are the key points in this document?
```

## Notes

- Keep real API keys in `.env`; do not commit them.
- ChromaDB files are stored in `vector_store/`.
- This is intentionally small and beginner-friendly. TODO comments mark places
  where production features can be added later.

## Quick checks

Run these commands after installing dependencies to verify the project:

```bash
python -m compileall src
PYTHONPATH=src python - <<'PY'
import importlib
for module_name in [
    "app",
    "ats_engine",
    "chunker",
    "config",
    "document_loader",
    "embeddings",
    "jd_analyzer",
    "llm",
    "prompt_builder",
    "resume_analyzer",
    "retriever",
]:
    importlib.import_module(module_name)
print("All project modules imported successfully.")
PY
python src/app.py
```

If you only want to test the resume/JD analyzer, create `data/resumes/resume.txt`
and `data/jobs/job_description.txt` first. PDF Q&A additionally requires
`data/documents/sample.pdf`, `OPENAI_API_KEY`, and the embedding/vector-store
dependencies from `requirements.txt`.

## Troubleshooting

- `Missing dependency`: run `pip install -r requirements.txt` inside your active virtual environment.
- `OPENAI_API_KEY is missing`: copy `.env.example` to `.env` and add your key.
- `Missing resume or job description file`: create `data/resumes/resume.txt` and `data/jobs/job_description.txt`.
- `Missing sample PDF`: add `data/documents/sample.pdf` if you want to run PDF Q&A.
