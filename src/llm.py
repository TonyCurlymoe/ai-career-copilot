"""Small OpenAI chat helper.

This module intentionally keeps the LLM integration simple for the first
working version of the project. Later, this can grow into a provider interface
for local models, OpenAI, Azure OpenAI, Anthropic, and other backends.
"""

from importlib import import_module

from config import settings


def chat_with_openai(prompt: str) -> str:
    """Send a prompt to OpenAI Chat Completions and return the text response.

    TODO: Add retries, streaming, structured output, and provider selection.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example "
            "and add your OpenAI API key before running the app."
        )

    try:
        openai_module = import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency: openai. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    client = openai_module.OpenAI(api_key=settings.openai_api_key)
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as error:
        raise RuntimeError(f"OpenAI chat request failed: {error}") from error

    message = response.choices[0].message.content
    return message or ""
