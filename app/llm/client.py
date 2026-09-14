import requests

from app.config import settings


def generate(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response using the configured Ollama model.
    """

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = requests.post(
        settings.ollama_url,
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]

