import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3"


def generate(
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response using the local Ollama model.
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
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]
