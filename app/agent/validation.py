from app.generation.answer import Answer


def is_valid_answer(answer: Answer) -> bool:
    if not answer.text.strip():
        return False

    if not answer.citations:
        return False

    return True
