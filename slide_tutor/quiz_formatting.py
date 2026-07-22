"""Shared answer formatting for quiz UI, export, and diagnostic prompts."""

from __future__ import annotations

from slide_tutor.models import QuizQuestion


def format_quiz_answer(question: QuizQuestion | None, answer: str | None) -> str:
    """Expand option keys while preserving free text and legacy cache values."""
    if answer is None or not str(answer).strip():
        return "—"
    value = str(answer).strip()
    if question is None:
        return value
    if question.question_type in {"mcq", "true_false"} and value in question.options:
        return f"{value}. {question.options[value]}"
    return value


def question_by_id(questions: list[QuizQuestion]) -> dict[str, QuizQuestion]:
    return {question.question_id: question for question in questions}
