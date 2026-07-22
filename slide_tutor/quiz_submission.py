"""Streamlit-free quiz submission orchestration.

The UI collects answers, while this module owns the immutable submission
snapshot, type-specific grading order, final aggregation, diagnostics, and the
single persistence boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from slide_tutor.models import (
    QuizAttempt,
    QuizResult,
    QuizResultDetail,
    QuizSet,
    WeaknessReport,
)
from slide_tutor.projects import utc_now
from slide_tutor.quiz import (
    batch_wrong_reasons,
    generate_weakness_report,
    grade_fill_blank_answers,
    grade_short_answers,
    save_quiz_attempt,
    score_quiz,
)


@dataclass(frozen=True)
class QuizSubmissionOutcome:
    """Complete, persisted submission result returned to the UI."""

    attempt: QuizAttempt
    result: QuizResult
    weakness_report: WeaknessReport
    details: tuple[QuizResultDetail, ...]


def submit_quiz_answers(
    quiz_set: QuizSet,
    answers: dict[str, str | None],
    *,
    language: str | None = None,
    quiz_type: str | None = None,
    attempt_id: str | None = None,
) -> QuizSubmissionOutcome:
    """Grade and atomically save one submission, or raise before any save.

    Fill-blank and short-answer AI failures intentionally escape this function;
    callers can show a retry state while no provisional score or attempt exists.
    """
    # Freeze the user payload before any grader or diagnostic can mutate it.
    answer_snapshot = deepcopy(answers)
    details = score_quiz(quiz_set, answer_snapshot)
    target_language = language or quiz_set.language

    # Each AI answer grader runs before score aggregation and persistence.
    details = grade_fill_blank_answers(quiz_set, details, target_language)
    details = grade_short_answers(quiz_set, details, target_language)

    wrong_map = batch_wrong_reasons(quiz_set, details, target_language)
    for detail in details:
        if not detail.is_correct and detail.question_id in wrong_map:
            detail.wrong_reason = wrong_map[detail.question_id]

    score = sum(detail.points_awarded or 0.0 for detail in details)
    max_score = sum(detail.max_points for detail in details)
    resolved_attempt_id = attempt_id or f"attempt-{uuid4().hex}"
    result = QuizResult(
        attempt_id=resolved_attempt_id,
        score=score,
        total=len(details),
        max_score=max_score,
        details=details,
    )
    report = generate_weakness_report(quiz_set, details, target_language)

    resolved_quiz_type = quiz_type if quiz_type in {"per_pdf", "final_exam"} else None
    if resolved_quiz_type is None:
        resolved_quiz_type = "final_exam" if len(quiz_set.file_ids) > 1 else "per_pdf"
    attempt = QuizAttempt(
        attempt_id=resolved_attempt_id,
        quiz_set_id=quiz_set.quiz_set_id,
        quiz_type=resolved_quiz_type,
        project_id=quiz_set.project_id,
        file_ids=list(quiz_set.file_ids),
        total_count=quiz_set.total_count,
        answers=answer_snapshot,
        submitted_at=utc_now(),
    )
    # Save only after every required AI grading result is complete.
    save_quiz_attempt(quiz_set, attempt, result, report)
    return QuizSubmissionOutcome(
        attempt=attempt,
        result=result,
        weakness_report=report,
        details=tuple(details),
    )


__all__ = ["QuizSubmissionOutcome", "submit_quiz_answers"]
