"""Quiz generation pipeline — Per-PDF quizzes and cross-PDF Final Exams.

Pipeline: sample pages → assemble context → generate candidates via LLM →
validate + dedupe → repair pass → cache QuizSet.

Also handles scoring, batch wrong-reason generation, and Weakness Report
(LLM-preferred with rule-based fallback).
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any

from slide_tutor import cache, config
from slide_tutor.llm import LLMError, complete_json
from slide_tutor.models import (
    QUIZ_PROMPT_VERSION,
    Citation,
    FileSlidesPayload,
    FillBlankGradeBatch,
    GenerationSnapshot,
    QuizAttempt,
    QuizDiagnosisFact,
    QuizQuestion,
    QuizResult,
    QuizResultDetail,
    QuizReuseSummary,
    ReviewedExamBundle,
    ShortAnswerGradeBatch,
    QuizSet,
    WeaknessItem,
    WeaknessReport,
    SlidePage,
)
from slide_tutor.page_understanding import load_page_understanding
from slide_tutor.projects import load_project, utc_now
from slide_tutor.quiz_formatting import format_quiz_answer, question_by_id
from slide_tutor.prompts import (
    PROMPT_BATCH_WRONG_REASON,
    PROMPT_GRADE_FILL_BLANKS,
    PROMPT_GENERATE_QUIZ,
    PROMPT_GRADE_SHORT_ANSWERS,
    PROMPT_WEAKNESS_REPORT,
)

logger = logging.getLogger(__name__)

# ── Config ──

_PER_PDF_DEFAULT = 6
_PER_PDF_MIN = 3
_PER_PDF_MAX = 15
_FINAL_DEFAULT = 20
_FINAL_MIN = 10
_FINAL_MAX = 40
_CANDIDATE_SURPLUS = 4       # generate target + this many to survive dedupe
_PER_PDF_SAMPLE = 15          # max pages to sample per PDF
_FINAL_SAMPLE_PER_PDF = 8     # max pages per PDF for Final Exam
_DEDUPE_THRESHOLD = 0.85
_QUESTION_TYPE_ORDER = ["mcq", "true_false", "fill_blank", "short_answer"]
_QUESTION_TYPE_RATIOS = {
    "mcq": 60,
    "true_false": 15,
    "fill_blank": 15,
    "short_answer": 10,
}
_QUESTION_TYPE_ALIASES = {
    "mcq": "mcq",
    "tf": "true_false",
    "true_false": "true_false",
    "true-false": "true_false",
    "fill": "fill_blank",
    "fill_blank": "fill_blank",
    "fill-blank": "fill_blank",
    "short": "short_answer",
    "short_answer": "short_answer",
    "short-answer": "short_answer",
}

_FILL_ALTERNATIVE_RE = re.compile(r"(?:[（(].*[）)]|[/、,，;；]|\bor\b|或)", re.IGNORECASE)
_FILL_MATH_RE = re.compile(r"(?:\$|\\(?:frac|sqrt|begin|end|\[|\()|[=^_])")
_FILL_QUESTION_NOTATION_RE = re.compile(r"(?:\$|\\(?:frac|sqrt|begin|end|\[|\()|[=^])")
_FILL_MATH_QUESTION_RE = re.compile(
    r"(?:公式|方程|变量|符号|等式|latex|formula|equation|variable|symbol|数式|方程式|変数|記号)",
    re.IGNORECASE,
)


# ── Helpers ──


def build_question_type_mix(target_count: int) -> dict[str, int]:
    """Strict-ratio question distribution using deterministic largest remainder."""
    target_count = max(0, int(target_count))
    mix = {
        qtype: target_count * _QUESTION_TYPE_RATIOS[qtype] // 100
        for qtype in _QUESTION_TYPE_ORDER
    }
    remainders = {
        qtype: target_count * _QUESTION_TYPE_RATIOS[qtype] % 100
        for qtype in _QUESTION_TYPE_ORDER
    }
    remaining = target_count - sum(mix.values())
    ranked = sorted(
        _QUESTION_TYPE_ORDER,
        key=lambda qtype: (-remainders[qtype], _QUESTION_TYPE_ORDER.index(qtype)),
    )
    for qtype in ranked[:remaining]:
        mix[qtype] += 1
    return mix


def format_question_type_mix(question_type_mix: dict[str, int]) -> str:
    """Stable string form used for prompt text, cache keys, and QuizSet snapshots."""
    return ",".join(f"{qtype}:{int(question_type_mix.get(qtype, 0))}" for qtype in _QUESTION_TYPE_ORDER)


def fill_blank_quality_issues(question: QuizQuestion) -> list[str]:
    """Reject only strong signals that a fill blank is not one short concept."""
    if question.question_type != "fill_blank":
        return []
    answer = question.blank_answer.strip()
    issues: list[str] = []
    if _FILL_ALTERNATIVE_RE.search(answer):
        issues.append("alternative_answer_signal")
    if _FILL_MATH_RE.search(answer) or _FILL_QUESTION_NOTATION_RE.search(question.question):
        issues.append("math_notation_signal")
    if _FILL_MATH_QUESTION_RE.search(question.question):
        issues.append("math_request_signal")
    if len(answer) > 80 or len(answer.split()) > 12:
        issues.append("answer_too_long")
    return issues


def _ensure_question_origin(question: QuizQuestion) -> None:
    if question.question_origin_id:
        return
    citation_identity = "|".join(
        citation.evidence_id
        or f"{citation.file_id}:{citation.slide_id}:{citation.page_number}"
        or f"{citation.source_file}:{citation.page_number}"
        for citation in question.citations
    )
    raw = "|".join([question.question_type, question.question, citation_identity])
    question.question_origin_id = f"origin-{sha256(raw.encode()).hexdigest()[:16]}"


def _normalize_question_type_mix(
    question_type_mix: dict[str, int] | str | None,
    target_count: int,
) -> dict[str, int]:
    if question_type_mix is None:
        return build_question_type_mix(target_count)

    parsed = {qtype: 0 for qtype in _QUESTION_TYPE_ORDER}
    if isinstance(question_type_mix, str):
        for raw_part in question_type_mix.split(","):
            part = raw_part.strip()
            if not part:
                continue
            key, sep, raw_value = part.partition(":")
            if not sep:
                raise ValueError(f"Invalid question_type_mix part: {part}")
            qtype = _QUESTION_TYPE_ALIASES.get(key.strip())
            if qtype is None:
                raise ValueError(f"Unknown question type in mix: {key}")
            parsed[qtype] = int(raw_value.strip())
    else:
        for key, value in question_type_mix.items():
            qtype = _QUESTION_TYPE_ALIASES.get(str(key).strip())
            if qtype is None:
                raise ValueError(f"Unknown question type in mix: {key}")
            parsed[qtype] = int(value)

    if any(count < 0 for count in parsed.values()):
        raise ValueError("question_type_mix counts cannot be negative")
    if sum(parsed.values()) != target_count:
        raise ValueError(
            f"question_type_mix totals {sum(parsed.values())}, expected {target_count}"
        )
    return parsed


def _describe_question_type_mix(question_type_mix: dict[str, int]) -> str:
    labels = {
        "mcq": "mcq",
        "true_false": "true_false",
        "fill_blank": "fill_blank",
        "short_answer": "short_answer",
    }
    return "; ".join(
        f"exactly {question_type_mix.get(qtype, 0)} {labels[qtype]}"
        for qtype in _QUESTION_TYPE_ORDER
    )


def _all_slides(project_id: str) -> list[tuple[str, SlidePage]]:
    """Yield (file_id, SlidePage) for every ready file in the project."""
    project = load_project(project_id)
    pairs: list[tuple[str, SlidePage]] = []
    for f in project.files:
        if f.status != "ready":
            continue
        data = cache.load_json(
            cache.get_project_dir(project_id) / "files" / f.file_id / config.SLIDES_FILENAME
        )
        if not data:
            continue
        payload = FileSlidesPayload.model_validate(data)
        for slide in payload.slides:
            pairs.append((f.file_id, slide))
    return pairs


def _usable_text(project_id: str, file_id: str, slide: SlidePage) -> str | None:
    """Return the best available text for a slide, preferring clean_markdown.
    Returns None if no usable text exists (quality == 'failed' and no raw)."""
    ud = load_page_understanding(project_id, file_id, slide.slide_id)
    if ud is not None and (ud.canonical_markdown or ud.clean_markdown).strip():
        return ud.canonical_markdown or ud.clean_markdown
    if slide.text.strip():
        return slide.text
    return None


def _quality_note(project_id: str, file_id: str, slide: SlidePage) -> str:
    """Return a quality label for use in quiz metadata."""
    ud = load_page_understanding(project_id, file_id, slide.slide_id)
    if ud is not None:
        return ud.quality
    return slide.text_quality


# ── Sampling ──


def _sample_pages(
    slides: list[tuple[str, SlidePage]],
    count: int,
    project_id: str,
) -> list[tuple[str, SlidePage]]:
    """Uniformly sample up to `count` pages, skipping those with no usable text
    and those marked quality='failed'.  Tries to keep coverage even across files."""
    # Filter to pages with usable text
    usable: list[tuple[str, SlidePage]] = []
    for file_id, slide in slides:
        qual = _quality_note(project_id, file_id, slide)
        if qual == "failed":
            continue
        if _usable_text(project_id, file_id, slide):
            usable.append((file_id, slide))

    if len(usable) <= count:
        return usable

    # Uniform sampling
    step = max(1, len(usable) // count)
    sampled: list[tuple[str, SlidePage]] = []
    for i in range(0, len(usable), step):
        sampled.append(usable[i])
        if len(sampled) >= count:
            break
    return sampled


# ── Context assembly ──


def _build_quiz_context(pages: list[tuple[str, SlidePage]], project_id: str, max_chars: int = 0) -> str:
    """Format sampled pages into a prompt context block.  Truncates to max_chars
    if given, otherwise uses config.MAX_CONTEXT_CHARS."""
    limit = max_chars if max_chars > 0 else config.MAX_CONTEXT_CHARS
    parts: list[str] = []
    total = 0
    for file_id, slide in pages:
        text = _usable_text(project_id, file_id, slide) or ""
        header = f"[{slide.source_file}, page {slide.page_number}]"
        block = f"{header}\n{text}\n"
        if total + len(block) > limit:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


# ── Generation + dedupe ──


def _generate_candidates(
    context: str,
    difficulty_mix: str,
    language: str,
    question_type_mix: dict[str, int],
    synthesis_hint: str = "",
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Call the LLM to generate candidate questions.  Returns the raw list of dicts."""
    target_count = sum(question_type_mix.values())
    prompt = PROMPT_GENERATE_QUIZ.format(
        language=language,
        context=context,
        target_count=target_count,
        difficulty_mix=difficulty_mix,
        question_type_mix=_describe_question_type_mix(question_type_mix),
        synthesis_hint=synthesis_hint,
    )
    result = complete_json(
        "You are a quiz generator.  Output ONLY valid JSON.",
        prompt,
        max_retries=max_retries,
        reasoning_effort="medium",
    )
    return result.get("questions", []) if isinstance(result, dict) else []


def _validate_and_dedupe(candidates: list[dict[str, Any]]) -> list[QuizQuestion]:
    """Pydantic-validate each candidate, drop invalid ones, deduplicate by question text.
    Auto-generates question_id for any candidate that lacks one (LLMs often omit it)."""
    valid: list[QuizQuestion] = []
    for i, c in enumerate(candidates):
        # D8: question_id is a system identifier — we generate it, not the LLM.
        c.setdefault("question_id", f"q-{i + 1}")
        try:
            q = QuizQuestion.model_validate(c)
            if fill_blank_quality_issues(q):
                logger.warning("Dropping fill-blank question with strong quality signal")
                continue
            _ensure_question_origin(q)
            valid.append(q)
        except Exception as exc:
            logger.warning("Dropping invalid question: %s", exc)

    # Dedupe by question text similarity
    deduped: list[QuizQuestion] = []
    for q in valid:
        is_dup = False
        for existing in deduped:
            ratio = SequenceMatcher(None, q.question, existing.question).ratio()
            if ratio > _DEDUPE_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            deduped.append(q)

    return deduped


def _select_questions_by_mix(
    candidates: list[QuizQuestion],
    question_type_mix: dict[str, int],
) -> list[QuizQuestion]:
    counts = {qtype: 0 for qtype in _QUESTION_TYPE_ORDER}
    selected: list[QuizQuestion] = []
    for question in candidates:
        qtype = question.question_type
        if counts.get(qtype, 0) >= question_type_mix.get(qtype, 0):
            continue
        selected.append(question)
        counts[qtype] += 1
    return selected


def _missing_question_type_mix(
    questions: list[QuizQuestion],
    required_mix: dict[str, int],
) -> dict[str, int]:
    counts = {qtype: 0 for qtype in _QUESTION_TYPE_ORDER}
    for question in questions:
        counts[question.question_type] += 1
    return {
        qtype: max(required_mix.get(qtype, 0) - counts.get(qtype, 0), 0)
        for qtype in _QUESTION_TYPE_ORDER
    }


def _repair_pass(
    context: str,
    existing: list[QuizQuestion],
    missing_mix: dict[str, int],
    difficulty_mix: str,
    language: str,
    synthesis_hint: str = "",
) -> list[QuizQuestion]:
    """Second attempt: ask only for the missing question types."""
    if sum(missing_mix.values()) == 0:
        return []
    banned = list({q.topic_tag for q in existing})
    hint = (
        f"\nIMPORTANT: Do NOT recreate questions on these topics: {', '.join(banned)}.\n"
        f"Generate ONLY these missing question types: {format_question_type_mix(missing_mix)}.\n"
    )
    try:
        new_candidates = _generate_candidates(
            context,
            difficulty_mix,
            language,
            missing_mix,
            synthesis_hint + hint,
            max_retries=1,
        )
        return _validate_and_dedupe(new_candidates)
    except Exception as exc:
        logger.warning("Repair pass failed: %s", exc)
        return []


def _generate_questions_with_repair(
    context: str,
    target_count: int,
    difficulty_mix: str,
    language: str,
    required_mix: dict[str, int],
    synthesis_hint: str = "",
) -> list[QuizQuestion]:
    candidate_mix = build_question_type_mix(target_count + _CANDIDATE_SURPLUS)
    candidate_mix = {
        qtype: max(candidate_mix[qtype], required_mix[qtype])
        for qtype in _QUESTION_TYPE_ORDER
    }
    candidates = _generate_candidates(
        context, difficulty_mix, language, candidate_mix, synthesis_hint
    )
    valid = _validate_and_dedupe(candidates)
    questions = _select_questions_by_mix(valid, required_mix)

    missing = _missing_question_type_mix(questions, required_mix)
    if sum(missing.values()) > 0:
        extra = _repair_pass(context, questions, missing, difficulty_mix, language, synthesis_hint)
        merged = _validate_and_dedupe([q.model_dump(mode="json") for q in [*questions, *extra]])
        questions = _select_questions_by_mix(merged, required_mix)
        missing = _missing_question_type_mix(questions, required_mix)

    if sum(missing.values()) > 0:
        raise LLMError(
            "Quiz generation did not satisfy the required question type mix. "
            f"Missing: {format_question_type_mix(missing)}"
        )

    if len(questions) != target_count:
        raise LLMError(
            f"Quiz generation returned {len(questions)} usable questions, expected {target_count}."
        )

    return questions


# ── Public: generate ──


def generate_per_pdf_quiz(
    project_id: str,
    file_id: str,
    target_count: int = _PER_PDF_DEFAULT,
    language: str = "中文",
    question_type_mix: dict[str, int] | str | None = None,
    generation_snapshot: GenerationSnapshot | None = None,
    allow_partial: bool = False,
) -> QuizSet:
    """Generate a quiz covering a single PDF."""
    target_count = max(1 if allow_partial else _PER_PDF_MIN, min(_PER_PDF_MAX, target_count))
    required_mix = _normalize_question_type_mix(question_type_mix, target_count)
    question_type_mix_text = format_question_type_mix(required_mix)
    all_slides = [(fid, s) for fid, s in _all_slides(project_id) if fid == file_id]
    pages = _sample_pages(all_slides, _PER_PDF_SAMPLE, project_id)
    context = _build_quiz_context(pages, project_id)

    basic_count = max(1, round(target_count * 0.6))
    applied_count = target_count - basic_count
    difficulty_mix = f"{basic_count} basic + {applied_count} applied"

    questions = _generate_questions_with_repair(
        context, target_count, difficulty_mix, language, required_mix
    )
    quiz_set_id = _compute_quiz_set_id(
        [file_id], len(questions), difficulty_mix, language, question_type_mix_text,
        generation_snapshot=generation_snapshot,
    )

    quiz_set = QuizSet(
        quiz_set_id=quiz_set_id,
        project_id=project_id,
        file_ids=[file_id],
        total_count=len(questions),
        difficulty_mix=difficulty_mix,
        question_type_mix=question_type_mix_text,
        questions=questions,
        created_at=utc_now(),
        model=config.LLM_MODEL,
        prompt_version=QUIZ_PROMPT_VERSION,
        language=language,
        generation_snapshot=generation_snapshot,
    )
    from slide_tutor.topics import resolve_quiz_topics
    return resolve_quiz_topics(quiz_set)


def generate_final_exam(
    project_id: str,
    target_count: int = _FINAL_DEFAULT,
    language: str = "中文",
    question_type_mix: dict[str, int] | str | None = None,
    generation_snapshot: GenerationSnapshot | None = None,
    allow_partial: bool = False,
) -> QuizSet:
    """Generate a comprehensive exam covering ALL PDFs in the project, with
    proportional page allocation and cross-PDF synthesis questions."""
    target_count = max(1 if allow_partial else _FINAL_MIN, min(_FINAL_MAX, target_count))
    required_mix = _normalize_question_type_mix(question_type_mix, target_count)
    question_type_mix_text = format_question_type_mix(required_mix)
    all_slides = _all_slides(project_id)
    if not all_slides:
        raise ValueError("No ready PDFs in this project.")

    # Count pages per file for proportional allocation
    from collections import Counter
    file_page_counts: Counter[str] = Counter()
    for fid, _ in all_slides:
        file_page_counts[fid] += 1
    total_pages = sum(file_page_counts.values())

    pages: list[tuple[str, SlidePage]] = []
    for fid, count in file_page_counts.items():
        file_slides = [(f, s) for f, s in all_slides if f == fid]
        alloc = max(1, int(_FINAL_SAMPLE_PER_PDF * count / max(total_pages / len(file_page_counts), 1)))
        pages.extend(_sample_pages(file_slides, min(alloc, _FINAL_SAMPLE_PER_PDF), project_id))

    # Limit total context
    max_chars = config.MAX_CONTEXT_CHARS
    context = _build_quiz_context(pages, project_id, max_chars=max_chars)

    basic_count = max(1, round(target_count * 0.5))
    applied_count = target_count - basic_count
    difficulty_mix = f"{basic_count} basic + {applied_count} applied"
    synthesis_hint = (
        "\nInclude 2-3 synthesis questions that require connecting concepts "
        "from DIFFERENT lectures/pdfs to answer correctly.\n"
    )

    questions = _generate_questions_with_repair(
        context, target_count, difficulty_mix, language, required_mix, synthesis_hint
    )
    quiz_set_id = _compute_quiz_set_id(
        list(file_page_counts.keys()), len(questions), difficulty_mix, language,
        question_type_mix_text, generation_snapshot=generation_snapshot,
    )

    quiz_set = QuizSet(
        quiz_set_id=quiz_set_id,
        project_id=project_id,
        file_ids=list(file_page_counts.keys()),
        total_count=len(questions),
        difficulty_mix=difficulty_mix,
        question_type_mix=question_type_mix_text,
        questions=questions,
        created_at=utc_now(),
        model=config.LLM_MODEL,
        prompt_version=QUIZ_PROMPT_VERSION,
        language=language,
        generation_snapshot=generation_snapshot,
    )
    from slide_tutor.topics import resolve_quiz_topics
    return resolve_quiz_topics(quiz_set)


# ── Scoring ──


def score_quiz(quiz_set: QuizSet, answers: dict[str, str | None]) -> list[QuizResultDetail]:
    """Score all questions against the answer dict.  Pure rules, no LLM.
    MCQ/TF use deterministic option-key scoring; fill-blank and short-answer
    remain pending until their respective batch AI graders run."""
    details: list[QuizResultDetail] = []
    for q in quiz_set.questions:
        user_ans = answers.get(q.question_id)
        qtype = getattr(q, "question_type", "mcq") or "mcq"

        if qtype == "fill_blank":
            # Fill-blank answers are intentionally not interpreted locally.
            is_correct = False
            points = None
        elif qtype == "short_answer":
            # Deferred to batch LLM grading — mark as pending
            is_correct = False
            points = None
        else:
            # MCQ / True-False: exact match
            is_correct = user_ans == q.correct_option
            points = 1.0 if is_correct else 0.0

        details.append(QuizResultDetail(
            question_id=q.question_id,
            question_type=qtype,
            topic_tag=q.topic_tag,
            topic_key=q.topic_key,
            topic_names=dict(q.topic_names),
            question=q.question,
            user_answer=user_ans,
            correct_option=_expected_answer_for_result(q),
            is_correct=is_correct,
            explanation=q.explanation,
            wrong_reason=None,
            citations=q.citations,
            points_awarded=points,
            max_points=1.0 if qtype != "short_answer" else 3.0,
            grading_feedback=None,
        ))
    return details


def _expected_answer_for_result(question: QuizQuestion) -> str:
    if question.question_type in ("mcq", "true_false"):
        return question.correct_option
    if question.question_type == "fill_blank":
        return question.blank_answer
    if question.question_type == "short_answer":
        return question.model_answer
    return ""


def _grade_fill_blank(user_answer: str | None, expected: str) -> bool:
    """Legacy helper retained for old imports; product scoring never calls it."""
    import re
    if not user_answer or not expected:
        return False
    a = re.sub(r"\s+", " ", user_answer.strip().lower())
    b = re.sub(r"\s+", " ", expected.strip().lower())
    if a == b:
        return True
    compact_a = re.sub(r"[\s，。,.、；;：:（）()\[\]【】\"'“”‘’]", "", a)
    compact_b = re.sub(r"[\s，。,.、；;：:（）()\[\]【】\"'“”‘’]", "", b)
    if compact_a == compact_b:
        return True
    if _is_short_cjk_answer(compact_a) or _is_short_cjk_answer(compact_b):
        return False
    # Levenshtein distance ≤ 2
    if len(a) < 2 or len(b) < 2:
        return False
    # Simple Levenshtein for short strings
    n, m = len(a), len(b)
    if abs(n - m) > 2:
        return False
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] <= 2


def _is_short_cjk_answer(value: str) -> bool:
    return len(value) <= 8 and any("\u4e00" <= char <= "\u9fff" for char in value)


def grade_fill_blank_answers(
    quiz_set: QuizSet,
    details: list[QuizResultDetail],
    language: str = "中文",
) -> list[QuizResultDetail]:
    """Grade every fill-blank answer in one strict AI binary-grading call."""
    fill_details = [d for d in details if d.question_type == "fill_blank"]
    if not fill_details:
        return details

    questions_by_id = {q.question_id: q for q in quiz_set.questions}
    items: list[dict[str, Any]] = []
    for detail in fill_details:
        question = questions_by_id.get(detail.question_id)
        if question is None:
            raise LLMError(f"Cannot grade fill-blank {detail.question_id}: question snapshot missing.")
        items.append({
            "question_id": detail.question_id,
            "question": detail.question,
            "student_answer": detail.user_answer or "",
            # The canonical answer is reference context for the grader, not a
            # local comparison target or candidate-answer set.
            "canonical_concept": question.blank_answer,
        })

    result = complete_json(
        "You are a careful binary fill-blank grader. Output ONLY valid JSON.",
        PROMPT_GRADE_FILL_BLANKS.format(
            language=language,
            fill_blank_answers_json=json.dumps(items, ensure_ascii=False, indent=2),
        ),
        max_retries=2,
        reasoning_effort="high",
    )
    try:
        response = FillBlankGradeBatch.model_validate(result)
    except Exception as exc:
        raise LLMError(f"Invalid fill-blank grading response: {exc}") from exc

    expected_ids = {detail.question_id for detail in fill_details}
    actual_ids = {grade.question_id for grade in response.grades}
    if actual_ids != expected_ids or len(response.grades) != len(fill_details):
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise LLMError(
            "Fill-blank grading result is incomplete or has unknown question ids "
            f"(missing={missing}, extra={extra})."
        )

    grades = {grade.question_id: grade for grade in response.grades}
    for detail in fill_details:
        grade = grades[detail.question_id]
        detail.is_correct = grade.is_correct
        detail.points_awarded = 1.0 if grade.is_correct else 0.0
        detail.grading_feedback = grade.feedback.strip()
    return details


def grade_short_answers(
    quiz_set: QuizSet,
    details: list[QuizResultDetail],
    language: str = "中文",
) -> list[QuizResultDetail]:
    """Grade all short-answer questions in one LLM call.

    Raises LLMError when grading fails so callers can avoid saving partial or
    pending attempts.
    """
    short_details = [d for d in details if d.question_type == "short_answer"]
    if not short_details:
        return details

    questions_by_id = {q.question_id: q for q in quiz_set.questions}
    items = []
    for detail in short_details:
        question = questions_by_id.get(detail.question_id)
        if question is None:
            raise LLMError(f"Cannot grade short answer {detail.question_id}: question snapshot missing.")
        items.append({
            "question_id": detail.question_id,
            "question": detail.question,
            "student_answer": detail.user_answer or "",
            "model_answer": question.model_answer,
            "scoring_rubric": question.scoring_rubric,
            "max_points": detail.max_points,
        })

    result = complete_json(
        "You are a fair short-answer grader.  Output ONLY valid JSON.",
        PROMPT_GRADE_SHORT_ANSWERS.format(
            language=language,
            short_answers_json=json.dumps(items, ensure_ascii=False, indent=2),
        ),
        max_retries=2,
        reasoning_effort="high",
    )
    try:
        response = ShortAnswerGradeBatch.model_validate(result)
    except ValueError as exc:
        raise LLMError("Invalid short-answer grading response.") from exc

    grades = {grade.question_id: grade for grade in response.grades}
    expected_ids = {detail.question_id for detail in short_details}
    if grades.keys() != expected_ids or len(response.grades) != len(short_details):
        raise LLMError("Short-answer grades must include exactly one score per question.")
    if any(grades[detail.question_id].points_awarded > detail.max_points for detail in short_details):
        raise LLMError("A short-answer grade exceeded the question's maximum score.")

    for detail in short_details:
        grade = grades[detail.question_id]
        detail.points_awarded = grade.points_awarded
        detail.is_correct = grade.points_awarded == detail.max_points
        detail.grading_feedback = grade.grading_feedback.strip()

    return details


# ── Batch wrong reasons ──


def batch_wrong_reasons(
    quiz_set: QuizSet, details: list[QuizResultDetail], language: str = "中文",
) -> dict[str, str]:
    """Call the LLM once to generate wrong_reason for ALL incorrect answers.
    Returns a dict mapping question_id → wrong_reason."""
    wrong_details = [
        d for d in details
        if not d.is_correct and d.question_type != "short_answer"
    ]
    if not wrong_details:
        return {}

    # Build compact JSON with just what the LLM needs
    wrong_items = []
    questions = question_by_id(quiz_set.questions)
    for d in wrong_details:
        question = questions.get(d.question_id)
        wrong_items.append({
            "question_id": d.question_id,
            "question": d.question,
            "user_answer": format_quiz_answer(question, d.user_answer),
            "correct_option": format_quiz_answer(question, d.correct_option),
            "explanation": d.explanation,
        })

    prompt = PROMPT_BATCH_WRONG_REASON.format(
        language=language,
        wrong_questions_json=json.dumps(wrong_items, ensure_ascii=False, indent=2),
    )
    try:
        result = complete_json(
            "You are a patient tutor.  Output ONLY valid JSON.",
            prompt,
            max_retries=1,
        )
        if isinstance(result, dict):
            return {k: str(v) for k, v in result.items()}
        return {}
    except Exception as exc:
        logger.warning("Batch wrong reasons failed: %s", exc)
        return {}


# ── Weakness Report ──


def _rule_based_weakness_report(details: list[QuizResultDetail]) -> WeaknessReport:
    """Fallback: group wrong answers by topic_tag, list citations.  No LLM needed."""
    wrong_details = [d for d in details if not d.is_correct]
    correct_details = [d for d in details if d.is_correct]

    # Group wrong by topic
    topic_wrong: dict[str, list[QuizResultDetail]] = {}
    for d in wrong_details:
        topic_wrong.setdefault(d.topic_tag, []).append(d)

    weaknesses: list[WeaknessItem] = []
    for topic, items in topic_wrong.items():
        all_citations: list[Citation] = []
        for item in items:
            all_citations.extend(item.citations)
        weaknesses.append(WeaknessItem(
            topic=topic,
            error_count=len(items),
            question_ids=[i.question_id for i in items],
            diagnosis=f"You missed {len(items)} question(s) on {topic}. Review the cited pages.",
            review_pages=all_citations[:5],  # limit to 5 most relevant
        ))

    # Strengths from correct answers grouped by topic
    topic_correct: dict[str, list[QuizResultDetail]] = {}
    for d in correct_details:
        if d.topic_tag in topic_wrong:
            continue
        topic_correct.setdefault(d.topic_tag, []).append(d)
    strengths: list[WeaknessItem] = []
    for topic, items in topic_correct.items():
        strengths.append(WeaknessItem(
            topic=topic,
            error_count=0,
            question_ids=[i.question_id for i in items],
            diagnosis=f"You got all {len(items)} question(s) on {topic} correct.",
        ))

    report_kind = "weakness" if weaknesses else "mastery"
    return WeaknessReport(
        weaknesses=weaknesses,
        strengths=strengths,
        generated_by="rule",
        report_kind=report_kind,
        quality_mode="standard",
        support_reviewed=True,
        quality_metadata={
            "fact_count": len(details),
            "unsupported_claim_count": 0,
            "fallback": True,
        },
    )


def build_quiz_diagnosis_facts(
    details: list[QuizResultDetail],
) -> list[QuizDiagnosisFact]:
    """Freeze only fully graded facts before any diagnosis call or persistence."""
    pending = [detail.question_id for detail in details if detail.points_awarded is None]
    if pending:
        raise LLMError(
            "Weakness Report requires complete grading; pending question(s): "
            + ", ".join(pending)
        )
    return [
        QuizDiagnosisFact(
            question_id=detail.question_id,
            question_type=detail.question_type,
            topic=detail.topic_tag,
            is_correct=detail.is_correct,
            points_awarded=float(detail.points_awarded or 0.0),
            max_points=detail.max_points,
            citations=list(detail.citations),
        )
        for detail in details
    ]


def _supported_diagnosis_items(
    raw_items: list[Any],
    *,
    facts_by_id: dict[str, QuizDiagnosisFact],
    expected_correct: bool,
) -> tuple[list[WeaknessItem], int]:
    """Keep claims with valid fact ids and rebuild pages from those facts."""
    supported: list[WeaknessItem] = []
    rejected = 0
    for raw in raw_items:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        requested_ids = [str(item) for item in raw.get("question_ids", [])]
        question_ids = list(dict.fromkeys(
            question_id for question_id in requested_ids
            if question_id in facts_by_id
            and facts_by_id[question_id].is_correct is expected_correct
        ))
        if not question_ids or len(question_ids) != len(set(requested_ids)):
            rejected += 1
            continue
        citations: list[Citation] = []
        seen_citations: set[tuple[str, int, str]] = set()
        for question_id in question_ids:
            for citation in facts_by_id[question_id].citations:
                identity = (citation.source_file, citation.page_number, citation.evidence_id)
                if identity in seen_citations:
                    continue
                seen_citations.add(identity)
                citations.append(citation)
        supported.append(WeaknessItem(
            topic=str(raw.get("topic", "")).strip() or facts_by_id[question_ids[0]].topic,
            error_count=0 if expected_correct else len(question_ids),
            question_ids=question_ids,
            diagnosis=str(raw.get("diagnosis", "")).strip(),
            review_pages=citations[:5],
        ))
    return supported, rejected


def generate_weakness_report(
    quiz_set: QuizSet, details: list[QuizResultDetail], language: str = "中文",
) -> WeaknessReport:
    """Generate a knowledge diagnosis.  LLM preferred, rule-based fallback."""
    facts = build_quiz_diagnosis_facts(details)
    facts_by_id = {fact.question_id: fact for fact in facts}
    wrong_details = [d for d in details if not d.is_correct]
    if not wrong_details:
        return _rule_based_weakness_report(details)

    # Build quiz summary for the prompt
    summary_items = []
    questions = question_by_id(quiz_set.questions)
    for d in details:
        question = questions.get(d.question_id)
        summary_items.append({
            "question_id": d.question_id,
            "question_type": d.question_type,
            "topic_tag": d.topic_tag,
            "question": d.question,
            "user_answer": format_quiz_answer(question, d.user_answer),
            "correct_option": format_quiz_answer(question, d.correct_option),
            "is_correct": d.is_correct,
            "points_awarded": d.points_awarded,
            "max_points": d.max_points,
            "grading_feedback": d.grading_feedback,
            "citations": [c.model_dump(mode="json") for c in (d.citations or [])],
        })

    prompt = PROMPT_WEAKNESS_REPORT.format(
        language=language,
        quiz_summary_json=json.dumps(summary_items, ensure_ascii=False, indent=2),
    )
    unsupported_count = 0
    try:
        result = complete_json(
            "You are a course instructor.  Output ONLY valid JSON.",
            prompt,
            max_retries=1,
        )
        if isinstance(result, dict):
            weaknesses, rejected_weaknesses = _supported_diagnosis_items(
                result.get("weaknesses", []),
                facts_by_id=facts_by_id,
                expected_correct=False,
            )
            strengths, rejected_strengths = _supported_diagnosis_items(
                result.get("strengths", []),
                facts_by_id=facts_by_id,
                expected_correct=True,
            )
            rejected = rejected_weaknesses + rejected_strengths
            unsupported_count = rejected
            if weaknesses:
                return WeaknessReport(
                    weaknesses=weaknesses,
                    strengths=strengths,
                    generated_by="llm",
                    report_kind="weakness",
                    quality_mode="reviewed",
                    support_reviewed=True,
                    quality_metadata={
                        "fact_count": len(facts),
                        "unsupported_claim_count": rejected,
                        "fallback": False,
                    },
                )
    except Exception as exc:
        logger.warning("LLM Weakness Report failed, using rule-based fallback: %s", exc)

    fallback = _rule_based_weakness_report(details)
    fallback.quality_metadata["unsupported_claim_count"] = unsupported_count
    return fallback


# ── Cache ──


def _content_quiz_set_id(quiz_set: QuizSet) -> str:
    payload = {
        "request": quiz_set.generation_request_signature,
        "questions": [
            {
                "origin": question.question_origin_id,
                "type": question.question_type,
                "question": question.question,
            }
            for question in quiz_set.questions
        ],
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def _matching_fast_snapshot(
    snapshot: QuizSet,
    *,
    project_id: str,
    file_ids: list[str],
    language: str,
    question_type_mix: str,
    request_signature: str,
    generation_snapshot: GenerationSnapshot | None,
) -> bool:
    if snapshot.quality_mode != "fast" or snapshot.project_id != project_id:
        return False
    if sorted(snapshot.file_ids) != sorted(file_ids):
        return False
    if snapshot.language != language or snapshot.question_type_mix != question_type_mix:
        return False
    if snapshot.generation_request_signature:
        return snapshot.generation_request_signature == request_signature
    if snapshot.generation_snapshot and generation_snapshot:
        return snapshot.generation_snapshot == generation_snapshot
    return True


def collect_reusable_wrong_questions(
    project_id: str,
    *,
    file_ids: list[str],
    language: str,
    question_type_mix: str,
    request_signature: str,
    generation_snapshot: GenerationSnapshot | None,
    required_mix: dict[str, int],
) -> tuple[list[QuizQuestion], list[str]]:
    """Return unresolved Fast questions, respecting the requested type quotas."""
    states: dict[str, dict[str, Any]] = {}
    records = sorted(
        list_quiz_attempts(project_id, limit=None),
        key=lambda record: str(record.get("attempt", {}).get("submitted_at", "")),
    )
    for record in records:
        raw_snapshot = record.get("quiz_set_snapshot")
        if not raw_snapshot:
            continue
        try:
            snapshot = QuizSet.model_validate(raw_snapshot)
        except Exception:
            continue
        if not _matching_fast_snapshot(
            snapshot,
            project_id=project_id,
            file_ids=file_ids,
            language=language,
            question_type_mix=question_type_mix,
            request_signature=request_signature,
            generation_snapshot=generation_snapshot,
        ):
            continue
        questions = {question.question_id: question for question in snapshot.questions}
        attempt_id = str(record.get("attempt", {}).get("attempt_id", ""))
        submitted_at = str(record.get("attempt", {}).get("submitted_at", ""))
        for detail in record.get("result", {}).get("details", []):
            question = questions.get(str(detail.get("question_id", "")))
            if question is None or fill_blank_quality_issues(question):
                continue
            _ensure_question_origin(question)
            state = states.setdefault(question.question_origin_id, {
                "question": question, "is_correct": False, "wrong_count": 0,
                "submitted_at": "", "attempt_id": "",
            })
            state["question"] = question
            state["is_correct"] = bool(detail.get("is_correct"))
            state["wrong_count"] += 0 if detail.get("is_correct") else 1
            state["submitted_at"] = submitted_at
            state["attempt_id"] = attempt_id

    candidates = sorted(
        (state for state in states.values() if not state["is_correct"]),
        key=lambda state: (state["wrong_count"], state["submitted_at"]),
        reverse=True,
    )
    counts = {question_type: 0 for question_type in _QUESTION_TYPE_ORDER}
    reused: list[QuizQuestion] = []
    source_attempt_ids: list[str] = []
    for state in candidates:
        question = state["question"]
        if counts[question.question_type] >= required_mix.get(question.question_type, 0):
            continue
        reused.append(question.model_copy(deep=True, update={
            "reused_from_attempt_id": state["attempt_id"] or None,
        }))
        counts[question.question_type] += 1
        if state["attempt_id"] and state["attempt_id"] not in source_attempt_ids:
            source_attempt_ids.append(state["attempt_id"])
    return reused, source_attempt_ids


def finalize_fast_quiz_set(
    quiz_set: QuizSet,
    *,
    request_signature: str,
    generation_snapshot: GenerationSnapshot | None,
    reused_count: int,
    generated_count: int,
    source_attempt_ids: list[str],
) -> QuizSet:
    for index, question in enumerate(quiz_set.questions, start=1):
        _ensure_question_origin(question)
        question.question_id = f"q-{index}"
    quiz_set.generation_request_signature = request_signature
    quiz_set.generation_snapshot = generation_snapshot
    quiz_set.reuse_summary = QuizReuseSummary(
        reused_count=reused_count,
        generated_count=generated_count,
        source_attempt_ids=source_attempt_ids,
    )
    quiz_set.quiz_set_id = _content_quiz_set_id(quiz_set)
    return quiz_set


def _compute_quiz_set_id(file_ids: list[str], target_count: int,
                         difficulty_mix: str, language: str,
                         question_type_mix: str,
                         *, quality_mode: str = "fast",
                         generation_snapshot: GenerationSnapshot | None = None) -> str:
    """Stable cache key from all parameters that affect quiz output.  Changing
    model, prompt version, language, or question distribution = different quiz."""
    key_parts = [
        *sorted(file_ids),
        str(target_count),
        difficulty_mix,
        question_type_mix,
        language,
        quality_mode,
        generation_snapshot.model_dump_json() if generation_snapshot else "",
        config.LLM_MODEL,
        QUIZ_PROMPT_VERSION,
    ]
    raw = "|".join(key_parts)
    return sha256(raw.encode()).hexdigest()[:12]


def _quiz_cache_path(project_id: str, quiz_set_id: str) -> Path:
    return cache.get_project_dir(project_id) / f"quiz_cache_{quiz_set_id}.json"


def _reviewed_metadata_path(project_id: str, quiz_set_id: str) -> Path:
    return cache.get_project_dir(project_id) / f"reviewed_exam_metadata_{quiz_set_id}.json"


def _reviewed_bundle_path(project_id: str, quiz_set_id: str) -> Path:
    return cache.get_project_dir(project_id) / f"reviewed_exam_bundle_{quiz_set_id}.json"


def _save_json_atomic(path: Path, payload: Any) -> None:
    """Expose a complete cache artifact only after serialization succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{sha256(os.urandom(16)).hexdigest()[:10]}")
    try:
        cache.save_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cached_quiz(project_id: str, quiz_set_id: str) -> QuizSet | None:
    """Try to load a cached QuizSet by its stable id.  Returns None on miss."""
    bundle_raw = cache.load_json(_reviewed_bundle_path(project_id, quiz_set_id), default=None)
    if bundle_raw:
        try:
            return ReviewedExamBundle.model_validate(bundle_raw).quiz_set
        except Exception:
            return None
    path = _quiz_cache_path(project_id, quiz_set_id)
    data = cache.load_json(path)
    if not data:
        return None
    try:
        return QuizSet.model_validate(data)
    except Exception:
        return None


def save_cached_quiz(quiz_set: QuizSet) -> None:
    """Persist a QuizSet to the project cache.  The quiz_set.quiz_set_id field
    must be set before calling this."""
    if quiz_set.quality_mode == "reviewed" and quiz_set.reviewed_metadata is not None:
        bundle = ReviewedExamBundle(
            quiz_set=quiz_set,
            metadata=quiz_set.reviewed_metadata,
        )
        _save_json_atomic(
            _reviewed_bundle_path(quiz_set.project_id, quiz_set.quiz_set_id), bundle,
        )
        return
    path = _quiz_cache_path(quiz_set.project_id, quiz_set.quiz_set_id)
    _save_json_atomic(path, quiz_set)


def load_reviewed_exam_metadata(project_id: str, quiz_set_id: str):
    """Load Reviewed metadata separately for audit/UI consumers."""
    from slide_tutor.models import ReviewedExamMetadata

    bundle_raw = cache.load_json(_reviewed_bundle_path(project_id, quiz_set_id), default=None)
    if bundle_raw:
        try:
            return ReviewedExamBundle.model_validate(bundle_raw).metadata
        except Exception:
            return None
    raw = cache.load_json(_reviewed_metadata_path(project_id, quiz_set_id), default=None)
    if not raw:
        return None
    try:
        return ReviewedExamMetadata.model_validate(raw)
    except Exception:
        return None


# ── Attempt storage ──


def _attempts_dir(project_id: str) -> Path:
    return cache.get_project_dir(project_id) / "quiz_attempts"


def _attempt_path(project_id: str, attempt_id: str) -> Path:
    return _attempts_dir(project_id) / f"{attempt_id}.json"


def save_quiz_attempt(quiz_set: QuizSet, attempt: QuizAttempt, result: QuizResult,
                      weakness_report: WeaknessReport | None) -> None:
    """Persist a completed quiz attempt + its result + a QuizSet snapshot so
    history can fully recover questions even if the cache is later deleted."""
    attempts_dir = _attempts_dir(attempt.project_id)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt": attempt.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "weakness_report": weakness_report.model_dump(mode="json") if weakness_report else None,
        "quiz_set_snapshot": quiz_set.model_dump(mode="json"),
    }
    cache.save_json(attempts_dir / f"{attempt.attempt_id}.json", record)


def _read_quiz_attempt(path: Path) -> dict | None:
    """Correct historical partial-credit classifications in memory."""
    record = cache.load_json(path)
    if not record:
        return record
    result = record.get("result") or {}
    corrected = False
    for detail in result.get("details", []):
        points = detail.get("points_awarded")
        maximum = detail.get("max_points")
        if (detail.get("question_type") == "short_answer" and detail.get("is_correct")
                and isinstance(points, (int, float)) and isinstance(maximum, (int, float))
                and points < maximum):
            detail["is_correct"] = False
            corrected = True
    if corrected and record.get("weakness_report"):
        details = QuizResult.model_validate(result).details
        record["weakness_report"] = _rule_based_weakness_report(details).model_dump(mode="json")
    return record


def load_quiz_attempt(project_id: str, attempt_id: str) -> dict | None:
    """Load a saved attempt record."""
    return _read_quiz_attempt(_attempt_path(project_id, attempt_id))


def _quiz_set_has_attempts(project_id: str, quiz_set_id: str) -> bool:
    attempts_dir = _attempts_dir(project_id)
    if not attempts_dir.exists():
        return False
    for path in attempts_dir.glob("*.json"):
        data = cache.load_json(path)
        if not data:
            continue
        attempt = data.get("attempt", {})
        if attempt.get("quiz_set_id") == quiz_set_id:
            return True
    return False


def delete_quiz_attempt(project_id: str, attempt_id: str) -> bool:
    """Delete one attempt and remove its QuizSet cache if it becomes orphaned."""
    path = _attempt_path(project_id, attempt_id)
    if not path.exists():
        return False

    data = cache.load_json(path) or {}
    attempt = data.get("attempt", {}) if isinstance(data, dict) else {}
    quiz_set_id = str(attempt.get("quiz_set_id", "")).strip()

    path.unlink()
    if quiz_set_id and not _quiz_set_has_attempts(project_id, quiz_set_id):
        _quiz_cache_path(project_id, quiz_set_id).unlink(missing_ok=True)
        _reviewed_bundle_path(project_id, quiz_set_id).unlink(missing_ok=True)
        _reviewed_metadata_path(project_id, quiz_set_id).unlink(missing_ok=True)
    return True


def list_quiz_attempts(project_id: str, limit: int | None = 20) -> list[dict]:
    """List recent quiz attempts for the history sidebar."""
    attempts_dir = _attempts_dir(project_id)
    if not attempts_dir.exists():
        return []
    records: list[dict] = []
    for f in sorted(attempts_dir.glob("*.json"), reverse=True):
        data = _read_quiz_attempt(f)
        if data:
            records.append(data)
        if limit is not None and len(records) >= limit:
            break
    return records
