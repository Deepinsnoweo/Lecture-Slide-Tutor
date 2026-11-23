"""Learning Dashboard — aggregates QuizAttempt history into lesson-level,
final-test, topic-level weakness, and review-priority summaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log1p
from typing import Any

from slide_tutor.projects import load_project, ordered_files
from slide_tutor.quiz import list_quiz_attempts


def _parse_date(date_str: str) -> datetime:
    """Parse ISO-ish date strings flexibly."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d"):
        try:
            # Strip timezone suffix if present
            clean = date_str.replace("Z", "").split("+")[0].split("[")[0]
            return datetime.strptime(clean[:26], fmt[:len(clean)] if len(clean) <= 10 else fmt).replace(tzinfo=UTC)
        except (ValueError, IndexError):
            continue
    return datetime.now(UTC) - timedelta(days=365)  # very old fallback


def _today() -> datetime:
    return datetime.now(UTC)


def _attempt_quiz_type(attempt: dict[str, Any]) -> str:
    """Classify current and legacy attempt records."""
    quiz_type = attempt.get("quiz_type")
    if quiz_type in ("per_pdf", "final_exam"):
        return str(quiz_type)
    file_ids = attempt.get("file_ids", [])
    return "final_exam" if len(file_ids) > 1 else "per_pdf"


def _score_pct(score: Any, total: Any) -> float:
    score_f = _safe_float(score)
    total_f = _safe_float(total)
    if total_f <= 0:
        return 0.0
    return max(0.0, min(score_f / total_f, 1.0))


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _citation_key(citation: dict[str, Any]) -> tuple[str, int, str]:
    try:
        page_number = int(citation.get("page_number", 0))
    except (TypeError, ValueError):
        page_number = 0
    return (
        str(citation.get("file_id") or citation.get("source_file", "")),
        page_number,
        str(citation.get("snippet_reference", ""))[:80],
    )


def _append_unique_citations(
    target: list[dict[str, Any]],
    seen: set[tuple[str, int, str]],
    citations: list[dict[str, Any]],
) -> None:
    for citation in citations:
        key = _citation_key(citation)
        if key in seen:
            continue
        seen.add(key)
        target.append(citation)


def _project_file_labels(project_id: str) -> dict[str, str]:
    try:
        project = load_project(project_id)
    except Exception:
        return {}
    return {file.file_id: file.original_name for file in ordered_files(project)}


def build_dashboard_data(
    project_id: str, limit: int = 50, language: str = "简体中文",
) -> dict[str, Any]:
    """Aggregate quiz history into dashboard-ready data structures."""
    del limit  # Topic state requires chronological project history, not a sliding score window.
    attempts = list_quiz_attempts(project_id, limit=None)
    attempts.sort(key=lambda rec: _parse_date(str(rec.get("attempt", {}).get("submitted_at", ""))))
    today = _today()
    file_labels = _project_file_labels(project_id)

    # Topic-level aggregation intentionally includes both per-PDF quizzes and
    # final exams: final-test mistakes are course-level weakness evidence.
    topics: dict[str, dict[str, Any]] = {}
    score_trend: list[dict[str, Any]] = []
    per_lesson_raw: dict[str, dict[str, Any]] = {}
    final_tests: list[dict[str, Any]] = []

    for rec in attempts:
        att = rec.get("attempt", {})
        res = rec.get("result", {})
        details = res.get("details", [])
        snapshot = rec.get("quiz_set_snapshot", {}) or {}
        attempt_language = str(snapshot.get("language") or "简体中文")

        def topic_identity(detail: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
            from slide_tutor.topics import display_topic_name, legacy_topic_key

            raw_tag = str(detail.get("topic_tag") or "Unknown")
            names = {
                str(key): str(value) for key, value in (detail.get("topic_names") or {}).items()
                if str(value).strip()
            }
            topic_key = str(detail.get("topic_key") or "").strip() or legacy_topic_key(
                attempt_language, raw_tag,
            )
            return topic_key, display_topic_name(topic_key, names, language, raw_tag), names
        quiz_type = _attempt_quiz_type(att)
        file_ids = list(att.get("file_ids", []) or [])
        sub_date = att.get("submitted_at", "")
        total = res.get("total", att.get("total_count", 0))
        max_score = res.get("max_score") or total
        score = res.get("score", 0)
        score_pct = _score_pct(score, max_score)

        # Legacy overview trend, kept for compatibility but no longer used as
        # the primary dashboard diagnosis.
        if sub_date:
            try:
                score_trend.append({
                    "date": _parse_date(sub_date),
                    "score": float(score or 0),
                    "total": float(max_score or 0),
                    "score_pct": score_pct,
                    "quiz_type": quiz_type,
                })
            except (ValueError, TypeError):
                pass

        wrong_details = [d for d in details if not d.get("is_correct")]
        weak_topics = sorted({
            topic_identity(d)[1]
            for d in wrong_details
        })

        if quiz_type == "per_pdf" and file_ids:
            file_id = file_ids[0]
            lesson = per_lesson_raw.setdefault(file_id, {
                "file_id": file_id,
                "lesson_name": file_labels.get(file_id, file_id),
                "attempts": [],
                "wrong_count": 0,
                "weak_topic_counts": {},
                "citations": [],
                "_citation_seen": set(),
            })
            lesson["attempts"].append({
                    "submitted_at": sub_date,
                    "score_pct": score_pct,
                    "score": _safe_float(score),
                    "total": _safe_float(max_score),
                })
            lesson["wrong_count"] += len(wrong_details)
            for detail in wrong_details:
                _topic_key, topic, _topic_names = topic_identity(detail)
                lesson["weak_topic_counts"][topic] = lesson["weak_topic_counts"].get(topic, 0) + 1
                _append_unique_citations(
                    lesson["citations"], lesson["_citation_seen"],
                    detail.get("citations", []) or [],
                )
        elif quiz_type == "final_exam":
            final_tests.append({
                "attempt_id": att.get("attempt_id", ""),
                "submitted_at": sub_date,
                "score_pct": score_pct,
                "score": _safe_float(score),
                "total": _safe_float(max_score),
                "weak_topics": weak_topics,
            })

        for detail in details:
            topic_key, topic, topic_names = topic_identity(detail)
            t = topics.setdefault(topic_key, {
                "topic": topic, "topic_names": topic_names,
                "correct": 0, "wrong": 0,
                "last_seen": "", "citations": [],
                "last_wrong": "", "weakness_debt": 0, "consecutive_correct": 0,
                "file_ids": [], "source_lessons": set(),
                "_citation_seen": set(),
            })
            t["topic"] = topic
            t["topic_names"].update(topic_names)
            if detail.get("is_correct"):
                t["correct"] += 1
                t["weakness_debt"] = max(t["weakness_debt"] - 1, 0)
                t["consecutive_correct"] += 1
            else:
                t["wrong"] += 1
                t["weakness_debt"] += 1
                t["consecutive_correct"] = 0
                if sub_date:
                    t["last_wrong"] = sub_date
            if sub_date and sub_date > t["last_seen"]:
                t["last_seen"] = sub_date
            detail_citations = detail.get("citations", []) or []
            for file_id in file_ids:
                t["file_ids"].append(file_id)
            citation_sources = {
                str(citation.get("source_file"))
                for citation in detail_citations
                if citation.get("source_file")
            }
            if citation_sources:
                t["source_lessons"].update(citation_sources)
            else:
                for file_id in file_ids:
                    t["source_lessons"].add(file_labels.get(file_id, file_id))
            _append_unique_citations(t["citations"], t["_citation_seen"], detail_citations)

    score_trend.sort(key=lambda x: x["date"])
    final_tests.sort(key=lambda x: _parse_date(str(x.get("submitted_at", ""))))

    per_lesson = []
    for data in per_lesson_raw.values():
        attempts_for_lesson = sorted(
            data["attempts"],
            key=lambda x: _parse_date(str(x.get("submitted_at", ""))),
        )
        scores = [a["score_pct"] for a in attempts_for_lesson]
        latest = attempts_for_lesson[-1] if attempts_for_lesson else {}
        weak_topics_sorted = sorted(
            data["weak_topic_counts"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        per_lesson.append({
            "file_id": data["file_id"],
            "lesson_name": data["lesson_name"],
            "attempt_count": len(attempts_for_lesson),
            "latest_score_pct": latest.get("score_pct", 0.0),
            "average_score_pct": sum(scores) / max(len(scores), 1),
            "lowest_score_pct": min(scores) if scores else 0.0,
            "last_attempt_at": latest.get("submitted_at", ""),
            "wrong_count": data["wrong_count"],
            "weak_topics": [topic for topic, _count in weak_topics_sorted[:5]],
            "citations": data["citations"][:5],
            "confidence": "low" if len(attempts_for_lesson) < 2 else "normal",
        })

    max_wrong = max((item["wrong_count"] for item in per_lesson), default=0)
    for item in per_lesson:
        normalized_wrong = 0.0
        if max_wrong > 0:
            normalized_wrong = min(log1p(item["wrong_count"]) / log1p(max_wrong), 1.0)
        item["priority"] = round(
            (1 - item["average_score_pct"]) * 0.6
            + (1 - item["latest_score_pct"]) * 0.3
            + normalized_wrong * 0.1,
            3,
        )

    per_lesson.sort(key=lambda x: (x["lesson_name"].lower(), x["last_attempt_at"]))
    needs_attention = sorted(
        [item for item in per_lesson if item["attempt_count"] > 0 and item["priority"] > 0],
        key=lambda x: x["priority"],
        reverse=True,
    )

    weaknesses = []
    for topic_key, data in topics.items():
        if data["weakness_debt"] < 1 or not data["last_wrong"]:
            continue
        try:
            days_gap = max(0, (today - _parse_date(data["last_wrong"])).days)
        except Exception:
            days_gap = 30
        if days_gap > 30:
            continue
        priority = data["weakness_debt"] / (days_gap + 1)
        weaknesses.append({
            "topic_key": topic_key,
            "topic": data["topic"],
            "priority": round(priority, 2),
            "correct": data["correct"],
            "wrong": data["wrong"],
            "weakness_debt": data["weakness_debt"],
            "last_seen": data["last_seen"],
            "citations": data["citations"][:5],
            "file_ids": list(set(data["file_ids"])),
            "source_lessons": sorted(data["source_lessons"]),
        })
    weaknesses.sort(key=lambda x: x["priority"], reverse=True)
    weaknesses = weaknesses[:5]

    # Mastered (100% correct)
    mastered = []
    for topic_key, data in topics.items():
        if data["weakness_debt"] == 0 and data["consecutive_correct"] >= 2:
            mastered.append({
                "topic_key": topic_key,
                "topic": data["topic"],
                "correct": data["correct"],
                "consecutive_correct": data["consecutive_correct"],
            })

    # Due for review (spaced repetition formula)
    due = []
    for topic_key, data in topics.items():
        try:
            days_gap = max(0, (today - _parse_date(data["last_seen"])).days)
        except Exception:
            days_gap = 30
        total_answers = data["correct"] + data["wrong"]
        error_rate = data["wrong"] / max(total_answers, 1)
        priority = error_rate * (1 + log1p(data["wrong"])) * (1 + min(days_gap, 14) / 14)
        if priority <= 0:
            continue
        due.append({
            "topic_key": topic_key,
            "topic": data["topic"],
            "priority": round(priority, 2),
            "error_rate": round(error_rate, 2),
            "wrong": data["wrong"],
            "correct": data["correct"],
            "days_gap": days_gap,
            "citations": data["citations"][:5],
            "file_ids": list(set(data["file_ids"])),
            "source_lessons": sorted(data["source_lessons"]),
        })
    due.sort(key=lambda x: x["priority"], reverse=True)

    return {
        "score_trend": score_trend,
        "per_lesson": per_lesson,
        "final_tests": final_tests,
        "needs_attention": needs_attention,
        "weaknesses": weaknesses,
        "mastered": mastered,
        "due": due,
        "total_attempts": len(attempts),
    }
