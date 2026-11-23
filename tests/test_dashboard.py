from datetime import UTC, datetime
from types import SimpleNamespace

from slide_tutor import dashboard


def _record(
    attempt_id: str,
    quiz_type: str,
    file_ids: list[str],
    score: int,
    total: int,
    submitted_at: str,
    details: list[dict],
) -> dict:
    return {
        "attempt": {
            "attempt_id": attempt_id,
            "quiz_type": quiz_type,
            "project_id": "project-1",
            "file_ids": file_ids,
            "total_count": total,
            "submitted_at": submitted_at,
        },
        "result": {
            "attempt_id": attempt_id,
            "score": score,
            "total": total,
            "details": details,
        },
    }


def _detail(topic: str, is_correct: bool, source_file: str, page_number: int) -> dict:
    return {
        "question_id": f"{topic}-{page_number}",
        "topic_tag": topic,
        "is_correct": is_correct,
        "citations": [{
            "source_file": source_file,
            "page_number": page_number,
            "snippet_reference": topic,
        }],
    }


def test_dashboard_splits_lesson_and_final_exam(monkeypatch):
    monkeypatch.setattr(dashboard, "_today", lambda: datetime(2026, 7, 4, tzinfo=UTC))
    records = [
        _record(
            "a1", "per_pdf", ["f1"], 3, 6, "2026-07-01T10:00:00Z",
            [
                _detail("supervised learning", False, "Lecture01.pdf", 8),
                _detail("classification", True, "Lecture01.pdf", 12),
            ],
        ),
        _record(
            "a2", "per_pdf", ["f1"], 5, 6, "2026-07-02T10:00:00Z",
            [_detail("supervised learning", False, "Lecture01.pdf", 8)],
        ),
        _record(
            "a3", "final_exam", ["f1", "f2"], 10, 20, "2026-07-03T10:00:00Z",
            [_detail("classification", False, "Lecture02.pdf", 3)],
        ),
    ]

    monkeypatch.setattr(dashboard, "list_quiz_attempts", lambda project_id, limit=50: records)
    monkeypatch.setattr(dashboard, "load_project", lambda project_id: SimpleNamespace())
    monkeypatch.setattr(
        dashboard,
        "ordered_files",
        lambda project: [
            SimpleNamespace(file_id="f1", original_name="Lecture01.pdf"),
            SimpleNamespace(file_id="f2", original_name="Lecture02.pdf"),
        ],
    )

    data = dashboard.build_dashboard_data("project-1")

    assert [item["lesson_name"] for item in data["per_lesson"]] == ["Lecture01.pdf"]
    lesson = data["per_lesson"][0]
    assert lesson["attempt_count"] == 2
    assert round(lesson["latest_score_pct"], 2) == 0.83
    assert round(lesson["average_score_pct"], 2) == 0.67
    assert round(lesson["lowest_score_pct"], 2) == 0.50
    assert lesson["confidence"] == "normal"

    assert len(data["final_tests"]) == 1
    assert data["final_tests"][0]["attempt_id"] == "a3"
    assert data["final_tests"][0]["weak_topics"] == ["classification"]

    weakness_topics = {item["topic"] for item in data["weaknesses"]}
    assert "classification" in weakness_topics
    assert "supervised learning" in weakness_topics

    assert data["needs_attention"][0]["priority"] <= 1.0


def test_dashboard_marks_single_attempt_low_confidence(monkeypatch):
    records = [
        _record(
            "a1", "per_pdf", ["f1"], 2, 6, "2026-07-01T10:00:00Z",
            [_detail("regression", False, "Lecture01.pdf", 4)],
        ),
    ]

    monkeypatch.setattr(dashboard, "list_quiz_attempts", lambda project_id, limit=50: records)
    monkeypatch.setattr(dashboard, "load_project", lambda project_id: SimpleNamespace())
    monkeypatch.setattr(
        dashboard,
        "ordered_files",
        lambda project: [SimpleNamespace(file_id="f1", original_name="Lecture01.pdf")],
    )

    data = dashboard.build_dashboard_data("project-1")

    assert data["per_lesson"][0]["confidence"] == "low"
    assert data["needs_attention"][0]["lesson_name"] == "Lecture01.pdf"


def test_dashboard_omits_perfect_lessons_from_needs_attention(monkeypatch):
    monkeypatch.setattr(dashboard, "_today", lambda: datetime(2026, 7, 4, tzinfo=UTC))
    records = [
        _record(
            "a1", "per_pdf", ["f1"], 6, 6, "2026-07-01T10:00:00Z",
            [_detail("regression", True, "Lecture01.pdf", 4)],
        ),
        _record(
            "a2", "final_exam", ["f1", "f2"], 18, 20, "2026-07-02T10:00:00Z",
            [_detail("classification", False, "Lecture02.pdf", 3)],
        ),
    ]

    monkeypatch.setattr(dashboard, "list_quiz_attempts", lambda project_id, limit=50: records)
    monkeypatch.setattr(dashboard, "load_project", lambda project_id: SimpleNamespace())
    monkeypatch.setattr(
        dashboard,
        "ordered_files",
        lambda project: [
            SimpleNamespace(file_id="f1", original_name="Lecture01.pdf"),
            SimpleNamespace(file_id="f2", original_name="Lecture02.pdf"),
        ],
    )

    data = dashboard.build_dashboard_data("project-1")

    assert data["needs_attention"] == []
    classification = next(item for item in data["weaknesses"] if item["topic"] == "classification")
    assert classification["source_lessons"] == ["Lecture02.pdf"]


def test_dashboard_weakness_debt_and_mastery_recovery(monkeypatch):
    monkeypatch.setattr(dashboard, "_today", lambda: datetime(2026, 7, 23, tzinfo=UTC))
    records = [
        _record("a1", "per_pdf", ["f1"], 0, 1, "2026-07-20T10:00:00Z", [_detail("regression", False, "L.pdf", 1)]),
        _record("a2", "per_pdf", ["f1"], 1, 1, "2026-07-21T10:00:00Z", [_detail("regression", True, "L.pdf", 1)]),
        _record("a3", "per_pdf", ["f1"], 1, 1, "2026-07-22T10:00:00Z", [_detail("regression", True, "L.pdf", 1)]),
    ]
    monkeypatch.setattr(dashboard, "list_quiz_attempts", lambda project_id, limit=None: records)
    monkeypatch.setattr(dashboard, "load_project", lambda project_id: SimpleNamespace())
    monkeypatch.setattr(dashboard, "ordered_files", lambda project: [SimpleNamespace(file_id="f1", original_name="L.pdf")])

    data = dashboard.build_dashboard_data("project-1")
    assert data["weaknesses"] == []
    assert data["mastered"][0]["topic"] == "regression"
    assert data["mastered"][0]["consecutive_correct"] == 2
