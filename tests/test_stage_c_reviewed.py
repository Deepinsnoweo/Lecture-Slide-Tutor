from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from slide_tutor import quiz
from slide_tutor.llm import LLMError
from slide_tutor.models import (
    BoundExamBlueprint,
    BoundQuestionSlot,
    Citation,
    EvidenceQualitySummary,
    EvidenceRef,
    ExamBlueprint,
    GenerationSnapshot,
    FileSlidesPayload,
    ProjectFile,
    ProjectManifest,
    RetrievalChunk,
    PlannedQuestionSlot,
    QuizQuestion,
    QuizSet,
    ReviewedExamMetadata,
)
from slide_tutor.quiz_quality import (
    _questions_from_raw,
    ReviewSummary,
    ValidationIssue,
    bind_exam_blueprint,
    run_reviewed_final_exam,
    ensure_required_cross_file_slot,
    repair_reviewed_mcq_questions,
    sanitize_planned_blueprint,
    validate_reviewed_mcq_questions,
)
from slide_tutor.quiz_submission import submit_quiz_answers
from slide_tutor.quiz_markdown import normalize_quiz_markdown


def _fill_set() -> QuizSet:
    return QuizSet(
        quiz_set_id="fill-set",
        project_id="p",
        file_ids=["f"],
        total_count=1,
        difficulty_mix="1 basic",
        question_type_mix="mcq:0,true_false:0,fill_blank:1,short_answer:0",
        language="中文",
        questions=[QuizQuestion(
            question_id="fill-1",
            question_type="fill_blank",
            topic_tag="topic",
            difficulty="basic",
            question="____ is the concept.",
            blank_answer="canonical concept",
            explanation="because",
            citations=[Citation(source_file="f.pdf", page_number=1, snippet_reference="concept")],
        )],
    )


def test_fill_blank_grading_is_one_batched_ai_call_and_strict(monkeypatch):
    calls: list[str] = []

    def fake_complete(_system, prompt, **_kwargs):
        calls.append(prompt)
        return {"grades": [{"question_id": "fill-1", "is_correct": True, "feedback": "正确"}]}

    monkeypatch.setattr(quiz, "complete_json", fake_complete)
    details = quiz.score_quiz(_fill_set(), {"fill-1": "学生答案"})
    assert details[0].points_awarded is None
    graded = quiz.grade_fill_blank_answers(_fill_set(), details, "中文")
    assert len(calls) == 1
    assert graded[0].is_correct is True
    assert graded[0].points_awarded == 1.0


def test_fill_blank_incomplete_ai_result_does_not_save_submission(monkeypatch):
    monkeypatch.setattr(quiz, "complete_json", lambda *_args, **_kwargs: {"grades": []})
    saved = False

    def fail_if_saved(*_args, **_kwargs):
        nonlocal saved
        saved = True

    monkeypatch.setattr("slide_tutor.quiz_submission.save_quiz_attempt", fail_if_saved)
    with pytest.raises(LLMError):
        submit_quiz_answers(_fill_set(), {"fill-1": "answer"})
    assert saved is False


def _planned(*, cross_file: bool = False) -> ExamBlueprint:
    slot = PlannedQuestionSlot(
        slot_id="slot-1",
        file_ids=["file-a", "file-b"] if cross_file else ["file-a"],
        topic_hint="topic",
        retrieval_query="query",
        difficulty="basic",
        question_type="mcq",
        coverage_role="cross_file" if cross_file else "single_file",
    )
    return ExamBlueprint(
        blueprint_id="blueprint-1",
        project_id="p",
        language="中文",
        target_count=1,
        question_type_mix={"mcq": 1},
        slots=[slot],
    )


def test_planner_sanitize_migrates_scope_and_rejects_unknown_file():
    result = sanitize_planned_blueprint(
        "p",
        {"slots": [{"slot_id": "s", "file_ids": ["file-a"], "topic_hint": "T", "retrieval_query": "q", "difficulty": "basic", "question_type": "mcq"}]},
        valid_file_ids={"file-a"},
        target_count=1,
        language="中文",
        question_type_mix={"mcq": 1},
    )
    assert result.slots[0].file_ids == ["file-a"]
    with pytest.raises(LLMError, match="unknown file"):
        sanitize_planned_blueprint(
            "p",
            {"slots": [{"slot_id": "s", "file_ids": ["missing"], "topic_hint": "T", "retrieval_query": "q", "difficulty": "basic", "question_type": "mcq"}]},
            valid_file_ids={"file-a"},
            target_count=1,
            language="中文",
            question_type_mix={"mcq": 1},
        )


def test_binding_empty_is_explicit_and_replans_at_most_once():
    blueprint = _planned()
    calls = 0

    def replan(slots):
        nonlocal calls
        calls += 1
        return slots

    empty_index = ([], TfidfVectorizer(), np.zeros((0, 0)))
    outcome = bind_exam_blueprint(blueprint, "p", index=empty_index, replan=replan)
    assert outcome.bound_blueprint is None
    assert outcome.replanned_once is True
    assert calls == 1
    assert outcome.issues[0].code == "binding_empty_or_ineligible"


def _bound_question(*, evidence_id: str = "file-a:slide-1") -> tuple[BoundExamBlueprint, QuizQuestion]:
    ref = EvidenceRef(
        project_id="p", file_id="file-a", slide_id="slide-1", source_file="a.pdf",
        page_number=1, snippet="evidence", evidence_eligibility="eligible",
    )
    slot = BoundQuestionSlot(
        slot_id="slot-1", file_ids=["file-a"], topic_hint="topic", retrieval_query="query",
        difficulty="basic", question_type="mcq", evidence_refs=[ref], evidence_summary="evidence",
        evidence_quality=EvidenceQualitySummary(status="eligible"),
    )
    bound = BoundExamBlueprint(
        blueprint_id="bound-1", planned_blueprint_id="blueprint-1", project_id="p",
        target_count=1, slots=[slot],
    )
    question = QuizQuestion(
        question_id="reviewed-slot-1", slot_id="slot-1", question_type="mcq", topic_tag="topic",
        difficulty="basic", question="Which answer is grounded?",
        options={"A": "one", "B": "two", "C": "three", "D": "four"}, correct_option="A",
        explanation="evidence", citations=[Citation(
            source_file="a.pdf", page_number=1, snippet_reference="evidence", slide_id="slide-1",
            project_id="p", file_id="file-a", evidence_id=evidence_id,
        )],
    )
    return bound, question


def test_reviewed_validator_rejects_citation_not_in_bound_evidence():
    bound, question = _bound_question(evidence_id="unknown")
    issues = validate_reviewed_mcq_questions([question], bound)
    assert any(issue.code == "citation_not_subset" for issue in issues)


def test_reviewed_mixed_types_validate_against_bound_slots():
    slots = []
    questions = []
    for index, question_type in enumerate(
        ("mcq", "true_false", "fill_blank", "short_answer"), start=1,
    ):
        question_text = {
            "mcq": "Which task predicts discrete labels?",
            "true_false": "Regression predicts continuous values.",
            "fill_blank": "The task that predicts categories is ____.",
            "short_answer": "Explain how validation evidence supports model selection.",
        }[question_type]
        ref = EvidenceRef(
            project_id="p", file_id="file-a", slide_id=f"slide-{index}",
            source_file="a.pdf", page_number=index, snippet="grounded evidence",
            evidence_eligibility="eligible",
        )
        slots.append(BoundQuestionSlot(
            slot_id=f"slot-{index}", file_ids=["file-a"], topic_hint="topic",
            retrieval_query="query", difficulty="basic", question_type=question_type,
            evidence_refs=[ref], evidence_summary="evidence",
            evidence_quality=EvidenceQualitySummary(status="eligible"),
        ))
        common = {
            "question_id": f"q-{index}", "slot_id": f"slot-{index}",
            "question_type": question_type, "topic_tag": "topic", "difficulty": "basic",
            "question": question_text, "explanation": "grounded",
            "citations": [Citation(
                source_file="a.pdf", page_number=index, snippet_reference="grounded evidence",
                project_id="p", file_id="file-a", slide_id=f"slide-{index}",
                evidence_id=ref.evidence_id,
            )],
        }
        if question_type == "mcq":
            common.update(options={"A": "one", "B": "two", "C": "three", "D": "four"}, correct_option="A")
        elif question_type == "true_false":
            common.update(options={"A": "True", "B": "False"}, correct_option="A")
        elif question_type == "fill_blank":
            common.update(blank_answer="classification")
        else:
            common.update(model_answer="A grounded answer.", scoring_rubric="Up to 3 points for the grounded idea.")
        questions.append(QuizQuestion.model_validate(common))

    bound = BoundExamBlueprint(
        blueprint_id="bound-mixed", planned_blueprint_id="planned-mixed",
        project_id="p", target_count=4, slots=slots,
    )
    assert validate_reviewed_mcq_questions(questions, bound) == []

    bad_fill = questions[2].model_copy(update={"blank_answer": "$x_i$"})
    issues = validate_reviewed_mcq_questions([*questions[:2], bad_fill, questions[3]], bound)
    assert any(issue.code == "fill_blank_math_notation_signal" for issue in issues)


def test_quiz_math_markdown_normalizes_delimiters_only():
    source = r"Use \(x^2\) and then \[y = mx + b\]."
    normalized = normalize_quiz_markdown(source)
    assert "$x^2$" in normalized
    assert "$$\ny = mx + b\n$$" in normalized


def test_reviewed_mixed_parser_normalizes_inapplicable_null_fields():
    bound, _question = _bound_question()
    slot = bound.slots[0].model_copy(update={"question_type": "short_answer"})
    raw = [{
        "slot_id": "slot-1", "question_type": "short_answer",
        "topic_tag": "validation", "difficulty": "applied",
        "question": "Explain why validation estimates generalization.",
        "options": None, "correct_option": None, "blank_answer": None,
        "model_answer": "It evaluates the model on held-out data.",
        "scoring_rubric": "Up to 3 points for held-out evaluation and generalization.",
        "explanation": "Grounded in the evidence.", "distractor_rationales": None,
        "evidence_ids": [slot.evidence_refs[0].evidence_id],
    }]

    parsed = _questions_from_raw(raw, [slot])

    assert len(parsed) == 1
    assert parsed[0].question_type == "short_answer"
    assert parsed[0].options == {}


def test_reviewed_repair_requires_every_slot_without_deleting_original(monkeypatch):
    bound, original = _bound_question()
    monkeypatch.setattr(
        "slide_tutor.quiz_quality.complete_json",
        lambda *_args, **_kwargs: {"questions": []},
    )

    with pytest.raises(LLMError, match="did not cover every requested slot"):
        repair_reviewed_mcq_questions(
            bound,
            [original],
            {"slot-1"},
            language="中文",
        )

    assert original.slot_id == "slot-1"


def test_legacy_planned_cache_is_readable_but_serializes_new_contract():
    slot = PlannedQuestionSlot.model_validate({
        "slot_id": "s", "file_id": "file-a", "topic": "topic", "retrieval_query": "q",
        "difficulty": "basic", "question_type": "mcq",
    })
    assert slot.file_ids == ["file-a"]
    assert "file_id" not in slot.model_dump()


def test_reviewed_mcq_orchestrator_binds_two_files_and_emits_metadata(
    isolated_data_root, monkeypatch,
):
    from slide_tutor import cache
    from slide_tutor.projects import save_project
    import slide_tutor.quiz_quality as quality
    import slide_tutor.index_store as index_store

    project = ProjectManifest(
        project_id="p", name="Representative", created_at="now", updated_at="now",
        file_order=["file-a", "file-b"], files=[
            ProjectFile(file_id="file-a", original_name="A.pdf", safe_name="A.pdf", sha256="a", size_bytes=1, page_count=1, created_at="now", status="ready"),
            ProjectFile(file_id="file-b", original_name="B.pdf", safe_name="B.pdf", sha256="b", size_bytes=1, page_count=1, created_at="now", status="ready"),
        ],
    )
    save_project(project)
    for file_id, name, text in [
        ("file-a", "A.pdf", "Classification predicts discrete labels."),
        ("file-b", "B.pdf", "Regression predicts continuous values."),
    ]:
        cache.save_json(
            cache.get_project_dir("p") / "files" / file_id / "slides.json",
            FileSlidesPayload(
                project_id="p", file_id=file_id, slides=[{
                    "slide_id": f"{file_id}-slide-1", "module_id": "m", "project_id": "p", "file_id": file_id,
                    "source_file": name, "source_type": "pdf", "page_number": 1, "global_index": 1,
                    "title_guess": text.split()[0], "text": text, "text_quality": "ok",
                }],
            ),
        )
    chunks = [
        RetrievalChunk(chunk_id="a", project_id="p", file_id="file-a", slide_id="file-a-slide-1", source_file="A.pdf", page_number=1, text="Classification predicts discrete labels."),
        RetrievalChunk(chunk_id="b", project_id="p", file_id="file-b", slide_id="file-b-slide-1", source_file="B.pdf", page_number=1, text="Regression predicts continuous values."),
    ]
    vectorizer = TfidfVectorizer().fit([chunk.text for chunk in chunks])
    index = (chunks, vectorizer, vectorizer.transform([chunk.text for chunk in chunks]))
    monkeypatch.setattr(index_store, "load_index", lambda _project_id: index)
    snapshot = GenerationSnapshot(
        source_signature="source", active_layer="baseline", page_generation_id="page-1",
        index_generation_id="index-1", page_content_contract_signature="page", retrieval_contract_signature="retrieval",
    )
    monkeypatch.setattr(quality, "active_generation_snapshot", lambda _project_id: snapshot)
    responses = [
        {"slots": [
            {"slot_id": "slot-1", "file_ids": ["file-a"], "topic_hint": "classification", "retrieval_query": "classification labels", "difficulty": "basic", "question_type": "mcq", "coverage_role": "single_file"},
            {"slot_id": "slot-2", "file_ids": ["file-a", "file-b"], "topic_hint": "classification and regression", "retrieval_query": "classification regression", "difficulty": "applied", "question_type": "mcq", "coverage_role": "cross_file"},
        ]},
        {"questions": [
            {"slot_id": "slot-1", "topic_tag": "classification", "difficulty": "basic", "question": "What does classification predict?", "options": {"A": "discrete labels", "B": "continuous values", "C": "nothing", "D": "timestamps"}, "correct_option": "A", "explanation": "The evidence says labels.", "distractor_rationales": {}, "evidence_ids": ["file-a:file-a-slide-1"]},
            {"slot_id": "slot-2", "topic_tag": "comparison", "difficulty": "applied", "question": "Which pair matches the two prediction tasks?", "options": {"A": "labels and values", "B": "values and labels", "C": "both labels", "D": "both values"}, "correct_option": "A", "explanation": "The two evidence pages differ.", "distractor_rationales": {}, "evidence_ids": ["file-a:file-a-slide-1", "file-b:file-b-slide-1"]},
        ]},
        {"findings": []},
    ]
    monkeypatch.setattr(quality, "complete_json", lambda *_args, **_kwargs: responses.pop(0))
    result = run_reviewed_final_exam(
        "p", target_count=2, language="中文", generation_snapshot=snapshot,
    )
    assert result.quality_mode == "reviewed"
    assert result.total_count == 2
    assert result.reviewed_metadata is not None
    assert result.reviewed_metadata.bound_blueprint.slots[1].coverage_role == "cross_file"
    assert {citation.file_id for citation in result.questions[1].citations} == {"file-a", "file-b"}


def test_cross_file_requirement_corrects_one_slot_once(monkeypatch):
    blueprint = _planned()
    cross = _planned(cross_file=True)
    cross.slots[0].difficulty = "applied"
    monkeypatch.setattr(
        "slide_tutor.quiz_quality.plan_exam_blueprint",
        lambda *args, **kwargs: cross,
    )
    catalog = [
        {"file_id": "file-a", "source_file": "A.pdf", "topic": "A", "page_numbers": [1], "summary": "A"},
        {"file_id": "file-b", "source_file": "B.pdf", "topic": "B", "page_numbers": [1], "summary": "B"},
    ]
    from slide_tutor.models import TopicCatalogEntry
    corrected = ensure_required_cross_file_slot(
        blueprint,
        project_id="p",
        language="中文",
        catalog=[TopicCatalogEntry.model_validate(item) for item in catalog],
    )
    assert corrected.slots[0].coverage_role == "cross_file"
    assert corrected.slots[0].difficulty == "applied"


def test_reviewed_bundle_is_single_canonical_cache(isolated_data_root):
    from slide_tutor import cache
    from slide_tutor.quiz import load_cached_quiz, load_reviewed_exam_metadata, save_cached_quiz

    quiz_set = _fill_set().model_copy(update={
        "quiz_set_id": "reviewed-bundle",
        "quality_mode": "reviewed",
        "reviewed_metadata": ReviewedExamMetadata(),
    })
    save_cached_quiz(quiz_set)
    project_dir = cache.get_project_dir("p")
    assert (project_dir / "reviewed_exam_bundle_reviewed-bundle.json").exists()
    assert not (project_dir / "quiz_cache_reviewed-bundle.json").exists()
    assert load_cached_quiz("p", "reviewed-bundle") is not None
    assert load_reviewed_exam_metadata("p", "reviewed-bundle") is not None


def test_fast_reuse_keeps_only_latest_unresolved_wrong(monkeypatch):
    questions = [
        QuizQuestion(
            question_id=f"q-{index}", question_origin_id=f"origin-{index}",
            question_type="mcq", topic_tag="topic", difficulty="basic",
            question=f"Question {index}?", options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_option="A", citations=[Citation(
                source_file="f.pdf", page_number=1, snippet_reference="topic",
            )],
        )
        for index in (1, 2)
    ]
    snapshot = QuizSet(
        quiz_set_id="old", project_id="p", file_ids=["f"], total_count=2,
        difficulty_mix="mixed", question_type_mix="mcq:2,true_false:0,fill_blank:0,short_answer:0",
        questions=questions, language="English", generation_request_signature="same",
    )
    records = [
        {
            "attempt": {"attempt_id": "a1", "submitted_at": "2026-07-20T00:00:00Z"},
            "quiz_set_snapshot": snapshot.model_dump(mode="json"),
            "result": {"details": [
                {"question_id": "q-1", "is_correct": False},
                {"question_id": "q-2", "is_correct": False},
            ]},
        },
        {
            "attempt": {"attempt_id": "a2", "submitted_at": "2026-07-21T00:00:00Z"},
            "quiz_set_snapshot": snapshot.model_dump(mode="json"),
            "result": {"details": [{"question_id": "q-1", "is_correct": True}]},
        },
    ]
    monkeypatch.setattr(quiz, "list_quiz_attempts", lambda project_id, limit=None: records)
    reused, attempt_ids = quiz.collect_reusable_wrong_questions(
        "p", file_ids=["f"], language="English",
        question_type_mix=snapshot.question_type_mix,
        request_signature="same", generation_snapshot=None,
        required_mix={"mcq": 2, "true_false": 0, "fill_blank": 0, "short_answer": 0},
    )
    assert [question.question_origin_id for question in reused] == ["origin-2"]
    assert attempt_ids == ["a1"]


def test_topic_registry_unifies_localized_names(isolated_data_root):
    from slide_tutor.topics import resolve_quiz_topics

    questions = []
    for index, tag in enumerate(("线性回归", "Linear Regression"), start=1):
        questions.append(QuizQuestion(
            question_id=f"q-{index}", question_type="mcq", topic_tag=tag,
            topic_names={"简体中文": "线性回归", "English": "Linear Regression", "日本語": "線形回帰"},
            difficulty="basic", question=f"Question {index}?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="A",
            citations=[Citation(source_file="f.pdf", page_number=1, snippet_reference="regression")],
        ))
    quiz_set = QuizSet(
        project_id="p", file_ids=["f"], total_count=2, difficulty_mix="basic",
        question_type_mix="mcq:2,true_false:0,fill_blank:0,short_answer:0",
        questions=questions, language="English",
    )
    resolved = resolve_quiz_topics(quiz_set)
    assert resolved.questions[0].topic_key == resolved.questions[1].topic_key
    assert resolved.questions[0].topic_tag == "Linear Regression"
