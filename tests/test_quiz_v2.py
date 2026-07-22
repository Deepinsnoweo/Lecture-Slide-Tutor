import pytest

from slide_tutor import quiz
from slide_tutor.llm import LLMError
from slide_tutor.models import QuizQuestion, QuizSet
from slide_tutor.quiz_export import export_quiz_report_markdown
from slide_tutor.quiz_formatting import format_quiz_answer


def _citation() -> dict:
    return {
        "source_file": "Lecture01.pdf",
        "page_number": 1,
        "snippet_reference": "Relevant source text.",
    }


def _question(question_id: str, question_type: str, **overrides) -> dict:
    base = {
        "question_id": question_id,
        "question_type": question_type,
        "topic_tag": "topic",
        "difficulty": "basic",
        "question": f"Question {question_id}?",
        "explanation": "Because the slides say so.",
        "distractor_rationales": {},
        "citations": [_citation()],
    }
    if question_type == "mcq":
        base |= {
            "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
            "correct_option": "B",
        }
    elif question_type == "true_false":
        base |= {
            "options": {"A": "True", "B": "False"},
            "correct_option": "A",
        }
    elif question_type == "fill_blank":
        base |= {
            "options": {},
            "correct_option": "",
            "blank_answer": "loss",
        }
    elif question_type == "short_answer":
        base |= {
            "options": {},
            "correct_option": "",
            "blank_answer": "",
            "model_answer": "Validation data estimates generalization.",
            "scoring_rubric": "3 points for generalization; partial credit for related ideas.",
        }
    return base | overrides


def _quiz_set(questions: list[QuizQuestion]) -> QuizSet:
    return QuizSet(
        quiz_set_id="quiz-v3",
        project_id="project-1",
        file_ids=["f1"],
        total_count=len(questions),
        difficulty_mix="mixed",
        question_type_mix="mcq:1,true_false:1,fill_blank:1,short_answer:1",
        questions=questions,
        created_at="2026-07-08T00:00:00Z",
        model="test-model",
        prompt_version="v3",
        language="中文",
    )


def test_question_type_mix_uses_strict_ratio():
    assert quiz.build_question_type_mix(6) == {
        "mcq": 4,
        "true_false": 1,
        "fill_blank": 1,
        "short_answer": 0,
    }
    assert quiz.build_question_type_mix(10) == {
        "mcq": 6,
        "true_false": 2,
        "fill_blank": 1,
        "short_answer": 1,
    }
    assert quiz.build_question_type_mix(20) == {
        "mcq": 12,
        "true_false": 3,
        "fill_blank": 3,
        "short_answer": 2,
    }


def test_option_answer_formatter_is_shared_with_markdown_export():
    question = QuizQuestion.model_validate(_question("m1", "mcq"))
    quiz_set = _quiz_set([question])
    details = quiz.score_quiz(quiz_set, {"m1": "A"})
    from slide_tutor.models import QuizResult
    result = QuizResult(score=0, total=1, max_score=1, details=details)

    assert format_quiz_answer(question, "A") == "A. Alpha"
    assert format_quiz_answer(question, "B") == "B. Beta"
    report = export_quiz_report_markdown(quiz_set, result)
    assert "Your answer: **A. Alpha**" in report
    assert "Correct option: **B. Beta**" in report
    assert format_quiz_answer(None, "legacy-value") == "legacy-value"


def test_diagnostic_prompts_receive_complete_option_text(monkeypatch):
    question = QuizQuestion.model_validate(_question("m1", "mcq"))
    quiz_set = _quiz_set([question])
    details = quiz.score_quiz(quiz_set, {"m1": "A"})
    prompts = []

    def fake_complete_json(_system, prompt, **_kwargs):
        prompts.append(prompt)
        if "wrong_reason" in prompt:
            return {"m1": "Review the distinction."}
        return {"weaknesses": [], "strengths": []}

    monkeypatch.setattr(quiz, "complete_json", fake_complete_json)
    quiz.batch_wrong_reasons(quiz_set, details, "中文")
    quiz.generate_weakness_report(quiz_set, details, "中文")

    combined = "\n".join(prompts)
    assert "A. Alpha" in combined
    assert "B. Beta" in combined


def test_weakness_report_rejects_unsupported_claim_and_falls_back(monkeypatch):
    question = QuizQuestion.model_validate(_question("m1", "mcq", topic_tag="classification"))
    quiz_set = _quiz_set([question])
    details = quiz.score_quiz(quiz_set, {"m1": "A"})
    monkeypatch.setattr(quiz, "complete_json", lambda *_args, **_kwargs: {
        "weaknesses": [{
            "topic": "invented", "question_ids": ["unknown"],
            "diagnosis": "Invented error.", "review_pages": [],
        }],
        "strengths": [],
    })

    report = quiz.generate_weakness_report(quiz_set, details, "English")

    assert report.generated_by == "rule"
    assert report.support_reviewed is True
    assert report.weaknesses[0].question_ids == ["m1"]
    assert report.quality_metadata["unsupported_claim_count"] == 1


def test_all_correct_produces_mastery_summary_without_provider(monkeypatch):
    question = QuizQuestion.model_validate(_question("m1", "mcq", topic_tag="classification"))
    quiz_set = _quiz_set([question])
    details = quiz.score_quiz(quiz_set, {"m1": "B"})
    monkeypatch.setattr(
        quiz, "complete_json",
        lambda *_args, **_kwargs: pytest.fail("perfect score must not call Provider"),
    )

    report = quiz.generate_weakness_report(quiz_set, details, "English")

    assert report.report_kind == "mastery"
    assert report.weaknesses == []
    assert report.strengths[0].question_ids == ["m1"]


def test_weakness_report_requires_complete_ai_grading():
    question = QuizQuestion.model_validate(_question("fill1", "fill_blank"))
    quiz_set = _quiz_set([question])
    details = quiz.score_quiz(quiz_set, {"fill1": "loss"})

    with pytest.raises(LLMError, match="requires complete grading"):
        quiz.generate_weakness_report(quiz_set, details, "English")


def test_generation_repair_fills_missing_question_types(monkeypatch):
    calls = []

    def fake_complete_json(system_prompt, user_prompt, max_retries=2, **_kwargs):
        calls.append(user_prompt)
        if len(calls) == 1:
            return {
                    "questions": [
                    _question("m1", "mcq", question="Which item defines supervised learning?"),
                    _question("m2", "mcq", question="Which metric evaluates classification accuracy?"),
                    _question("m3", "mcq", question="What does a training label represent?"),
                    _question("m4", "mcq", question="Which example describes regression?"),
                    _question("tf1", "true_false", question="Classification predicts discrete labels."),
                ]
            }
        return {
            "questions": [
                _question("fill1", "fill_blank", question="Fill in the blank: ____ measures error.")
            ]
        }

    monkeypatch.setattr(quiz, "complete_json", fake_complete_json)

    required = quiz.build_question_type_mix(6)
    questions = quiz._generate_questions_with_repair(  # noqa: SLF001
        "context", 6, "4 basic + 2 applied", "中文", required
    )

    assert [q.question_type for q in questions].count("mcq") == 4
    assert [q.question_type for q in questions].count("true_false") == 1
    assert [q.question_type for q in questions].count("fill_blank") == 1
    assert "Question type mix:" in calls[0]
    assert "Generate ONLY these missing question types" in calls[1]


def test_generation_raises_when_quota_still_missing(monkeypatch):
    monkeypatch.setattr(quiz, "complete_json", lambda *args, **kwargs: {"questions": []})

    with pytest.raises(LLMError):
        quiz._generate_questions_with_repair(  # noqa: SLF001
            "context", 6, "4 basic + 2 applied", "中文", quiz.build_question_type_mix(6)
        )


def test_cache_key_changes_with_question_type_mix():
    base = quiz._compute_quiz_set_id(  # noqa: SLF001
        ["f1"], 6, "4 basic + 2 applied", "中文",
        "mcq:4,true_false:1,fill_blank:1,short_answer:0",
    )
    other = quiz._compute_quiz_set_id(  # noqa: SLF001
        ["f1"], 6, "4 basic + 2 applied", "中文",
        "mcq:6,true_false:0,fill_blank:0,short_answer:0",
    )
    assert base != other


def test_fill_blank_rejects_different_short_chinese_terms():
    assert quiz._grade_fill_blank("预测变量", "预测变量") is True  # noqa: SLF001
    assert quiz._grade_fill_blank(" 响应变量 ", "预测变量") is False  # noqa: SLF001


def test_fill_blank_quality_gate_uses_only_strong_signals():
    valid = QuizQuestion.model_validate(_question(
        "fill-valid", "fill_blank", question="Fill in the blank: ____ measures error.",
    ))
    assert quiz.fill_blank_quality_issues(valid) == []

    alternative = valid.model_copy(update={"blank_answer": "loss or error"})
    formula = valid.model_copy(update={"blank_answer": "$x_i$"})
    assert "alternative_answer_signal" in quiz.fill_blank_quality_issues(alternative)
    assert "math_notation_signal" in quiz.fill_blank_quality_issues(formula)


def test_score_quiz_and_grade_short_answers(monkeypatch):
    questions = [
        QuizQuestion.model_validate(_question("m1", "mcq")),
        QuizQuestion.model_validate(_question("tf1", "true_false")),
        QuizQuestion.model_validate(_question("fill1", "fill_blank")),
        QuizQuestion.model_validate(_question("short1", "short_answer")),
    ]
    quiz_set = _quiz_set(questions)
    details = quiz.score_quiz(
        quiz_set,
        {"m1": "B", "tf1": "B", "fill1": "los", "short1": "It checks generalization."},
    )

    short_detail = next(d for d in details if d.question_type == "short_answer")
    assert short_detail.points_awarded is None

    def fake_complete_json(system_prompt, user_prompt, max_retries=2, **_kwargs):
        if "canonical_concept" in user_prompt:
            return {"grades": [{"question_id": "fill1", "is_correct": True, "feedback": "Correct concept."}]}
        assert "short1" in user_prompt
        return {
            "grades": [{
                "question_id": "short1",
                "points_awarded": 2.5,
                "grading_feedback": "Good answer; mention model selection for full credit.",
            }]
        }

    monkeypatch.setattr(quiz, "complete_json", fake_complete_json)
    graded = quiz.grade_fill_blank_answers(quiz_set, details, "中文")
    graded = quiz.grade_short_answers(quiz_set, graded, "中文")

    by_id = {d.question_id: d for d in graded}
    assert by_id["m1"].is_correct is True
    assert by_id["tf1"].is_correct is False
    assert by_id["fill1"].is_correct is True
    assert by_id["short1"].points_awarded == 2.5
    assert by_id["short1"].is_correct is True
    assert by_id["short1"].grading_feedback


def test_short_answer_grading_missing_grade_raises(monkeypatch):
    questions = [QuizQuestion.model_validate(_question("short1", "short_answer"))]
    quiz_set = _quiz_set(questions)
    details = quiz.score_quiz(quiz_set, {"short1": "A short answer."})
    monkeypatch.setattr(quiz, "complete_json", lambda *args, **kwargs: {"grades": []})

    with pytest.raises(LLMError):
        quiz.grade_short_answers(quiz_set, details, "中文")
