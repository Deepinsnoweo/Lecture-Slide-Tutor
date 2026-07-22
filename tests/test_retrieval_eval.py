from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from slide_tutor import cache, index_store
from slide_tutor.models import RetrievalChunk, SlidePage
from slide_tutor.retrieval_eval import (
    GoldReferenceError,
    RetrievalGoldSet,
    evaluate_gold_set,
    load_gold_set,
    make_project_retriever,
    validate_gold_references,
)
from slide_tutor.retrieval_fixture import build_retrieval_gold_project
from slide_tutor.projects import load_file_slides, load_project


def _chunk(slide_id: str, page: int, *, methods: list[str] | None = None) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=slide_id,
        project_id="retrieval-gold-fixture",
        file_id="file-neuro",
        slide_id=slide_id,
        source_file="neuro.pdf",
        page_number=page,
        text=f"content for {slide_id}",
        retrieval_methods=methods or ["controlled"],
    )


def test_gold_fixture_has_required_versioned_cases():
    gold_set = load_gold_set(min_cases=30)

    assert gold_set.fixture_version == "retrieval-gold-v1"
    assert len(gold_set.cases) >= 30
    assert {case.category for case in gold_set.cases} >= {
        "exact_title",
        "acronym",
        "cross_lingual",
        "semantic",
        "distractor",
        "positional",
        "typo",
    }


def test_evaluator_reports_metrics_breakdowns_and_failures():
    gold_set = RetrievalGoldSet(
        fixture_version="test-v1",
        name="test",
        cases=[
            {
                "case_id": "hit",
                "query": "hit",
                "category": "exact_title",
                "project_id": "p",
                "scope": "whole_project",
                "relevant_slide_ids": ["slide-1"],
                "required_method": "exact_title",
            },
            {
                "case_id": "miss",
                "query": "miss",
                "category": "semantic",
                "project_id": "p",
                "scope": "whole_project",
                "relevant_slide_ids": ["slide-2"],
                "required_method": "semantic",
            },
        ],
    )

    def retrieve(case):
        if case.case_id == "hit":
            return [_chunk("slide-1", 1, methods=["exact_title"])]
        return [_chunk("distractor", 2)]

    report = evaluate_gold_set(
        gold_set,
        retrieve,
        machine="test-machine",
        model_contract={"backend": "controlled"},
        index_signatures={"p": "index-v1"},
    )

    assert report.machine == "test-machine"
    assert report.model_contract["backend"] == "controlled"
    assert report.index_signatures["p"] == "index-v1"
    assert report.overall.case_count == 2
    assert report.overall.recall_at_3 == 0.5
    assert report.by_category["exact_title"].case_count == 1
    assert len(report.failures) == 1
    assert report.failures[0].actual_top_k == ["distractor"]
    assert "distractor" in report.failures[0].ranks_and_scores


def test_gold_reference_validation_fails_before_evaluation():
    gold_set = RetrievalGoldSet(
        fixture_version="test-v1",
        name="test",
        cases=[
            {
                "case_id": "missing",
                "query": "missing",
                "category": "semantic",
                "project_id": "p",
                "scope": "explicit_file",
                "file_id": "missing-file",
                "current_slide_id": "missing-slide",
                "relevant_slide_ids": ["missing-slide"],
                "required_method": "semantic",
            },
        ],
    )

    project = SimpleNamespace(
        files=[SimpleNamespace(file_id="known-file", status="ready")],
    )
    with pytest.raises(GoldReferenceError, match="missing file"):
        validate_gold_references(
            gold_set,
            lambda _project_id: project,
            lambda _project_id, _file_id: [],
        )


def test_report_is_json_serializable(tmp_path):
    gold_set = RetrievalGoldSet(
        fixture_version="test-v1",
        name="test",
        cases=[
            {
                "case_id": "one",
                "query": "one",
                "category": "exact_title",
                "project_id": "p",
                "scope": "whole_project",
                "relevant_slide_ids": ["slide-1"],
                "required_method": "any",
            },
        ],
    )
    report = evaluate_gold_set(gold_set, lambda _case: [_chunk("slide-1", 1)])
    output = tmp_path / "report.json"
    output.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["overall"]["mrr"] == 1.0


@pytest.mark.parametrize(
    ("required_method", "methods", "expected"),
    [
        ("exact_title", ["dense"], False),
        ("lexical", ["bm25"], True),
        ("semantic", ["bm25"], False),
        ("positional", ["positional"], True),
        ("positional", ["dense"], False),
        ("any", ["controlled"], True),
    ],
)
def test_required_method_is_a_real_gate(required_method, methods, expected):
    gold_set = RetrievalGoldSet(
        fixture_version="test-v1",
        name="method-test",
        cases=[{
            "case_id": "case",
            "query": "case",
            "category": "semantic",
            "project_id": "p",
            "scope": "whole_project",
            "relevant_slide_ids": ["slide-1"],
            "required_method": required_method,
        }],
    )
    report = evaluate_gold_set(
        gold_set,
        lambda _case: [_chunk("slide-1", 1, methods=methods)],
    )
    assert report.cases[0].method_satisfied is expected
    assert bool(report.failures) is (not expected)
    if not expected:
        assert report.failures[0].failure_kind in {"method", "retrieval_and_method"}


def test_deterministic_fixture_builder_is_repeatable_and_validates_references(
    isolated_data_root,
):
    gold_set = load_gold_set()
    first = build_retrieval_gold_project()
    validate_gold_references(gold_set, load_project, load_file_slides)
    first_payload = [
        slide.model_dump(mode="json")
        for file in first.files
        for slide in load_file_slides(first.project_id, file.file_id)
    ]
    second = build_retrieval_gold_project()
    second_payload = [
        slide.model_dump(mode="json")
        for file in second.files
        for slide in load_file_slides(second.project_id, file.file_id)
    ]
    assert [file.file_id for file in first.files] == ["file-neuro", "file-ml"]
    assert first_payload == second_payload


def test_positional_adapter_hits_current_and_requested_pages_without_llm(
    isolated_data_root,
):
    from slide_tutor import index_store

    build_retrieval_gold_project()
    chunks = index_store.build_chunks("retrieval-gold-fixture")
    backend, matrix = index_store._build_tfidf_index(chunks)
    retriever = make_project_retriever(
        lambda _project_id: (chunks, backend, matrix),
    )
    gold = load_gold_set()
    cases = {
        case.case_id: case for case in gold.cases
        if case.category == "positional"
    }
    for case_id, expected in {
        "positional-next-page": {"slide-neuro-050", "slide-neuro-051"},
        "positional-explicit-page": {"slide-neuro-017", "slide-neuro-050"},
        "positional-previous-ml": {"slide-ml-014", "slide-ml-015"},
        "positional-current": {"slide-neuro-050"},
    }.items():
        actual = retriever(cases[case_id])
        assert {chunk.slide_id for chunk in actual} >= expected
        assert all("positional" in chunk.retrieval_methods for chunk in actual)
