from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from slide_tutor.models import RetrievalChunk, RetrievalScope
from slide_tutor.retrieval import (
    bm25_build_count,
    clear_bm25_cache,
    retrieve_for_question,
)


def _chunk(slide_id, file_id, text, page):
    return RetrievalChunk(
        chunk_id=slide_id, project_id="p", file_id=file_id, slide_id=slide_id,
        source_file=f"{file_id}.pdf", page_number=page, title="Introduction",
        normalized_title="introduction", text=text,
    )


def _index():
    chunks = [
        _chunk("a-1", "file-a", "Introduction LTD synaptic depression", 1),
        _chunk("a-2", "file-a", "Gradient descent parameters", 2),
        _chunk("b-1", "file-b", "Introduction LTD unrelated file", 1),
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return chunks, vectorizer, matrix


def test_scope_contract_requires_and_forbids_expected_fields():
    with pytest.raises(ValueError):
        RetrievalScope(kind="current_file", project_id="p")
    with pytest.raises(ValueError):
        RetrievalScope(kind="selected_files", project_id="p", file_ids=[])
    with pytest.raises(ValueError):
        RetrievalScope(kind="whole_project", project_id="p", file_id="file-a")
    scope = RetrievalScope(kind="selected_files", project_id="p", file_ids=["file-a"])
    assert scope.summary == "selected_files:file-a"


def test_current_file_scope_is_applied_before_exact_bm25_and_dense():
    index = _index()
    results = retrieve_for_question(
        "Introduction LTD", None, index,
        scope=RetrievalScope(kind="current_file", project_id="p", file_id="file-a"),
    )
    assert results
    assert all(result.file_id == "file-a" for result in results)


def test_selected_files_no_match_does_not_fill_from_forbidden_file():
    index = _index()
    results = retrieve_for_question(
        "parameters", None, index,
        scope=RetrievalScope(kind="selected_files", project_id="p", file_ids=["file-b"]),
    )
    assert all(result.file_id == "file-b" for result in results)
    assert not any(result.slide_id == "a-2" for result in results)


def test_bm25_corpus_is_built_once_for_multiple_scopes():
    clear_bm25_cache()
    index = _index()
    retrieve_for_question(
        "LTD", None, index,
        scope=RetrievalScope(kind="current_file", project_id="p", file_id="file-a"),
    )
    retrieve_for_question(
        "LTD", None, index,
        scope=RetrievalScope(kind="whole_project", project_id="p"),
    )
    assert bm25_build_count("p") == 1
    clear_bm25_cache("p")
