import numpy as np

from slide_tutor.models import RetrievalChunk
from slide_tutor.retrieval import (
    _lexical_tokens,
    _merge_title_rankings,
    retrieve_for_question,
)


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        # Deliberately favors the distractor so exact/BM25 behavior is observable.
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def _chunk(page: int, title: str, body: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=f"slide-{page}",
        project_id="project-1",
        file_id="file-1",
        slide_id=f"slide-{page}",
        source_file="neuro.pdf",
        page_number=page,
        title=title,
        normalized_title=title.casefold(),
        text=f"{title}\n\n{body}".strip(),
    )


def test_exact_title_pins_concept_page_ahead_of_dense_distractor():
    chunks = [
        _chunk(5, "Short-Term Memories", "memory consolidation"),
        _chunk(50, "Long-Term Depression", "low frequency stimulation causes LTD"),
        _chunk(29, "Bradykinesia", "slow movement"),
    ]
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)

    results = retrieve_for_question(
        "Long-term depression在哪一页？",
        None,
        (chunks, FakeEmbeddingModel(), matrix),
    )

    assert results[0].page_number == 50
    assert "exact_title" in results[0].retrieval_methods
    assert "title_bm25" in results[0].retrieval_methods
    assert "bm25" in results[0].retrieval_methods


def test_original_query_supplies_lexical_terms_when_planner_rewrites_query():
    chunks = [
        _chunk(50, "Long-Term Depression", "LTD weakens synapses"),
        _chunk(51, "Potentiation", "LTP strengthens synapses"),
        _chunk(52, "Habituation", "Repeated stimulation changes response"),
    ]
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)

    results = retrieve_for_question(
        "relationship to synaptic plasticity",
        None,
        (chunks, FakeEmbeddingModel(), matrix),
        lexical_query="LTD 在哪一页？",
    )

    ltd = next(result for result in results if result.page_number == 50)
    assert "bm25" in ltd.retrieval_methods


def test_mixed_language_tokenizer_preserves_terms_and_cjk_bigrams():
    tokens = _lexical_tokens("Long-Term Depression 与长期抑制")

    assert "long-term" in tokens
    assert "long" in tokens
    assert "term" in tokens
    assert "长期" in tokens
    assert "抑制" in tokens


def test_title_lexical_branch_promotes_matching_title_without_case_rule():
    chunks = [
        _chunk(1, "Distributed Systems", "unrelated body text"),
        _chunk(2, "Gradient Descent", "optimization details"),
        _chunk(3, "Attention Mechanism", "query discusses attention"),
    ]
    matrix = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.99, 0.01]],
        dtype=np.float32,
    )

    results = retrieve_for_question(
        "What is the gradient descent procedure?",
        None,
        (chunks, FakeEmbeddingModel(), matrix),
    )

    gradient = next(result for result in results if result.page_number == 2)
    assert "title_bm25" in gradient.retrieval_methods


def test_generic_titles_do_not_create_title_lexical_matches():
    chunks = [
        _chunk(1, "Introduction", "background"),
        _chunk(2, "Ablation Study", "remove one component"),
    ]
    matrix = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

    results = retrieve_for_question(
        "introduction",
        None,
        (chunks, FakeEmbeddingModel(), matrix),
    )

    intro = next(result for result in results if result.page_number == 1)
    assert "title_bm25" not in intro.retrieval_methods


def test_controller_query_can_supply_cross_language_title_terms():
    chunks = [
        _chunk(31, "LLMは「世界モデル」を持っているか？", "related discussion"),
        _chunk(32, "２種類の「世界モデル」", "direct two-type definition"),
        _chunk(39, "集合的世界モデル仮説", "long related explanation"),
    ]
    matrix = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        dtype=np.float32,
    )

    results = retrieve_for_question(
        "LLM world model 世界モデル 2種類",
        None,
        (chunks, FakeEmbeddingModel(), matrix),
        lexical_query="LLM 是否拥有 world model？课件区分了哪两种？",
    )

    page32 = next(result for result in results if result.page_number == 32)
    assert "title_bm25" in page32.retrieval_methods
    assert "exact_title" not in page32.retrieval_methods


def test_matching_both_title_queries_contributes_only_one_title_branch_weight():
    page1 = _chunk(1, "Gradient Descent", "optimization")
    page2 = _chunk(2, "Attention", "neural attention")

    merged = _merge_title_rankings([
        [page2, page1],
        [page1],
    ])

    assert len(merged) == 2
    assert {chunk.page_number for _rank, chunk in merged} == {1, 2}
    assert next(rank for rank, chunk in merged if chunk.page_number == 1) == 1
