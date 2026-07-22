"""Context assembly for page-level RAG.

Supports two index backends transparently:
  - Embedding mode: SentenceTransformer + cosine similarity on dense vectors
  - TF-IDF mode: TfidfVectorizer + cosine similarity on sparse vectors
"""

from __future__ import annotations

import re
import hashlib
import json
import unicodedata
from threading import Lock
from typing import Any, Union

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from slide_tutor import config
from slide_tutor.models import RetrievalChunk, RetrievalScope, SlidePage

Index = tuple[list[RetrievalChunk], Any, Union[csr_matrix, np.ndarray]]


class ProjectBM25Cache:
    """Thread-safe project corpus cache; scope filtering happens at query time."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[str, tuple[BM25Okapi, list[RetrievalChunk], int]] = {}
        self._build_counts: dict[str, int] = {}

    @staticmethod
    def _key(project_id: str, chunks: list[RetrievalChunk]) -> str:
        payload = {
            "project_id": project_id,
            "lexical_version": config.LEXICAL_INDEX_VERSION,
            "tokenizer_version": config.LEXICAL_TOKENIZER_VERSION,
            "retrieval_contract": {
                "schema": config.RETRIEVAL_SCHEMA_VERSION,
                "rrf_k": config.RETRIEVAL_RRF_K,
                "title_index": config.RETRIEVAL_TITLE_INDEX_VERSION,
            },
            "chunks": [
                (
                    chunk.slide_id,
                    chunk.file_id,
                    chunk.generation_id,
                    chunk.text,
                    chunk.page_content_contract_version,
                )
                for chunk in chunks
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def get(self, project_id: str, chunks: list[RetrievalChunk]) -> tuple[BM25Okapi, list[RetrievalChunk]]:
        key = self._key(project_id, chunks)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry[0], entry[1]
        corpus = [_lexical_tokens(chunk.text) for chunk in chunks]
        model = BM25Okapi(corpus)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing[0], existing[1]
            self._entries[key] = (model, chunks, 0)
            self._build_counts[project_id] = self._build_counts.get(project_id, 0) + 1
            return model, chunks

    def clear(self, project_id: str | None = None) -> None:
        with self._lock:
            if project_id is None:
                self._entries.clear()
                self._build_counts.clear()
            else:
                keys = [
                    key for key, (_model, chunks, _count) in self._entries.items()
                    if chunks and chunks[0].project_id == project_id
                ]
                for key in keys:
                    self._entries.pop(key, None)
                self._build_counts.pop(project_id, None)

    def build_count(self, project_id: str) -> int:
        with self._lock:
            return self._build_counts.get(project_id, 0)


_project_bm25_cache = ProjectBM25Cache()


def clear_bm25_cache(project_id: str | None = None) -> None:
    _project_bm25_cache.clear(project_id)


def bm25_build_count(project_id: str) -> int:
    return _project_bm25_cache.build_count(project_id)


def scope_allows_chunk(scope: RetrievalScope | None, chunk: RetrievalChunk) -> bool:
    if scope is None:
        return True
    if chunk.project_id != scope.project_id:
        return False
    if scope.kind == "current_page":
        return chunk.slide_id == scope.slide_id and chunk.page_number == scope.page_number
    if scope.kind == "current_file":
        return chunk.file_id == scope.file_id
    if scope.kind == "selected_files":
        return chunk.file_id in scope.file_ids
    if scope.kind == "topic" and scope.file_ids:
        return chunk.file_id in scope.file_ids
    return True


def apply_retrieval_scope(
    index: Index,
    scope: RetrievalScope | None,
) -> Index:
    """Filter before exact/BM25/dense/RRF so no branch can leak files."""
    if scope is None:
        return index
    chunks, backend, matrix = index
    positions = [i for i, chunk in enumerate(chunks) if scope_allows_chunk(scope, chunk)]
    scoped_chunks = [chunks[i] for i in positions]
    if isinstance(matrix, np.ndarray):
        scoped_matrix = matrix[positions]
    else:
        scoped_matrix = matrix.tocsr()[positions]
    return scoped_chunks, backend, scoped_matrix


def _semantic_search(
    query: str,
    chunks: list[RetrievalChunk],
    backend: Any,
    matrix: Union[csr_matrix, np.ndarray],
    top_k: int = config.RETRIEVAL_TOP_K,
    exclude_slide_ids: set[str] | None = None,
) -> list[RetrievalChunk]:
    """Search chunks by cosine similarity to query, dispatching on backend type."""
    if not chunks:
        return []
    exclude = exclude_slide_ids or set()

    if isinstance(matrix, np.ndarray):
        # Embedding mode: encode query → cosine similarity
        query_vec = backend.encode(
            [f"{config.EMBEDDING_QUERY_PREFIX}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=config.EMBEDDING_NORMALIZE,
        )
        scores = cosine_similarity(query_vec, matrix).flatten()
    else:
        # TF-IDF mode: transform query → cosine similarity
        query_vec = backend.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()

    ranked_indices = scores.argsort()[::-1]
    results: list[RetrievalChunk] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]
        if chunk.slide_id not in exclude and len(results) < top_k:
            method = "dense" if isinstance(matrix, np.ndarray) else "tfidf"
            results.append(chunk.model_copy(update={
                "retrieval_methods": [method],
                "dense_rank": rank,
                "dense_score": float(scores[idx]),
            }))
            exclude.add(chunk.slide_id)
    return results


def _lexical_tokens(text: str) -> list[str]:
    """Tokenize mixed Latin/CJK text deterministically for project-level BM25."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized):
        tokens.append(word)
        if "-" in word:
            tokens.extend(part for part in word.split("-") if part)
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", normalized):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _bm25_search(
    query: str,
    chunks: list[RetrievalChunk],
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
    *,
    cache_corpus: list[RetrievalChunk] | None = None,
) -> list[RetrievalChunk]:
    """Run the always-on sparse branch; zero-score pages are never returned."""
    query_tokens = _lexical_tokens(query)
    corpus_chunks = cache_corpus or chunks
    corpus = [_lexical_tokens(chunk.text) for chunk in corpus_chunks]
    if not query_tokens or not any(corpus):
        return []
    if cache_corpus is not None:
        bm25 = _project_bm25_cache.get(corpus_chunks[0].project_id, corpus_chunks)[0]
    else:
        bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1]
    results: list[RetrievalChunk] = []
    for rank, idx in enumerate(ranked, start=1):
        chunk = corpus_chunks[idx]
        allowed_ids = {item.slide_id for item in chunks}
        if scores[idx] <= 0 or len(results) >= top_k:
            break
        if chunk.slide_id not in allowed_ids:
            continue
        results.append(chunk.model_copy(update={
            "retrieval_methods": ["bm25"],
            "bm25_rank": rank,
            "bm25_score": float(scores[idx]),
        }))
    return results


_GENERIC_TITLES = {"introduction", "summary", "overview", "contents", "references"}


def _title_bm25_search(
    query: str,
    chunks: list[RetrievalChunk],
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> list[RetrievalChunk]:
    """Search title metadata as a separate lexical signal.

    Title matching is intentionally generic: it boosts meaningful title terms
    for any deck while ignoring common section labels that otherwise create
    broad false positives.  The body BM25 branch remains responsible for
    evidence recall when a page has no useful title.
    """
    query_tokens = _lexical_tokens(query)
    title_chunks = [
        chunk for chunk in chunks
        if (chunk.title or chunk.normalized_title).strip()
        and (chunk.title or chunk.normalized_title).strip().casefold()
        not in _GENERIC_TITLES
    ]
    if not query_tokens or not title_chunks:
        return []
    corpus = [
        _lexical_tokens(chunk.title or chunk.normalized_title)
        for chunk in title_chunks
    ]
    if not any(corpus):
        return []
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1]
    results: list[RetrievalChunk] = []
    for rank, idx in enumerate(ranked, start=1):
        if scores[idx] <= 0 or len(results) >= top_k:
            break
        chunk = title_chunks[idx]
        results.append(chunk.model_copy(update={
            "retrieval_methods": ["title_bm25"],
            "bm25_rank": rank,
            "bm25_score": float(scores[idx]),
        }))
    return results


def _merge_title_rankings(
    rankings: list[list[RetrievalChunk]],
    top_k: int = config.RETRIEVAL_CANDIDATE_K,
) -> list[tuple[int, RetrievalChunk]]:
    """Merge title-query rankings using each slide's best rank.

    A slide receives the title branch weight only once even when both the
    original question and the controller query match it.  Raw BM25 scores are
    not compared across queries because query length and token composition make
    those scores non-comparable.
    """
    best: dict[str, tuple[int, int, RetrievalChunk]] = {}
    for query_index, ranked in enumerate(rankings):
        for rank, chunk in enumerate(ranked, start=1):
            candidate = (rank, query_index, chunk)
            current = best.get(chunk.slide_id)
            if current is None or candidate[:2] < current[:2]:
                best[chunk.slide_id] = candidate
    merged = sorted(
        best.values(),
        key=lambda item: (item[0], item[1], item[2].slide_id),
    )
    return [(rank, chunk) for rank, _query_index, chunk in merged[:top_k]]


def _exact_title_search(query: str, chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    """Pin exact title/phrase matches without using edit distance as a retriever."""
    from slide_tutor.index_store import normalize_title

    normalized_query = normalize_title(query)
    matches: list[RetrievalChunk] = []
    for chunk in chunks:
        title = chunk.normalized_title or normalize_title(chunk.title)
        if not title or title in _GENERIC_TITLES or len(title) < 3:
            continue
        if title == normalized_query or title in normalized_query:
            matches.append(chunk.model_copy(update={"retrieval_methods": ["exact_title"]}))
    return matches[:config.RETRIEVAL_CANDIDATE_K]


def _hybrid_search(
    semantic_query: str,
    lexical_query: str,
    chunks: list[RetrievalChunk],
    backend: Any,
    matrix: Union[csr_matrix, np.ndarray],
    top_k: int = config.RETRIEVAL_TOP_K,
    exclude_slide_ids: set[str] | None = None,
    cache_corpus: list[RetrievalChunk] | None = None,
) -> list[RetrievalChunk]:
    """Fuse exact-title, title/body BM25, and semantic candidates with RRF."""
    exact = _exact_title_search(lexical_query, chunks)
    title_queries = [lexical_query]
    if sorted(_lexical_tokens(semantic_query)) != sorted(_lexical_tokens(lexical_query)):
        title_queries.append(semantic_query)
    title_ranked = _merge_title_rankings([
        _title_bm25_search(query, chunks)
        for query in title_queries
    ])
    sparse = _bm25_search(lexical_query, chunks, cache_corpus=cache_corpus)
    semantic = _semantic_search(
        semantic_query,
        chunks,
        backend,
        matrix,
        top_k=config.RETRIEVAL_CANDIDATE_K,
    )
    by_id: dict[str, RetrievalChunk] = {}
    methods: dict[str, list[str]] = {}
    scores: dict[str, float] = {}
    details: dict[str, dict[str, int | float | None]] = {}

    ranked_branches = (
        (title_ranked, config.RETRIEVAL_TITLE_LEXICAL_WEIGHT),
        (list(enumerate(sparse, start=1)), config.RETRIEVAL_BODY_LEXICAL_WEIGHT),
        (list(enumerate(semantic, start=1)), config.RETRIEVAL_DENSE_WEIGHT),
    )
    for ranked, branch_weight in ranked_branches:
        for rank, chunk in ranked:
            by_id.setdefault(chunk.slide_id, chunk)
            methods.setdefault(chunk.slide_id, [])
            for method in chunk.retrieval_methods:
                if method not in methods[chunk.slide_id]:
                    methods[chunk.slide_id].append(method)
            detail = details.setdefault(chunk.slide_id, {})
            for field in ("dense_rank", "dense_score", "bm25_rank", "bm25_score"):
                value = getattr(chunk, field)
                if value is not None:
                    detail[field] = value
            scores[chunk.slide_id] = scores.get(chunk.slide_id, 0.0) + (
                branch_weight / (config.RETRIEVAL_RRF_K + rank)
            )

    exact_ids: list[str] = []
    for chunk in exact:
        exact_ids.append(chunk.slide_id)
        by_id.setdefault(chunk.slide_id, chunk)
        methods.setdefault(chunk.slide_id, [])
        if "exact_title" not in methods[chunk.slide_id]:
            methods[chunk.slide_id].insert(0, "exact_title")

    excluded = exclude_slide_ids or set()
    fused_ids = exact_ids + [
        slide_id for slide_id, _score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True,
        ) if slide_id not in exact_ids
    ]
    results: list[RetrievalChunk] = []
    for slide_id in fused_ids:
        if slide_id in excluded or len(results) >= top_k:
            continue
        chunk = by_id[slide_id]
        results.append(chunk.model_copy(update={
            **details.get(slide_id, {}),
            "retrieval_methods": methods.get(slide_id, []),
            "rrf_score": scores.get(slide_id),
        }))
    return results


def _dedupe_chunks(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    """Remove duplicate slide_ids, keeping first occurrence."""
    seen: set[str] = set()
    result: list[RetrievalChunk] = []
    for chunk in chunks:
        if chunk.slide_id not in seen:
            seen.add(chunk.slide_id)
            result.append(chunk)
    return result


def _find_neighbor(
    chunks: list[RetrievalChunk],
    current_slide: SlidePage,
    offset: int,
) -> RetrievalChunk | None:
    """Find a chunk whose file_id matches current_slide and whose page_number
    is current + offset.  Returns None if no such neighbor exists."""
    target_page = current_slide.page_number + offset
    if target_page < 1:
        return None
    for chunk in chunks:
        if (chunk.file_id == current_slide.file_id
                and chunk.page_number == target_page):
            return chunk
    return None


def retrieve_for_current_page(
    current_slide: SlidePage,
    index: Index,
    scope: RetrievalScope | None = None,
) -> list[RetrievalChunk]:
    """Retrieve context for explaining the currently selected slide page.

    Priority order:
      1. current page (always first)
      2. previous page (same file, page-1)
      3. next page (same file, page+1)
      4. semantic top-k from the index
    """
    original_chunks = index[0]
    chunks, backend, matrix = apply_retrieval_scope(index, scope)
    if not chunks:
        return []

    result: list[RetrievalChunk] = []
    exclude: set[str] = set()

    # 1. Current page
    current = _find_neighbor(chunks, current_slide, offset=0)
    if current:
        result.append(current)
        exclude.add(current.slide_id)

    # 2. Previous page
    prev_chunk = _find_neighbor(chunks, current_slide, offset=-1)
    if prev_chunk:
        result.append(prev_chunk)
        exclude.add(prev_chunk.slide_id)

    # 3. Next page
    next_chunk = _find_neighbor(chunks, current_slide, offset=1)
    if next_chunk:
        result.append(next_chunk)
        exclude.add(next_chunk.slide_id)

    # 4. Semantic top-k
    query = current_slide.text if current_slide.text else current_slide.title_guess or ""
    semantic = _hybrid_search(
        query, query, chunks, backend, matrix,
        exclude_slide_ids=exclude,
        cache_corpus=original_chunks,
    )
    result.extend(semantic)

    return _dedupe_chunks(result)


def retrieve_for_question(
    question: str,
    current_slide: SlidePage | None,
    index: Index,
    lexical_query: str | None = None,
    scope: RetrievalScope | None = None,
) -> list[RetrievalChunk]:
    """Retrieve context with hybrid recall, optionally preserving original wording."""
    original_chunks = index[0]
    chunks, backend, matrix = apply_retrieval_scope(index, scope)
    if not chunks:
        return []

    result: list[RetrievalChunk] = []
    exclude: set[str] = set()

    if current_slide is not None:
        current = _find_neighbor(chunks, current_slide, offset=0)
        if current:
            result.append(current)
            exclude.add(current.slide_id)

    semantic = _hybrid_search(
        question,
        lexical_query or question,
        chunks,
        backend,
        matrix,
        exclude_slide_ids=exclude,
        cache_corpus=original_chunks,
    )
    result.extend(semantic)
    return _dedupe_chunks(result)


def format_context(chunks: list[RetrievalChunk]) -> str:
    """Format a list of retrieval chunks into a plain-text context block
    suitable for insertion into an LLM prompt.  Truncates to MAX_CONTEXT_CHARS."""
    parts: list[str] = []
    total = 0
    max_chars = config.MAX_CONTEXT_CHARS

    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.source_file}, page {chunk.page_number}"
        body = chunk.text.strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts)
