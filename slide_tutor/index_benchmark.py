"""Local benchmark runner for Stage A index rebuild reliability.

Benchmarks are intentionally opt-in and operate on a temporary absence of the
project index.  The original chunks/index artefacts are restored after each
sample, so measuring a real project does not change user data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from slide_tutor import config
from slide_tutor import index_store
from slide_tutor.index_store import (
    IndexProgress,
    ensure_index,
    rebuild_index,
    reset_embedding_model_cache,
)
from slide_tutor.projects import load_project


class IndexBenchmarkSample(BaseModel):
    project_id: str
    ready_page_count: int = Field(ge=0)
    mode: Literal["cold", "warm"]
    operation: Literal["explicit_preindex", "lazy_rebuild"]
    status: Literal["passed", "failed"]
    elapsed_seconds: float = Field(ge=0)
    phase_seconds: dict[str, float] = Field(default_factory=dict)
    chunk_count: int = Field(ge=0)
    backend: str | None = None
    error: str | None = None


class IndexBenchmarkReport(BaseModel):
    report_version: str = "index-benchmark-v1"
    generated_at: str
    samples: list[IndexBenchmarkSample]


@contextmanager
def _temporarily_remove_index(project_id: str) -> Iterator[None]:
    """Hide current index artefacts and restore them even if a build fails."""
    project_dir = config.PROJECTS_DIR / project_id
    backup_root = Path(tempfile.mkdtemp(prefix=".benchmark-index-", dir=project_dir))
    chunks = project_dir / config.CHUNKS_FILENAME
    index_dir = project_dir / config.INDEX_DIRNAME
    backup_chunks = backup_root / config.CHUNKS_FILENAME
    backup_index = backup_root / config.INDEX_DIRNAME
    had_chunks = chunks.exists()
    had_index = index_dir.exists()
    try:
        if had_chunks:
            chunks.rename(backup_chunks)
        if had_index:
            index_dir.rename(backup_index)
        yield
    finally:
        if chunks.exists():
            chunks.unlink(missing_ok=True)
        if index_dir.exists():
            shutil.rmtree(index_dir, ignore_errors=True)
        if had_chunks and backup_chunks.exists():
            backup_chunks.rename(chunks)
        if had_index and backup_index.exists():
            backup_index.rename(index_dir)
        shutil.rmtree(backup_root, ignore_errors=True)


def _phase_durations(events: list[IndexProgress]) -> dict[str, float]:
    starts: dict[str, float] = {}
    durations: dict[str, float] = {}
    for event in events:
        if event.status == "started":
            starts[event.stage] = event.elapsed_seconds
        elif event.status == "completed" and event.stage in starts:
            durations[event.stage] = round(
                max(0.0, event.elapsed_seconds - starts[event.stage]), 4,
            )
    return durations


def _run_sample(
    project_id: str,
    mode: Literal["cold", "warm"],
    operation: Literal["explicit_preindex", "lazy_rebuild"],
) -> IndexBenchmarkSample:
    project = load_project(project_id)
    ready_page_count = sum(
        file.page_count for file in project.files if file.status == "ready"
    )
    if mode == "cold":
        reset_embedding_model_cache()
    events: list[IndexProgress] = []
    started = time.perf_counter()
    try:
        with _temporarily_remove_index(project_id):
            if operation == "explicit_preindex":
                index = rebuild_index(project_id, progress_callback=events.append)
            else:
                index = ensure_index(project_id, progress_callback=events.append)
        elapsed = time.perf_counter() - started
        backend = events[-1].backend if events else None
        return IndexBenchmarkSample(
            project_id=project_id,
            ready_page_count=ready_page_count,
            mode=mode,
            operation=operation,
            status="passed",
            elapsed_seconds=round(elapsed, 4),
            phase_seconds=_phase_durations(events),
            chunk_count=len(index[0]),
            backend=backend,
        )
    except Exception as exc:
        return IndexBenchmarkSample(
            project_id=project_id,
            ready_page_count=ready_page_count,
            mode=mode,
            operation=operation,
            status="failed",
            elapsed_seconds=round(time.perf_counter() - started, 4),
            phase_seconds=_phase_durations(events),
            chunk_count=0,
            backend=events[-1].backend if events else None,
            error=f"{type(exc).__name__}: {exc}",
        )


def benchmark_projects(
    project_ids: list[str],
    *,
    repeats: int = 1,
    include_lazy: bool = True,
    backend: Literal["configured", "tfidf"] = "configured",
) -> IndexBenchmarkReport:
    """Measure cold/warm explicit and lazy rebuild paths for selected projects."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if backend not in {"configured", "tfidf"}:
        raise ValueError(f"unsupported benchmark backend: {backend}")
    original_loader = index_store._load_embedding_model
    if backend == "tfidf":
        # Controlled fallback mode makes the 65/128/735-scale pipeline
        # benchmark independent of network/model availability.
        index_store._load_embedding_model = lambda: None
    samples: list[IndexBenchmarkSample] = []
    operations = ["explicit_preindex", "lazy_rebuild"] if include_lazy else [
        "explicit_preindex"
    ]
    try:
        for project_id in project_ids:
            for mode in ("cold", "warm"):
                for operation in operations:
                    for _ in range(repeats):
                        samples.append(_run_sample(project_id, mode, operation))
        return IndexBenchmarkReport(
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            samples=samples,
        )
    finally:
        index_store._load_embedding_model = original_loader


def save_benchmark_report(path: str | Path, report: IndexBenchmarkReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark slide index rebuild paths")
    parser.add_argument("--project-id", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=["configured", "tfidf"], default="configured")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark_projects(
        args.project_id,
        repeats=args.repeats,
        backend=args.backend,
    )
    save_benchmark_report(args.output, report)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
