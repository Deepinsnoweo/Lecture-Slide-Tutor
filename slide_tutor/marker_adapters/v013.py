"""Marker 0.1.3 adapter using the established bounded-page pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from threading import RLock

import fitz as pymupdf

from slide_tutor import config
from slide_tutor.marker_adapters.base import ParserPageResult

logger = logging.getLogger(__name__)

# These strings are persisted in cache identities.  Keep the established
# implementation label even though the code now lives behind an adapter.
PAGE_BATCH_ADAPTER_IMPLEMENTATION = (
    "slide_tutor.page_parsers._convert_marker_document_pages_v013"
)
PAGE_BATCH_ADAPTER_VERSION = "v0.1.3-batch-adapter-v1"


class V013MarkerAdapter:
    """Production Marker 0.1.3 candidate provider.

    Marker 0.1.3's public converter loses page boundaries.  The verified batch
    path therefore retains the existing private pipeline, while unknown Marker
    versions use the public one-page conversion path.
    """

    _batch_versions = {"0.1.3"}

    def __init__(self) -> None:
        self._available: bool | None = None
        self._models: list | None = None
        self._runtime_lock = RLock()

    def marker_version(self) -> str | None:
        try:
            return version("marker-pdf")
        except PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._available = (
                version("marker-pdf") is not None
                and find_spec("marker") is not None
            )
        except Exception as exc:
            logger.info(
                "marker-pdf unavailable — PyMuPDF fallback will be used (%s)",
                exc,
            )
            self._available = False
        return self._available

    def cache_identity(self) -> dict[str, str]:
        return {
            "parser_implementation": "marker-pdf",
            "parser_version": self.marker_version() or "unavailable",
            "adapter_implementation": PAGE_BATCH_ADAPTER_IMPLEMENTATION,
            "adapter_version": PAGE_BATCH_ADAPTER_VERSION,
        }

    def batch_supported(self) -> bool:
        return self.marker_version() in self._batch_versions

    @contextmanager
    def local_runtime(self):
        """Temporarily disable model network access when configured offline."""
        if not config.MARKER_LOCAL_FILES_ONLY:
            yield
            return

        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        previous_env = {key: os.environ.get(key) for key in keys}
        hub_constants = sys.modules.get("huggingface_hub.constants")
        transformers_hub = sys.modules.get("transformers.utils.hub")
        previous_hub_offline = getattr(hub_constants, "HF_HUB_OFFLINE", None)
        previous_transformers_offline = getattr(
            transformers_hub, "_is_offline_mode", None,
        )
        try:
            for key in keys:
                os.environ[key] = "1"
            if hub_constants is not None:
                hub_constants.HF_HUB_OFFLINE = True
            if transformers_hub is not None:
                transformers_hub._is_offline_mode = True
            yield
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if hub_constants is not None and previous_hub_offline is not None:
                hub_constants.HF_HUB_OFFLINE = previous_hub_offline
            if (
                transformers_hub is not None
                and previous_transformers_offline is not None
            ):
                transformers_hub._is_offline_mode = previous_transformers_offline

    def _load_models(self) -> list:
        with self._runtime_lock:
            if self._models is not None:
                return self._models
            with self.local_runtime():
                from marker.models import load_all_models

                logger.info("Loading Marker models (one-time, ~2GB)...")
                self._models = load_all_models()
            logger.info("Marker models loaded")
            return self._models

    @staticmethod
    def _write_pages_pdf(
        pdf_path: Path,
        page_numbers: Sequence[int],
        output_path: Path,
    ) -> None:
        if not page_numbers:
            raise ValueError("Marker page batch cannot be empty")
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("Marker page batch contains duplicate page numbers")
        source_doc = pymupdf.open(str(pdf_path))
        batch_doc = pymupdf.open()
        try:
            for page_number in page_numbers:
                page_index = page_number - 1
                if page_index < 0 or page_index >= source_doc.page_count:
                    raise ValueError(
                        f"Page {page_number} out of range "
                        f"(1-{source_doc.page_count})"
                    )
                batch_doc.insert_pdf(
                    source_doc,
                    from_page=page_index,
                    to_page=page_index,
                )
            batch_doc.save(str(output_path))
        finally:
            batch_doc.close()
            source_doc.close()

    def _extract_single(
        self,
        pdf_path: Path,
        page_number: int,
    ) -> ParserPageResult:
        warnings: list[str] = []
        with self._runtime_lock:
            with self.local_runtime():
                from marker.convert import convert_single_pdf

                models = self._load_models()
                with TemporaryDirectory(prefix="slide-tutor-marker-") as tmp_dir:
                    single_page_path = Path(tmp_dir) / f"page-{page_number}.pdf"
                    self._write_pages_pdf(
                        pdf_path,
                        [page_number],
                        single_page_path,
                    )
                    markdown, metadata = convert_single_pdf(
                        str(single_page_path),
                        models,
                        max_pages=1,
                        parallel_factor=1,
                    )

        ocr_stats = metadata.get("ocr_stats", {}) if isinstance(metadata, dict) else {}
        if ocr_stats.get("ocr_pages", 0) > 0:
            warnings.append(f"OCR used on {ocr_stats['ocr_pages']} page(s)")
        if ocr_stats.get("ocr_failed", 0) > 0:
            warnings.append(f"OCR failed on {ocr_stats['ocr_failed']} page(s)")
        if isinstance(metadata, dict) and metadata.get("pages") not in (None, 1):
            warnings.append(
                f"Marker returned {metadata['pages']} pages for a single-page parse"
            )
        if not markdown.strip():
            warnings.append("Marker returned empty output")
        return self._result(
            page_number,
            markdown.strip(),
            warnings,
            ocr_stats=ocr_stats,
        )

    def _convert_document_pages_v013(
        self,
        pdf_path: Path,
    ) -> tuple[list[str], dict]:
        """Run the verified 0.1.3 document stages once per bounded batch."""
        with self.local_runtime():
            from marker.cleaners.bullets import replace_bullets
            from marker.cleaners.code import identify_code_blocks, indent_blocks
            from marker.cleaners.equations import replace_equations
            from marker.cleaners.headers import (
                filter_common_titles,
                filter_header_footer,
            )
            from marker.cleaners.table import create_new_tables, merge_table_blocks
            from marker.convert import annotate_spans
            from marker.debug.data import dump_bbox_debug_data
            from marker.extract_text import get_text_blocks
            from marker.markdown import get_full_text, merge_lines, merge_spans
            from marker.ordering import order_blocks
            from marker.postprocessors.editor import edit_full_text
            from marker.segmentation import detect_document_block_types
            from marker.settings import settings

        with self._runtime_lock:
            models = self._load_models()
            texify_model, layout_model, order_model, edit_model = models
            doc = pymupdf.open(str(pdf_path))
            try:
                language = settings.DEFAULT_LANG
                tess_language = settings.TESSERACT_LANGUAGES.get(language, "eng")
                spell_language = settings.SPELLCHECK_LANGUAGES.get(language)
                if "eng" not in tess_language:
                    tess_language = f"eng+{tess_language}"
                blocks, _toc, ocr_stats = get_text_blocks(
                    doc,
                    tess_language,
                    spell_language,
                    max_pages=None,
                    parallel=settings.OCR_PARALLEL_WORKERS,
                )
                metadata = {"pages": len(blocks), "ocr_stats": ocr_stats}
                if not any(page.blocks for page in blocks):
                    return ["" for _page in blocks], metadata

                block_types = detect_document_block_types(
                    doc,
                    blocks,
                    layout_model,
                    batch_size=settings.LAYOUT_BATCH_SIZE,
                )
                bad_span_ids = filter_header_footer(blocks)
                annotate_spans(blocks, block_types)
                dump_bbox_debug_data(doc, blocks)
                blocks = order_blocks(
                    doc,
                    blocks,
                    order_model,
                    batch_size=settings.ORDERER_BATCH_SIZE,
                )
                identify_code_blocks(blocks)
                indent_blocks(blocks)
                merge_table_blocks(blocks)
                create_new_tables(blocks)
                for page in blocks:
                    for block in page.blocks:
                        block.filter_spans(bad_span_ids)
                        block.filter_bad_span_types()

                filtered, _equation_stats = replace_equations(
                    doc,
                    blocks,
                    block_types,
                    texify_model,
                    batch_size=settings.TEXIFY_BATCH_SIZE,
                )
                merged_pages = merge_spans(filtered)
                if len(merged_pages) != len(filtered):
                    raise RuntimeError("Marker changed page count during span merge")

                page_block_groups = [
                    merge_lines([merged_page], [page]) if merged_page else []
                    for merged_page, page in zip(merged_pages, filtered)
                ]
                all_text_blocks = [
                    block
                    for page_blocks in page_block_groups
                    for block in page_blocks
                ]
                kept_ids = {
                    id(block) for block in filter_common_titles(all_text_blocks)
                }

                page_markdown: list[str] = []
                for page_blocks in page_block_groups:
                    kept_blocks = [
                        block for block in page_blocks if id(block) in kept_ids
                    ]
                    text = get_full_text(kept_blocks)
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    text = re.sub(r"(\n\s){3,}", "\n\n", text)
                    text = replace_bullets(text)
                    text, _edit_stats = edit_full_text(
                        text,
                        edit_model,
                        batch_size=settings.EDITOR_BATCH_SIZE,
                    )
                    page_markdown.append(text.strip())
                return page_markdown, metadata
            finally:
                doc.close()

    def _extract_batch(
        self,
        pdf_path: Path,
        page_numbers: Sequence[int],
    ) -> list[ParserPageResult]:
        if not self.batch_supported():
            raise RuntimeError(
                f"Marker {self.marker_version() or 'unknown'} "
                "has no verified batch adapter"
            )
        with TemporaryDirectory(prefix="slide-tutor-marker-batch-") as tmp_dir:
            batch_path = Path(tmp_dir) / "batch.pdf"
            self._write_pages_pdf(pdf_path, page_numbers, batch_path)
            markdown_pages, metadata = self._convert_document_pages_v013(batch_path)

        if len(markdown_pages) != len(page_numbers):
            raise RuntimeError(
                f"Marker returned {len(markdown_pages)} pages for a "
                f"{len(page_numbers)}-page batch"
            )
        ocr_stats = metadata.get("ocr_stats", {}) if isinstance(metadata, dict) else {}
        batch_warnings: list[str] = []
        if ocr_stats.get("ocr_pages", 0) > 0:
            batch_warnings.append(
                f"Marker batch used OCR on {ocr_stats['ocr_pages']} page(s)"
            )
        if ocr_stats.get("ocr_failed", 0) > 0:
            batch_warnings.append(
                f"Marker batch OCR failed on {ocr_stats['ocr_failed']} page(s)"
            )
        results: list[ParserPageResult] = []
        for page_number, markdown in zip(page_numbers, markdown_pages):
            warnings = list(batch_warnings)
            if not markdown.strip():
                warnings.append("Marker returned empty output")
            results.append(self._result(
                page_number,
                markdown.strip(),
                warnings,
                ocr_stats=ocr_stats,
            ))
        return results

    def _result(
        self,
        page_number: int,
        markdown: str,
        warnings: list[str],
        *,
        ocr_stats: dict | None = None,
    ) -> ParserPageResult:
        ocr_stats = ocr_stats or {}
        return ParserPageResult(
            source_page_number=page_number,
            markdown=markdown,
            structured_blocks=[],
            parser_family="marker",
            parser_version=self.marker_version(),
            mode="legacy-v013",
            ocr_used=bool(ocr_stats.get("ocr_pages", 0)),
            warnings=warnings,
            runtime_facts={"ocr_stats": dict(ocr_stats)},
        )

    def _parse_resilient(
        self,
        pdf_path: Path,
        page_numbers: Sequence[int],
        cancel_check: Callable[[], None] | None,
    ) -> list[ParserPageResult]:
        if cancel_check is not None:
            cancel_check()
        try:
            if len(page_numbers) == 1:
                return [self._extract_single(pdf_path, page_numbers[0])]
            return self._extract_batch(pdf_path, page_numbers)
        except Exception as exc:
            if len(page_numbers) == 1:
                return [ParserPageResult(
                    source_page_number=page_numbers[0],
                    markdown="",
                    parser_family="marker",
                    parser_version=None,
                    mode="legacy-v013",
                    warnings=[f"Marker error: {exc}"],
                    runtime_facts={"error_type": type(exc).__name__},
                )]
            logger.warning(
                "Marker batch failed for pages %s; retrying smaller batches (%s)",
                list(page_numbers),
                exc,
            )
            midpoint = len(page_numbers) // 2
            return [
                *self._parse_resilient(
                    pdf_path,
                    page_numbers[:midpoint],
                    cancel_check,
                ),
                *self._parse_resilient(
                    pdf_path,
                    page_numbers[midpoint:],
                    cancel_check,
                ),
            ]

    def parse_pages(
        self,
        pdf_path: Path,
        page_numbers: Sequence[int],
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> list[ParserPageResult]:
        ordered_page_numbers = list(page_numbers)
        if not ordered_page_numbers:
            return []
        if not self.is_available():
            return []
        if len(ordered_page_numbers) == 1 or not self.batch_supported():
            results: list[ParserPageResult] = []
            for page_number in ordered_page_numbers:
                results.extend(self._parse_resilient(
                    pdf_path,
                    [page_number],
                    cancel_check,
                ))
            return results
        return self._parse_resilient(
            pdf_path,
            ordered_page_numbers,
            cancel_check,
        )
