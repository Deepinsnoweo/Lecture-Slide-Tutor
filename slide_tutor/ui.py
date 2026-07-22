from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from streamlit_sortables import sort_items

from slide_tutor import cache, config
from slide_tutor.ingestion import IngestionError, add_uploads_to_project
from slide_tutor.models import ProjectFile, ProjectManifest, SlidePage
from slide_tutor.projects import (
    create_project,
    delete_project,
    delete_project_file,
    list_projects,
    load_file_slides,
    load_project,
    ordered_files,
    utc_now,
    update_file_order,
)

# Phase 2 imports
from slide_tutor.index_store import (
    IndexProgress,
    cancel_index_job,
    clear_finished_index_job,
    drain_index_job_progress,
    get_index_job_snapshot,
    mark_index_job_terminal_delivered,
    start_index_job,
)

# Phase 2 Advance imports
from slide_tutor.page_understanding import load_page_understanding

# Phase 4 Advance Stage A imports
from slide_tutor.qa_jobs import QAJobRequest, qa_job_registry
from slide_tutor.qa_pipelines import run_qa_pipeline
from slide_tutor.qa_records import (
    current_response_from_snapshot,
    clear_project_answer_history,
    delete_project_answer_record,
    finalize_qa_snapshot,
    load_project_answer_history,
    migrate_legacy_answer_state,
    project_answer_records,
    persist_answer_record,
    remove_answer_records,
    save_project_answer_history,
    upsert_answer_record,
)
from slide_tutor.thumbnails import ensure_page_image

# Phase 3 imports
from slide_tutor.quiz import (
    build_question_type_mix,
    delete_quiz_attempt,
    format_question_type_mix,
    list_quiz_attempts,
    load_cached_quiz,
    load_quiz_attempt,
)
from slide_tutor.models import QuizSet
from slide_tutor.quiz_formatting import format_quiz_answer, question_by_id
from slide_tutor.quiz_markdown import normalize_quiz_markdown, render_quiz_markdown
from slide_tutor.quiz_quality import (
    PLANNER_PROMPT_VERSION,
    REVIEW_PROMPT_VERSION,
    active_generation_snapshot,
)
from slide_tutor.quiz_jobs import (
    QuizGenerationRequest,
    quiz_generation_registry,
)
from slide_tutor.quiz_submission import submit_quiz_answers


_QUESTION_TYPE_DISPLAY = {
    "mcq": "MCQ",
    "true_false": "T/F",
    "fill_blank": "Fill",
    "short_answer": "Short",
}

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
_APP_LOGO_PATH = _ASSET_DIR / "lecture-slide-tutor-logo.svg"
_APP_ICON_PATH = _ASSET_DIR / "lecture-slide-tutor-icon.svg"


def _render_app_shell() -> None:
    """Render compact, product-owned chrome without hiding Streamlit controls."""
    st.logo(
        str(_APP_LOGO_PATH),
        size="large",
        icon_image=str(_APP_ICON_PATH),
    )
    st.html(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 4rem;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        </style>
        """
    )


def _resolve_file_path(project_id: str, file_id: str, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    return cache.get_project_file_dir(project_id, file_id) / relative_path


def _quality_label(slide: SlidePage) -> str:
    if slide.text_quality == "ok":
        return "text ok"
    if slide.text_quality == "low":
        return "low text"
    return "no text"


def _project_label(project: ProjectManifest) -> str:
    return f"{project.name} ({len(project.files)} PDFs)"


def _ordered_ready_files(project: ProjectManifest) -> list[ProjectFile]:
    return [file for file in ordered_files(project) if file.status == "ready"]


def _quiz_generation_quality_notice(lifecycle_status: str) -> str | None:
    """Return the PLAN-required quality hint when quiz surfaces use baseline.

    Quiz, Final exam and Weakness use the active page/index snapshot.  Until a
    coherent enhanced snapshot is active, the user must be told that those
    surfaces are operating on baseline content rather than silently implying
    enhanced formula/table/OCR coverage.
    """
    if lifecycle_status == "enhanced_current":
        return None
    if lifecycle_status == "enhancing":
        return (
            "Using baseline content while enhancement runs. "
            "Formulas, tables, and OCR may need review."
        )
    if lifecycle_status == "enhancement_failed":
        return (
            "Using baseline content because enhancement failed. "
            "Formulas, tables, and OCR may need review."
        )
    if lifecycle_status == "baseline_current":
        return (
            "Using baseline content. Formulas, tables, and OCR may need review."
        )
    return None


_GENERATION_LIFECYCLE_LABELS = {
    "baseline_building": "Building baseline · Existing snapshot remains searchable",
    "baseline_current": "Baseline ready · Enhanced processing not ready",
    "enhancing": "Baseline ready · Enhancing formulas, tables, and OCR",
    "enhanced_current": "Enhanced index ready",
    "enhancement_failed": "Enhancement failed · Baseline remains searchable",
    "stale": "Source snapshot out of date · Refresh required",
}


def _generation_lifecycle_label(lifecycle_status: str) -> str:
    """Return compact English-only status text for generation lifecycle UI."""
    return _GENERATION_LIFECYCLE_LABELS.get(
        lifecycle_status, "Generation status unavailable",
    )


def _index_progress_ui() -> tuple[object, object, object]:
    """Create a compact progress renderer shared by lazy and explicit rebuilds."""
    bar = st.sidebar.progress(0)
    message = st.sidebar.empty()
    stages = {
        "load shared model": 0,
        "build chunks": 1,
        "encode passages": 2,
        "persist index": 3,
        "activate index": 4,
    }

    def render(event: IndexProgress) -> None:
        stage_number = stages.get(event.stage, 0)
        ratio = event.completed / event.total if event.total else (
            1.0 if event.status == "completed" else 0.0
        )
        fraction = min(1.0, max(0.0, (stage_number + ratio) / len(stages)))
        bar.progress(fraction)
        detail = event.detail or event.status
        progress_text = (
            f"{event.stage} · {event.completed}/{event.total} · "
            f"{event.elapsed_seconds:.1f}s"
            if event.total else f"{event.stage} · {event.elapsed_seconds:.1f}s"
        )
        message.caption(f"{progress_text} · {detail}")

    def clear() -> None:
        bar.empty()
        message.empty()

    return render, bar, clear


@st.fragment(run_every=1)
def _render_preindex_job_fragment(
    project_id: str,
    total_pages: int,
    operation_label: str = "Pre-index all pages",
) -> None:
    """Render one project's worker state; the worker never touches Streamlit."""
    snapshot = get_index_job_snapshot(project_id)
    if snapshot is None:
        if st.button(
            operation_label,
            use_container_width=True,
            key=f"preindex-start-{project_id}",
            help=(
                "Build a coherent page-content and retrieval generation in the background; "
                "the previous generation remains available until activation succeeds"
            ),
        ):
            try:
                start_index_job(project_id)
            except RuntimeError as exc:
                st.warning(str(exc))
            else:
                st.rerun()
        return

    events = drain_index_job_progress(project_id)
    snapshot = get_index_job_snapshot(project_id)
    if snapshot is None:
        return
    event = events[-1] if events else snapshot.last_progress
    if event is not None:
        stage_ranges = {
            "queued": (0.0, 0.0),
            "build page content": (0.0, 0.55),
            "load shared model": (0.55, 0.60),
            "build chunks": (0.60, 0.72),
            "encode passages": (0.72, 0.90),
            "persist index": (0.90, 0.97),
            "activate index": (0.97, 1.0),
        }
        ratio = event.completed / event.total if event.total else (
            1.0 if event.status == "completed" else 0.0
        )
        start, end = stage_ranges.get(event.stage, (0.0, 0.0))
        st.progress(min(1.0, max(0.0, start + (end - start) * ratio)))
        counts = f"{event.completed}/{event.total}" if event.total else "working"
        reused = f" · reused={event.reused}" if event.reused else ""
        backend = f" · backend={event.backend}" if event.backend else ""
        dimension = f" · dimension={event.dimension}" if event.dimension else ""
        detail = f" · {event.detail}" if event.detail else ""
        st.caption(
            f"{event.stage} · {counts} · {event.elapsed_seconds:.1f}s"
            f"{reused}{backend}{dimension}{detail}"
        )
        if event.backend == "tfidf" and event.fallback_reason:
            st.warning(f"Embedding fallback: {event.fallback_reason}")

    operation_name = (
        "Pre-index" if snapshot.operation == "explicit_preindex" else "Lazy index"
    )
    if snapshot.status in {"queued", "running"}:
        st.button(
            f"Cancel {operation_name.lower()}",
            use_container_width=True,
            key=f"preindex-cancel-{project_id}",
            on_click=cancel_index_job,
            args=(project_id,),
        )
        state = "queued" if snapshot.status == "queued" else "running"
        st.caption(f"{operation_name} {state} · {total_pages} source pages")
    elif mark_index_job_terminal_delivered(project_id):
        # The sidebar status is computed outside this fragment.  A full rerun
        # lets it observe the activated generation before the job is removed.
        st.rerun(scope="app")
    elif snapshot.status == "completed":
        st.success(f"{operation_name} complete · {total_pages} pages")
        clear_finished_index_job(project_id)
    elif snapshot.status == "cancelled":
        st.warning(f"{operation_name} cancelled; previous compatible index preserved")
        clear_finished_index_job(project_id)
    else:
        st.error(f"{operation_name} failed: {snapshot.error or 'unknown error'}")
        clear_finished_index_job(project_id)


def _ensure_selected_file(project: ProjectManifest) -> None:
    current_project = st.session_state.get("selected_project_id")
    ready_files = _ordered_ready_files(project)
    ready_ids = {file.file_id for file in ready_files}
    selected_file = st.session_state.get("selected_file_id")

    if current_project != project.project_id:
        st.session_state["selected_project_id"] = project.project_id
        st.session_state["selected_file_id"] = ready_files[0].file_id if ready_files else None
        st.session_state["expanded_file_id"] = ready_files[0].file_id if ready_files else None
        st.session_state["selected_page_number"] = 1
        return

    if selected_file not in ready_ids:
        st.session_state["selected_file_id"] = ready_files[0].file_id if ready_files else None
        st.session_state["expanded_file_id"] = ready_files[0].file_id if ready_files else None
        st.session_state["selected_page_number"] = 1

    expanded_file = st.session_state.get("expanded_file_id")
    if expanded_file is not None and expanded_file not in ready_ids:
        st.session_state["expanded_file_id"] = st.session_state.get("selected_file_id")


def _selected_file(project: ProjectManifest) -> ProjectFile | None:
    selected_id = st.session_state.get("selected_file_id")
    for file in project.files:
        if file.file_id == selected_id:
            return file
    ready_files = _ordered_ready_files(project)
    return ready_files[0] if ready_files else None


def _selected_slide(slides: list[SlidePage]) -> SlidePage | None:
    if not slides:
        return None
    page_number = int(st.session_state.get("selected_page_number", 1))
    page_number = min(max(page_number, 1), len(slides))
    st.session_state["selected_page_number"] = page_number
    return slides[page_number - 1]


def _find_cited_file(project: ProjectManifest, source_file: str) -> ProjectFile | None:
    """Resolve an LLM citation filename to a project file with tolerant matching."""
    src_lower = (source_file or "").strip().lower()
    if not src_lower:
        return None
    for file in project.files:
        if file.original_name.strip().lower() == src_lower:
            return file
    for file in project.files:
        name_lower = file.original_name.strip().lower()
        if src_lower in name_lower or name_lower in src_lower:
            return file
    return None


_PROJECT_PICKER_KEY = "project_picker_id"


def _clear_project_scoped_state() -> None:
    """Reset UI state that should not carry across projects."""
    for key in (
        "selected_file_id",
        "expanded_file_id",
        "selected_page_number",
        "delete_confirm_file_id",
        "delete_confirm_project_id",
        "qa_memory",
        "qa_last_answer",
        "qa_last_question",
        "qa_last_bundle",
        "quiz_file_select",
        "quiz_file_id",
        "quiz_type",
        "quiz_target_count",
    ):
        st.session_state.pop(key, None)
    _clear_quiz_session_state()


def _on_project_picker_change() -> None:
    """Synchronize the project selectbox and clear stale per-project state."""
    selected_id = st.session_state.get(_PROJECT_PICKER_KEY)
    if not selected_id or selected_id == st.session_state.get("selected_project_id"):
        return
    st.session_state["selected_project_id"] = selected_id
    _clear_project_scoped_state()


def _consume_pending_browse_jump() -> None:
    """Apply deferred quiz-to-browse navigation before Streamlit widgets render."""
    pending = st.session_state.pop("_pending_browse_jump", None)
    if not isinstance(pending, dict):
        return

    file_id = pending.get("file_id")
    if not file_id:
        return

    try:
        page_number = int(pending.get("page_number", 1))
    except (TypeError, ValueError):
        page_number = 1

    project_id = pending.get("project_id")
    if project_id:
        st.session_state["selected_project_id"] = project_id
    st.session_state["app_mode"] = "browse"
    st.session_state["app_mode_radio"] = "Browse slides"
    st.session_state["selected_file_id"] = file_id
    st.session_state["expanded_file_id"] = file_id
    st.session_state["selected_page_number"] = max(page_number, 1)
    if pending.get("return_to") == "quiz":
        st.session_state["_return_to_quiz_results"] = True
    elif pending.get("return_to") == "dashboard":
        st.session_state["_return_to_dashboard"] = True


def _consume_pending_quiz_return() -> None:
    """Apply deferred browse-to-quiz or browse-to-dashboard navigation
    before Streamlit widgets render."""
    if not st.session_state.pop("_pending_quiz_return", False):
        return

    st.session_state["app_mode"] = "quiz"
    st.session_state["app_mode_radio"] = "Take quiz"
    st.session_state.pop("_return_to_quiz_results", None)


def _consume_pending_dashboard_return() -> None:
    """Apply deferred browse-to-dashboard navigation before widgets render."""
    if not st.session_state.pop("_pending_dashboard_return", False):
        return

    st.session_state["app_mode"] = "dashboard"
    st.session_state["app_mode_radio"] = "Dashboard"
    st.session_state.pop("_return_to_dashboard", None)


def _render_project_picker() -> ProjectManifest | None:
    st.sidebar.subheader("Project")
    projects = list_projects()

    with st.sidebar.form("new-project-form", clear_on_submit=True):
        project_name = st.text_input("New project", placeholder="Machine Learning")
        create_clicked = st.form_submit_button("Create project", use_container_width=True)
        if create_clicked:
            try:
                project = create_project(project_name)
                st.session_state["selected_project_id"] = project.project_id
                st.session_state[_PROJECT_PICKER_KEY] = project.project_id
                _clear_project_scoped_state()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))

    projects = list_projects()
    if not projects:
        st.info("Create a project before uploading PDFs.")
        return None

    project_ids = [project.project_id for project in projects]
    current_id = st.session_state.get("selected_project_id")
    if current_id not in project_ids:
        current_id = project_ids[0]
        st.session_state["selected_project_id"] = current_id
        _clear_project_scoped_state()

    if st.session_state.get(_PROJECT_PICKER_KEY) != current_id:
        st.session_state[_PROJECT_PICKER_KEY] = current_id

    selected_id = st.sidebar.selectbox(
        "Current project",
        project_ids,
        index=project_ids.index(current_id),
        key=_PROJECT_PICKER_KEY,
        format_func=lambda project_id: _project_label(next(p for p in projects if p.project_id == project_id)),
        on_change=_on_project_picker_change,
    )
    selected_id = st.session_state.get("selected_project_id", selected_id)
    project = load_project(selected_id)

    # Delete project button with two-step confirmation
    _render_delete_project(project)

    return project


def _render_pdf_upload(project: ProjectManifest) -> ProjectManifest:
    st.sidebar.subheader("Add PDFs")
    upload_key = f"pdf-upload-{project.project_id}-{st.session_state.get('upload_nonce', 0)}"
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF slides",
        type=["pdf"],
        accept_multiple_files=True,
        key=upload_key,
    )
    if st.sidebar.button("Add to project", use_container_width=True, disabled=not uploaded_files):
        try:
            with st.spinner("Processing PDFs..."):
                result = add_uploads_to_project(project.project_id, list(uploaded_files or []))
            if result.added_files:
                st.sidebar.success(f"Added {len(result.added_files)} PDF(s).")
            if result.skipped_files:
                st.sidebar.info(f"Skipped existing PDF(s): {', '.join(result.skipped_files)}")
            if result.failed_files:
                st.sidebar.error("\n".join(result.failed_files))
            st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
            st.rerun()
        except ModuleNotFoundError as exc:
            st.sidebar.error(f"Missing dependency: {exc.name}. Install Phase 1 requirements first.")
        except (IngestionError, ValueError) as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # pragma: no cover - last-resort UI guard
            st.sidebar.error(f"Unexpected processing error: {exc}")
    return load_project(project.project_id)


def _render_pdf_sorter(project: ProjectManifest) -> ProjectManifest:
    files = ordered_files(project)
    if not files:
        st.sidebar.subheader("PDFs")
        st.sidebar.info("No PDFs in this project yet.")
        return project

    # Phase 4 Advance: Pre-index status and button
    from slide_tutor.index_store import get_preindex_status
    ready_files = _ordered_ready_files(project)
    total_pages = sum(f.page_count for f in ready_files)
    preindex_status = get_preindex_status(project.project_id)

    if preindex_status == "current":
        index_meta_path = cache.get_project_dir(project.project_id) / "index" / "index_meta.json"
        meta = cache.load_json(index_meta_path, default={}) or {}
        indexed_at = meta.get("indexed_at", "")[:16].replace("T", " ")
        st.sidebar.caption(
            f"Pre-indexed · {indexed_at} · {total_pages} source pages"
        )
    elif preindex_status == "lazy_current":
        st.sidebar.caption(f"Lazy index ready · {total_pages} source pages")
    elif preindex_status == "not_ready":
        st.sidebar.caption("No PDFs ready for indexing")
    elif preindex_status == "stale":
        st.sidebar.caption("Pre-index is out of date — project files changed")
    elif preindex_status == "not_indexed":
        st.sidebar.caption("Not pre-indexed — Q&A will build a lazy index on first use")

    if preindex_status != "not_ready":
        from slide_tutor.generation import generation_state
        from slide_tutor.index_store import compute_source_signature
        generation = generation_state(
            project.project_id, compute_source_signature(project.project_id),
        )
        st.sidebar.caption(_generation_lifecycle_label(generation.lifecycle_status))
        if generation.page_status in {"missing", "stale"}:
            operation_label = "Refresh sources"
        elif generation.index_status in {"missing", "stale"}:
            operation_label = "Rebuild index"
        else:
            operation_label = "Pre-index all pages"
        with st.sidebar:
            _render_preindex_job_fragment(project.project_id, total_pages, operation_label)

    labels_by_id = {
        file.file_id: f"{file.original_name} · {file.page_count}p · {file.file_id[:6]}"
        for file in files
    }
    ids_by_label = {label: file_id for file_id, label in labels_by_id.items()}
    labels = [labels_by_id[file.file_id] for file in files]

    if len(files) > 1:
        with st.sidebar:
            sorted_labels = sort_items(
                labels,
                header="Drag to reorder PDFs",
                direction="vertical",
                custom_style="""
                .sortable-component {
                    background: transparent;
                    border: 0;
                    padding: 0;
                    margin: 0 0 12px 0;
                }
                .sortable-container {
                    background: transparent;
                    border: 0;
                    padding: 0;
                }
                .sortable-container-header {
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #31333f;
                    padding: 0 0 4px 0;
                }
                .sortable-container-body {
                    background: transparent;
                    border: 0;
                    padding: 0;
                }
                .sortable-item {
                    background: transparent !important;
                    color: #31333f !important;
                    border: 0 !important;
                    box-shadow: none !important;
                    border-radius: 0 !important;
                    padding: 2px 0 !important;
                    margin: 0 !important;
                    font-size: 0.82rem !important;
                    font-weight: 400 !important;
                    text-align: left !important;
                    line-height: 1.25 !important;
                }
                .sortable-item:hover {
                    text-decoration: underline;
                }
                """,
                key=f"pdf-order-{project.project_id}",
            )
    else:
        sorted_labels = labels

    sorted_ids = [ids_by_label[label] for label in sorted_labels if label in ids_by_label]
    current_ids = [file.file_id for file in files]
    if sorted_ids and sorted_ids != current_ids:
        project = update_file_order(project.project_id, sorted_ids)
        st.rerun()

    st.sidebar.subheader("PDFs")
    selected_id = st.session_state.get("selected_file_id")
    expanded_id = st.session_state.get("expanded_file_id")
    confirm_key = "delete_confirm_file_id"

    for file in ordered_files(project):
        status = "" if file.status == "ready" else f" · {file.status}"
        label = f"{file.original_name} · {file.page_count} pages{status}"

        # Render file button and delete button side by side
        btn_col, del_col = st.sidebar.columns([10, 1])
        with btn_col:
            if st.button(
                label,
                key=f"select-file-{file.file_id}",
                type="primary" if file.file_id == selected_id else "secondary",
                use_container_width=True,
                disabled=file.status != "ready",
            ):
                if expanded_id == file.file_id:
                    st.session_state["expanded_file_id"] = None
                    st.session_state["selected_file_id"] = file.file_id
                else:
                    if selected_id != file.file_id:
                        st.session_state["selected_page_number"] = 1
                    st.session_state["selected_file_id"] = file.file_id
                    st.session_state["expanded_file_id"] = file.file_id
                st.rerun()
        with del_col:
            if st.button("✕", key=f"delete-file-{file.file_id}", help="Delete this PDF"):
                st.session_state[confirm_key] = file.file_id
                st.rerun()

        # Two-step confirmation for file deletion
        if st.session_state.get(confirm_key) == file.file_id:
            st.sidebar.warning(f"Delete **{file.original_name}**?")
            cf_col, cc_col = st.sidebar.columns(2)
            with cf_col:
                if st.button("Yes", key=f"confirm-delete-file-{file.file_id}", type="primary", use_container_width=True):
                    project = delete_project_file(project.project_id, file.file_id)
                    st.session_state.pop(confirm_key, None)
                    # Let _ensure_selected_file fix stale selected/expanded file IDs
                    st.rerun()
            with cc_col:
                if st.button("No", key=f"cancel-delete-file-{file.file_id}", use_container_width=True):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

        if file.file_id == st.session_state.get("expanded_file_id") and file.status == "ready":
            _render_page_picker(project.project_id, file, load_file_slides(project.project_id, file.file_id))

    return project


def _render_delete_project(project: ProjectManifest) -> None:
    """Two-step confirmation: first click sets flag, second click confirms deletion."""
    confirm_key = "delete_confirm_project_id"

    # Step 1: show the delete button (only if not already confirming for this project)
    if st.session_state.get(confirm_key) != project.project_id:
        if st.sidebar.button("🗑 Delete project", key="delete-project-btn", use_container_width=True):
            st.session_state[confirm_key] = project.project_id
            st.rerun()
        return

    # Step 2: confirmation
    st.sidebar.warning(f"Delete **{project.name}** and all its PDFs? This cannot be undone.")
    col_confirm, col_cancel = st.sidebar.columns(2)
    with col_confirm:
        if st.button("Yes, delete", key="delete-project-confirm", type="primary", use_container_width=True):
            delete_project(project.project_id)
            # Clear all project-related session state so picker shows empty state
            for key in ("selected_project_id", "selected_file_id", "expanded_file_id", "selected_page_number", confirm_key):
                st.session_state.pop(key, None)
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key="delete-project-cancel", use_container_width=True):
            st.session_state.pop(confirm_key, None)
            st.rerun()


def _render_page_picker(project_id: str, file: ProjectFile, slides: list[SlidePage]) -> None:
    if not slides:
        return

    jump_key = _page_jump_key(file.file_id)
    for slide in slides:
        thumb_path = _resolve_file_path(project_id, file.file_id, slide.thumbnail_path)
        if thumb_path and thumb_path.exists():
            st.sidebar.image(str(thumb_path), width=120)
        label = f"Page {slide.page_number} · {_quality_label(slide)}"
        st.sidebar.button(
            label,
            key=f"select-page-{file.file_id}-{slide.page_number}",
            use_container_width=True,
            on_click=_set_selected_page,
            args=(st.session_state, slide.page_number, len(slides), jump_key),
        )


def _page_jump_key(file_id: str) -> str:
    return f"jump-to-page-{file_id}"


def _set_selected_page(
    state, page_number: int, total_pages: int, jump_key: str,
) -> None:
    """Commit navigation in a widget callback so the next run renders every widget."""
    page_number = min(max(int(page_number), 1), max(int(total_pages), 1))
    state["selected_page_number"] = page_number
    state[jump_key] = page_number


def _apply_page_jump(state, jump_key: str, total_pages: int) -> None:
    """Copy the number-input value into canonical page state before rerendering."""
    _set_selected_page(
        state, int(state.get(jump_key, 1)), total_pages, jump_key,
    )


def _render_navigation(file: ProjectFile, total_pages: int) -> None:
    current_page = int(st.session_state.get("selected_page_number", 1))
    current_page = min(max(current_page, 1), max(total_pages, 1))
    st.session_state["selected_page_number"] = current_page
    jump_key = _page_jump_key(file.file_id)
    # Keep programmatic/citation navigation and the number input in one state.
    st.session_state[jump_key] = current_page

    st.caption(f"{file.original_name} · Page {current_page} / {total_pages}")
    prev_col, page_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        st.button(
            "Previous page",
            disabled=current_page <= 1,
            use_container_width=True,
            on_click=_set_selected_page,
            args=(st.session_state, current_page - 1, total_pages, jump_key),
        )
    with page_col:
        st.number_input(
            "Jump to page",
            min_value=1,
            max_value=max(total_pages, 1),
            step=1,
            key=jump_key,
            on_change=_apply_page_jump,
            args=(st.session_state, jump_key, total_pages),
            label_visibility="collapsed",
        )
    with next_col:
        st.button(
            "Next page",
            disabled=current_page >= total_pages,
            use_container_width=True,
            on_click=_set_selected_page,
            args=(st.session_state, current_page + 1, total_pages, jump_key),
        )


_PREVIEW_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _preview_image_data_uri(path: Path) -> str | None:
    """Encode only cache image formats accepted by the contained preview."""
    mime = _PREVIEW_MIME_BY_SUFFIX.get(path.suffix.casefold())
    if mime is None:
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime};base64,{encoded}"


def _preview_state_key(project_id: str, file_id: str) -> str:
    return f"slide-preview-full-width-{project_id}-{file_id}"


def _preview_preference_key(project_id: str, file_id: str) -> str:
    return f"slide-preview-full-width-preference-{project_id}-{file_id}"


def _ensure_preview_preference_state(state, project_id: str, file_id: str) -> tuple[str, str]:
    """Restore a PDF preference even after Streamlit cleans the widget-owned key."""
    widget_key = _preview_state_key(project_id, file_id)
    preference_key = _preview_preference_key(project_id, file_id)
    if preference_key not in state:
        state[preference_key] = bool(state.get(widget_key, False))
    if widget_key not in state:
        state[widget_key] = bool(state[preference_key])
    return widget_key, preference_key


def _persist_preview_preference(
    state, project_id: str, file_id: str, widget_key: str,
) -> None:
    """Copy the toggle into PDF-owned state before another widget can disappear."""
    state[_preview_preference_key(project_id, file_id)] = bool(
        state.get(widget_key, False)
    )


def _render_selected_slide(project_id: str, file: ProjectFile, slide: SlidePage) -> None:
    st.subheader(file.original_name)
    if slide.title_guess:
        st.caption(slide.title_guess)

    image_path = ensure_page_image(project_id, file.file_id, slide)
    if image_path:
        widget_key, preference_key = _ensure_preview_preference_state(
            st.session_state, project_id, file.file_id,
        )
        st.toggle(
            "Full-width preview",
            key=widget_key,
            on_change=_persist_preview_preference,
            args=(st.session_state, project_id, file.file_id, widget_key),
            help="Switch from viewport-fit mode to the original full-width view",
        )
        # Layout follows durable PDF preference, never transient widget hydration.
        full_width = bool(st.session_state[preference_key])
        if full_width:
            st.image(str(image_path), width="stretch")
        else:
            preview = _preview_image_data_uri(Path(image_path))
            if preview is None:
                st.warning("This cached page image format cannot be previewed safely.")
            else:
                alt = html.escape(
                    f"{file.original_name}, page {slide.page_number}", quote=True,
                )
                st.html(f"""
                <style>
                .slide-tutor-preview {{
                    height: min(68vh, 760px);
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    padding: 12px;
                    border: 1px solid #d9dde3;
                    border-radius: 8px;
                    background: #f7f7f8;
                }}
                .slide-tutor-preview img {{
                    display: block;
                    width: 100%;
                    height: 100%;
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                }}
                @media (max-width: 768px) {{
                    .slide-tutor-preview {{
                        height: 55vh;
                        padding: 6px;
                    }}
                }}
                </style>
                <div class="slide-tutor-preview">
                    <img src="{preview}" alt="{alt}">
                </div>
                """)
    else:
        st.warning("Page image is missing from the project cache.")

    if slide.text_quality == "empty":
        st.warning("No extractable text was found on this page. It may be scanned or image-only.")
    elif slide.text_quality == "low":
        st.info("Only a small amount of text was extracted from this page.")

    # Preview should only display cached parser quality; it must not trigger
    # Marker model loading while the user is simply browsing pages.
    understanding = load_page_understanding(project_id, file.file_id, slide.slide_id)
    if understanding is not None:
        quality_label = understanding.quality
        if quality_label == "ok":
            st.caption(f"Page understanding: {understanding.selected_provider} · ok")
        elif quality_label in {"fallback", "degraded", "sparse_valid"}:
            st.info(
                f"Page understanding: {understanding.selected_provider} · {quality_label}. "
                "Answers may use degraded source text."
            )
        elif quality_label == "needs_vlm":
            st.warning("This page is visually complex or sparse; a VLM would improve completeness.")
        else:
            st.warning("Page understanding is empty; AI answers may be incomplete.")

    with st.expander("Extracted text", expanded=False):
        st.text_area(
            "Text",
            value=slide.text or "(No extracted text)",
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )


# ── Phase 4 Advance: Q&A memory v2 helpers ──

_MAX_MEMORY_ROUNDS = 5  # max Q&A rounds (question+answer pair = 1 round)
_MEMORY_STORED_QUESTION_CHAR_LIMIT = 1200
_MEMORY_STORED_ANSWER_CHAR_LIMIT = 8000
_MEMORY_PROMPT_CHAR_BUDGET = 2000
_MEMORY_PROMPT_QUESTION_CHAR_LIMIT = 120
_MEMORY_PROMPT_ANSWER_CHAR_LIMIT = 200
_QA_LANGUAGES = ("简体中文", "English", "日本語")
_QA_LANGUAGE_ALIASES = {"中文": "简体中文"}


def _normalize_qa_language(value) -> str:
    selected = _QA_LANGUAGE_ALIASES.get(value, value)
    return selected if selected in _QA_LANGUAGES else "简体中文"


def _clip_text(value: str, limit: int) -> str:
    """Keep stored memory bounded without cutting ordinary answers short."""
    if len(value) <= limit:
        return value
    return value[:max(0, limit - 3)].rstrip() + "..."


def _build_memory_turn(question: str, answer: str, file: ProjectFile,
                       slide: SlidePage, language: str) -> dict:
    """Create one Q&A memory turn.  Storage keeps near-full text; prompt
    trimming happens later in _format_memory_for_prompt()."""
    return {
        "question": _clip_text(question, _MEMORY_STORED_QUESTION_CHAR_LIMIT),
        "answer": _clip_text(answer, _MEMORY_STORED_ANSWER_CHAR_LIMIT),
        "file_id": file.file_id,
        "page_number": slide.page_number,
        "source_file": file.original_name,
        "language": language,
    }


def _append_to_memory(question: str, answer: str, file: ProjectFile,
                      slide: SlidePage, language: str) -> None:
    """Add one Q&A round to the short-term memory (max _MAX_MEMORY_ROUNDS rounds).
    Each round is a question+answer pair — not separate user/assistant messages."""
    memory = st.session_state.get("qa_memory", [])
    memory.append(_build_memory_turn(question, answer, file, slide, language))
    if len(memory) > _MAX_MEMORY_ROUNDS:
        memory = memory[-_MAX_MEMORY_ROUNDS:]
    st.session_state["qa_memory"] = memory


def _format_memory_for_prompt(memory: list[dict]) -> str:
    """Build a compact memory block for prompt injection from QAMemoryTurn-style
    dicts.  Each round is labelled with its chronological index.  Returns empty
    string if memory is empty."""
    if not memory:
        return "(No previous conversation)"
    lines = [
        "Recent conversation memory (for pronoun resolution — facts come from slide context):"
    ]
    total = 0
    for i, turn in enumerate(memory, start=1):
        q = turn["question"][:_MEMORY_PROMPT_QUESTION_CHAR_LIMIT]
        a = turn["answer"][:_MEMORY_PROMPT_ANSWER_CHAR_LIMIT]
        block = f"[{i}] Q: {q}\n[{i}] A: {a}"
        if total + len(block) > _MEMORY_PROMPT_CHAR_BUDGET:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


def _qa_action_visibility(answer: str | None, memory: list[dict]) -> dict[str, bool]:
    """Decouple Clear memory from the current answer so hidden memory remains
    user-clearable after Clear answer is clicked."""
    return {
        "clear_answer": bool(answer) and "❌" not in answer,
        "clear_memory": bool(memory),
    }


def _clear_qa_output_state(state: dict) -> None:
    """Clear an answer together with the context bundle that produced it."""
    for key in ("qa_last_answer", "qa_last_question", "qa_last_bundle"):
        state.pop(key, None)


def _ensure_qa_language_state(state, project_id: str) -> str:
    """Restore a project language even if Streamlit cleaned the widget key."""
    widget_key = f"qa-target-language-{project_id}"
    project_languages = dict(state.get("qa_target_languages", {}))
    selected = project_languages.get(
        project_id,
        state.get(widget_key, state.get("qa_target_language", "简体中文")),
    )
    selected = _normalize_qa_language(selected)
    if project_languages.get(project_id) != selected:
        project_languages[project_id] = selected
        state["qa_target_languages"] = project_languages
    if widget_key not in state or state[widget_key] not in _QA_LANGUAGES:
        state[widget_key] = selected
    return widget_key


def _persist_qa_language_state(state, project_id: str, widget_key: str) -> None:
    """Copy the widget value into project-owned state before a navigation rerun."""
    selected = _normalize_qa_language(state.get(widget_key, "简体中文"))
    project_languages = dict(state.get("qa_target_languages", {}))
    project_languages[project_id] = selected
    state["qa_target_languages"] = project_languages


def _render_context_bundle(bundle) -> None:
    """Render context metadata belonging to one answer record only."""
    if bundle is None:
        return
    summary = ", ".join(
        f"{kind}: {sum(source.kind == kind for source in bundle.sources)}"
        for kind in sorted({source.kind for source in bundle.sources})
    )
    if not summary:
        summary = "no page sources"
    with st.expander(f"Context used: {summary}"):
        if bundle.planner_mode:
            st.caption(f"Planner: {bundle.planner_mode}")
        if bundle.agent_summary is not None:
            st.caption(_agent_summary_text(bundle.agent_summary))
            for round_trace in bundle.agent_summary.round_traces:
                st.caption(
                    f"Round {round_trace.round_number}: "
                    f"{len(round_trace.tool_traces)} tool call(s) · "
                    f"{round_trace.observation_source_count} source(s)"
                )
        if bundle.tool_traces:
            st.caption("Tools used")
            for trace in bundle.tool_traces:
                status = trace.status
                detail = f" — {trace.summary}" if trace.summary else ""
                if trace.error:
                    detail = f" — {trace.error}"
                st.caption(f"{trace.name} · {status}{detail}")
        for source in bundle.sources:
            location = (
                f"{source.source_file}, page {source.page_number}"
                if source.source_file and source.page_number else "metadata"
            )
            methods = f" · methods={source.retrieval_methods}" if source.retrieval_methods else ""
            st.caption(f"{source.kind}: {location} — {source.reason}{methods}")
            if source.snippet:
                st.markdown(source.snippet[:300])


def _agent_summary_text(summary) -> str:
    """Render only sanitized Agent provenance; never expose prompts/observations."""
    mode = "Agent v2 → v1" if summary.mode == "bounded_agent_v2_fallback" else "Agent v2"
    usage = summary.usage_status
    token_text = ""
    if summary.total_tokens is not None:
        token_text = f" · tokens={summary.total_tokens}"
    return (
        f"{mode} · rounds={summary.rounds} · "
        f"tools={summary.executed_tool_calls} · stop={summary.stop_reason} · "
        f"usage={usage}{token_text}"
    )


def _commit_finished_qa_snapshot(snapshot) -> bool:
    """Finalize a task once; repeated fragment polls are idempotent."""
    transition = finalize_qa_snapshot(
        snapshot,
        st.session_state.setdefault("qa_answer_records", []),
        st.session_state.setdefault("qa_memory", []),
        st.session_state.setdefault("qa_submitted_tasks", set()),
    )
    if not transition.changed:
        return False
    if transition.record is not None:
        # Persist before marking the task submitted so fragment reruns cannot
        # acknowledge a terminal answer that was never added to Project History.
        persist_answer_record(transition.record)
    st.session_state["qa_answer_records"] = transition.records
    st.session_state["qa_memory"] = transition.memory
    st.session_state["qa_submitted_tasks"] = transition.submitted_task_ids
    if st.session_state.get("qa_active_task_id") == snapshot.task_id:
        st.session_state.pop("qa_active_task_id", None)
    qa_job_registry.cleanup(session_id=snapshot.session_id)
    return True


def _store_current_response(snapshot) -> None:
    responses = dict(st.session_state.setdefault("qa_current_responses", {}))
    responses[snapshot.project_id] = current_response_from_snapshot(snapshot)
    st.session_state["qa_current_responses"] = responses


def _render_current_response(project_id: str) -> None:
    response = st.session_state.get("qa_current_responses", {}).get(project_id)
    if response is None:
        return
    st.markdown("#### Current response")
    st.caption(
        f"{response.action_type.title()} · {response.source_file}, "
        f"page {response.page_number} · {response.language} · {response.status}"
    )
    if response.answer_markdown:
        st.markdown(response.answer_markdown)
    elif response.status in {"pending", "running"}:
        st.caption("Waiting for generated text…")
    if response.quality_warning:
        st.warning(response.quality_warning)
    if response.error:
        st.error(response.error)
    _render_context_bundle(response.context_bundle)
    if st.button(
        "Clear current output", key=f"clear-current-output-{project_id}",
        disabled=response.status in {"pending", "running"},
    ):
        responses = dict(st.session_state.get("qa_current_responses", {}))
        responses.pop(project_id, None)
        st.session_state["qa_current_responses"] = responses
        st.rerun()


@st.fragment(run_every=0.5)
def _render_qa_job_fragment(session_id: str, project_id: str) -> None:
    """Poll a worker snapshot and render accumulated Markdown, never a generator."""
    task_id = st.session_state.get("qa_active_task_id")
    if not task_id:
        return
    snapshot = qa_job_registry.snapshot(task_id)
    if snapshot is None:
        return
    _store_current_response(snapshot)
    if snapshot.project_id != project_id and snapshot.status in {"pending", "running"}:
        st.info(
            f"This browser session is busy with a {snapshot.action_type} task for "
            f"project {snapshot.project_id} · {snapshot.source_file}, "
            f"page {snapshot.page_number}. One generation task can run at a time."
        )
        if st.button("Stop generation", key=f"stop-generation-{task_id}"):
            qa_job_registry.cancel(task_id)
        return
    if snapshot.status in {"pending", "running"}:
        st.info(
            f"{snapshot.action_type.title()} · {snapshot.source_file}, page {snapshot.page_number} · "
            f"generating from the captured request"
        )
        _render_current_response(snapshot.project_id)
        if st.button("Stop generation", key=f"stop-generation-{task_id}"):
            qa_job_registry.cancel(task_id)
        return
    if _commit_finished_qa_snapshot(snapshot):
        st.rerun(scope="app")


def _delete_answer_record(record_id: str) -> None:
    records = st.session_state.get("qa_answer_records", [])
    record = next((item for item in records if item.record_id == record_id), None)
    if record is not None:
        delete_project_answer_record(record.project_id, record_id)
    st.session_state["qa_answer_records"] = [
        item for item in records if item.record_id != record_id
    ]
    st.rerun()


def _render_projected_record(record, *, label: str) -> None:
    with st.expander(label, expanded=False):
        if record.question:
            st.markdown(f"**Q:** {record.question}")
        if record.answer_markdown:
            st.markdown(record.answer_markdown)
        if record.error:
            st.error(record.error)
        _render_context_bundle(record.context_bundle)
        if st.button("Delete answer", key=f"delete-answer-{record.record_id}"):
            _delete_answer_record(record.record_id)


def _has_history_or_memory(records, project_id: str, memory: list[dict]) -> bool:
    return bool(memory) or any(record.project_id == project_id for record in records)


def _render_answer_history(
    project_id: str,
    current_file_id: str,
    current_page: int,
    *,
    busy: bool = False,
) -> None:
    records = st.session_state.get("qa_answer_records", [])
    scope_options = ["Current page", "Current PDF"]
    scope_map = {
        "Current page": "current_page",
        "Current PDF": "current_file",
    }
    scope_key = f"qa-history-scope-{project_id}"
    scope_label = st.session_state.get(scope_key, "Current page")
    if scope_label not in scope_options:
        scope_label = "Current page"
        st.session_state[scope_key] = scope_label
    projection = project_answer_records(
        records,
        project_id=project_id,
        scope=scope_map[scope_label],
        current_file_id=current_file_id,
        current_page_number=current_page,
    )
    st.caption(
        f"Answers · {projection.visible_record_count} visible / "
        f"{projection.project_record_count} project records"
    )
    if projection.project_record_count == 0:
        return

    file_count = sum(
        record.project_id == project_id and record.file_id == current_file_id
        for record in records
    )
    toolbar = st.columns([1, 4], vertical_alignment="center")
    with toolbar[0]:
        with st.popover("Manage answers", use_container_width=True):
            if st.button(
                f"Clear current PDF ({file_count})", key=f"clear-file-answers-{project_id}",
                disabled=busy or file_count == 0, use_container_width=True,
            ):
                st.session_state[f"confirm-clear-file-answers-{project_id}"] = True
    with toolbar[1]:
        st.radio(
            "Answer range", scope_options, horizontal=True, key=scope_key,
            label_visibility="collapsed",
        )

    for delete_scope, label, count in (("current_file", "current PDF", file_count),):
        confirm_key = f"confirm-clear-file-answers-{project_id}"
        if not st.session_state.get(confirm_key):
            continue
        st.warning(f"Clear {count} answer record(s) from the {label}? Memory and Current response stay intact.")
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if st.button("Confirm clear", key=f"confirm-{delete_scope}-{project_id}", use_container_width=True):
                updated_records = remove_answer_records(
                    records, project_id=project_id, scope=delete_scope,
                    current_file_id=current_file_id,
                )
                save_project_answer_history(project_id, updated_records)
                st.session_state["qa_answer_records"] = updated_records
                st.session_state.pop(confirm_key, None)
                st.rerun()
        with confirm_cols[1]:
            if st.button("Cancel", key=f"cancel-{delete_scope}-{project_id}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    if not projection.groups:
        st.caption("No saved answers in this range.")
        return

    for group in projection.groups:
        if projection.scope == "current_file":
            st.markdown(f"#### Page {group.page_number}")
        elif projection.scope == "whole_project":
            st.markdown(f"#### {group.source_file} · page {group.page_number}")
        if group.explain is not None:
            _render_projected_record(group.explain, label="Explain · latest")
        for record in group.translations:
            _render_projected_record(record, label=f"Translate · {record.language} · latest")
        if group.asks:
            with st.expander(f"Ask · {len(group.asks)} question(s)", expanded=False):
                for position, record in enumerate(group.asks, start=1):
                    if position > 1:
                        st.divider()
                    st.caption(record.created_at.replace("T", " ")[:19])
                    st.markdown(f"**Q:** {record.question}")
                    st.markdown(record.answer_markdown)
                    _render_context_bundle(record.context_bundle)
                    if st.button("Delete answer", key=f"delete-answer-{record.record_id}"):
                        _delete_answer_record(record.record_id)
        if group.recent_tasks:
            with st.expander(f"Recent task status · {len(group.recent_tasks)}", expanded=False):
                for record in group.recent_tasks:
                    st.caption(
                        f"{record.action_type.title()} · {record.status} · "
                        f"{record.created_at.replace('T', ' ')[:19]}"
                    )
                    if record.answer_markdown:
                        st.markdown(record.answer_markdown)
                    if record.error:
                        st.error(record.error)
                    if st.button("Delete record", key=f"delete-answer-{record.record_id}"):
                        _delete_answer_record(record.record_id)
        if group.previous_versions:
            with st.expander(f"Previous versions · {len(group.previous_versions)}", expanded=False):
                for record in group.previous_versions:
                    st.caption(
                        f"{record.action_type.title()} · {record.language} · "
                        f"{record.created_at.replace('T', ' ')[:19]}"
                    )
                    st.markdown(record.answer_markdown)
                    if st.button("Delete version", key=f"delete-answer-{record.record_id}"):
                        _delete_answer_record(record.record_id)


def _render_project_history(project_id: str, *, busy: bool = False) -> None:
    """Independent durable-history entry, separate from Q&A answer range."""
    records = sorted(
        (
            record for record in st.session_state.get("qa_answer_records", [])
            if record.project_id == project_id
        ),
        key=lambda record: record.created_at,
        reverse=True,
    )
    with st.expander(f"Project History · {len(records)} record(s)", expanded=False):
        if not records:
            st.caption("No persisted answers for this project.")
            return
        action = st.selectbox(
            "Action filter", ["All", "Ask", "Explain", "Translate"],
            key=f"project-history-action-{project_id}",
        )
        visible = records if action == "All" else [
            record for record in records if record.action_type == action.casefold()
        ]
        if st.button(
            "Clear Project History", key=f"clear-project-history-{project_id}",
            disabled=busy, use_container_width=True,
        ):
            st.session_state[f"confirm-project-history-clear-{project_id}"] = True
        if st.session_state.get(f"confirm-project-history-clear-{project_id}"):
            st.warning(
                "Clear all persisted answers for this project? Current response and "
                "conversation memory remain unchanged."
            )
            confirm, cancel = st.columns(2)
            with confirm:
                if st.button("Confirm clear", key=f"confirm-project-history-{project_id}"):
                    clear_project_answer_history(project_id)
                    st.session_state["qa_answer_records"] = [
                        record for record in st.session_state.get("qa_answer_records", [])
                        if record.project_id != project_id
                    ]
                    st.session_state.pop(f"confirm-project-history-clear-{project_id}", None)
                    st.rerun()
            with cancel:
                if st.button("Cancel", key=f"cancel-project-history-{project_id}"):
                    st.session_state.pop(f"confirm-project-history-clear-{project_id}", None)
                    st.rerun()
        for record in visible:
            timestamp = record.created_at.replace("T", " ")[:19]
            label = (
                f"{timestamp} · {record.action_type.title()} · "
                f"{record.source_file}, page {record.page_number}"
            )
            _render_projected_record(record, label=label)


def _render_qa_panel(project_id: str, file: ProjectFile, slide: SlidePage) -> None:
    """Q&A actions create rerun-safe background jobs and independent records."""
    st.subheader("🤖 Q&A")
    st.session_state.setdefault("qa_memory", [])
    st.session_state.setdefault("qa_answer_records", [])
    st.session_state.setdefault("qa_submitted_tasks", set())
    st.session_state.setdefault("qa_current_responses", {})
    st.session_state.setdefault("qa_session_id", f"session-{uuid4().hex}")
    loaded_projects = set(st.session_state.setdefault("qa_history_loaded_projects", set()))
    if project_id not in loaded_projects:
        records = list(st.session_state["qa_answer_records"])
        for record in load_project_answer_history(project_id):
            records = upsert_answer_record(records, record)
        st.session_state["qa_answer_records"] = records
        loaded_projects.add(project_id)
        st.session_state["qa_history_loaded_projects"] = loaded_projects
    if not st.session_state.get("qa_migration_done"):
        legacy = migrate_legacy_answer_state(st.session_state)
        if legacy is not None:
            st.session_state["qa_answer_records"] = upsert_answer_record(
                st.session_state["qa_answer_records"], legacy,
            )
            persist_answer_record(legacy)
        st.session_state["qa_migration_done"] = True

    active = qa_job_registry.active_snapshot(st.session_state["qa_session_id"])
    if active is not None:
        st.session_state["qa_active_task_id"] = active.task_id
    busy = active is not None

    language_key = _ensure_qa_language_state(st.session_state, project_id)
    lang = st.selectbox(
        "Answer language", list(_QA_LANGUAGES), index=0,
        key=language_key,
        disabled=busy,
        on_change=_persist_qa_language_state,
        args=(st.session_state, project_id, language_key),
    )

    def start(action_type: str, question: str = "") -> bool:
        request = QAJobRequest(
            session_id=st.session_state["qa_session_id"],
            project_id=project_id,
            file_id=file.file_id,
            source_file=slide.source_file,
            page_number=slide.page_number,
            action_type=action_type,  # type: ignore[arg-type]
            language=lang,
            question=question,
            current_slide=slide.model_copy(deep=True),
            file_page_count=file.page_count,
            memory=[dict(item) for item in st.session_state.get("qa_memory", [])],
        )
        try:
            snapshot = qa_job_registry.start(request, run_qa_pipeline)
        except RuntimeError as exc:
            st.warning(str(exc))
            return False
        st.session_state["qa_active_task_id"] = snapshot.task_id
        _store_current_response(snapshot)
        return True

    if st.button("Explain this page", type="primary", use_container_width=True,
                 key="explain-page-btn", disabled=busy):
        if start("explain", f"Explain: {slide.source_file}, page {slide.page_number}"):
            st.rerun()
    if st.button("Translate this page", use_container_width=True,
                 key="translate-page-btn", disabled=busy):
        if start("translate", f"Translate: {slide.source_file}, page {slide.page_number} → {lang}"):
            st.rerun()

    question_key = f"qa-question-input-{project_id}"
    if st.session_state.pop(f"qa-clear-question-{project_id}", False):
        st.session_state[question_key] = ""
    with st.form(f"qa-question-form-{project_id}", clear_on_submit=False):
        q_input = st.text_input(
            "Ask a question about the course material",
            placeholder="e.g. What is supervised learning?",
            label_visibility="collapsed", key=question_key, disabled=busy,
        )
        ask_submitted = st.form_submit_button(
            "Ask", use_container_width=True, disabled=busy,
        )
    if ask_submitted and not busy and q_input.strip():
        question = q_input.strip()
        if start("ask", question):
            st.session_state[f"qa-clear-question-{project_id}"] = True
            st.rerun()

    # Keep running and terminal output in one stable location below the controls.
    # The outer container survives the fragment-to-full-rerun transition, so the
    # completed response does not jump from above the controls to below them.
    with st.container(key=f"qa-current-response-slot-{project_id}"):
        if st.session_state.get("qa_active_task_id"):
            _render_qa_job_fragment(st.session_state["qa_session_id"], project_id)
        active = qa_job_registry.active_snapshot(st.session_state["qa_session_id"])
        if active is None or active.project_id != project_id:
            _render_current_response(project_id)

    records = st.session_state.get("qa_answer_records", [])
    memory = st.session_state.get("qa_memory", [])
    if _has_history_or_memory(records, project_id, memory):
        st.divider()
        st.subheader("History & memory")
        _render_answer_history(
            project_id, file.file_id, slide.page_number, busy=busy,
        )
        _render_project_history(project_id, busy=busy)
        if memory:
            with st.expander(f"Conversation memory · {len(memory)} round(s)"):
                if st.button("Clear conversation memory", key="clear-memory", disabled=busy):
                    st.session_state["qa_memory"] = []
                    st.rerun()
                for turn in reversed(memory):
                    st.caption(f"**{turn['source_file']}**, page {turn['page_number']}")
                    st.markdown(f"**Q:** {turn['question']}")
                    st.markdown(turn["answer"])


# ── Phase 3: Quiz sidebar ──


_QUIZ_SESSION_KEYS = (
    "quiz_set",
    "quiz_answers",
    "quiz_submitted",
    "quiz_result",
    "weakness_report",
    "quiz_attempt_id",
    "active_quiz_type",
)


def _clear_quiz_session_state() -> None:
    for key in _QUIZ_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("_return_to_quiz_results", None)
    st.session_state.pop("_pending_browse_jump", None)
    st.session_state.pop("_pending_quiz_return", None)


def _attempt_quiz_type(attempt: dict) -> str:
    quiz_type = attempt.get("quiz_type")
    if quiz_type in ("per_pdf", "final_exam"):
        return quiz_type
    file_ids = attempt.get("file_ids", [])
    return "final_exam" if len(file_ids) > 1 else "per_pdf"


def _attempt_time_label(value: str) -> str:
    if not value:
        return "unknown time"
    return value.replace("T", " ")[:16]


def _group_quiz_attempts(
    project: ProjectManifest, attempts: list[dict],
) -> tuple[list[dict], list[tuple[ProjectFile, list[dict]]], list[dict]]:
    files = ordered_files(project)
    by_file: dict[str, list[dict]] = {file.file_id: [] for file in files}
    final_exams: list[dict] = []
    other: list[dict] = []

    for record in attempts:
        attempt = record.get("attempt", {})
        file_ids = attempt.get("file_ids", [])
        if _attempt_quiz_type(attempt) == "final_exam":
            final_exams.append(record)
        elif file_ids and file_ids[0] in by_file:
            by_file[file_ids[0]].append(record)
        else:
            other.append(record)

    per_pdf = [(file, by_file[file.file_id]) for file in files if by_file[file.file_id]]
    return final_exams, per_pdf, other


def _restore_quiz_attempt(project: ProjectManifest, attempt_id: str) -> bool:
    data = load_quiz_attempt(project.project_id, attempt_id)
    if not data:
        return False

    attempt = data.get("attempt", {})
    snapshot = data.get("quiz_set_snapshot")
    if snapshot:
        st.session_state["quiz_set"] = QuizSet.model_validate(snapshot)
    else:
        qsid = attempt.get("quiz_set_id", "")
        cached = load_cached_quiz(project.project_id, qsid) if qsid else None
        if cached:
            st.session_state["quiz_set"] = cached
        else:
            st.sidebar.warning("Quiz cache unavailable for this attempt.")
            return False

    st.session_state["quiz_answers"] = attempt.get("answers", {})
    st.session_state["quiz_submitted"] = True
    st.session_state["quiz_attempt_id"] = attempt_id
    st.session_state["active_quiz_type"] = _attempt_quiz_type(attempt)

    from slide_tutor.models import QuizResult as QR, WeaknessReport as WR
    result_raw = data.get("result")
    wr_raw = data.get("weakness_report")
    st.session_state["quiz_result"] = QR.model_validate(result_raw) if result_raw else None
    st.session_state["weakness_report"] = WR.model_validate(wr_raw) if wr_raw else None
    return True


def _render_history_record(project: ProjectManifest, record: dict) -> None:
    attempt = record.get("attempt", {})
    result = record.get("result", {})
    attempt_id = attempt.get("attempt_id", "")
    if not attempt_id:
        st.caption("Invalid history record")
        return

    score = result.get("score")
    total = result.get("max_score") or result.get("total", attempt.get("total_count"))
    submitted = _attempt_time_label(str(attempt.get("submitted_at", "")))
    label = f"{_format_points(score)}/{_format_points(total)} · {submitted}"

    cols = st.columns([3, 1])
    with cols[0]:
        if st.button(label, key=f"hist-open-{attempt_id}", use_container_width=True):
            if _restore_quiz_attempt(project, attempt_id):
                st.rerun()
    with cols[1]:
        if st.button("Delete", key=f"hist-delete-{attempt_id}", use_container_width=True):
            st.session_state[f"_confirm_delete_attempt_{attempt_id}"] = True
            st.rerun()

    confirm_key = f"_confirm_delete_attempt_{attempt_id}"
    if st.session_state.get(confirm_key):
        st.warning("Delete this quiz history?")
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if st.button("Yes", key=f"hist-confirm-delete-{attempt_id}", use_container_width=True):
                deleted = delete_quiz_attempt(project.project_id, attempt_id)
                st.session_state.pop(confirm_key, None)
                if deleted and st.session_state.get("quiz_attempt_id") == attempt_id:
                    _clear_quiz_session_state()
                st.rerun()
        with confirm_cols[1]:
            if st.button("No", key=f"hist-cancel-delete-{attempt_id}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()


def _render_quiz_history(project: ProjectManifest) -> None:
    attempts = list_quiz_attempts(project.project_id, limit=100)
    if not attempts:
        return

    final_exams, per_pdf, other = _group_quiz_attempts(project, attempts)
    st.sidebar.divider()
    st.sidebar.caption("History")

    if final_exams:
        with st.sidebar.expander(f"Final exams ({len(final_exams)})", expanded=False):
            for record in final_exams:
                _render_history_record(project, record)

    for file, records in per_pdf:
        with st.sidebar.expander(f"{file.original_name} ({len(records)})", expanded=False):
            for record in records:
                _render_history_record(project, record)

    if other:
        with st.sidebar.expander(f"Other quizzes ({len(other)})", expanded=False):
            for record in other:
                _render_history_record(project, record)


@st.fragment(run_every=0.5)
def _render_quiz_generation_fragment(session_id: str) -> None:
    task_id = st.session_state.get("quiz_generation_task_id")
    if not task_id:
        return
    snapshot = quiz_generation_registry.snapshot(task_id)
    if snapshot is None:
        return
    label = "Final exam" if snapshot.quiz_type == "final_exam" else "Quiz"
    if snapshot.status in {"pending", "running"}:
        st.info(
            f"{label} generation · {snapshot.language} · "
            f"{snapshot.target_count} questions · {snapshot.phase}"
        )
        if snapshot.phase_detail:
            st.caption(snapshot.phase_detail)
        if snapshot.progress_total:
            st.progress(
                min(snapshot.progress_current / snapshot.progress_total, 1.0),
                text=f"Batch {snapshot.progress_current}/{snapshot.progress_total}",
            )
        if st.button("Stop generation", key=f"quiz-stop-{snapshot.task_id}", use_container_width=True):
            quiz_generation_registry.cancel(snapshot.task_id)
            st.rerun(scope="app")
        return
    terminal = quiz_generation_registry.claim_terminal(task_id)
    if terminal is None:
        return
    st.session_state.pop("quiz_generation_task_id", None)
    if terminal.status == "completed" and terminal.result is not None:
        _clear_quiz_session_state()
        st.session_state["quiz_set"] = terminal.result
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["active_quiz_type"] = terminal.quiz_type
        st.session_state.pop("quiz_generation_error", None)
    elif terminal.status == "cancelled":
        st.session_state["quiz_generation_error"] = "Quiz generation cancelled. No cache was activated."
    else:
        st.session_state["quiz_generation_error"] = terminal.error or "Quiz generation failed"
    st.rerun(scope="app")


def _render_quiz_sidebar(project: ProjectManifest) -> None:
    """Quiz-mode sidebar: type selection, PDF picker, question count, Generate."""
    st.sidebar.subheader("Quiz")
    from slide_tutor.generation import generation_state
    from slide_tutor.index_store import compute_source_signature

    generation = generation_state(
        project.project_id, compute_source_signature(project.project_id),
    )
    quality_notice = _quiz_generation_quality_notice(generation.lifecycle_status)
    if quality_notice:
        st.sidebar.warning(quality_notice)
    st.session_state.setdefault("quiz_generation_session_id", f"quiz-session-{uuid4().hex}")
    session_id = st.session_state["quiz_generation_session_id"]
    with st.sidebar:
        _render_quiz_generation_fragment(session_id)
    active = quiz_generation_registry.active_snapshot(session_id)
    busy = active is not None

    error = st.session_state.get("quiz_generation_error")
    if error:
        st.sidebar.error(error)

    quiz_language_key = f"quiz-language-{project.project_id}"
    if st.session_state.get(quiz_language_key) == "中文":
        st.session_state[quiz_language_key] = "简体中文"
    quiz_language = st.sidebar.selectbox(
        "Quiz language", list(_QA_LANGUAGES), index=0,
        key=quiz_language_key, disabled=busy,
    )

    quiz_type = st.sidebar.radio(
        "Quiz type",
        ["Per-PDF quiz", "Final exam"],
        key=f"quiz-type-radio-{project.project_id}", disabled=busy,
    )
    is_final = quiz_type == "Final exam"

    quality_mode = "fast"
    if is_final:
        quality_choice = st.sidebar.radio(
            "Final exam quality",
            ["Fast", "Reviewed"],
            key=f"quiz-quality-radio-{project.project_id}",
            disabled=busy,
            help="Reviewed uses a bounded blueprint, evidence binding, deterministic validation, and one semantic review/repair pass.",
        )
        quality_mode = "reviewed" if quality_choice == "Reviewed" else "fast"
        if quality_mode == "reviewed":
            st.sidebar.info(
                "Reviewed binds every mixed-format question to evidence and may take longer/cost more."
            )

    # File picker (only for Per-PDF)
    quiz_file_id: str | None = None
    if not is_final:
        ready_files = _ordered_ready_files(project)
        if not ready_files:
            st.sidebar.info("No PDFs ready for quiz generation.")
            return
        file_labels = {f.file_id: f"{f.original_name} ({f.page_count}p)" for f in ready_files}
        quiz_file_id = st.sidebar.selectbox(
            "Select PDF",
            list(file_labels.keys()),
            format_func=lambda fid: file_labels[fid],
            key=f"quiz-file-select-{project.project_id}", disabled=busy,
        )
    else:
        quiz_file_id = None

    # Question count slider
    if is_final:
        min_q, max_q, default_q = 10, 40, 20
    else:
        min_q, max_q, default_q = 3, 15, 6
    target_count = st.sidebar.slider(
        "Number of questions", min_q, max_q, default_q, step=1,
        key=f"quiz-target-slider-{project.project_id}-{is_final}", disabled=busy,
    )
    question_type_mix = build_question_type_mix(target_count)
    question_type_mix_text = format_question_type_mix(question_type_mix)
    st.sidebar.caption(
        "Question mix: "
        + " · ".join(
            f"{_QUESTION_TYPE_DISPLAY[qtype]} {question_type_mix[qtype]}"
            for qtype in _QUESTION_TYPE_DISPLAY
        )
    )

    # Store choices in session_state so the main area can use them
    st.session_state["quiz_type"] = "final_exam" if is_final else "per_pdf"
    st.session_state["quiz_file_id"] = quiz_file_id
    st.session_state["quiz_target_count"] = target_count
    st.session_state["quiz_question_type_mix"] = question_type_mix_text

    # Generate button
    gen_label = "Generate quiz" if not is_final else "Generate final exam"
    if st.sidebar.button(
        gen_label, type="primary", use_container_width=True,
        key=f"quiz-gen-btn-{project.project_id}", disabled=busy,
    ):
        request = QuizGenerationRequest(
            session_id=session_id,
            project_id=project.project_id,
            quiz_type="final_exam" if is_final else "per_pdf",
            file_id=quiz_file_id,
            target_count=target_count,
            language=quiz_language,
            question_type_mix=question_type_mix_text,
            quality_mode=quality_mode,
            generation_snapshot=active_generation_snapshot(project.project_id),
            blueprint_schema_version="planned-v2" if quality_mode == "reviewed" else "",
            planner_prompt_version=PLANNER_PROMPT_VERSION if quality_mode == "reviewed" else "",
            review_prompt_version=REVIEW_PROMPT_VERSION if quality_mode == "reviewed" else "",
        )
        try:
            snapshot = quiz_generation_registry.start(request)
        except RuntimeError as exc:
            st.session_state["quiz_generation_error"] = str(exc)
        else:
            st.session_state["quiz_generation_task_id"] = snapshot.task_id
            st.session_state.pop("quiz_generation_error", None)
        st.rerun()

    _render_quiz_history(project)


# ── Phase 3: Quiz main area ──


@st.fragment(run_every=0.5)
def _render_quiz_main_generation_fragment(session_id: str) -> None:
    active = quiz_generation_registry.active_snapshot(session_id)
    if active is None:
        return
    label = "Reviewed Final Exam" if active.quality_mode == "reviewed" else (
        "Fast Final Exam" if active.quiz_type == "final_exam" else "Fast Quiz"
    )
    elapsed = 0
    if active.started_at:
        try:
            elapsed = max(0, int((datetime.now(UTC) - datetime.fromisoformat(
                active.started_at.replace("Z", "+00:00"),
            )).total_seconds()))
        except ValueError:
            elapsed = 0
    st.info(f"{label} is generating · {active.phase} · {elapsed}s elapsed")
    if active.phase_detail:
        st.caption(active.phase_detail)
    if active.progress_total:
        st.progress(
            min(active.progress_current / active.progress_total, 1.0),
            text=f"Batch {active.progress_current}/{active.progress_total}",
        )
    st.caption("Stop takes effect after the current Provider batch finishes.")
    if st.button(
        "Stop after current batch",
        key=f"quiz-main-stop-{active.task_id}",
        type="secondary",
    ):
        quiz_generation_registry.cancel(active.task_id)
        st.rerun(scope="app")


def _render_quiz_main(project: ProjectManifest) -> None:
    """Quiz-mode main content: form (before submit) or results (after submit)."""
    session_id = st.session_state.get("quiz_generation_session_id", "")
    active = quiz_generation_registry.active_snapshot(session_id) if session_id else None
    if active is not None:
        _render_quiz_main_generation_fragment(session_id)
        return
    quiz_set: QuizSet | None = st.session_state.get("quiz_set")
    if quiz_set is None:
        st.info("Select quiz options in the sidebar and click Generate.")
        return

    if quiz_set.quality_mode == "reviewed" and quiz_set.reviewed_metadata is not None:
        metadata = quiz_set.reviewed_metadata
        st.success("Reviewed Final Exam generated with bound evidence.")
        with st.expander("Reviewed exam audit summary"):
            st.write(
                f"Generation: {metadata.active_layer} · page {metadata.page_generation_id or 'legacy'} "
                f"· index {metadata.index_generation_id or 'legacy'}"
            )
            st.write(f"Validation: {metadata.validation_summary}")
            st.write(f"Review: {metadata.review_findings_summary}")
            st.write(f"Repair: {metadata.repair_summary}")

    submitted = st.session_state.get("quiz_submitted", False)

    if not submitted:
        _render_quiz_form(quiz_set)
    else:
        _render_quiz_results(quiz_set, project)


def _render_quiz_form(quiz_set: QuizSet) -> None:
    """Render the quiz as a form.  BEFORE submission: no citations, no answers."""
    st.subheader(
        f"Quiz · {quiz_set.total_count} questions · {quiz_set.difficulty_mix}"
    )
    st.caption(f"Question mix: {quiz_set.question_type_mix}")
    # Streamlit can retain a pre-schema QuizSet instance in session_state across
    # hot reloads. Disk-loaded legacy JSON is revalidated, but the in-memory
    # object needs a defensive compatibility read until the next app restart.
    reuse_summary = getattr(quiz_set, "reuse_summary", None)
    if reuse_summary is not None and reuse_summary.reused_count:
        st.caption(
            f"Reused {reuse_summary.reused_count} previous wrong question(s) · "
            f"generated {reuse_summary.generated_count} new question(s)."
        )
    answers: dict[str, str | None] = st.session_state.get("quiz_answers", {})

    with st.form("quiz-form"):
        for i, q in enumerate(quiz_set.questions, start=1):
            qtype = getattr(q, "question_type", "mcq") or "mcq"
            qtype_label = {"mcq": "", "true_false": "[T/F]", "fill_blank": "[Fill]", "short_answer": "[Short]"}.get(qtype, "")
            label = normalize_quiz_markdown(
                f"**Q{i}** {qtype_label} [{q.difficulty}] {q.question}"
            )
            current_val = answers.get(q.question_id)

            if qtype in ("mcq", "true_false"):
                options_keys = list(q.options.keys())
                idx = options_keys.index(current_val) if current_val in options_keys else None
                choice = st.radio(
                    label, options_keys, index=idx,
                    format_func=lambda k, opts=q.options: normalize_quiz_markdown(
                        f"{k}: {opts.get(k, '')}"
                    ),
                    key=f"quiz-q-{q.question_id}",
                )
                answers[q.question_id] = choice
            elif qtype == "fill_blank":
                choice = st.text_input(
                    label, value=current_val or "",
                    placeholder="Type your answer...",
                    key=f"quiz-q-{q.question_id}",
                )
                answers[q.question_id] = choice.strip() if choice else None
            elif qtype == "short_answer":
                choice = st.text_area(
                    label, value=current_val or "",
                    placeholder="Write your answer...",
                    height=120,
                    key=f"quiz-q-{q.question_id}",
                )
                answers[q.question_id] = choice.strip() if choice else None
            else:
                # Legacy MCQ fallback
                options_keys = list(q.options.keys()) if q.options else ["A", "B", "C", "D"]
                idx = options_keys.index(current_val) if current_val in options_keys else None
                choice = st.radio(label, options_keys, index=idx, key=f"quiz-q-{q.question_id}")
                answers[q.question_id] = choice

            st.caption(f"_{q.topic_tag}_")
            st.divider()

        if st.form_submit_button("Submit answers", type="primary", use_container_width=True):
            st.session_state["quiz_answers"] = answers
            try:
                outcome = submit_quiz_answers(
                    quiz_set,
                    answers,
                    language=quiz_set.language,
                    quiz_type=st.session_state.get("active_quiz_type"),
                )
            except Exception as exc:
                # AI grader failures leave the form unsubmitted and therefore
                # retryable; the service has not persisted a partial attempt.
                st.error(f"Answer grading failed; please retry: {exc}")
                return
            st.session_state["quiz_result"] = outcome.result
            st.session_state["weakness_report"] = outcome.weakness_report
            st.session_state["quiz_submitted"] = True
            st.session_state["quiz_attempt_id"] = outcome.attempt.attempt_id
            st.rerun()


def _correct_answer_label(question_type: str) -> str:
    if question_type == "fill_blank":
        return "Expected answer"
    if question_type == "short_answer":
        return "Model answer"
    return "Correct option"


def _format_points(value: float | int | None) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _render_quiz_results(quiz_set: QuizSet, project: ProjectManifest) -> None:
    """Render quiz results AFTER submission: score, Weakness Report, per-question
    feedback with citations and jump-to-page links.

    session_state holds Pydantic objects (QuizResult, WeaknessReport), not dicts.
    Access fields with attribute notation, not .get()."""
    result = st.session_state.get("quiz_result")
    wr = st.session_state.get("weakness_report")
    if result is None:
        return

    score = result.score
    total = result.total
    max_score = getattr(result, "max_score", None)
    details: list[QuizResultDetail] = result.details or []

    # Score header
    if max_score is not None and max_score > 0:
        pct = round(100 * score / max_score)
        st.subheader(f"Results: {_format_points(score)} / {_format_points(max_score)} ({pct}%)")
    else:
        pct = round(100 * score / max(total, 1))
        st.subheader(f"Results: {int(score)} / {total} ({pct}%)")

    # Weakness Report
    if wr is not None:
        gen_label = "AI diagnosis" if wr.generated_by == "llm" else "By topic"
        report_title = "Mastery Summary" if getattr(wr, "report_kind", "weakness") == "mastery" else "Weakness Report"
        support_label = " · support reviewed" if getattr(wr, "support_reviewed", False) else ""
        st.caption(f"{report_title} ({gen_label}{support_label})")
        for weakness_index, w in enumerate(wr.weaknesses):
            with st.expander(f"⚠️ {w.topic} — {w.error_count} error(s)"):
                st.write(w.diagnosis)
                for citation_index, rp in enumerate(w.review_pages):
                    _render_citation_review(
                        rp, project, f"weak-{weakness_index}-{citation_index}"
                    )
        for s in wr.strengths:
            with st.expander(f"✅ {s.topic} — all correct"):
                st.write(s.diagnosis)

    st.divider()

    # Per-question feedback
    questions = question_by_id(quiz_set.questions)
    for i, d in enumerate(details, start=1):
        icon = "✅" if d.is_correct else ("⏳" if d.points_awarded is None else "❌")
        question_preview = d.question[:80] + "..." if len(d.question) > 80 else d.question
        qtype = getattr(d, "question_type", "mcq") or "mcq"
        type_label = _QUESTION_TYPE_DISPLAY.get(qtype, qtype)
        with st.expander(f"{icon} Q{i} [{type_label}]: {question_preview}"):
            question = questions.get(d.question_id)
            render_quiz_markdown(
                st, f"**Your answer:** {format_quiz_answer(question, d.user_answer)}"
            )
            if d.correct_option:
                correct = format_quiz_answer(question, d.correct_option)
                render_quiz_markdown(st, f"**{_correct_answer_label(qtype)}:** {correct}")
            # Quiz v3: show partial credit where grading provides it.
            if d.points_awarded is not None:
                st.caption(f"Score: {_format_points(d.points_awarded)}/{_format_points(d.max_points)}")
            if d.grading_feedback:
                st.info(f"📝 {d.grading_feedback}")
            render_quiz_markdown(st, d.explanation)
            if d.wrong_reason:
                st.info(f"💡 {d.wrong_reason}")
            # Show citations as clickable links AFTER submission
            for citation_index, c in enumerate(d.citations):
                _render_citation_review(
                    c, project, f"q-{i}-{d.question_id}-{citation_index}"
                )

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retake same quiz", use_container_width=True):
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.session_state.pop("quiz_result", None)
            st.session_state.pop("weakness_report", None)
            st.session_state.pop("quiz_attempt_id", None)
            st.rerun()
    with col2:
        if st.button("New quiz", use_container_width=True):
            _clear_quiz_session_state()
            st.rerun()

    # Export as Markdown (Phase 4)
    from slide_tutor.quiz_export import export_quiz_report_markdown
    report_md = export_quiz_report_markdown(quiz_set, result, wr)
    safe_name = _safe_export_filename(quiz_set)
    st.download_button("Export as Markdown", report_md, file_name=safe_name,
                       mime="text/markdown", use_container_width=True, key="export-quiz-md")


def _safe_export_filename(quiz_set: QuizSet) -> str:
    """Build a filesystem-safe Markdown filename from quiz metadata."""
    import re
    label = quiz_set.file_ids[0][:8] if quiz_set.file_ids else "quiz"
    total = quiz_set.total_count
    date_str = quiz_set.created_at[:10] if quiz_set.created_at else ""
    raw = f"quiz_report_{label}_{total}q_{date_str}.md"
    return re.sub(r"[/\\:*?\"<>|]", "_", raw)[:120]


def _render_cited_page_preview(project: ProjectManifest, file: ProjectFile,
                               page_num: int) -> None:
    """Render an inline preview for a cited page."""
    slides = load_file_slides(project.project_id, file.file_id)
    slide = slides[page_num - 1] if 1 <= page_num <= len(slides) else None

    with st.expander(f"Preview page {page_num}", expanded=False):
        if slide is None:
            st.warning("The cited page is outside the available page range.")
        else:
            image_path = ensure_page_image(project.project_id, file.file_id, slide)
            if image_path:
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Page image is missing from the project cache.")


def _render_citation_review(citation, project: ProjectManifest, key_suffix: str) -> None:
    """Render an inline cited-page preview plus a stable Browse jump action."""
    source_file = citation.source_file
    page_num = citation.page_number
    st.caption(f"📖 {source_file}, page {page_num}")

    file = _find_cited_file(project, source_file)
    if file is None:
        st.warning("The cited PDF is no longer available in this project.")
        return

    _render_cited_page_preview(project, file, page_num)

    if st.button(
        "Open in Browse",
        key=f"open-browse-{key_suffix}-{file.file_id}-{page_num}",
        use_container_width=True,
    ):
        st.session_state["_pending_browse_jump"] = {
            "project_id": project.project_id,
            "file_id": file.file_id,
            "page_number": page_num,
            "return_to": "quiz",
        }
        st.rerun()


def _new_attempt_id() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("attempt-%Y%m%d%H%M%S%f")


def _format_pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _render_dashboard(project: ProjectManifest) -> None:
    """Phase 4: Learning Dashboard with score trend, recent weaknesses, and
    due-for-review recommendations."""
    from slide_tutor.dashboard import build_dashboard_data

    dashboard_language = st.selectbox(
        "Dashboard language",
        list(_QA_LANGUAGES),
        key=f"dashboard-language-{project.project_id}",
    )
    data = build_dashboard_data(project.project_id, language=dashboard_language)
    if data["total_attempts"] == 0:
        st.info("Complete at least one quiz to see your dashboard.")
        return

    per_lesson = data.get("per_lesson", [])
    final_tests = data.get("final_tests", [])
    needs_attention = data.get("needs_attention", [])

    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lessons quizzed", len(per_lesson))
    c2.metric("Final tests", len(final_tests))
    c3.metric("Needs attention", len(needs_attention))

    st.subheader("Per-Lesson Quizzes")
    if per_lesson:
        lesson_rows = []
        for item in per_lesson:
            confidence = "low" if item["confidence"] == "low" else "normal"
            lesson_rows.append({
                "Lesson": item["lesson_name"],
                "Attempts": item["attempt_count"],
                "Latest": _format_pct(item["latest_score_pct"]),
                "Average": _format_pct(item["average_score_pct"]),
                "Lowest": _format_pct(item["lowest_score_pct"]),
                "Weak topics": ", ".join(item["weak_topics"]) or "—",
                "Confidence": confidence,
            })
        st.dataframe(lesson_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No per-lesson quiz attempts yet.")

    st.subheader("Needs Attention")
    if needs_attention:
        for li, item in enumerate(needs_attention[:5]):
            label = (
                f"{item['lesson_name']} — avg {_format_pct(item['average_score_pct'])}, "
                f"latest {_format_pct(item['latest_score_pct'])}, priority {item['priority']}"
            )
            with st.expander(label):
                if item["confidence"] == "low":
                    st.info("Low confidence: only one quiz attempt for this lesson.")
                st.write(
                    f"Attempts: {item['attempt_count']}, "
                    f"Lowest: {_format_pct(item['lowest_score_pct'])}, "
                    f"Wrong answers: {item['wrong_count']}"
                )
                if item["weak_topics"]:
                    st.caption(f"Weak topics: {', '.join(item['weak_topics'])}")
                for ci, c in enumerate(item.get("citations", [])):
                    st.caption(f"📖 {c.get('source_file', '')}, page {c.get('page_number', '')}")
                    _render_review_link_from_citation(c, project, f"attention-{li}-{ci}")
    else:
        st.caption("No lesson-level weak spots yet.")

    st.subheader("Final Tests")
    if final_tests:
        trend_data = {
            "Final test": list(range(1, len(final_tests) + 1)),
            "Score %": [round(100 * item["score_pct"]) for item in final_tests],
        }
        st.line_chart(trend_data, x="Final test", y="Score %")
        latest = final_tests[-1]
        st.caption(
            f"Latest final test: {_format_pct(latest['score_pct'])} "
            f"({latest['score']}/{latest['total']})"
        )
        if latest.get("weak_topics"):
            st.caption(f"Latest weak topics: {', '.join(latest['weak_topics'])}")
    else:
        st.caption("No final tests yet.")

    # Recent weaknesses (4.4)
    st.subheader("Recent Weaknesses")
    weaknesses = data["weaknesses"]
    if weaknesses:
        for wi, w in enumerate(weaknesses[:5]):
            with st.expander(f"⚠️ {w['topic']} — {w['wrong']} error(s), priority {w['priority']}"):
                st.write(f"Correct: {w['correct']}, Wrong: {w['wrong']}")
                if w.get("source_lessons"):
                    st.caption(f"Sources: {', '.join(w['source_lessons'])}")
                if w.get("citations"):
                    for ci, c in enumerate(w["citations"]):
                        st.caption(f"📖 {c.get('source_file', '')}, page {c.get('page_number', '')}")
                        _render_review_link_from_citation(c, project, f"weak-{wi}-{ci}")
    else:
        st.caption("No weaknesses — great job!")

    # Review priority is a ranking heuristic, not a next-review scheduler.
    st.subheader("Review Priority")
    due = data["due"]
    if due:
        for di, d in enumerate(due[:5]):
            with st.expander(f"🔄 {d['topic']} — {d['days_gap']}d ago, priority {d['priority']}"):
                st.write(f"Error rate: {d['error_rate']}, Wrong: {d['wrong']}, Correct: {d['correct']}")
                if d.get("source_lessons"):
                    st.caption(f"Sources: {', '.join(d['source_lessons'])}")
                if d.get("citations"):
                    for ci, c in enumerate(d["citations"]):
                        st.caption(f"📖 {c.get('source_file', '')}, page {c.get('page_number', '')}")
                        _render_review_link_from_citation(c, project, f"due-{di}-{ci}")
    else:
        st.caption("No review priorities yet.")

    # Mastered
    st.subheader("Mastered")
    mastered = data["mastered"]
    if mastered:
        for m in mastered:
            st.caption(f"✅ {m['topic']} — {m['correct']} correct")


def _render_review_link_from_citation(citation: dict, project: ProjectManifest,
                                       key_suffix: str = "") -> None:
    """Minimal jump-to-page link for dashboard citations (dict-based).
    key_suffix disambiguates buttons when the same page is cited in multiple rows."""
    source_file = citation.get("source_file", "")
    try:
        page_num = int(citation.get("page_number", 0))
    except (TypeError, ValueError):
        page_num = 0

    file = _find_cited_file(project, source_file)
    if file is None:
        st.warning("The cited PDF is no longer available in this project.")
        return

    _render_cited_page_preview(project, file, page_num)

    if st.button(f"↗ {source_file}, p.{page_num}",
                 key=f"dash-jump-{key_suffix}-{file.file_id}-{page_num}"):
        # Use deferred navigation — same pattern as quiz jump-to-page.
        # Setting app_mode directly here races with the radio widget.
        st.session_state["_pending_browse_jump"] = {
            "project_id": project.project_id,
            "file_id": file.file_id,
            "page_number": page_num,
            "return_to": "dashboard",
        }
        st.rerun()


def render_app() -> None:
    st.set_page_config(page_title=config.APP_NAME, layout="wide")
    _render_app_shell()
    _consume_pending_browse_jump()
    _consume_pending_quiz_return()
    _consume_pending_dashboard_return()

    # ── App mode toggle ──
    # Pending quiz review jumps must update the radio key before this widget
    # exists; otherwise Streamlit can preserve the previous "Take quiz" value.
    mode = st.session_state.get("app_mode", "browse")
    mode_labels = ["Browse slides", "Take quiz", "Dashboard"]
    mode_map = {"Browse slides": "browse", "Take quiz": "quiz", "Dashboard": "dashboard"}
    idx_map = {"browse": 0, "quiz": 1, "dashboard": 2}
    new_mode = st.sidebar.radio(
        "Mode", mode_labels,
        index=idx_map.get(mode, 0),
        key="app_mode_radio",
    )
    mode = mode_map.get(new_mode, "browse")
    st.session_state["app_mode"] = mode

    if mode == "dashboard":
        st.header("Learning Dashboard", anchor="learning-dashboard")
        project = _render_project_picker()
        if project is None:
            return
        _render_dashboard(project)
        return

    if mode == "quiz":
        st.session_state.pop("_return_to_quiz_results", None)
        st.header("Quiz", anchor="quiz")
        project = _render_project_picker()
        if project is None:
            return
        _render_quiz_sidebar(project)
        _render_quiz_main(project)
        return

    # ── Browse mode (Phase 1 + Phase 2 Advance) ──
    st.caption(
        "Browse course slides, ask grounded questions, and review explanations or translations."
    )
    if (
        st.session_state.get("_return_to_quiz_results")
        and st.session_state.get("quiz_result") is not None
    ):
        if st.button("Back to quiz results", use_container_width=True):
            st.session_state["_pending_quiz_return"] = True
            st.rerun()
    if st.session_state.get("_return_to_dashboard"):
        if st.button("Back to dashboard", use_container_width=True):
            st.session_state["_pending_dashboard_return"] = True
            st.rerun()

    project = _render_project_picker()
    if project is None:
        return

    project = _render_pdf_upload(project)
    _ensure_selected_file(project)
    project = _render_pdf_sorter(project)
    _ensure_selected_file(project)

    file = _selected_file(project)
    if file is None:
        st.info("Upload PDFs to this project, then select one to browse.")
        return

    slides = load_file_slides(project.project_id, file.file_id)
    if not slides:
        st.warning("No pages were found for the selected PDF.")
        return

    slide = _selected_slide(slides)
    if slide is None:
        st.warning("No page is selected.")
        return

    # Commit the complete identity before any page-scoped projection renders.
    st.session_state["selected_page_identity"] = (
        project.project_id, file.file_id, slide.page_number,
    )

    _render_navigation(file, len(slides))
    _render_selected_slide(project.project_id, file, slide)

    # ── Phase 2: Q&A panel ──
    st.divider()
    _render_qa_panel(project.project_id, file, slide)
