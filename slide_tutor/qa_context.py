"""Bounded context planner for Phase 4 Advance Stage A.

Decides which context sources (current page, explicit pages, neighbors,
semantic search, memory) to load before the answer model runs.  Uses a
lightweight LLM call with JSON mode + Pydantic validation; falls back to a
deterministic plan on any failure so Q&A is never blocked.

Also provides build_qa_context_bundle() to execute the plan — loading pages,
running semantic search, and assembling a sectioned prompt context with
source metadata for the UI."""

from __future__ import annotations

import logging
from collections.abc import Callable

from slide_tutor import config
from slide_tutor.llm import complete_json, complete_tool_calls
from slide_tutor.models import (
    QAContextBundle,
    QAContextPlanningResult,
    QAContextPlan,
    QAContextSource,
    QAMemoryTurn,
    LLMCallMetrics,
    RetrievalChunk,
    RetrievalScope,
    SlidePage,
    ToolExecutionContext,
)
from slide_tutor.prompts import PROMPT_PLAN_QA_CONTEXT
from slide_tutor.qa_tools import (
    QA_TOOL_REGISTRY,
    tool_results_to_qa_context_plan,
)
from slide_tutor.tool_runtime import (
    build_openai_tool_schemas,
    execute_tool_calls,
)

logger = logging.getLogger(__name__)

_ALLOWED_PAGE_OFFSETS = {-3, -2, -1, 1, 2, 3}


def _default_plan(question: str) -> QAContextPlan:
    """Deterministic fallback: current page + neighbours + semantic search +
    recent memory.  Mirrors the pre-planner retrieval behaviour so the user
    experience is never worse than before the planner existed."""
    return QAContextPlan(
        include_current_page=True,
        page_numbers=[],
        page_offsets=[-1, 1],
        semantic_query=question,
        use_memory=True,
        ambiguity_note=None,
    )


def _dedupe_ints(values: list[int]) -> list[int]:
    """Preserve planner order while removing duplicate integer tool args."""
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _sanitize_plan(plan: QAContextPlan, current_slide: SlidePage, max_page: int) -> QAContextPlan:
    """Enforce dynamic tool bounds that the static Pydantic schema cannot know."""
    safe_max_page = max(max_page, 1)
    page_numbers = [
        page for page in _dedupe_ints(plan.page_numbers)
        if 1 <= page <= safe_max_page
    ]
    page_offsets = [
        offset for offset in _dedupe_ints(plan.page_offsets)
        if (
            offset in _ALLOWED_PAGE_OFFSETS
            and 1 <= current_slide.page_number + offset <= safe_max_page
        )
    ]
    return plan.model_copy(update={
        "page_numbers": page_numbers,
        "page_offsets": page_offsets,
    })


def _format_memory_summary(memory: list[dict]) -> str:
    """Summarize recent questions for the planner without trusting old session shape."""
    questions: list[str] = []
    for turn in memory[-5:]:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("question") or "(missing question)")[:150]
        questions.append(q)
    if not questions:
        return "(no recent conversation)"

    lines = ["  (oldest → newest)"]
    for i, question in enumerate(questions, start=1):
        markers: list[str] = []
        if i == 1:
            markers.append("oldest")
        if i == len(questions):
            markers.append("MOST RECENT")
        suffix = f" — {', '.join(markers)}" if markers else ""
        lines.append(f"  [{i}{suffix}] {question}")
    return "\n".join(lines)


def _planner_prompts(
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    *,
    native_tools: bool = False,
) -> tuple[str, str]:
    """Build shared planner inputs for JSON and native tool-call paths."""
    memory_summary = _format_memory_summary(memory)
    page_snippet = (current_slide.text or "(no extractable text)")[:300]
    if native_tools:
        return (
            "You are a bounded context planner. Use only the supplied Q&A tools "
            "to select context for the student's question. Do not answer the "
            "question and do not return JSON or prose; your response must be one "
            "or more tool calls. The application supplies the current PDF scope "
            "and current page, so never invent file IDs, project IDs, or paths.",
            f"""Current page:
  File: {current_slide.source_file}
  Page: {current_slide.page_number} / {max(max_page, 1)}
  Text snippet: {page_snippet}

Recent Q&A memory:
{memory_summary}

Student question: {question}

Choose the smallest useful set of supplied tools. Use load_pages for explicit
page requests, load_neighbor_pages for previous/next-page requests, and
search_slides for topic retrieval. The current page and memory policy are
enforced by the application.""",
        )
    user_prompt = PROMPT_PLAN_QA_CONTEXT.format(
        question=question,
        current_file=current_slide.source_file,
        current_page=current_slide.page_number,
        max_page=max(max_page, 1),
        page_snippet=page_snippet,
        memory_summary=memory_summary,
    )
    return (
        "You are a bounded context planner. Choose only the supplied context "
        "tools and do not answer the student's question. Keep the selection minimal.",
        user_prompt,
    )


# ── Positional constraint enforcement (A4.4) ──

# Regex patterns for explicit positional/directional mentions in student questions.
# These are deterministic — no LLM guessing needed.
_POSITIONAL_PATTERNS: list[tuple[str, str, int | None]] = [
    # (regex, action_type, value)
    # action_type: "offset" → page_offsets, "page" → page_numbers
    # Chinese
    (r"上一页|前[一1]页|前[一1]张|上[一1]张", "offset", -1),
    (r"下一页|下[一1]页|下[一1]张|后[一1]页|后[一1]张", "offset", 1),
    (r"本页|当前页|这(?:[一1])?页|这(?:[一1])?张|当前这张", "current", None),
    (r"第\s*(\d+)\s*页", "page", None),   # captures page number
    (r"第\s*(\d+)\s*张", "page", None),   # 第N张 slides
    # English
    (r"previous\s+(page|slide)", "offset", -1),
    (r"next\s+(page|slide)", "offset", 1),
    (r"(this|current)\s+(page|slide)", "current", None),
    (r"page\s+(\d+)", "page", None),
    (r"slide\s+(\d+)", "page", None),
]


def _enforce_positional_constraints(
    plan: QAContextPlan, question: str, current_page: int, max_page: int,
) -> QAContextPlan:
    """Augment the planner output with deterministic positional constraints.

    Extracts explicit page numbers and directional references (上一页, page 17,
    next slide, etc.) from the question text.  These are mandatory additions —
    they never replace planner-suggested pages, only add to them."""
    parsed_current, added_pages, added_offsets = deterministic_positional_targets(
        question, current_page, max_page,
    )
    force_current = plan.include_current_page or parsed_current

    # Merge: positional extras are additions, not replacements
    offsets = _dedupe_ints(list(plan.page_offsets) + list(added_offsets))
    pages = _dedupe_ints(list(plan.page_numbers) + list(added_pages))

    return plan.model_copy(update={
        "include_current_page": force_current,
        "page_numbers": pages,
        "page_offsets": offsets,
    })


def deterministic_positional_targets(
    question: str,
    current_page: int,
    max_page: int,
) -> tuple[bool, list[int], list[int]]:
    """Parse the same positional vocabulary used by production Q&A.

    This is deliberately LLM-free and is shared by planner enforcement and the
    retrieval evaluator.  The return value is ``(include_current, pages,
    offsets)`` in stable first-seen order.
    """
    import re

    safe_max = max(max_page, 1)
    force_current = False
    added_pages: list[int] = []
    added_offsets: list[int] = []
    for pattern, action, value in _POSITIONAL_PATTERNS:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            if action == "current":
                force_current = True
            elif action == "offset" and value is not None:
                target = current_page + value
                if 1 <= target <= safe_max and value not in added_offsets:
                    added_offsets.append(value)
            elif action == "page":
                try:
                    page_num = int(match.group(1))
                except (IndexError, ValueError):
                    continue
                if 1 <= page_num <= safe_max and page_num not in added_pages:
                    added_pages.append(page_num)
    # A directional or explicit-page question is a request for context around
    # the selected page, so the selected page is always the first tool result.
    if added_pages or added_offsets:
        force_current = True
    return force_current, added_pages, added_offsets


def plan_qa_context(
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    *,
    metrics_callback: Callable[[LLMCallMetrics], None] | None = None,
) -> QAContextPlan:
    """Run the bounded context planner to decide which sources to load.

    Uses the configured lightweight planner model (temperature 0, JSON mode,
    Pydantic validation).
    On any failure the deterministic fallback plan is returned — Q&A is never
    blocked by a planner error.

    Args:
        question: The student's free-form question.
        current_slide: The slide currently displayed in the UI.
        max_page: Total pages in the current PDF (for page-number validation).
        memory: Recent Q&A rounds as dicts (question/answer/file_id/…).

    Returns:
        A validated QAContextPlan (planner output or fallback).
    """
    try:
        # ── Build planner inputs ──
        _system_prompt, user_prompt = _planner_prompts(
            question, current_slide, max_page, memory,
        )

        system_prompt = (
            "You are a context planner.  Output ONLY valid JSON — no explanation, "
            "no markdown fences.  Choose the minimal set of context sources needed "
            "to answer the question accurately."
        )

        # ── Planner call (lightweight: single attempt, temperature 0) ──
        plan = complete_json(
            system_prompt,
            user_prompt,
            pydantic_model=QAContextPlan,
            max_retries=1,  # planner is meant to be lightweight; one shot is enough
            model=config.QA_PLANNER_MODEL,
            temperature=0.0,
            metrics_callback=metrics_callback,
        )
        if not isinstance(plan, QAContextPlan):
            plan = QAContextPlan.model_validate(plan)
        sanitized = _sanitize_plan(plan, current_slide, max_page)
        enforced = _enforce_positional_constraints(
            sanitized, question, current_slide.page_number, max_page,
        )
        logger.debug("Planner returned: %s", enforced.model_dump())
        return enforced
    except Exception as exc:
        logger.warning("Context planner failed, using fallback plan: %s", exc)
        fallback = _sanitize_plan(_default_plan(question), current_slide, max_page)
        return _enforce_positional_constraints(
            fallback, question, current_slide.page_number, max_page,
        )


def _tool_execution_context(
    current_slide: SlidePage,
    max_page: int,
) -> ToolExecutionContext:
    """Derive trusted Q&A scope from the immutable selected slide."""
    project_id = current_slide.project_id
    file_id = current_slide.file_id
    if not project_id or not file_id:
        raise ValueError("selected slide lacks project/file identity")
    return ToolExecutionContext(
        project_id=project_id,
        current_file_id=file_id,
        current_page_number=current_slide.page_number,
        max_page=max(max_page, 1),
        retrieval_scope=RetrievalScope(
            kind="current_file",
            project_id=project_id,
            file_id=file_id,
        ),
        allowed_file_ids=[file_id],
    )


def build_qa_tool_execution_context(
    current_slide: SlidePage,
    max_page: int,
) -> ToolExecutionContext:
    """Expose the single trusted current-file scope builder to Agent adapters."""
    return _tool_execution_context(current_slide, max_page)


def plan_qa_context_with_tools(
    question: str,
    current_slide: SlidePage,
    max_page: int,
    memory: list[dict],
    *,
    metrics_callback: Callable[[LLMCallMetrics], None] | None = None,
    tool_calling_enabled: bool | None = None,
) -> QAContextPlanningResult:
    """Use one native tool-call batch, falling back to the legacy planner.

    The Function Calling path is deliberately selection-only.  It produces
    generic ToolResults, then this module's Q&A adapter produces the existing
    QAContextPlan.  Context loading and final answering remain unchanged.
    """
    enabled = (
        config.QA_TOOL_CALLING_ENABLED
        if tool_calling_enabled is None
        else tool_calling_enabled
    )
    if not enabled:
        return QAContextPlanningResult(
            plan=plan_qa_context(
                question,
                current_slide,
                max_page,
                memory,
                metrics_callback=metrics_callback,
            ),
            mode="json_planner",
        )

    try:
        system_prompt, user_prompt = _planner_prompts(
            question, current_slide, max_page, memory, native_tools=True,
        )
        raw_calls = complete_tool_calls(
            system_prompt,
            user_prompt,
            build_openai_tool_schemas(QA_TOOL_REGISTRY),
            model=config.QA_PLANNER_MODEL,
            temperature=0.0,
            metrics_callback=metrics_callback,
        )
        results, traces = execute_tool_calls(
            raw_calls,
            registry=QA_TOOL_REGISTRY,
            context=_tool_execution_context(current_slide, max_page),
            max_calls=config.QA_TOOL_CALLING_MAX_CALLS,
        )
        plan = tool_results_to_qa_context_plan(results)
        search_trace_seen = False
        for result in results:
            if result.name != "search_slides":
                continue
            trace = next((item for item in traces if item.call_id == result.call_id), None)
            if trace is None:
                continue
            if search_trace_seen:
                trace.status = "deduplicated"
                trace.summary = "additional search query ignored; plan keeps first"
            else:
                search_trace_seen = True
        plan = _sanitize_plan(plan, current_slide, max_page)
        plan = _enforce_positional_constraints(
            plan, question, current_slide.page_number, max_page,
        )
        logger.debug("Function Calling planner returned: %s", plan.model_dump())
        return QAContextPlanningResult(
            plan=plan,
            mode="function_calling",
            tool_traces=traces,
        )
    except Exception as exc:
        logger.warning("Function Calling planner failed; using legacy fallback: %s", exc)
        return QAContextPlanningResult(
            plan=plan_qa_context(
                question,
                current_slide,
                max_page,
                memory,
                metrics_callback=metrics_callback,
            ),
            mode="function_calling_fallback",
            fallback_reason=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


# ── Context bundle builder (A4.3) ──

_MAX_SOURCE_SNIPPET_CHARS = 200


def _section_header(title: str) -> str:
    """Mark a prompt context section so the answer model sees the structure."""
    return f"\n--- {title} ---\n"


def _chunks_to_context(
    chunks: list[RetrievalChunk], max_chars: int,
) -> tuple[str, list[RetrievalChunk]]:
    """Format chunks within one shared budget and return those actually used."""
    parts: list[str] = []
    included: list[RetrievalChunk] = []
    total = 0
    for chunk in chunks:
        body = chunk.text.strip()
        if not body:
            continue
        header = f"[{len(included) + 1}] {chunk.source_file}, page {chunk.page_number}"
        prefix = f"{header}\n"
        remaining = max_chars - total
        if remaining <= len(prefix) + 1:
            break
        body_budget = remaining - len(prefix) - 1
        used_body = body[:body_budget]
        block = f"{prefix}{used_body}\n"
        parts.append(block)
        total += len(block)
        included.append(chunk)
        if len(used_body) < len(body):
            break
    return "\n".join(parts).rstrip(), included


def _valid_memory_turns(memory: list[dict]) -> list[QAMemoryTurn]:
    """Keep only valid memory turns so stale session state cannot break Q&A."""
    turns: list[QAMemoryTurn] = []
    for raw_turn in memory[-5:]:
        try:
            if isinstance(raw_turn, QAMemoryTurn):
                turns.append(raw_turn)
            elif isinstance(raw_turn, dict):
                turns.append(QAMemoryTurn.model_validate(raw_turn))
        except ValueError:
            logger.debug("Skipping malformed Q&A memory turn")
    return turns


def _memory_to_context(
    memory_turns: list[QAMemoryTurn], max_chars: int,
) -> tuple[str, list[QAMemoryTurn]]:
    """Keep newest rounds within budget, then render selected turns chronologically."""
    header = (
        "Recent conversation memory (for pronoun resolution; oldest → newest; "
        "most recent explicitly marked):"
    )
    if len(header) >= max_chars:
        return "", []

    selected: list[tuple[int, QAMemoryTurn, str]] = []
    total = len(header)
    turn_count = len(memory_turns)
    for index in range(turn_count - 1, -1, -1):
        turn = memory_turns[index]
        marker = " — MOST RECENT" if index == turn_count - 1 else ""
        label = f"{index + 1}{marker}"
        block = (
            f"\n[{label}] Q: {turn.question[:120]}"
            f"\n[{label}] A: {turn.answer[:200]}"
        )
        if total + len(block) > max_chars:
            continue
        selected.append((index, turn, block))
        total += len(block)

    selected.sort(key=lambda item: item[0])
    parts = [header, *(block for _index, _turn, block in selected)]
    included = [turn for _index, turn, _block in selected]
    return "".join(parts), included


def _slide_text_for_context(project_id: str, file_id: str, slide: SlidePage) -> str:
    """Get the best available text for a slide — PageUnderstanding preferred."""
    try:
        from slide_tutor.page_understanding import get_page_text_for_prompt
        resolution = get_page_text_for_prompt(project_id, file_id, slide)
        return resolution.text
    except Exception:
        return slide.text or "(no extractable text)"


def build_qa_context_bundle(
    plan: QAContextPlan,
    project_id: str,
    current_slide: SlidePage,
    current_file_id: str,
    max_page: int,
    memory: list[dict],
    index: tuple[list[RetrievalChunk], object, object],
    original_question: str | None = None,
) -> QAContextBundle:
    """Execute the context plan and assemble a sectioned prompt context.

    Steps (in order):
      1. Selected page (current)
      2. Explicitly requested pages (by page number)
      3. Nearby pages (by offset)
      4. Related pages (semantic search from project index)
      5. Recent conversation memory

    Returns a QAContextBundle with the formatted prompt_context and source
    metadata for the 'Context used' expander in the UI.
    """
    sources: list[QAContextSource] = []
    sections: list[str] = []
    loaded_slide_ids: set[str] = set()
    budget = config.MAX_CONTEXT_CHARS
    memory_budget = min(1500, budget // 5)
    # Coverage-first: selected, explicit, and nearby pages each keep a useful
    # independent excerpt. Semantic results share one budget as supplementary context.
    per_page_budget = max(budget // 3, 800)
    semantic_budget = per_page_budget

    def add_page_source(kind: str, target: SlidePage, reason: str, title: str) -> None:
        """Add one high-priority page once, only when its text enters the prompt."""
        if target.slide_id in loaded_slide_ids:
            return
        text = _slide_text_for_context(project_id, current_file_id, target).strip()
        block = text[:per_page_budget]
        if not block:
            return
        sections.append(f"{_section_header(title)}\n{block}")
        sources.append(QAContextSource(
            kind=kind,
            source_file=target.source_file,
            page_number=target.page_number,
            title=target.title_guess,
            reason=reason,
            snippet=block[:_MAX_SOURCE_SNIPPET_CHARS],
        ))
        loaded_slide_ids.add(target.slide_id)

    # ── 1. Selected page ──
    if plan.include_current_page:
        add_page_source(
            "selected_page", current_slide, "Currently displayed slide",
            f"Selected page ({current_slide.source_file}, page {current_slide.page_number})",
        )

    # ── 2. Explicit pages ──
    from slide_tutor.projects import load_file_slides
    slides = load_file_slides(project_id, current_file_id)
    slide_by_page: dict[int, SlidePage] = {s.page_number: s for s in slides}
    for page_num in plan.page_numbers:
        target = slide_by_page.get(page_num)
        if target is not None:
            add_page_source(
                "explicit_page", target, f"Explicitly requested page {page_num}",
                f"Page {page_num} ({target.source_file})",
            )

    # ── 3. Nearby pages ──
    for offset in plan.page_offsets:
        target_page = current_slide.page_number + offset
        target = slide_by_page.get(target_page)
        if target is None:
            continue
        direction = "previous" if offset < 0 else "next"
        if abs(offset) > 1:
            direction = f"{direction} (offset {offset})"
        add_page_source(
            "nearby_page", target, f"Nearby page: {direction}",
            f"{direction.title()} page ({target.source_file}, page {target_page})",
        )

    # ── 4. Semantic search ──
    if plan.semantic_query.strip():
        from slide_tutor.retrieval import retrieve_for_question
        retrieved = retrieve_for_question(
            plan.semantic_query,
            None,
            index,
            lexical_query=original_question or plan.semantic_query,
            scope=RetrievalScope(
                kind="current_file",
                project_id=project_id,
                file_id=current_file_id,
            ),
        )
        fresh = [chunk for chunk in retrieved if chunk.slide_id not in loaded_slide_ids]
        context_block, included_chunks = _chunks_to_context(fresh, semantic_budget)
        if context_block:
            sections.append(_section_header("Related pages (semantic search)") + context_block)
            for chunk in included_chunks:
                loaded_slide_ids.add(chunk.slide_id)
                sources.append(QAContextSource(
                    kind="retrieved_page",
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    title=chunk.title or None,
                    reason=(
                        f"{' + '.join(chunk.retrieval_methods) or 'retrieved'} match: "
                        f"'{(original_question or plan.semantic_query)[:60]}'"
                    ),
                    snippet=chunk.text.strip()[:_MAX_SOURCE_SNIPPET_CHARS],
                    retrieval_methods=chunk.retrieval_methods,
                ))

    # ── 5. Memory ──
    memory_turns: list[QAMemoryTurn] = []
    if plan.use_memory and memory:
        candidate_turns = _valid_memory_turns(memory)
        memory_block, memory_turns = _memory_to_context(candidate_turns, memory_budget)
        if memory_block:
            sections.append(_section_header("Recent conversation memory") + memory_block)
            for turn in memory_turns:
                sources.append(QAContextSource(
                    kind="memory",
                    source_file=turn.source_file,
                    page_number=turn.page_number,
                    reason=f"Memory: {turn.question[:80]}",
                    snippet=turn.answer[:_MAX_SOURCE_SNIPPET_CHARS],
                ))

    # ── Assemble ──
    prompt_context = "\n".join(sections).strip()
    # Build a human-readable planner summary for dev/debug
    parts = []
    if plan.include_current_page:
        parts.append("current page")
    if plan.page_numbers:
        parts.append(f"page(s) {plan.page_numbers}")
    if plan.page_offsets:
        parts.append(f"neighbors {plan.page_offsets}")
    if plan.semantic_query.strip():
        parts.append(f"search: '{plan.semantic_query[:50]}'")
    if plan.use_memory:
        parts.append("memory")
    planner_summary = "; ".join(parts) if parts else None

    return QAContextBundle(
        prompt_context=prompt_context,
        sources=sources,
        memory_turns=memory_turns,
        planner_summary=planner_summary,
    )
