"""Contracts and bounded quality pipeline for Stage C Reviewed Final Exams.

The module keeps planning, evidence binding, deterministic validation, and the
single semantic review/repair boundary together so each transition is explicit
and independently testable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

from pydantic import BaseModel, Field

from slide_tutor import cache, config
from slide_tutor.llm import LLMError, complete_json
from slide_tutor.models import (
    BoundExamBlueprint,
    BoundQuestionSlot,
    BindingIssue,
    BindingOutcome,
    Citation,
    EvidenceQualitySummary,
    EvidenceRef,
    ExamBlueprint,
    GenerationSnapshot,
    PageQuality,
    PlannedQuestionSlot,
    QuizQuestion,
    QuizSet,
    ReviewedExamMetadata,
    TopicCatalogEntry,
)
from slide_tutor.prompts import (
    PROMPT_GENERATE_REVIEWED_MCQ,
    PROMPT_PLAN_REVIEWED_BLUEPRINT,
    PROMPT_REPAIR_REVIEWED_MCQ,
    PROMPT_REVIEW_REVIEWED_EXAM,
)
from slide_tutor.projects import load_project, ordered_files, utc_now
from slide_tutor.retrieval import retrieve_for_question

logger = logging.getLogger(__name__)

PLANNER_PROMPT_VERSION = "reviewed-planner-v2"
REVIEWED_GENERATION_PROMPT_VERSION = "reviewed-mixed-v1"
REVIEW_PROMPT_VERSION = "reviewed-review-v3"
BLUEPRINT_SCHEMA_VERSION = "planned-v2"
REVIEWED_GENERATION_BATCH_SIZE = max(
    1, int(os.getenv("SLIDE_TUTOR_REVIEWED_BATCH_SIZE", "4"))
)

_QUESTION_TYPE_ORDER = ("mcq", "true_false", "fill_blank", "short_answer")
_REVIEWED_TYPE_RULES = {
    "mcq": (
        "MCQ: options must be exactly A/B/C/D, with one correct_option and "
        "distractor_rationales for the incorrect choices."
    ),
    "true_false": (
        "True/False: options must be exactly A/B representing True and False "
        "in the exam language; correct_option must be A or B."
    ),
    "fill_blank": (
        "Fill Blank: ask for one short non-mathematical concept term; options and "
        "correct_option must be empty; blank_answer is one canonical concept only. "
        "No alternatives, parentheses, lists, variables, symbols, equations, or LaTeX."
    ),
    "short_answer": (
        "Short Answer: options, correct_option, and blank_answer must be empty; "
        "include a grounded model_answer and an explicit scoring_rubric for up to 3 points."
    ),
}


def _question_type_rules(slots: list[BoundQuestionSlot]) -> str:
    """Return only the concrete type contracts needed by one Provider batch."""
    present = {slot.question_type for slot in slots}
    return "\n".join(
        f"- {_REVIEWED_TYPE_RULES[question_type]}"
        for question_type in _QUESTION_TYPE_ORDER
        if question_type in present
    )


class ContractIssue(BaseModel):
    severity: str = "blocking"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    slot_id: str | None = None


class ValidationIssue(BaseModel):
    severity: str = "blocking"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    question_id: str | None = None
    slot_id: str | None = None


class ReviewFinding(BaseModel):
    slot_id: str = Field(min_length=1)
    severity: str = "blocking"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ReviewSummary(BaseModel):
    findings: list[ReviewFinding] = Field(default_factory=list)


@dataclass(frozen=True)
class _GenerationContext:
    progress: Callable[..., None] | None = None
    cancel_requested: Callable[[], bool] | None = None

    def stage(self, name: str, detail: str = "", current: int = 0, total: int = 0) -> None:
        if self.cancel_requested and self.cancel_requested():
            raise LLMError("Reviewed exam generation cancelled")
        if self.progress:
            try:
                self.progress(name, detail, current, total)
            except TypeError:
                self.progress(name, detail)


def planning_contract_signature(
    *,
    planner_prompt_version: str,
    planner_model_contract: str,
    scope_contract_version: str,
    blueprint_schema_version: str = BLUEPRINT_SCHEMA_VERSION,
) -> str:
    """Hash only inputs that can change planned slots and retrieval intent."""
    payload = {
        "blueprint_schema_version": blueprint_schema_version,
        "planner_prompt_version": planner_prompt_version,
        "planner_model_contract": planner_model_contract,
        "scope_contract_version": scope_contract_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def planned_slot_has_no_evidence(slot: PlannedQuestionSlot) -> bool:
    """Keep planned serialization free of page-level citation fields."""
    return "evidence_refs" not in slot.model_dump(mode="json")


def build_topic_catalog(project_id: str, *, max_entries: int = 80) -> list[TopicCatalogEntry]:
    """Build a deterministic, bounded topic catalog without sending full pages to the planner."""
    from slide_tutor.models import FileSlidesPayload
    from slide_tutor.page_understanding import load_page_understanding

    project = load_project(project_id)
    entries: list[TopicCatalogEntry] = []
    seen: dict[tuple[str, str], TopicCatalogEntry] = {}
    for project_file in ordered_files(project):
        if project_file.status != "ready":
            continue
        raw = cache.load_json(
            cache.get_project_dir(project_id)
            / "files"
            / project_file.file_id
            / config.SLIDES_FILENAME,
            default=None,
        )
        if not raw:
            continue
        try:
            slides = FileSlidesPayload.model_validate(raw).slides
        except Exception:
            continue
        for slide in slides:
            understanding = load_page_understanding(
                project_id, project_file.file_id, slide.slide_id,
            )
            text = (
                understanding.canonical_markdown
                if understanding is not None
                else slide.text
            ).strip()
            if not text:
                continue
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            topic = (slide.title_guess or first_line or f"Page {slide.page_number}").strip()
            topic = topic[:100]
            key = (project_file.file_id, topic.casefold())
            summary = " ".join(text.split())[:240]
            previous = seen.get(key)
            if previous is not None:
                previous.page_numbers.append(slide.page_number)
                continue
            entry = TopicCatalogEntry(
                file_id=project_file.file_id,
                source_file=project_file.original_name,
                topic=topic,
                page_numbers=[slide.page_number],
                summary=summary,
            )
            seen[key] = entry
            entries.append(entry)
            if len(entries) >= max_entries:
                return entries
    return entries


def _topic_catalog_json(catalog: list[TopicCatalogEntry]) -> str:
    return json.dumps(
        [entry.model_dump(mode="json") for entry in catalog],
        ensure_ascii=False,
        indent=2,
    )


def sanitize_planned_blueprint(
    project_id: str,
    raw: dict[str, Any],
    *,
    valid_file_ids: set[str],
    target_count: int,
    language: str,
    question_type_mix: dict[str, int],
    source_signature: str = "",
    page_content_contract_signature: str = "",
    retrieval_contract_signature: str = "",
    planning_signature: str = "",
) -> ExamBlueprint:
    """Normalize planner output and reject scope/coverage drift before binding."""
    raw_slots = raw.get("slots") if isinstance(raw, dict) else None
    if not isinstance(raw_slots, list) or len(raw_slots) != target_count:
        raise LLMError(
            f"Blueprint planner returned {len(raw_slots) if isinstance(raw_slots, list) else 0} "
            f"slots; expected {target_count}."
        )
    expected_types = [
        question_type
        for question_type in _QUESTION_TYPE_ORDER
        for _ in range(int(question_type_mix.get(question_type, 0)))
    ]
    if len(expected_types) != target_count:
        raise LLMError("Blueprint question_type_mix does not match target_count.")
    slots: list[PlannedQuestionSlot] = []
    seen_ids: set[str] = set()
    for index, raw_slot in enumerate(raw_slots, start=1):
        if not isinstance(raw_slot, dict):
            raise LLMError(f"Blueprint slot {index} is not an object.")
        payload = dict(raw_slot)
        payload.setdefault("slot_id", f"slot-{index}")
        if payload["slot_id"] in seen_ids:
            raise LLMError(f"Blueprint contains duplicate slot id: {payload['slot_id']}")
        seen_ids.add(payload["slot_id"])
        raw_file_ids = payload.get("file_ids") or (
            [payload["file_id"]] if payload.get("file_id") else []
        )
        payload["file_ids"] = list(dict.fromkeys(str(file_id) for file_id in raw_file_ids))
        unknown = set(payload["file_ids"]) - valid_file_ids
        if unknown:
            raise LLMError(f"Blueprint slot {payload['slot_id']} has unknown file ids: {sorted(unknown)}")
        payload["topic_hint"] = str(payload.get("topic_hint") or payload.get("topic") or "topic").strip()
        payload["retrieval_query"] = str(payload.get("retrieval_query") or payload["topic_hint"]).strip()
        payload["question_type"] = str(
            payload.get("question_type") or expected_types[index - 1]
        ).strip()
        payload["coverage_role"] = (
            "cross_file" if len(payload["file_ids"]) >= 2 else "single_file"
        )
        try:
            slots.append(PlannedQuestionSlot.model_validate(payload))
        except Exception as exc:
            raise LLMError(f"Invalid blueprint slot {payload['slot_id']}: {exc}") from exc

    actual_mix = {
        question_type: sum(slot.question_type == question_type for slot in slots)
        for question_type in _QUESTION_TYPE_ORDER
    }
    expected_mix = {
        question_type: int(question_type_mix.get(question_type, 0))
        for question_type in _QUESTION_TYPE_ORDER
    }
    if actual_mix != expected_mix:
        raise LLMError(
            f"Blueprint question type mix {actual_mix} does not match requested {expected_mix}."
        )

    return ExamBlueprint(
        blueprint_id=f"blueprint-{hashlib.sha256(json.dumps([s.model_dump(mode='json') for s in slots], sort_keys=True).encode()).hexdigest()[:12]}",
        project_id=project_id,
        language=language,
        target_count=target_count,
        question_type_mix=question_type_mix,
        slots=slots,
        blueprint_schema_version=BLUEPRINT_SCHEMA_VERSION,
        source_signature=source_signature,
        planning_contract_signature=planning_signature,
        page_content_contract_signature=page_content_contract_signature,
        retrieval_contract_signature=retrieval_contract_signature,
    )


def plan_exam_blueprint(
    project_id: str,
    *,
    target_count: int,
    language: str,
    question_type_mix: dict[str, int],
    source_signature: str = "",
    page_content_contract_signature: str = "",
    retrieval_contract_signature: str = "",
    catalog: list[TopicCatalogEntry] | None = None,
) -> ExamBlueprint:
    """Plan slots from bounded catalog metadata; no page正文 is passed to the planner."""
    catalog = catalog if catalog is not None else build_topic_catalog(project_id)
    if not catalog:
        raise LLMError("Cannot plan a Reviewed exam without a usable topic catalog.")
    project = load_project(project_id)
    valid_file_ids = {file.file_id for file in project.files if file.status == "ready"}
    planning_signature = planning_contract_signature(
        planner_prompt_version=PLANNER_PROMPT_VERSION,
        planner_model_contract=config.LLM_MODEL,
        scope_contract_version="retrieval-scope-v1",
    )
    result = complete_json(
        "You are a bounded exam planner. Output ONLY valid JSON.",
        PROMPT_PLAN_REVIEWED_BLUEPRINT.format(
            language=language,
            target_count=target_count,
            question_type_mix=json.dumps(question_type_mix, ensure_ascii=False, sort_keys=True),
            topic_catalog_json=_topic_catalog_json(catalog),
        ),
        max_retries=2,
        reasoning_effort="high",
    )
    return sanitize_planned_blueprint(
        project_id,
        result if isinstance(result, dict) else {},
        valid_file_ids=valid_file_ids,
        target_count=target_count,
        language=language,
        question_type_mix=question_type_mix,
        source_signature=source_signature,
        page_content_contract_signature=page_content_contract_signature,
        retrieval_contract_signature=retrieval_contract_signature,
        planning_signature=planning_signature,
    )


def ensure_required_cross_file_slot(
    blueprint: ExamBlueprint,
    *,
    project_id: str,
    language: str,
    catalog: list[TopicCatalogEntry],
) -> ExamBlueprint:
    """Correct one slot once when a multi-file Reviewed plan lacks synthesis."""
    catalog_file_ids = {entry.file_id for entry in catalog}
    if len(catalog_file_ids) < 2 or any(
        slot.coverage_role == "cross_file" and slot.difficulty == "applied"
        for slot in blueprint.slots
    ):
        return blueprint
    replace_index = next(
        (index for index, slot in enumerate(blueprint.slots) if slot.difficulty == "applied"),
        len(blueprint.slots) - 1,
    )
    replacement_type = blueprint.slots[replace_index].question_type
    correction_mix = {question_type: 0 for question_type in _QUESTION_TYPE_ORDER}
    correction_mix[replacement_type] = 1
    correction = plan_exam_blueprint(
        project_id,
        target_count=1,
        language=language,
        question_type_mix=correction_mix,
        source_signature=blueprint.source_signature,
        page_content_contract_signature=blueprint.page_content_contract_signature,
        retrieval_contract_signature=blueprint.retrieval_contract_signature,
        catalog=catalog,
    )
    candidate = correction.slots[0]
    if candidate.coverage_role != "cross_file" or candidate.difficulty != "applied":
        raise LLMError(
            "Reviewed blueprint correction did not produce the required cross-file applied slot."
        )
    slots = list(blueprint.slots)
    slots[replace_index] = candidate.model_copy(update={"slot_id": slots[replace_index].slot_id})
    return blueprint.model_copy(update={"slots": slots})


def validate_evidence_ref(ref: EvidenceRef, slide_loader) -> list[ContractIssue]:
    """Validate page identity and eligibility without performing binding."""
    issues: list[ContractIssue] = []
    try:
        slide = slide_loader(ref.project_id, ref.file_id, ref.slide_id)
    except Exception as exc:
        issues.append(ContractIssue(code="missing_slide", message=str(exc)))
        return issues
    if slide is None or slide.page_number != ref.page_number:
        issues.append(ContractIssue(
            code="invalid_page_identity",
            message=f"Evidence page does not match slide identity: {ref.slide_id}/{ref.page_number}",
        ))
    if ref.evidence_eligibility in {"ineligible", "needs_vlm_only"}:
        issues.append(ContractIssue(
            code="ineligible_evidence",
            message=f"Evidence eligibility is {ref.evidence_eligibility}",
        ))
    return issues


def validate_bound_slot(slot: BoundQuestionSlot, slide_loader) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for ref in slot.evidence_refs:
        issues.extend(
            issue.model_copy(update={"slot_id": slot.slot_id})
            for issue in validate_evidence_ref(ref, slide_loader)
        )
        if ref.file_id not in slot.file_ids:
            issues.append(ContractIssue(
                code="scope_leak",
                message=f"Evidence file {ref.file_id} is outside slot scope",
                slot_id=slot.slot_id,
            ))
    if not slot.evidence_summary.strip():
        issues.append(ContractIssue(
            code="missing_evidence_summary",
            message="Bound slot requires a non-empty evidence_summary",
            slot_id=slot.slot_id,
        ))
    if slot.coverage_role == "cross_file" and len({ref.file_id for ref in slot.evidence_refs}) < 2:
        issues.append(ContractIssue(
            code="insufficient_cross_file_evidence",
            message="Cross-file slot requires evidence from at least two files",
            slot_id=slot.slot_id,
        ))
    return issues


def validate_bound_blueprint(blueprint: BoundExamBlueprint, slide_loader) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for slot in blueprint.slots:
        issues.extend(validate_bound_slot(slot, slide_loader))
        for ref in slot.evidence_refs:
            if ref.project_id != blueprint.project_id:
                issues.append(ContractIssue(
                    code="project_scope_leak",
                    message=f"Evidence project {ref.project_id} is outside blueprint project {blueprint.project_id}",
                    slot_id=slot.slot_id,
                ))
    return issues


def bound_blueprint_is_stale(
    blueprint: BoundExamBlueprint,
    *,
    planned_blueprint_id: str,
    source_signature: str,
    planning_contract_signature: str,
    page_content_contract_signature: str,
    retrieval_contract_signature: str,
) -> bool:
    return any([
        not blueprint.planned_blueprint_id,
        blueprint.planned_blueprint_id != planned_blueprint_id,
        blueprint.source_signature != source_signature,
        not blueprint.planning_contract_signature,
        blueprint.planning_contract_signature != planning_contract_signature,
        blueprint.page_content_contract_signature != page_content_contract_signature,
        blueprint.retrieval_contract_signature != retrieval_contract_signature,
    ])


def planned_blueprint_is_stale(
    blueprint: ExamBlueprint,
    *,
    source_signature: str,
    planning_contract_signature: str,
) -> bool:
    return any([
        blueprint.source_signature != source_signature,
        not blueprint.planning_contract_signature,
        blueprint.planning_contract_signature != planning_contract_signature,
    ])


def active_generation_snapshot(project_id: str) -> GenerationSnapshot:
    """Capture the current coherent generation identity without mutating caches."""
    from slide_tutor.generation import generation_state
    from slide_tutor.index_store import compute_source_signature

    source_signature = compute_source_signature(project_id)
    state = generation_state(project_id, source_signature)
    return GenerationSnapshot(
        source_signature=source_signature,
        active_layer=state.active_layer,
        page_generation_id=state.active_generation_id,
        index_generation_id=state.active_index_generation_id,
        page_content_contract_signature=state.page_content_contract_signature,
        retrieval_contract_signature=state.retrieval_contract_signature,
    )


def _evidence_ref_from_chunk(chunk) -> EvidenceRef:
    return EvidenceRef(
        project_id=chunk.project_id,
        file_id=chunk.file_id,
        slide_id=chunk.slide_id,
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        snippet=chunk.snippet_reference or chunk.text[:240],
        retrieval_methods=list(chunk.retrieval_methods),
        page_quality=chunk.page_quality,
        evidence_eligibility=chunk.evidence_eligibility,
    )


def _bind_once(
    blueprint: ExamBlueprint,
    project_id: str,
    *,
    index,
    expected_generation: GenerationSnapshot | None,
    current_generation: Callable[[], GenerationSnapshot] | None,
) -> tuple[BoundExamBlueprint | None, list[BindingIssue]]:
    from slide_tutor.models import RetrievalScope
    from slide_tutor.projects import load_file_slides

    issues: list[BindingIssue] = []
    if index is None:
        return None, [BindingIssue(
            code="index_unavailable",
            message="No active retrieval index is available for evidence binding.",
            slot_id=slot.slot_id,
            retryable=True,
        ) for slot in blueprint.slots]
    if expected_generation is not None and current_generation is not None:
        now = current_generation()
        if now != expected_generation:
            return None, [BindingIssue(
                code="generation_changed",
                message="Page/index generation changed during binding; retry with one coherent snapshot.",
                slot_id=slot.slot_id,
                retryable=True,
            ) for slot in blueprint.slots]

    bound_slots: list[BoundQuestionSlot] = []
    for slot in blueprint.slots:
        scope = RetrievalScope(
            kind="selected_files",
            project_id=project_id,
            file_ids=list(slot.file_ids),
        )
        chunks = retrieve_for_question(
            slot.retrieval_query,
            None,
            index,
            lexical_query=slot.topic_hint,
            scope=scope,
        )
        eligible = [
            chunk for chunk in chunks
            if chunk.file_id in slot.file_ids
            and chunk.evidence_eligibility in {"eligible", "degraded"}
        ]
        if slot.coverage_role == "cross_file":
            selected: list[Any] = []
            seen_files: set[str] = set()
            for chunk in eligible:
                if chunk.file_id not in seen_files:
                    selected.append(chunk)
                    seen_files.add(chunk.file_id)
            if len(seen_files) < 2:
                issues.append(BindingIssue(
                    code="insufficient_cross_file_evidence",
                    message="Retrieval did not find eligible evidence from two files.",
                    slot_id=slot.slot_id,
                    retryable=True,
                ))
                continue
            eligible = selected
        elif not eligible:
            issues.append(BindingIssue(
                code="binding_empty_or_ineligible",
                message="Retrieval returned no eligible evidence inside the slot scope.",
                slot_id=slot.slot_id,
                retryable=True,
            ))
            continue

        refs = [_evidence_ref_from_chunk(chunk) for chunk in eligible[:5]]
        status = "degraded" if any(ref.evidence_eligibility == "degraded" for ref in refs) else "eligible"
        bound_slots.append(BoundQuestionSlot(
            slot_id=slot.slot_id,
            file_ids=list(slot.file_ids),
            topic_hint=slot.topic_hint,
            retrieval_query=slot.retrieval_query,
            difficulty=slot.difficulty,
            question_type=slot.question_type,
            coverage_role=slot.coverage_role,
            evidence_refs=refs,
            evidence_summary=" ".join(ref.snippet for ref in refs)[:1000],
            evidence_quality=EvidenceQualitySummary(
                status=status,
                warnings=["degraded page" for ref in refs if ref.evidence_eligibility == "degraded"],
                blocking=False,
            ),
        ))

    if issues:
        return None, issues
    bound = BoundExamBlueprint(
        blueprint_id=f"bound-{blueprint.blueprint_id}",
        planned_blueprint_id=blueprint.blueprint_id,
        project_id=blueprint.project_id,
        language=blueprint.language,
        target_count=blueprint.target_count,
        slots=bound_slots,
        blueprint_schema_version="bound-v2",
        source_signature=blueprint.source_signature,
        planning_contract_signature=blueprint.planning_contract_signature,
        page_content_contract_signature=blueprint.page_content_contract_signature,
        retrieval_contract_signature=blueprint.retrieval_contract_signature,
    )
    slide_loader = lambda p, f, s: next(
        (slide for slide in load_file_slides(p, f) if slide.slide_id == s), None
    )
    contract_issues = validate_bound_blueprint(bound, slide_loader)
    if contract_issues:
        return None, [BindingIssue(
            code=issue.code,
            message=issue.message,
            slot_id=issue.slot_id or "unknown",
            retryable=False,
        ) for issue in contract_issues]
    return bound, []


def bind_exam_blueprint(
    blueprint: ExamBlueprint,
    project_id: str,
    *,
    index=None,
    replan: Callable[[list[PlannedQuestionSlot]], list[PlannedQuestionSlot]] | None = None,
    expected_generation: GenerationSnapshot | None = None,
    current_generation: Callable[[], GenerationSnapshot] | None = None,
) -> BindingOutcome:
    """Bind evidence, optionally replan failed slots once, and never fake Bound slots."""
    if index is None:
        from slide_tutor.index_store import load_index

        index = load_index(project_id)
    bound, issues = _bind_once(
        blueprint,
        project_id,
        index=index,
        expected_generation=expected_generation,
        current_generation=current_generation,
    )
    if bound is not None:
        return BindingOutcome(
            planned_blueprint_id=blueprint.blueprint_id,
            bound_blueprint=bound,
            replanned_once=False,
        )
    if not replan or not all(issue.retryable for issue in issues):
        return BindingOutcome(
            planned_blueprint_id=blueprint.blueprint_id,
            issues=issues,
            replanned_once=False,
        )

    failed_ids = {issue.slot_id for issue in issues}
    failed_slots = [slot for slot in blueprint.slots if slot.slot_id in failed_ids]
    replacements = replan(failed_slots)
    if len(replacements) != len(failed_slots):
        return BindingOutcome(
            planned_blueprint_id=blueprint.blueprint_id,
            issues=issues,
            replanned_once=True,
        )
    replacement_by_old_id = {
        old.slot_id: new.model_copy(update={"slot_id": old.slot_id})
        for old, new in zip(failed_slots, replacements, strict=True)
    }
    replanned_slots = [replacement_by_old_id.get(slot.slot_id, slot) for slot in blueprint.slots]
    replanned = blueprint.model_copy(update={"slots": replanned_slots})
    bound, retry_issues = _bind_once(
        replanned,
        project_id,
        index=index,
        expected_generation=expected_generation,
        current_generation=current_generation,
    )
    return BindingOutcome(
        planned_blueprint_id=blueprint.blueprint_id,
        bound_blueprint=bound,
        issues=retry_issues,
        replanned_once=True,
    )


def _slot_payload(slot: BoundQuestionSlot) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "question_type": slot.question_type,
        "topic_hint": slot.topic_hint,
        "difficulty": slot.difficulty,
        "coverage_role": slot.coverage_role,
        "allowed_file_ids": slot.file_ids,
        "evidence": [
            {
                "evidence_id": ref.evidence_id,
                "file_id": ref.file_id,
                "source_file": ref.source_file,
                "page_number": ref.page_number,
                "snippet": ref.snippet,
                "page_quality": ref.page_quality,
                "evidence_eligibility": ref.evidence_eligibility,
            }
            for ref in slot.evidence_refs
        ],
    }


def _citation_from_ref(ref: EvidenceRef) -> Citation:
    return Citation(
        source_file=ref.source_file,
        page_number=ref.page_number,
        snippet_reference=ref.snippet or "Evidence from the bound slide.",
        slide_id=ref.slide_id,
        project_id=ref.project_id,
        file_id=ref.file_id,
        evidence_id=ref.evidence_id,
        page_quality=ref.page_quality,
        evidence_eligibility=ref.evidence_eligibility,
        retrieval_provenance=list(ref.retrieval_methods),
    )


def _placeholder_citation(evidence_id: str) -> Citation:
    """Keep malformed model evidence representable so the validator can block it."""
    return Citation(
        source_file="unresolved-evidence",
        page_number=1,
        snippet_reference="Unresolved evidence id.",
        evidence_id=evidence_id,
    )


def _questions_from_raw(
    raw_questions: list[Any],
    slots: list[BoundQuestionSlot],
) -> list[QuizQuestion]:
    refs_by_slot = {
        slot.slot_id: {ref.evidence_id: ref for ref in slot.evidence_refs}
        for slot in slots
    }
    questions: list[QuizQuestion] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            continue
        slot_id = str(raw.get("slot_id", "")).strip()
        evidence_ids = [str(item) for item in raw.get("evidence_ids", []) if str(item).strip()]
        allowed = refs_by_slot.get(slot_id, {})
        citations = [
            _citation_from_ref(allowed[evidence_id]) if evidence_id in allowed
            else _placeholder_citation(evidence_id)
            for evidence_id in evidence_ids
        ]
        if not citations:
            citations = [_placeholder_citation("")]
        payload = dict(raw)
        payload.pop("slot_id", None)
        payload.pop("evidence_ids", None)
        payload["slot_id"] = slot_id or None
        payload["question_id"] = f"reviewed-{slot_id or len(questions) + 1}"
        origin_raw = "|".join([slot_id, str(raw.get("question", "")), *sorted(evidence_ids)])
        payload["question_origin_id"] = f"origin-{hashlib.sha256(origin_raw.encode()).hexdigest()[:16]}"
        payload["citations"] = [citation.model_dump(mode="json") for citation in citations]
        slot = next((item for item in slots if item.slot_id == slot_id), None)
        payload["question_type"] = slot.question_type if slot is not None else str(
            raw.get("question_type", "mcq")
        )
        # Providers commonly emit JSON null for fields that are inapplicable to
        # one mixed question type. Normalize only neutral empty values; required
        # answers/rubrics remain subject to the Pydantic and deterministic gates.
        for field_name in ("correct_option", "blank_answer", "model_answer", "scoring_rubric"):
            if payload.get(field_name) is None:
                payload[field_name] = ""
        for field_name in ("options", "distractor_rationales", "topic_names"):
            if payload.get(field_name) is None:
                payload[field_name] = {}
        try:
            questions.append(QuizQuestion.model_validate(payload))
        except Exception as exc:
            logger.warning("Dropping malformed Reviewed question: %s", exc)
    return questions


def generate_reviewed_mcq_questions(
    bound_blueprint: BoundExamBlueprint,
    *,
    language: str,
    batch_size: int = REVIEWED_GENERATION_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[QuizQuestion]:
    """Generate evidence-bound mixed questions in small configurable batches.

    The historical function name remains as a compatibility import for G1 code.
    """
    questions: list[QuizQuestion] = []
    total_batches = max(1, (len(bound_blueprint.slots) + max(1, batch_size) - 1) // max(1, batch_size))
    for batch_index, start in enumerate(
        range(0, len(bound_blueprint.slots), max(1, batch_size)), start=1,
    ):
        if cancel_requested and cancel_requested():
            raise LLMError("Reviewed exam generation cancelled")
        if progress:
            progress(batch_index - 1, total_batches)
        slots = bound_blueprint.slots[start:start + max(1, batch_size)]
        result = complete_json(
            "You are an evidence-bound mixed-format exam generator. Output ONLY valid JSON.",
            PROMPT_GENERATE_REVIEWED_MCQ.format(
                language=language,
                question_type_rules=_question_type_rules(slots),
                slot_evidence_json=json.dumps(
                    [_slot_payload(slot) for slot in slots],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            max_retries=2,
            reasoning_effort="medium",
        )
        raw_questions = result.get("questions", []) if isinstance(result, dict) else []
        questions.extend(_questions_from_raw(raw_questions, slots))
        if progress:
            progress(batch_index, total_batches)
    return questions


def validate_reviewed_mcq_questions(
    questions: list[QuizQuestion],
    bound_blueprint: BoundExamBlueprint,
) -> list[ValidationIssue]:
    """Validate mixed Reviewed questions; name retained for compatibility."""
    from slide_tutor.quiz import fill_blank_quality_issues

    issues: list[ValidationIssue] = []
    expected = {slot.slot_id: slot for slot in bound_blueprint.slots}
    actual_by_slot: dict[str, QuizQuestion] = {}
    for question in questions:
        slot_id = question.slot_id or ""
        if slot_id not in expected:
            issues.append(ValidationIssue(
                code="unknown_slot", message=f"Question references unknown slot {slot_id!r}.",
                question_id=question.question_id, slot_id=slot_id or None,
            ))
            continue
        if slot_id in actual_by_slot:
            issues.append(ValidationIssue(
                code="duplicate_slot", message=f"Slot {slot_id} has multiple questions.",
                question_id=question.question_id, slot_id=slot_id,
            ))
        actual_by_slot[slot_id] = question
        slot = expected[slot_id]
        if question.question_type != slot.question_type:
            issues.append(ValidationIssue(
                code="wrong_question_type",
                message=(
                    f"Question type {question.question_type} does not match "
                    f"planned type {slot.question_type}."
                ),
                question_id=question.question_id, slot_id=slot_id,
            ))
        if question.question_type == "mcq":
            if set(question.options) != {"A", "B", "C", "D"}:
                issues.append(ValidationIssue(
                    code="invalid_options", message="MCQ must have exactly A/B/C/D options.",
                    question_id=question.question_id, slot_id=slot_id,
                ))
            if len(set(question.options.values())) != 4:
                issues.append(ValidationIssue(
                    code="duplicate_options", message="MCQ options must be distinct.",
                    question_id=question.question_id, slot_id=slot_id,
                ))
            correct_text = question.options.get(question.correct_option, "").strip().casefold()
            if len(correct_text) >= 4 and correct_text in question.question.casefold():
                issues.append(ValidationIssue(
                    code="answer_leakage",
                    message="The correct option text is repeated in the question stem.",
                    question_id=question.question_id,
                    slot_id=slot_id,
                ))
        elif question.question_type == "true_false":
            if set(question.options) != {"A", "B"} or len(set(question.options.values())) != 2:
                issues.append(ValidationIssue(
                    code="invalid_true_false_options",
                    message="True/False must have two distinct A/B options.",
                    question_id=question.question_id, slot_id=slot_id,
                ))
        elif question.question_type == "fill_blank":
            for quality_code in fill_blank_quality_issues(question):
                issues.append(ValidationIssue(
                    code=f"fill_blank_{quality_code}",
                    message="Fill Blank violates the single non-mathematical concept contract.",
                    question_id=question.question_id, slot_id=slot_id,
                ))
        elif question.question_type == "short_answer":
            if not question.model_answer.strip() or not question.scoring_rubric.strip():
                issues.append(ValidationIssue(
                    code="short_answer_rubric_missing",
                    message="Short Answer requires a model answer and scoring rubric.",
                    question_id=question.question_id, slot_id=slot_id,
                ))
        allowed_ids = {ref.evidence_id for ref in slot.evidence_refs}
        citation_ids = [citation.evidence_id for citation in question.citations]
        if not citation_ids or any(evidence_id not in allowed_ids for evidence_id in citation_ids):
            issues.append(ValidationIssue(
                code="citation_not_subset", message="Citations must be a non-empty subset of bound evidence ids.",
                question_id=question.question_id, slot_id=slot_id,
            ))
        cited_files = {
            citation.file_id for citation in question.citations if citation.file_id
        }
        if not cited_files.issubset(set(slot.file_ids)):
            issues.append(ValidationIssue(
                code="citation_scope_leak", message="Citation file is outside the slot scope.",
                question_id=question.question_id, slot_id=slot_id,
            ))
        if slot.coverage_role == "cross_file" and len(cited_files) < 2:
            issues.append(ValidationIssue(
                code="cross_file_coverage_missing", message="Cross-file slot needs citations from two files.",
                question_id=question.question_id, slot_id=slot_id,
            ))

    missing = set(expected) - set(actual_by_slot)
    for slot_id in sorted(missing):
        issues.append(ValidationIssue(
            code="missing_slot_question", message=f"No generated question for slot {slot_id}.", slot_id=slot_id,
        ))
    for left_index, left in enumerate(questions):
        for right in questions[left_index + 1:]:
            if SequenceMatcher(None, left.question, right.question).ratio() > 0.85:
                issues.append(ValidationIssue(
                    code="duplicate_question", message="Generated questions are too similar.",
                    question_id=right.question_id, slot_id=right.slot_id,
                ))
                break
    return issues


def review_reviewed_exam(
    bound_blueprint: BoundExamBlueprint,
    questions: list[QuizQuestion],
    validation_issues: list[ValidationIssue],
    *,
    language: str,
) -> ReviewSummary:
    """Ask one bounded semantic reviewer for findings, never for final answers."""
    payload = {
        "slots": [_slot_payload(slot) for slot in bound_blueprint.slots],
        "questions": [question.model_dump(mode="json") for question in questions],
    }
    result = complete_json(
        "You are a strict exam quality reviewer. Output ONLY valid JSON.",
        PROMPT_REVIEW_REVIEWED_EXAM.format(
            language=language,
            review_payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
            validation_summary_json=json.dumps(
                [issue.model_dump(mode="json") for issue in validation_issues],
                ensure_ascii=False,
                indent=2,
            ),
        ),
        max_retries=1,
        reasoning_effort="high",
    )
    try:
        summary = ReviewSummary.model_validate(result)
    except Exception as exc:
        raise LLMError(f"Invalid Reviewed exam review response: {exc}") from exc
    valid_slots = {slot.slot_id for slot in bound_blueprint.slots}
    unknown = [finding for finding in summary.findings if finding.slot_id not in valid_slots]
    if unknown:
        raise LLMError("Reviewed exam reviewer returned an unknown slot id.")
    return summary


def repair_reviewed_mcq_questions(
    bound_blueprint: BoundExamBlueprint,
    questions: list[QuizQuestion],
    slot_ids: set[str],
    *,
    language: str,
) -> list[QuizQuestion]:
    """Repair only blocking slots; callers must not invoke this twice."""
    slots = [slot for slot in bound_blueprint.slots if slot.slot_id in slot_ids]
    if not slots:
        return questions
    result = complete_json(
        "You repair only listed evidence-bound mixed-format slots. Output ONLY valid JSON.",
        PROMPT_REPAIR_REVIEWED_MCQ.format(
            language=language,
            question_type_rules=_question_type_rules(slots),
            required_slot_ids_json=json.dumps(sorted(slot_ids), ensure_ascii=False),
            repair_payload_json=json.dumps(
                [_slot_payload(slot) for slot in slots],
                ensure_ascii=False,
                indent=2,
            ),
        ),
        max_retries=1,
        reasoning_effort="medium",
    )
    repaired = _questions_from_raw(
        result.get("questions", []) if isinstance(result, dict) else [], slots,
    )
    replacements = {question.slot_id: question for question in repaired}
    missing_replacements = slot_ids - set(replacements)
    if missing_replacements:
        raise LLMError(
            "Reviewed repair response did not cover every requested slot: "
            f"{sorted(missing_replacements)}"
        )
    return [
        replacements.get(question.slot_id, question)
        for question in questions
    ] + [
        question for slot_id, question in replacements.items()
        if not any(existing.slot_id == slot_id for existing in questions)
    ]


def _reviewed_quiz_set_id(
    project_id: str,
    file_ids: list[str],
    target_count: int,
    language: str,
    question_type_mix: dict[str, int],
    snapshot: GenerationSnapshot,
) -> str:
    payload = {
        "project_id": project_id,
        "file_ids": sorted(file_ids),
        "target_count": target_count,
        "language": language,
        "question_type_mix": question_type_mix,
        "quality_mode": "reviewed",
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "generation_prompt_version": REVIEWED_GENERATION_PROMPT_VERSION,
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "snapshot": snapshot.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_reviewed_final_exam(
    project_id: str,
    *,
    target_count: int,
    language: str,
    question_type_mix: dict[str, int] | None = None,
    generation_snapshot: GenerationSnapshot | None = None,
    progress: Callable[[str, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> QuizSet:
    """Execute a bounded mixed-format Reviewed exam with one repair maximum."""
    context = _GenerationContext(progress=progress, cancel_requested=cancel_requested)
    snapshot = generation_snapshot or active_generation_snapshot(project_id)
    mix = question_type_mix or {"mcq": target_count, "true_false": 0, "fill_blank": 0, "short_answer": 0}
    normalized_mix = {
        question_type: int(mix.get(question_type, 0))
        for question_type in _QUESTION_TYPE_ORDER
    }
    if any(count < 0 for count in normalized_mix.values()) or sum(normalized_mix.values()) != target_count:
        raise ValueError("Reviewed question_type_mix must be non-negative and sum to target_count.")
    mix = normalized_mix

    context.stage("planning", "building bounded topic catalog")
    catalog = build_topic_catalog(project_id)
    planned = plan_exam_blueprint(
        project_id,
        target_count=target_count,
        language=language,
        question_type_mix=mix,
        source_signature=snapshot.source_signature,
        page_content_contract_signature=snapshot.page_content_contract_signature,
        retrieval_contract_signature=snapshot.retrieval_contract_signature,
        catalog=catalog,
    )
    planned = ensure_required_cross_file_slot(
        planned, project_id=project_id, language=language, catalog=catalog,
    )

    context.stage("binding", "retrieving eligible evidence")

    def replan_failed_slots(failed_slots: list[PlannedQuestionSlot]) -> list[PlannedQuestionSlot]:
        """Replace only failed slots using the same bounded catalog, once."""
        retry_mix = {
            question_type: sum(slot.question_type == question_type for slot in failed_slots)
            for question_type in _QUESTION_TYPE_ORDER
        }
        retry = plan_exam_blueprint(
            project_id,
            target_count=len(failed_slots),
            language=language,
            question_type_mix=retry_mix,
            source_signature=snapshot.source_signature,
            page_content_contract_signature=snapshot.page_content_contract_signature,
            retrieval_contract_signature=snapshot.retrieval_contract_signature,
            catalog=catalog,
        )
        return retry.slots

    outcome = bind_exam_blueprint(
        planned,
        project_id,
        expected_generation=snapshot,
        current_generation=lambda: active_generation_snapshot(project_id),
        replan=replan_failed_slots,
    )
    if outcome.bound_blueprint is None:
        raise LLMError(
            "Reviewed evidence binding failed: "
            + "; ".join(f"{issue.slot_id}: {issue.message}" for issue in outcome.issues)
        )
    bound = outcome.bound_blueprint

    context.stage("generation", "generating evidence-bound mixed questions")

    def batch_progress(current: int, total: int) -> None:
        context.stage(
            "generation", f"generating evidence-bound questions · batch {current}/{total}", current, total,
        )

    questions = generate_reviewed_mcq_questions(
        bound,
        language=language,
        progress=batch_progress,
        cancel_requested=cancel_requested,
    )
    validation = validate_reviewed_mcq_questions(questions, bound)
    initial_validation = list(validation)

    context.stage("review", "running one semantic review")
    review = review_reviewed_exam(bound, questions, validation, language=language)
    blocking_slots = {
        issue.slot_id for issue in validation if issue.severity == "blocking" and issue.slot_id
    }
    blocking_slots.update(
        finding.slot_id for finding in review.findings if finding.severity == "blocking"
    )
    repair_summary: dict[str, Any] = {"attempted": False, "slot_ids": [], "remaining_blocking": 0}
    if blocking_slots:
        context.stage("review", f"repairing {len(blocking_slots)} slot(s) once")
        repair_summary = {"attempted": True, "slot_ids": sorted(blocking_slots), "remaining_blocking": 0}
        questions = repair_reviewed_mcq_questions(
            bound, questions, blocking_slots, language=language,
        )
        # The repaired set gets the full deterministic validator again; no
        # second semantic reviewer is allowed by the Stage C contract.
        validation = validate_reviewed_mcq_questions(questions, bound)
        repair_summary["remaining_blocking"] = sum(
            issue.severity == "blocking" for issue in validation
        )
        if repair_summary["remaining_blocking"]:
            issue_codes = sorted({issue.code for issue in validation if issue.severity == "blocking"})
            issue_slots = sorted({issue.slot_id for issue in validation if issue.severity == "blocking" and issue.slot_id})
            raise LLMError(
                "Reviewed exam remained invalid after its single repair pass: "
                f"codes={issue_codes}, slots={issue_slots}"
            )

    context.stage("finalizing", "assembling reviewed metadata and cache payload")
    now = utc_now()
    metadata = ReviewedExamMetadata(
        quality_mode="reviewed",
        blueprint_schema_version=BLUEPRINT_SCHEMA_VERSION,
        planner_prompt_version=PLANNER_PROMPT_VERSION,
        planner_model=config.LLM_MODEL,
        generation_prompt_version=REVIEWED_GENERATION_PROMPT_VERSION,
        generation_model=config.LLM_MODEL,
        review_prompt_version=REVIEW_PROMPT_VERSION,
        review_model=config.LLM_MODEL,
        source_signature=snapshot.source_signature,
        page_generation_id=snapshot.page_generation_id,
        index_generation_id=snapshot.index_generation_id,
        active_layer=snapshot.active_layer,
        planned_blueprint=planned,
        bound_blueprint=bound,
        validation_summary={
            "initial_issue_count": len(initial_validation),
            "initial_blocking_issue_count": sum(
                issue.severity == "blocking" for issue in initial_validation
            ),
            "post_repair_issue_count": len(validation),
            "post_repair_blocking_issue_count": sum(
                issue.severity == "blocking" for issue in validation
            ),
            "cross_file_slot_count": sum(
                slot.coverage_role == "cross_file" for slot in bound.slots
            ),
        },
        review_findings_summary={
            "finding_count": len(review.findings),
            "blocking_finding_count": sum(finding.severity == "blocking" for finding in review.findings),
            "codes": sorted({finding.code for finding in review.findings}),
        },
        repair_summary={
            **repair_summary,
            "semantic_recheck_performed": False,
        },
        created_at=now,
        completed_at=now,
    )
    project = load_project(project_id)
    file_ids = [file.file_id for file in ordered_files(project) if file.status == "ready"]
    quiz_set = QuizSet(
        quiz_set_id=_reviewed_quiz_set_id(
            project_id, file_ids, target_count, language, mix, snapshot,
        ),
        project_id=project_id,
        file_ids=file_ids,
        total_count=len(questions),
        difficulty_mix="reviewed blueprint",
        question_type_mix=",".join(
            f"{question_type}:{mix[question_type]}" for question_type in _QUESTION_TYPE_ORDER
        ),
        questions=questions,
        created_at=now,
        model=config.LLM_MODEL,
        prompt_version=REVIEWED_GENERATION_PROMPT_VERSION,
        language=language,
        quality_mode="reviewed",
        reviewed_metadata=metadata,
        generation_snapshot=snapshot,
    )
    from slide_tutor.topics import resolve_quiz_topics
    return resolve_quiz_topics(quiz_set)


__all__ = [
    "BLUEPRINT_SCHEMA_VERSION",
    "PLANNER_PROMPT_VERSION",
    "REVIEWED_GENERATION_PROMPT_VERSION",
    "REVIEW_PROMPT_VERSION",
    "BindingOutcome",
    "ContractIssue",
    "ReviewFinding",
    "ReviewSummary",
    "ValidationIssue",
    "active_generation_snapshot",
    "bind_exam_blueprint",
    "bound_blueprint_is_stale",
    "build_topic_catalog",
    "generate_reviewed_mcq_questions",
    "plan_exam_blueprint",
    "planned_blueprint_is_stale",
    "planned_slot_has_no_evidence",
    "repair_reviewed_mcq_questions",
    "review_reviewed_exam",
    "run_reviewed_final_exam",
    "sanitize_planned_blueprint",
    "validate_bound_blueprint",
    "validate_bound_slot",
    "validate_evidence_ref",
    "validate_reviewed_mcq_questions",
]
