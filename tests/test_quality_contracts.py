from __future__ import annotations

import pytest

from slide_tutor.models import (
    BoundExamBlueprint,
    BoundQuestionSlot,
    EvidenceQualitySummary,
    EvidenceRef,
    ExamBlueprint,
    PlannedQuestionSlot,
)
from slide_tutor.quiz_quality import (
    bound_blueprint_is_stale,
    planned_blueprint_is_stale,
    planned_slot_has_no_evidence,
    planning_contract_signature,
    validate_bound_blueprint,
)


def _planned(slot_id="s1"):
    return PlannedQuestionSlot(
        slot_id=slot_id, file_id="file-a", topic="topic",
        retrieval_query="query", difficulty="basic", question_type="mcq",
    )


def _ref(eligibility="eligible"):
    return EvidenceRef(
        project_id="p", file_id="file-a", slide_id="slide-1",
        source_file="lecture.pdf", page_number=1,
        evidence_eligibility=eligibility,
    )


def test_planned_slot_has_no_evidence_and_blueprint_contract_is_strict():
    assert planned_slot_has_no_evidence(_planned())
    blueprint = ExamBlueprint(
        blueprint_id="b", project_id="p", target_count=1,
        slots=[_planned()], question_type_mix={"mcq": 1},
    )
    assert blueprint.slots[0].slot_id == "s1"
    with pytest.raises(ValueError):
        ExamBlueprint(
            blueprint_id="b", project_id="p", target_count=1,
            slots=[_planned("a"), _planned("a")],
        )


def test_bound_slot_rejects_blocking_or_missing_evidence():
    with pytest.raises(ValueError):
        EvidenceQualitySummary(status="needs_vlm_only", blocking=False)
    with pytest.raises(ValueError):
        BoundQuestionSlot(
            slot_id="s", file_id="f", topic="t", retrieval_query="q",
            difficulty="basic", question_type="mcq", evidence_refs=[_ref()],
            evidence_summary="summary",
            evidence_quality=EvidenceQualitySummary(status="needs_vlm_only", blocking=True),
        )


def test_bound_blueprint_validation_checks_page_identity_and_quality():
    planning = planning_contract_signature(
        planner_prompt_version="planner-v1",
        planner_model_contract="model-v1",
        scope_contract_version="scope-v1",
    )
    slot = BoundQuestionSlot(
        slot_id="s", file_id="file-a", topic="t", retrieval_query="q",
        difficulty="basic", question_type="mcq", evidence_refs=[_ref()],
        evidence_summary="summary",
        evidence_quality=EvidenceQualitySummary(status="eligible", blocking=False),
    )
    blueprint = BoundExamBlueprint(
        blueprint_id="b", project_id="p", target_count=1, slots=[slot],
        planned_blueprint_id="planned-b",
        source_signature="source", page_content_contract_signature="page",
        retrieval_contract_signature="retrieval", planning_contract_signature=planning,
    )
    issues = validate_bound_blueprint(
        blueprint,
        lambda project_id, file_id, slide_id: type("Slide", (), {"page_number": 1})(),
    )
    assert issues == []
    assert bound_blueprint_is_stale(
        blueprint,
        planned_blueprint_id="planned-b",
        source_signature="new-source",
        planning_contract_signature=planning,
        page_content_contract_signature="page",
        retrieval_contract_signature="retrieval",
    )


def test_planned_and_bound_stale_contracts_are_distinct():
    planning = planning_contract_signature(
        planner_prompt_version="planner-v1",
        planner_model_contract="model-v1",
        scope_contract_version="scope-v1",
    )
    planned = ExamBlueprint(
        blueprint_id="planned-b", project_id="p", target_count=1,
        slots=[_planned()], source_signature="source",
        planning_contract_signature=planning,
        page_content_contract_signature="old-page",
        retrieval_contract_signature="old-retrieval",
    )
    slot = BoundQuestionSlot(
        slot_id="s", file_id="file-a", topic="t", retrieval_query="q",
        difficulty="basic", question_type="mcq", evidence_refs=[_ref()],
        evidence_summary="summary",
        evidence_quality=EvidenceQualitySummary(status="eligible", blocking=False),
    )
    bound = BoundExamBlueprint(
        blueprint_id="bound-b", planned_blueprint_id="planned-b",
        project_id="p", target_count=1, slots=[slot],
        source_signature="source", planning_contract_signature=planning,
        page_content_contract_signature="page", retrieval_contract_signature="retrieval",
    )
    assert not planned_blueprint_is_stale(
        planned, source_signature="source", planning_contract_signature=planning,
    )
    assert bound_blueprint_is_stale(
        bound, planned_blueprint_id="planned-b", source_signature="source",
        planning_contract_signature=planning,
        page_content_contract_signature="new-page",
        retrieval_contract_signature="retrieval",
    )
    assert bound_blueprint_is_stale(
        bound, planned_blueprint_id="planned-b", source_signature="source",
        planning_contract_signature=planning,
        page_content_contract_signature="page",
        retrieval_contract_signature="new-retrieval",
    )


def test_planning_contract_changes_only_for_planning_inputs():
    first = planning_contract_signature(
        planner_prompt_version="planner-v1",
        planner_model_contract="model-v1",
        scope_contract_version="scope-v1",
    )
    same = planning_contract_signature(
        planner_prompt_version="planner-v1",
        planner_model_contract="model-v1",
        scope_contract_version="scope-v1",
    )
    changed = planning_contract_signature(
        planner_prompt_version="planner-v2",
        planner_model_contract="model-v1",
        scope_contract_version="scope-v1",
    )
    assert first == same
    assert first != changed


def test_legacy_empty_planning_signatures_are_stale():
    planned = ExamBlueprint(
        blueprint_id="legacy", project_id="p", target_count=1,
        slots=[_planned()], source_signature="source",
    )
    assert planned_blueprint_is_stale(
        planned, source_signature="source", planning_contract_signature="current",
    )
