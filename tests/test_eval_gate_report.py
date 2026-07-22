from __future__ import annotations

import json

from slide_tutor.eval_gate_report import (
    build_gate_report,
    evaluate_paths,
    format_gate_report,
)


def _report(*, execution_command: str, backend: str, gates: dict, ref_passed: bool = True) -> dict:
    return {
        "execution_command": execution_command,
        "reference_validation": {"passed": ref_passed, "case_count": 33},
        "evaluation": {
            "model_contract": {"backend": backend},
            "overall": {
                "case_count": 33,
                "recall_at_3": 0.98,
                "recall_at_5": 1.0,
                "mrr": 0.88,
                "precision_at_5": 0.26,
            },
        },
        "gates": gates,
    }


_ALL_TRUE_CONTROLLED = {
    "no_case_errors": True,
    "exact_title_recall_at_3": True,
    "positional_deterministic_hit_rate": True,
    "exact_title_method_compliance": True,
    "positional_method_compliance": True,
    "ltd_pages_in_top_k": True,
    "semantic_cross_lingual_no_regression": True,
}


def test_controlled_all_green_passes():
    report = _report(
        execution_command="python -m slide_tutor.retrieval_eval --backend controlled",
        backend="controlled",
        gates=dict(_ALL_TRUE_CONTROLLED),
    )
    summary = build_gate_report(report)
    assert summary.ok
    assert summary.backend == "controlled"
    assert not summary.is_configured_run


def test_configured_embedding_all_green_passes():
    gates = dict(_ALL_TRUE_CONTROLLED, configured_backend_is_embedding=True)
    report = _report(
        execution_command="python -m slide_tutor.retrieval_eval --backend configured",
        backend="embedding",
        gates=gates,
    )
    summary = build_gate_report(report)
    assert summary.ok
    assert summary.is_configured_run


def test_configured_silent_tfidf_fallback_fails():
    """The exact real-world bug: configured run fell back to tfidf, embedding gate false."""
    gates = dict(_ALL_TRUE_CONTROLLED, configured_backend_is_embedding=False)
    report = _report(
        execution_command="python -m slide_tutor.retrieval_eval --backend configured",
        backend="tfidf",
        gates=gates,
    )
    summary = build_gate_report(report)
    assert not summary.ok
    # Both the red gate and the backend mismatch must be flagged.
    joined = " | ".join(summary.problems)
    assert "configured_backend_is_embedding" in joined
    assert "expected 'embedding'" in joined


def test_configured_missing_embedding_gate_fails():
    report = _report(
        execution_command="python -m slide_tutor.retrieval_eval --backend configured",
        backend="embedding",
        gates=dict(_ALL_TRUE_CONTROLLED),  # gate absent
    )
    summary = build_gate_report(report)
    assert not summary.ok
    assert any("missing" in p for p in summary.problems)


def test_reference_validation_failure_fails():
    report = _report(
        execution_command="--backend controlled",
        backend="controlled",
        gates=dict(_ALL_TRUE_CONTROLLED),
        ref_passed=False,
    )
    summary = build_gate_report(report)
    assert not summary.ok
    assert any("reference_validation" in p for p in summary.problems)


def test_format_contains_backend_and_marks():
    gates = dict(_ALL_TRUE_CONTROLLED, configured_backend_is_embedding=False)
    report = _report(
        execution_command="--backend configured",
        backend="tfidf",
        gates=gates,
    )
    text = format_gate_report(build_gate_report(report))
    assert "backend: tfidf" in text
    assert "[FAIL] configured_backend_is_embedding" in text
    assert "RESULT:  FAIL" in text


def test_evaluate_paths_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps(_report(
        execution_command="--backend controlled",
        backend="controlled",
        gates=dict(_ALL_TRUE_CONTROLLED),
    )), encoding="utf-8")
    bad.write_text(json.dumps(_report(
        execution_command="--backend configured",
        backend="tfidf",
        gates=dict(_ALL_TRUE_CONTROLLED, configured_backend_is_embedding=False),
    )), encoding="utf-8")

    _text, code_good = evaluate_paths([str(good)])
    assert code_good == 0
    _text, code_bad = evaluate_paths([str(good), str(bad)])
    assert code_bad == 1
    _text, code_missing = evaluate_paths([str(tmp_path / "nope.json")])
    assert code_missing == 2
