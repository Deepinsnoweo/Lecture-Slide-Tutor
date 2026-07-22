"""Generate a retrieval-eval gate report *from the JSON artifact*, and fail loud.

Why this module exists
----------------------
`slide_tutor.retrieval_eval` already writes every gate result and the real
backend into its output JSON (``Docs/eval/*.json``).  The failure mode this
guards against is a human/agent *hand-writing* a summary ("configured backend =
embedding, all gates pass") that silently contradicts the artifact it just
produced — e.g. a run that fell back to TF-IDF still being reported as green.

This tool does two things, both mechanical:

1. **Report from evidence, not memory.**  It reads a report JSON and prints the
   backend, case count, headline metrics, and every gate as ``PASS``/``FAIL``.
   Because the text is *derived* from the file, it cannot disagree with the file.
2. **Red gate => non-zero exit.**  If any gate is false, reference validation
   failed, or a ``--backend configured`` run did not actually run on the
   embedding backend, the process exits non-zero so "done" is blocked.

Usage
-----
    python -m slide_tutor.eval_gate_report Docs/eval/retrieval-gold-stage-b-*.json

Exit codes: 0 = every report all-green; 1 = at least one gate/validation red;
2 = usage / unreadable file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateReport:
    """A faithful, derived-from-artifact summary of one eval report."""

    path: str
    is_configured_run: bool
    backend: str
    case_count: int | None
    reference_passed: bool
    metrics: dict[str, float]
    gates: dict[str, bool]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# Metrics we surface in the one-line headline (key -> label).
_HEADLINE_METRICS = {
    "recall_at_3": "Recall@3",
    "recall_at_5": "Recall@5",
    "mrr": "MRR",
    "precision_at_5": "P@5",
    "deterministic_hit_rate": "det_hit",
}


def _is_configured_run(report: dict) -> bool:
    cmd = str(report.get("execution_command") or "")
    return "--backend configured" in cmd or "backend=configured" in cmd


def build_gate_report(report: dict, *, path: str = "<report>") -> GateReport:
    """Derive a pass/fail summary from a single parsed report dict.

    Pure function: no I/O, so it is trivially testable.  Every ``problems``
    entry is a red flag that must make the CLI exit non-zero.
    """
    evaluation = report.get("evaluation") or {}
    model_contract = (
        report.get("model_contract")
        or evaluation.get("model_contract")
        or {}
    )
    overall = evaluation.get("overall") or report.get("overall") or {}
    gates = dict(report.get("gates") or {})
    ref = report.get("reference_validation") or {}
    ref_passed = bool(ref.get("passed", False))
    configured = _is_configured_run(report)
    backend = str(model_contract.get("backend") or "unknown")

    problems: list[str] = []

    # 1. Any explicitly-false gate is a failure.
    for name, value in gates.items():
        if not value:
            problems.append(f"gate red: {name}")

    # 2. Reference validation must have passed.
    if not ref_passed:
        problems.append("reference_validation.passed is false")

    # 3. A configured run must genuinely be on the embedding backend and must
    #    carry the embedding-identity gate.  A silent TF-IDF fallback is exactly
    #    what we refuse to let pass as "configured".
    if configured:
        if "configured_backend_is_embedding" not in gates:
            problems.append(
                "configured run is missing the "
                "'configured_backend_is_embedding' gate (cannot confirm backend)"
            )
        if backend != "embedding":
            problems.append(
                f"configured run reports backend='{backend}', expected 'embedding' "
                "(likely a silent fallback)"
            )

    metrics = {
        key: overall[key]
        for key in _HEADLINE_METRICS
        if isinstance(overall.get(key), (int, float))
    }

    return GateReport(
        path=path,
        is_configured_run=configured,
        backend=backend,
        case_count=overall.get("case_count"),
        reference_passed=ref_passed,
        metrics=metrics,
        gates=gates,
        problems=problems,
    )


def format_gate_report(summary: GateReport) -> str:
    """Render one GateReport as a human-readable block."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(Path(summary.path).name)
    run_kind = "configured (--backend configured)" if summary.is_configured_run else "controlled/other"
    lines.append(f"  run:     {run_kind}")
    backend_note = ""
    if summary.is_configured_run and summary.backend != "embedding":
        backend_note = "   <-- FAIL: expected embedding"
    lines.append(f"  backend: {summary.backend}{backend_note}")
    ref = "passed" if summary.reference_passed else "FAILED"
    lines.append(f"  cases:   {summary.case_count}   reference_validation: {ref}")
    if summary.metrics:
        metric_str = "  ".join(
            f"{_HEADLINE_METRICS[k]}={summary.metrics[k]:.3f}" for k in _HEADLINE_METRICS
            if k in summary.metrics
        )
        lines.append(f"  metrics: {metric_str}")
    lines.append("  gates:")
    if not summary.gates:
        lines.append("    (no gates present)")
    for name, value in summary.gates.items():
        mark = "PASS" if value else "FAIL"
        lines.append(f"    [{mark}] {name}")
    if summary.ok:
        lines.append("  RESULT:  PASS")
    else:
        lines.append(f"  RESULT:  FAIL ({len(summary.problems)} problem(s))")
        for problem in summary.problems:
            lines.append(f"    - {problem}")
    return "\n".join(lines)


def evaluate_paths(paths: list[str]) -> tuple[str, int]:
    """Return (rendered text, exit code) for a list of report paths."""
    blocks: list[str] = []
    summaries: list[GateReport] = []
    io_error = False
    for path in paths:
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blocks.append(f"{path}: cannot read report ({type(exc).__name__}: {exc})")
            io_error = True
            continue
        summary = build_gate_report(report, path=path)
        summaries.append(summary)
        blocks.append(format_gate_report(summary))

    failed = [s for s in summaries if not s.ok]
    blocks.append("=" * 78)
    if io_error:
        verdict = "ERROR — one or more reports could not be read"
    elif failed:
        verdict = f"OVERALL: FAIL — {len(failed)} of {len(summaries)} report(s) have red gates"
    else:
        verdict = f"OVERALL: PASS — {len(summaries)} report(s) all green"
    blocks.append(verdict)

    if io_error:
        return "\n".join(blocks), 2
    return "\n".join(blocks), (1 if failed else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print retrieval-eval gate results from JSON artifacts and "
        "exit non-zero if any gate is red.",
    )
    parser.add_argument("reports", nargs="+", help="Path(s) to retrieval_eval *.json report files")
    args = parser.parse_args(argv)
    text, code = evaluate_paths(args.reports)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
