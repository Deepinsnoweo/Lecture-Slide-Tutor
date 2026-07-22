from __future__ import annotations

from slide_tutor.storage_audit import audit_storage, classify_storage_file


def test_storage_categories_cover_stage_c_artifacts(tmp_path):
    assert classify_storage_file(tmp_path / "files" / "f" / "source.pdf") == "source"
    assert classify_storage_file(tmp_path / "files" / "f" / "pages" / "p.png") == "images"
    assert classify_storage_file(tmp_path / "index" / "embeddings.npy") == "index"
    assert classify_storage_file(tmp_path / "files" / "f" / "understanding" / "p.json") == "page_generation"
    assert classify_storage_file(tmp_path / "quiz_cache_x.json") == "quiz"
    assert classify_storage_file(tmp_path / "qa_answer_history_v1.json") == "history"


def test_storage_audit_is_read_only_and_uses_conservative_thresholds(tmp_path):
    project = tmp_path / "projects" / "p"
    project.mkdir(parents=True)
    (project / "quiz_cache_x.json").write_bytes(b"quiz")
    (project / "qa_answer_history_v1.json").write_bytes(b"history")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    report = audit_storage(tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert before == after
    assert report.total_bytes == 11
    assert report.projects[0].project_id == "p"
    assert report.decisions["implement_lru_now"] is False
    assert report.decisions["implement_heavy_vector_now"] is False
    assert report.decisions["delete_legacy_data"] is False
