from __future__ import annotations

from slide_tutor.prompts import build_qa_system_prompt


def test_tutor_system_prompt_enforces_simplified_chinese_over_source_language():
    prompt = build_qa_system_prompt("简体中文")

    assert "Simplified Chinese (简体中文)" in prompt
    assert "source material must not influence the response language" in prompt
    assert "course-material tutor" in prompt


def test_translator_system_prompt_uses_canonical_language_name():
    prompt = build_qa_system_prompt("日本語", translator=True)

    assert "Japanese (日本語)" in prompt
    assert "translate all explanatory content into" in prompt
    assert "course-material translator" in prompt


def test_tutor_system_prompt_teaches_canonical_citation_examples():
    prompt = build_qa_system_prompt("简体中文")

    assert "[Lecture01.pdf, page 17]" in prompt
    assert "[SDS-HRI tutorial.pdf, page 34]" in prompt
    assert "[第17页]" in prompt
    assert "Lecture01.pdf p.17" in prompt
    assert "only pages present in the supplied context" in prompt
