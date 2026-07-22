"""Project topic identity and localized display-name helpers."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

from slide_tutor import cache
from slide_tutor.models import QuizSet, TopicRegistry, TopicRegistryEntry

TOPIC_LANGUAGES = ("简体中文", "English", "日本語")
TOPIC_REGISTRY_FILENAME = "topic_registry.json"


def normalize_topic_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    value = re.sub(r"[\s\-_—–]+", " ", value)
    value = re.sub(r"[^\w\s\u3040-\u30ff\u3400-\u9fff]", "", value)
    return " ".join(value.split())


def _registry_path(project_id: str):
    return cache.get_project_dir(project_id) / TOPIC_REGISTRY_FILENAME


def load_topic_registry(project_id: str) -> TopicRegistry:
    raw = cache.load_json(_registry_path(project_id), default=None)
    if raw:
        try:
            return TopicRegistry.model_validate(raw)
        except Exception:
            pass
    return TopicRegistry(project_id=project_id)


def save_topic_registry(registry: TopicRegistry) -> None:
    cache.save_json(_registry_path(registry.project_id), registry)


def _question_evidence_ids(question) -> set[str]:
    values: set[str] = set()
    for citation in question.citations:
        evidence_id = citation.evidence_id or (
            f"{citation.file_id}:{citation.slide_id}"
            if citation.file_id and citation.slide_id else ""
        )
        if evidence_id:
            values.add(evidence_id)
    return values


def _new_topic_key(project_id: str, normalized_name: str, evidence_ids: set[str]) -> str:
    seed = "|".join([project_id, normalized_name, *sorted(evidence_ids)])
    return f"topic-{sha256(seed.encode()).hexdigest()[:16]}"


def resolve_quiz_topics(quiz_set: QuizSet) -> QuizSet:
    """Assign stable topic keys without merging ambiguous concepts."""
    registry = load_topic_registry(quiz_set.project_id)
    entries_by_key = {entry.topic_key: entry for entry in registry.entries}

    for question in quiz_set.questions:
        names = {
            language: str(question.topic_names.get(language, "")).strip()
            for language in TOPIC_LANGUAGES
            if str(question.topic_names.get(language, "")).strip()
        }
        names.setdefault(quiz_set.language, question.topic_tag.strip())
        normalized_names = {normalize_topic_name(name) for name in names.values() if name}
        evidence_ids = _question_evidence_ids(question)

        name_matches: set[str] = set()
        evidence_matches: set[str] = set()
        for entry in registry.entries:
            known = {normalize_topic_name(name) for name in entry.names.values()}
            for aliases in entry.aliases.values():
                known.update(normalize_topic_name(alias) for alias in aliases)
            if normalized_names & known:
                name_matches.add(entry.topic_key)
            if evidence_ids and evidence_ids & set(entry.evidence_ids):
                evidence_matches.add(entry.topic_key)

        candidates = name_matches & evidence_matches if name_matches and evidence_matches else name_matches or evidence_matches
        if len(candidates) == 1:
            topic_key = next(iter(candidates))
            entry = entries_by_key[topic_key]
        else:
            normalized = normalize_topic_name(names.get(quiz_set.language, question.topic_tag)) or "topic"
            topic_key = _new_topic_key(quiz_set.project_id, normalized, evidence_ids)
            entry = entries_by_key.get(topic_key)
            if entry is None:
                entry = TopicRegistryEntry(
                    topic_key=topic_key,
                    names=dict(names),
                    evidence_ids=sorted(evidence_ids),
                    provisional=len(candidates) > 1,
                )
                registry.entries.append(entry)
                entries_by_key[topic_key] = entry

        for language, name in names.items():
            entry.names.setdefault(language, name)
        entry.evidence_ids = sorted(set(entry.evidence_ids) | evidence_ids)
        question.topic_key = entry.topic_key
        question.topic_names = dict(entry.names)
        question.topic_tag = display_topic_name(
            entry.topic_key, entry.names, quiz_set.language, question.topic_tag,
        )

    save_topic_registry(registry)
    return quiz_set


def legacy_topic_key(language: str, topic_tag: str) -> str:
    normalized = normalize_topic_name(topic_tag) or "unknown"
    return f"legacy:{language}:{normalized}"


def display_topic_name(
    topic_key: str,
    names: dict[str, str],
    language: str,
    fallback: str,
) -> str:
    del topic_key
    return names.get(language) or next((names.get(item) for item in TOPIC_LANGUAGES if names.get(item)), None) or fallback


__all__ = [
    "TOPIC_LANGUAGES", "display_topic_name", "legacy_topic_key", "load_topic_registry",
    "normalize_topic_name", "resolve_quiz_topics", "save_topic_registry",
]
