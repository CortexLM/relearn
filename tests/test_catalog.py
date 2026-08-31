"""The slices the image ships, and the ones the operator can replace."""

from __future__ import annotations

import json

import pytest

from relearn_eval import catalog
from relearn_eval.decontam import contaminated_stems, is_contaminated
from relearn_eval.grading import grade_choice


def test_the_public_slice_covers_the_pinned_public_ids():
    items = catalog.public_items()
    assert {item.id for item in items} == set(range(1, 41))
    assert all(item.task == "text" for item in items)
    assert all(item.prompt.strip() for item in items)


def test_every_shipped_slice_is_present_and_unique():
    for load in (
        catalog.public_items,
        catalog.canary_items,
        catalog.general_canary_items,
        catalog.agent_trace_items,
    ):
        items = load()
        assert items
        assert len({item.id for item in items}) == len(items)


def test_no_shipped_item_names_an_official_bench():
    """The general canary is bench-*style*, never the bench.

    A canary lifted from a benchmark cannot detect regression on it, and
    training on official items is out of bounds for miners too.
    """
    bodies = [item.prompt for item in catalog.public_items()]
    bodies += [f"{item.prompt} {item.answer}" for item in catalog.canary_items()]
    for item in catalog.general_canary_items():
        bodies.append(item.rendered())
    for item in catalog.agent_trace_items():
        bodies.append(f"{item.prompt} {' '.join(item.must_include)}")
    for body in bodies:
        assert not is_contaminated(body), contaminated_stems(body)


def test_the_general_canary_answer_key_is_spread_across_the_letters():
    """A constant key would score a model that always answers the same letter."""
    items = catalog.general_canary_items()
    counts: dict[str, int] = {}
    for item in items:
        counts[item.answer] = counts.get(item.answer, 0) + 1
        assert item.answer in "ABCDE"
        assert len(item.choices) >= 2
    assert len(counts) >= 4
    assert max(counts.values()) <= len(items) // 2


def test_the_general_canary_renders_lettered_choices():
    item = catalog.general_canary_items()[0]
    rendered = item.rendered()
    assert "A. " in rendered and "B. " in rendered
    assert grade_choice(f"{item.answer}", item.answer) == 1.0


def test_the_operator_can_replace_a_slice(tmp_path, monkeypatch):
    path = tmp_path / "public.jsonl"
    path.write_text(
        json.dumps({"id": 7, "prompt": "an operator public item", "dataset_id": "live"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RELEARN_PUBLIC_FILE", str(path))
    items = catalog.public_items()
    assert [item.id for item in items] == [7]


def test_a_missing_or_broken_override_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("RELEARN_PUBLIC_FILE", str(tmp_path / "nope.jsonl"))
    with pytest.raises(catalog.CatalogError, match="is not a file"):
        catalog.public_items()

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    monkeypatch.setenv("RELEARN_PUBLIC_FILE", str(broken))
    with pytest.raises(catalog.CatalogError, match="not JSON"):
        catalog.public_items()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    monkeypatch.setenv("RELEARN_PUBLIC_FILE", str(empty))
    with pytest.raises(catalog.CatalogError, match="public split is empty"):
        catalog.public_items()


def test_a_canary_without_an_answer_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "canaries.jsonl"
    path.write_text(json.dumps({"id": 1, "prompt": "what is 2 + 2?"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("RELEARN_CANARY_FILE", str(path))
    with pytest.raises(catalog.CatalogError, match="has no answer"):
        catalog.canary_items()
