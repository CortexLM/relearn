"""Slices the image owns: public, canaries, general canary, agent trace.

The request carries the private holdout and nothing else, but the document has
to carry the series the gates read — a champion with no public split or no
general-bench canary is refused at boot on the control plane, so an image that
omitted them could never produce a usable score.

Two of these slices are operator-overridable, because only the operator knows
the live catalog:

* `RELEARN_PUBLIC_FILE` — the published split (same record shape as the
  holdout). The ids the pin publishes are the ones miners may train on, so a
  live host should point this at the real public records; the shipped file is
  the CI / local default, exactly like the committed holdout commitment in
  `config/relearn-pin.toml` is the CI seal rather than the live one.
* `RELEARN_CANARY_FILE`, `RELEARN_GENERAL_CANARY_FILE`,
  `RELEARN_AGENT_TRACE_FILE` — same idea for the graded slices.

The shipped items are synthetic and deliberately not drawn from any official
benchmark: `decontam` blocks official stems, and a canary lifted from MMLU
would put the bench inside the thing that is supposed to detect bench
regression.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .contract import ContractError
from .request import HoldoutItem


class CatalogError(ContractError):
    """A slice the image needs is missing or malformed."""


@dataclass(frozen=True)
class GradedItem:
    """One item with a reference answer."""

    id: int
    prompt: str
    answer: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChoiceItem:
    """One multiple-choice item, graded by letter."""

    id: int
    prompt: str
    choices: tuple[str, ...]
    answer: str

    def rendered(self) -> str:
        lettered = "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip("ABCDE", self.choices, strict=False)
        )
        return (
            f"{self.prompt}\n{lettered}\n"
            "Answer with the single letter of the correct choice."
        )


@dataclass(frozen=True)
class TraceItem:
    """One agent-trace task, graded by rubric."""

    id: int
    prompt: str
    must_include: tuple[str, ...]


def _read_lines(name: str, override_env: str) -> Iterator[dict[str, object]]:
    override = os.environ.get(override_env, "").strip()
    if override:
        path = Path(override)
        if not path.is_file():
            raise CatalogError(f"{override_env} {override} is not a file")
        body = path.read_text(encoding="utf-8")
    else:
        body = resources.files(__package__).joinpath(f"catalog/{name}").read_text(
            encoding="utf-8"
        )
    for number, line in enumerate(body.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"{name}:{number} is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise CatalogError(f"{name}:{number} is not an object")
        yield parsed


def _require(body: dict[str, object], key: str, name: str) -> object:
    if key not in body or body[key] in (None, ""):
        raise CatalogError(f"{name} item {body.get('id', '?')} has no {key}")
    return body[key]


def public_items() -> tuple[HoldoutItem, ...]:
    """The published split, in the same record shape as the holdout."""
    items = tuple(
        HoldoutItem.from_wire(body) for body in _read_lines("public.jsonl", "RELEARN_PUBLIC_FILE")
    )
    if not items:
        raise CatalogError("public split is empty; the memorization gap gate cannot run")
    _refuse_duplicates(items, "public")
    return items


def canary_items() -> tuple[GradedItem, ...]:
    items = tuple(
        GradedItem(
            id=int(_require(body, "id", "canaries")),  # type: ignore[arg-type]
            prompt=str(_require(body, "prompt", "canaries")),
            answer=str(_require(body, "answer", "canaries")),
            aliases=tuple(str(alias) for alias in body.get("aliases", ()) or ()),
        )
        for body in _read_lines("canaries.jsonl", "RELEARN_CANARY_FILE")
    )
    if not items:
        raise CatalogError("canary slice is empty")
    _refuse_duplicates(items, "canaries")
    return items


def general_canary_items() -> tuple[ChoiceItem, ...]:
    items = tuple(
        ChoiceItem(
            id=int(_require(body, "id", "general_canary")),  # type: ignore[arg-type]
            prompt=str(_require(body, "prompt", "general_canary")),
            choices=tuple(str(choice) for choice in _require(body, "choices", "general_canary")),  # type: ignore[call-overload]
            answer=str(_require(body, "answer", "general_canary")),
        )
        for body in _read_lines("general_canary.jsonl", "RELEARN_GENERAL_CANARY_FILE")
    )
    if not items:
        raise CatalogError("general-bench canary slice is empty")
    _refuse_duplicates(items, "general_canary")
    return items


def agent_trace_items() -> tuple[TraceItem, ...]:
    items = tuple(
        TraceItem(
            id=int(_require(body, "id", "agent_trace")),  # type: ignore[arg-type]
            prompt=str(_require(body, "prompt", "agent_trace")),
            must_include=tuple(
                str(step) for step in _require(body, "must_include", "agent_trace")  # type: ignore[call-overload]
            ),
        )
        for body in _read_lines("agent_trace.jsonl", "RELEARN_AGENT_TRACE_FILE")
    )
    if not items:
        raise CatalogError("agent-trace slice is empty")
    _refuse_duplicates(items, "agent_trace")
    return items


def _refuse_duplicates(items: Sequence[object], name: str) -> None:
    seen: set[int] = set()
    for item in items:
        item_id = int(item.id)
        if item_id in seen:
            raise CatalogError(f"duplicate {name} id {item_id}")
        seen.add(item_id)
