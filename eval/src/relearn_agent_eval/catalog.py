"""Slices the image owns: the published episodes and the canaries.

The request carries the private holdout and nothing else, but the document has
to carry the series the gates read. A champion with no public split has no
memorization gap to measure, and a champion with no canaries has nothing to
catch the failure mode a relative comparison cannot see: a model that has lost
the ability to emit a well-formed tool call at all scores badly against the
champion, and so does the champion, so the comparison says nothing.

Both slices are operator-overridable, because only the operator knows the live
catalog:

* `RELEARN_AGENT_PUBLIC_FILE` — the published episodes, in the same record
  shape as the holdout. These are the ids miners may train on, so a live host
  should point this at the real published records; the shipped file is the CI
  and local default, in the same way the committed holdout commitment in the
  control plane's pin is the CI seal rather than the live one.
* `RELEARN_AGENT_CANARY_FILE` — the canary episodes.

The shipped episodes are synthetic. They are deliberately not lifted from any
published agent benchmark: a canary drawn from a bench cannot detect regression
on that bench, and a public split drawn from one is already in every
candidate's training mix.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from relearn_common.env import env
from relearn_common.errors import ContractError

from .request import ToolSchema, Trace


class CatalogError(ContractError):
    """A slice the image needs is missing or malformed."""


@dataclass(frozen=True)
class CanaryEpisode:
    """One episode with exactly one unambiguous correct call."""

    id: int
    goal: str
    tools: tuple[ToolSchema, ...]
    tool: str
    arguments: dict[str, object] = field(default_factory=dict)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(schema.name for schema in self.tools)


def _read_lines(name: str, override_env: str) -> Iterator[dict[str, object]]:
    override = env(override_env)
    if override:
        path = Path(override)
        if not path.is_file():
            raise CatalogError(f"{override_env} {override} is not a file")
        body = path.read_text(encoding="utf-8")
    else:
        body = (
            resources.files(__package__).joinpath(f"catalog/{name}").read_text(encoding="utf-8")
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


def _refuse_duplicates(items: Sequence[object], name: str) -> None:
    seen: set[int] = set()
    for item in items:
        item_id = int(item.id)  # type: ignore[attr-defined]
        if item_id in seen:
            raise CatalogError(f"duplicate {name} id {item_id}")
        seen.add(item_id)


def public_episodes() -> tuple[Trace, ...]:
    """The published split, in the same record shape as the holdout."""
    traces = tuple(
        Trace.from_wire(body)
        for body in _read_lines("public.jsonl", "RELEARN_AGENT_PUBLIC_FILE")
    )
    if not traces:
        raise CatalogError("public split is empty; the memorization gap gate cannot run")
    _refuse_duplicates(traces, "public")
    for trace in traces:
        trace.validate(where="public")
    return traces


def canary_episodes() -> tuple[CanaryEpisode, ...]:
    """The shipped single-call episodes."""
    episodes: list[CanaryEpisode] = []
    for body in _read_lines("canaries.jsonl", "RELEARN_AGENT_CANARY_FILE"):
        raw_tools = body.get("tools") or []
        if not isinstance(raw_tools, list) or not raw_tools:
            raise CatalogError(f"canary {body.get('id', '?')} offers no tools")
        arguments = body.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise CatalogError(f"canary {body.get('id', '?')} arguments are not an object")
        tool = str(body.get("tool", "") or "").strip()
        tools = tuple(ToolSchema.from_wire(item) for item in raw_tools)
        if tool not in {schema.name for schema in tools}:
            raise CatalogError(
                f"canary {body.get('id', '?')} expects {tool!r}, which is not in its schema"
            )
        episodes.append(
            CanaryEpisode(
                id=int(body.get("id", 0) or 0),
                goal=str(body.get("goal", "") or ""),
                tools=tools,
                tool=tool,
                arguments=dict(arguments),
            )
        )
    if not episodes:
        raise CatalogError("canary slice is empty")
    _refuse_duplicates(episodes, "canaries")
    return tuple(episodes)
