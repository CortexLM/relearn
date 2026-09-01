"""Finding the JSON a model meant to emit, inside everything else it said.

Both new challenges ask a model for a JSON object and get back a chain of
thought with the object somewhere in it. Q-Judger emits a thinking trace and
then a score tree; an agent emits reasoning and then a tool call. In both cases
the reasoning contains braces of its own, so "the first `{`" is the wrong
answer and "everything between the first `{` and the last `}`" is worse.

The rule is: prefer a fenced block, otherwise take the *last* balanced
top-level object, because the thing a model emits last is the thing it decided
on.
"""

from __future__ import annotations


def balanced_objects(raw: str) -> list[str]:
    """Every balanced top-level `{…}` slice, string- and escape-aware.

    String-aware matters: a brace inside a quoted argument value must not open
    a nesting level, or a tool call carrying `{"query": "a {weird} string"}`
    would be truncated.
    """
    out: list[str] = []
    depth = 0
    start = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
            if depth == 0 and index >= start:
                out.append(raw[start : index + 1])
    return out


def fenced_json(raw: str) -> str | None:
    """The body of the first ```json fence that holds an object."""
    for fence in ("```json", "```JSON"):
        head = raw.find(fence)
        if head < 0:
            continue
        after = raw[head + len(fence) :]
        tail = after.find("```")
        if tail < 0:
            continue
        body = after[:tail].strip()
        if body.startswith("{"):
            return body
    return None


def extract_json_object(raw: str) -> str | None:
    """Locate the JSON object in a reply that begins with a thinking trace."""
    fenced = fenced_json(raw)
    if fenced is not None:
        return fenced
    objects = balanced_objects(raw)
    return objects[-1] if objects else None
