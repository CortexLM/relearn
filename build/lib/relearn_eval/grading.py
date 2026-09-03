"""Reference grading for the slices that ship with known answers.

The private holdout has no reference answers — it is judged by the frozen
teacher. The slices the image owns do have them, and they are graded here
without a network call:

* canaries — base-competence items the model must already get right, so the
  gate can tell "worse than the champion" from "broken".
* general canary — MMLU / MMMU-*style* multiple choice, off the visible score.
  Style, not the benches themselves: official items are blocklisted
  (`decontam`) and never scored or trained on here.
* agent trace — a rubric over an ordered plan, scored as the fraction of the
  rubric the answer satisfies.

Every grader returns a score in `[0, 1]`, and none of them invents one: an
unparseable answer is a zero for that item, which is a measurement, whereas a
missing item would be a hole in the document.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_LETTER = re.compile(r"\b([A-E])\b")


def normalize(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    without_punctuation = _PUNCTUATION.sub(" ", stripped.casefold())
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def grade_reference(answer: str, expected: str, aliases: tuple[str, ...] = ()) -> float:
    """1.0 when the answer contains the expected string (or an alias)."""
    haystack = f" {normalize(answer)} "
    for candidate in (expected, *aliases):
        needle = normalize(candidate)
        if needle and f" {needle} " in haystack:
            return 1.0
    return 0.0


def grade_choice(answer: str, expected_letter: str) -> float:
    """1.0 when the answer's first choice letter is the expected one."""
    wanted = expected_letter.strip().upper()[:1]
    if not wanted:
        return 0.0
    match = _LETTER.search(answer.upper())
    if match:
        return 1.0 if match.group(1) == wanted else 0.0
    condensed = answer.strip().upper()
    return 1.0 if condensed[:1] == wanted else 0.0


def grade_rubric(answer: str, must_include: tuple[str, ...], ordered: bool = True) -> float:
    """Fraction of the rubric the answer satisfies.

    `ordered` also requires the steps to appear in the rubric's order, so a
    plan that lists the right moves in an impossible sequence does not score as
    a correct trace.
    """
    if not must_include:
        return 0.0
    haystack = normalize(answer)
    hits = 0
    cursor = 0
    for step in must_include:
        needle = normalize(step)
        if not needle:
            continue
        position = haystack.find(needle, cursor if ordered else 0)
        if position < 0:
            continue
        hits += 1
        if ordered:
            cursor = position + len(needle)
    return hits / len(must_include)
