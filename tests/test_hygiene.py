"""What the image must never do.

Three properties, each of which has a way of quietly regressing:

* stdout carries the document and the markers, nothing else — in particular no
  holdout prompt, since the transcript is read, logged, and stored off the pod;
* nothing in the package can produce a score without the model or the judge;
* no secret, no teacher host, no Modal, no DFlash2.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from relearn_eval import METRICS_MARKER, OK_MARKER
from relearn_eval.cli import main

from .conftest import FakeRunner, FakeTeacher, holdout_items, make_request

PACKAGE = Path(__file__).resolve().parents[1] / "eval" / "src" / "relearn_eval"
REPO = Path(__file__).resolve().parents[1]


def package_sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


#: What the image is built from: the package, the build, the workflows. Prose
#: is excluded on purpose — a doc that says "never DFlash2" has to be able to
#: say it, and so does this test file.
BUILT_FROM_ROOTS = ("eval", ".github")
BUILT_FROM_FILES = ("pyproject.toml",)
BUILT_FROM_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".sh", ".jsonl", ""}


def built_from_files() -> list[Path]:
    paths = [REPO / name for name in BUILT_FROM_FILES]
    for root in BUILT_FROM_ROOTS:
        paths.extend(
            path
            for path in (REPO / root).rglob("*")
            if path.is_file()
            and path.suffix in BUILT_FROM_SUFFIXES
            and ".egg-info" not in str(path)
        )
    return sorted(path for path in paths if path.is_file())


def test_stdout_carries_no_holdout_prompt(tmp_path, monkeypatch, capsys):
    request = make_request(holdout_items(8))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request.to_wire()), encoding="utf-8")
    monkeypatch.setattr("relearn_eval.cli.build_runner", lambda *_: FakeRunner())
    monkeypatch.setattr("relearn_eval.cli.build_teacher", lambda *_: FakeTeacher())
    monkeypatch.setattr("relearn_eval.cli.resolve_artifact", lambda *_a, **_k: None)

    assert (
        main(
            [
                "score",
                "--request",
                str(path),
                "--out",
                str(tmp_path / "metrics.json"),
                "--workdir",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    for item in request.holdout:
        assert item.prompt not in captured.out
        assert item.prompt not in captured.err
        assert item.prompt not in (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert set(captured.out.splitlines()[1:]) == {OK_MARKER}
    assert captured.out.splitlines()[0].startswith(METRICS_MARKER)


def test_the_document_carries_ids_not_prompts(tmp_path, monkeypatch, capsys):
    request = make_request(holdout_items(4))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request.to_wire()), encoding="utf-8")
    monkeypatch.setattr("relearn_eval.cli.build_runner", lambda *_: FakeRunner())
    monkeypatch.setattr("relearn_eval.cli.build_teacher", lambda *_: FakeTeacher())
    monkeypatch.setattr("relearn_eval.cli.resolve_artifact", lambda *_a, **_k: None)
    main(
        [
            "score",
            "--request",
            str(path),
            "--out",
            str(tmp_path / "metrics.json"),
            "--workdir",
            str(tmp_path / "work"),
        ]
    )
    capsys.readouterr()
    body = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    for key in body["holdout"]:
        assert key == f"h{int(key[1:])}"


@pytest.mark.parametrize(
    "banned",
    [
        "import modal",
        "modal.com",
        "modal_token",
        "modal.App",
        "dflash",
        "flash-next",
        "libertaidai",
        "inferact",
    ],
)
def test_the_image_is_not_built_from_a_forbidden_dependency(banned: str):
    """Not Modal, not DFlash2, not the Flash variants, not an unpinned teacher."""
    for path in built_from_files():
        body = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert banned not in body, f"{path} mentions {banned}"


def test_the_prohibitions_are_written_down():
    """The docs have to be able to name what the build may not contain."""
    body = (REPO / "docs" / "EVAL-IMAGE.md").read_text(encoding="utf-8")
    assert "DFlash2" in body
    assert "no Modal" in body


def code_strings(path: Path) -> list[str]:
    """Every string literal in a module that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node not in docstrings
    ]


def test_no_teacher_endpoint_or_credential_is_baked_in():
    """No host and no key in the code, only in the operator's environment."""
    host = re.compile(r"https?://\S")
    for path in package_sources():
        for literal in code_strings(path):
            # A bare scheme is a scheme check; a scheme with a host after it
            # would be an endpoint baked into the image.
            assert not host.search(literal), f"{path}: {literal!r}"
            assert not literal.startswith("sk-"), f"{path}: {literal!r}"
    teacher = (PACKAGE / "teacher.py").read_text(encoding="utf-8")
    # The bearer token is interpolated from the environment, never a literal.
    assert 'f"Bearer {self.api_key}"' in teacher
    assert "RELEARN_TEACHER_API_KEY" in teacher


def test_the_package_ships_no_offline_scorer():
    """No hash-derived series, and no fallback that answers without the model.

    The deleted `eval/harness/eval.py` computed its numbers from
    `sha256(digest ‖ seed)`. That shape is what this test exists to keep out.
    """
    for path in package_sources():
        if path.name in {"commitment.py", "artifact.py", "images.py", "generators.py"}:
            continue  # these hash bytes for identity, not for scores
        body = path.read_text(encoding="utf-8")
        assert "hashlib" not in body, f"{path} hashes something on the scoring path"
    scoring = (PACKAGE / "scoring.py").read_text(encoding="utf-8")
    assert "runner" in scoring and "teacher" in scoring
    for word in ("sim_", "fallback_score", "random"):
        assert word not in scoring, f"scoring.py mentions {word}"


def test_the_teacher_has_no_offline_judge():
    body = (PACKAGE / "teacher.py").read_text(encoding="utf-8")
    # The deleted judge returned `1.0 if candidate.strip() else 0.0` when no
    # endpoint was configured. An unconfigured teacher must raise instead.
    assert "backend" not in body
    assert 'raise TeacherError(' in body


def test_pixel_shuffle_has_no_passthrough():
    """The shuffle either shuffles or fails.

    Returning the original bytes when the imaging runtime is missing would let
    a model that never looked at the image pass the pixel-shuffle gate.
    """
    tree = ast.parse((PACKAGE / "images.py").read_text(encoding="utf-8"))
    shuffle = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "shuffle_pixels"
    )
    returns = [
        ast.unparse(node.value)
        for node in ast.walk(shuffle)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert returns and all("body" not in expression for expression in returns), returns
