"""The entrypoint as the control plane drives it.

The harvest stages `request.json` over stdin, runs

    relearn-eval score --request request.json --out metrics.json

and then reads `metrics.json` and the markers back. So these tests care about
exactly what a pod produces on stdout and on disk — including in the failure
cases, where the pod must produce neither a sidecar nor a completion marker.
"""

from __future__ import annotations

import io
import json

import pytest

from relearn_eval import METRICS_MARKER, OK_MARKER, accept, decode_document
from relearn_eval.cli import EXIT_REFUSED, main
from relearn_eval.preflight import PreflightError
from relearn_eval.request import HarvestRequest
from relearn_eval.scoring import Slices

from .conftest import FakeRunner, FakeTeacher, holdout_items, make_request, ready


@pytest.fixture
def staged(tmp_path):
    request = make_request(holdout_items(8))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request.to_wire()), encoding="utf-8")
    return request, path


@pytest.fixture
def wired(monkeypatch):
    """Stand in for the two things the image cannot carry: model and judge.

    Patched at the preflight seam, which is where the real run proves the
    judge answers and the weights are on the pod.
    """
    runner = FakeRunner()
    teacher = FakeTeacher()
    monkeypatch.setattr("relearn_eval.cli.preflight", lambda *_a, **_k: ready(teacher))
    monkeypatch.setattr("relearn_eval.cli.build_runner", lambda *_: runner)
    return runner, teacher


def run_score(staged, tmp_path, extra: list[str] | None = None) -> int:
    _, path = staged
    return main(
        [
            "score",
            "--request",
            str(path),
            "--out",
            str(tmp_path / "metrics.json"),
            "--workdir",
            str(tmp_path / "work"),
            *(extra or []),
        ]
    )


def test_a_scored_run_prints_the_document_then_the_ok_marker(staged, tmp_path, wired, capsys):
    request, _ = staged
    assert run_score(staged, tmp_path) == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 2, out
    assert lines[0].startswith(METRICS_MARKER)
    # The completion marker is last: a consumer that truncates the transcript
    # loses diagnostics, never the ordering that makes a run acceptable.
    assert lines[1] == OK_MARKER
    accept(out, request)


def test_the_sidecar_is_one_line_with_no_trailing_newline(staged, tmp_path, wired):
    request, _ = staged
    assert run_score(staged, tmp_path) == 0

    body = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert body == body.rstrip("\n")
    assert "\n" not in body
    # Exactly how the harvest reconstructs the line.
    harvested = f"{METRICS_MARKER}{body}\n{OK_MARKER}\n"
    accept(harvested, request)
    assert decode_document(body).submission_digest == request.submission_digest


def test_a_judge_score_that_does_not_round_cleanly_still_publishes(
    staged, tmp_path, monkeypatch, capsys
):
    """Publishing rounds the scores, so the self-check compares encoded forms."""
    monkeypatch.setattr(
        "relearn_eval.cli.preflight", lambda *_a, **_k: ready(FakeTeacher(level=2 / 3))
    )
    monkeypatch.setattr("relearn_eval.cli.build_runner", lambda *_: FakeRunner())
    request, _ = staged

    assert run_score(staged, tmp_path) == 0
    accept(capsys.readouterr().out, request)
    body = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(body["holdout"].values()) == {0.666667}


def test_a_request_over_stdin_is_scored(staged, tmp_path, wired, monkeypatch, capsys):
    request, path = staged
    monkeypatch.setattr("sys.stdin", io.StringIO(path.read_text(encoding="utf-8")))
    assert (
        main(
            [
                "score",
                "--request",
                "-",
                "--out",
                str(tmp_path / "metrics.json"),
                "--workdir",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    accept(capsys.readouterr().out, request)


def test_an_unreachable_judge_refuses_without_a_marker_or_a_sidecar(
    staged, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "relearn_eval.cli.preflight",
        lambda *_a, **_k: (_ for _ in ()).throw(
            PreflightError("no judge: RELEARN_TEACHER_API_URL is unset")
        ),
    )
    assert run_score(staged, tmp_path) == EXIT_REFUSED
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "metrics.json").exists()


def test_an_unconfigured_pod_cannot_score_at_all(staged, tmp_path, monkeypatch, capsys):
    # Nothing stubbed: no teacher URL, no model runtime, no artifact store.
    for name in (
        "RELEARN_TEACHER_API_URL",
        "RELEARN_BASE_MODEL_DIR",
        "RELEARN_ARTIFACT_DIR",
        "RELEARN_ARTIFACT_URL_TEMPLATE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert run_score(staged, tmp_path) == EXIT_REFUSED
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "metrics.json").exists()


def test_a_request_for_another_image_is_refused(staged, tmp_path, wired, monkeypatch, capsys):
    monkeypatch.setenv("RELEARN_EVAL_IMAGE_DIGEST", f"sha256:{'cd' * 32}")
    assert run_score(staged, tmp_path) == EXIT_REFUSED
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "metrics.json").exists()


def test_a_tampered_request_is_refused_before_the_model_is_loaded(
    tmp_path, monkeypatch, capsys
):
    loaded: list[str] = []
    monkeypatch.setattr(
        "relearn_eval.cli.build_runner", lambda *_: loaded.append("loaded") or FakeRunner()
    )
    monkeypatch.setattr("relearn_eval.cli.preflight", lambda *_a, **_k: ready(FakeTeacher()))
    request = make_request(holdout_items(8))
    wire = request.to_wire()
    wire["holdout_commitment"] = "aa" * 32
    path = tmp_path / "request.json"
    path.write_text(json.dumps(wire), encoding="utf-8")

    assert (
        main(["score", "--request", str(path), "--out", str(tmp_path / "metrics.json")])
        == EXIT_REFUSED
    )
    assert loaded == []
    assert capsys.readouterr().out == ""


def test_the_runner_is_released_even_when_scoring_fails(staged, tmp_path, monkeypatch):
    runner = FakeRunner()

    def explode(*_args, **_kwargs):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr("relearn_eval.cli.preflight", lambda *_a, **_k: ready(FakeTeacher()))
    monkeypatch.setattr("relearn_eval.cli.build_runner", lambda *_: runner)
    monkeypatch.setattr("relearn_eval.cli.score_request", explode)
    assert run_score(staged, tmp_path) != 0
    assert runner.closed


def test_verify_accepts_this_images_own_output(staged, tmp_path, wired, capsys):
    _, path = staged
    assert run_score(staged, tmp_path) == 0
    transcript = tmp_path / "run.log"
    transcript.write_text(capsys.readouterr().out, encoding="utf-8")

    metrics = str(tmp_path / "metrics.json")
    assert main(["verify", "--request", str(path), "--metrics", metrics]) == 0
    assert main(["verify", "--request", str(path), "--transcript", str(transcript)]) == 0


def test_verify_refuses_another_runs_document(staged, tmp_path, wired, capsys):
    assert run_score(staged, tmp_path) == 0
    capsys.readouterr()

    other = make_request(holdout_items(8), submission_digest="an-earlier-run")
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other.to_wire()), encoding="utf-8")
    assert (
        main(["verify", "--request", str(other_path), "--metrics", str(tmp_path / "metrics.json")])
        == EXIT_REFUSED
    )


def test_the_boot_baseline_document_is_installable_as_a_champion_file(
    tmp_path, wired, capsys
):
    """`RELEARN_BASE_CHAMPION_FILE` is this image's output for the base model."""
    request = make_request(holdout_items(8), artifact_digest="base-relearn-champion")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request.to_wire()), encoding="utf-8")
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
    capsys.readouterr()

    body = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    # A `BaselineMeasurement` reads the same object: the envelope plus identity.
    for key in ("eval_image_digest", "holdout_commitment", "holdout", "public", "general_canary"):
        assert key in body
    assert body["holdout_commitment"] == request.holdout_commitment


def test_score_needs_no_slice_argument_beyond_the_request(staged, tmp_path, wired):
    # The image owns the public / canary / general slices, so the control plane
    # only has to deliver the holdout.
    assert run_score(staged, tmp_path) == 0
    document = decode_document((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    slices = Slices.load()
    assert len(document.measurement.public) == len(slices.public)
    staged_request = HarvestRequest.from_json(staged[1].read_text(encoding="utf-8"))
    assert "public" not in json.dumps(staged_request.to_wire())
