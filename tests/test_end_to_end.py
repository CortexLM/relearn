"""The whole run, with a real model and a real HTTP judge.

This is the test the live failure asked for. Everything else in the suite
patches the model and the judge; here a tiny transformers model is loaded for
real and a stub judge answers over a real socket, so what is under test is the
thing the harvest actually invokes:

    relearn-eval score --request request.json --out metrics.json

must exit 0, write a one-line `metrics.json`, print `RELEARN_METRICS=` and
`RELEARN_EVAL_OK` on its own stdout, and produce a document `verify` accepts —
for a champion-baseline request with no miner artifact.

Skipped when the model runtime is absent, which is how the contract-only image
and a plain dev box see it. CI installs a CPU torch for one job so this runs
on every change.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import typing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from relearn_eval import METRICS_MARKER, OK_MARKER, accept, decode_document
from relearn_eval.contract import BASE_CHAMPION_ARTIFACT, BASE_CHAMPION_RUN
from relearn_eval.request import HarvestRequest

from .conftest import holdout_items, make_request

torch = pytest.importorskip("torch", reason="the model runtime is a build extra")
pytest.importorskip("transformers", reason="the model runtime is a build extra")

#: A few-hundred-kilobyte real model. Its answers are gibberish, which is the
#: point: this test is about the run completing, not about scores being good.
TINY_MODEL = os.environ.get("RELEARN_TEST_MODEL", "hf-internal-testing/tiny-random-gpt2")


class StubJudge(BaseHTTPRequestHandler):
    """An OpenAI-compatible judge that scores everything the same."""

    score = 0.62
    delay = 0.0
    seen: typing.ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        if StubJudge.delay:
            time.sleep(StubJudge.delay)
        try:
            StubJudge.seen.append(json.loads(body))
        except json.JSONDecodeError:
            StubJudge.seen.append({})
        payload = json.dumps(
            {"choices": [{"message": {"content": json.dumps({"score": StubJudge.score})}}]}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        return

    def handle_one_request(self) -> None:
        # A killed run drops the connection mid-reply; that is the case under
        # test, not a stub failure worth printing.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


@pytest.fixture
def judge():
    StubJudge.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubJudge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


@pytest.fixture
def pod(judge, tmp_path, monkeypatch):
    """Environment of a primed pod: a judge, a warm model, a small decode."""
    monkeypatch.setenv("RELEARN_TEACHER_API_URL", judge)
    monkeypatch.setenv("RELEARN_TEACHER_API_KEY", "stub-key")
    monkeypatch.setenv("RELEARN_EVAL_BACKEND", "transformers")
    monkeypatch.setenv("RELEARN_MAX_NEW_TOKENS", "8")
    monkeypatch.setenv("RELEARN_ALLOW_MODEL_DOWNLOAD", "1")
    monkeypatch.setenv("RELEARN_LOG_LEVEL", "INFO")
    return tmp_path


def baseline_request(size: int = 4, base_model: str = TINY_MODEL) -> HarvestRequest:
    """What `boot_base_champion` asks a live pod for.

    `base_model` comes from the request, as it does live: the image scores the
    pinned base, and has no flag to point it at some other weights.
    """
    return make_request(
        holdout_items(size),
        submission_digest=BASE_CHAMPION_RUN,
        artifact_digest=BASE_CHAMPION_ARTIFACT,
        base_model=base_model,
    )


def stage(request: HarvestRequest, root: Path) -> Path:
    path = root / "request.json"
    path.write_text(json.dumps(request.to_wire()), encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the entrypoint as a process, the way the pod does."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "relearn_eval", *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def test_a_champion_baseline_request_scores_and_prints_the_markers(pod):
    request = baseline_request()
    staged = stage(request, pod)
    metrics = pod / "metrics.json"

    done = run_cli(
        "score",
        "--request",
        str(staged),
        "--out",
        str(metrics),
        "--workdir",
        str(pod / "work"),
    )

    assert done.returncode == 0, done.stderr[-4000:]

    # 2. The markers are on the process's own stdout, in order, last.
    lines = done.stdout.splitlines()
    assert lines[0].startswith(METRICS_MARKER), done.stdout[:400]
    assert lines[-1] == OK_MARKER
    accept(done.stdout, request)

    # 1. And the sidecar the harvest cats is one line with no trailing newline.
    body = metrics.read_text(encoding="utf-8")
    assert "\n" not in body
    accept(f"{METRICS_MARKER}{body}\n{OK_MARKER}\n", request)

    document = decode_document(body)
    assert document.artifact_digest == BASE_CHAMPION_ARTIFACT
    assert len(document.measurement.holdout) == len(request.holdout)
    assert document.measurement.public
    assert document.measurement.general_canary
    assert StubJudge.seen, "the judge was never called"
    # The judge is asked about the answers, never handed weights.
    for call in StubJudge.seen:
        assert call.get("model") == "glm-5.3"


def test_the_documents_verify_command_accepts_the_run(pod):
    request = baseline_request()
    staged = stage(request, pod)
    metrics = pod / "metrics.json"
    assert (
        run_cli(
            "score",
            "--request",
            str(staged),
            "--out",
            str(metrics),
            "--workdir",
            str(pod / "work"),
        ).returncode
        == 0
    )
    checked = run_cli("verify", "--request", str(staged), "--metrics", str(metrics))
    assert checked.returncode == 0, checked.stderr[-2000:]


def test_a_pod_with_no_judge_says_so_once_and_writes_nothing(pod, monkeypatch):
    monkeypatch.delenv("RELEARN_TEACHER_API_URL")
    staged = stage(baseline_request(), pod)
    metrics = pod / "metrics.json"

    done = run_cli(
        "score", "--request", str(staged), "--out", str(metrics), "--workdir", str(pod / "work")
    )

    assert done.returncode != 0
    assert done.stdout == ""
    assert not metrics.exists()
    assert "no judge" in done.stderr
    assert "RELEARN_TEACHER_API_URL" in done.stderr
    # One stated reason, not a stack trace and not silence.
    reasons = [line for line in done.stderr.splitlines() if " ERROR " in line]
    assert len(reasons) == 1, done.stderr[-2000:]


def test_a_pod_without_the_weights_says_so_before_it_spends_the_run(pod, monkeypatch):
    monkeypatch.delenv("RELEARN_ALLOW_MODEL_DOWNLOAD")
    staged = stage(baseline_request(base_model="an-org/a-model-this-pod-has-never-seen"), pod)
    metrics = pod / "metrics.json"

    done = run_cli(
        "score",
        "--request",
        str(staged),
        "--out",
        str(metrics),
        "--workdir",
        str(pod / "work"),
    )

    assert done.returncode != 0
    assert done.stdout == ""
    assert not metrics.exists()
    assert "no model" in done.stderr
    assert "RELEARN_BASE_MODEL_DIR" in done.stderr


def test_a_killed_run_states_the_phase_it_died_in(pod, monkeypatch):
    """The live symptom: the harvest's `timeout` fires and nothing is printed.

    A signalled run has to leave a reason in the log tail, because the tail is
    the operator's whole diagnosis once the pod is gone.
    """
    StubJudge.delay = 0.5
    monkeypatch.setenv("RELEARN_JUDGE_CONCURRENCY", "1")
    staged = stage(baseline_request(size=60), pod)
    metrics = pod / "metrics.json"
    # stderr to a file, as the harvest does (`> run.log 2>&1`), so nothing
    # competes for the pipe while the run is still going.
    log = pod / "run.log"
    try:
        with log.open("w", encoding="utf-8") as sink:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    "-m",
                    "relearn_eval",
                    "score",
                    "--request",
                    str(staged),
                    "--out",
                    str(metrics),
                    "--workdir",
                    str(pod / "work"),
                ],
                stdout=subprocess.PIPE,
                stderr=sink,
                text=True,
            )
            # Signal it mid-holdout, which is where a real overrun happens.
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if "phase: holdout" in log.read_text(encoding="utf-8"):
                    break
                assert process.poll() is None, log.read_text(encoding="utf-8")[-2000:]
                time.sleep(0.2)
            process.send_signal(signal.SIGTERM)
            stdout, _ = process.communicate(timeout=180)
        stderr = log.read_text(encoding="utf-8")
    finally:
        StubJudge.delay = 0.0

    assert process.returncode != 0
    assert OK_MARKER not in stdout
    assert METRICS_MARKER not in stdout
    assert not metrics.exists()
    assert "terminated by SIGTERM" in stderr, stderr[-2000:]
    assert "holdout" in stderr
    assert "run timeout" in stderr


def test_a_run_that_cannot_fit_the_budget_refuses_between_phases(pod, monkeypatch):
    monkeypatch.setenv("RELEARN_RUN_BUDGET_SECS", "0.001")
    staged = stage(baseline_request(), pod)
    metrics = pod / "metrics.json"

    done = run_cli(
        "score", "--request", str(staged), "--out", str(metrics), "--workdir", str(pod / "work")
    )

    assert done.returncode != 0
    assert done.stdout == ""
    assert not metrics.exists()
    assert "run budget" in done.stderr
    assert "RELEARN_RUN_BUDGET_SECS" in done.stderr


def test_preflight_answers_without_scoring(pod):
    staged = stage(baseline_request(), pod)
    done = run_cli("preflight", "--request", str(staged), "--workdir", str(pod / "work"))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "this pod can score" in done.stderr
    assert not (pod / "metrics.json").exists()
