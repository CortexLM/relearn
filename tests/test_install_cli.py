"""`/usr/bin/relearn-eval` must be a real file the harvest PATH can run.

Live 127 on `sha256:00839671` and again on `sha256:86240d76`: the pod came
RUNNING, SSH `PATH` was `/usr/bin:/bin`, and `timeout` could not exec
`/usr/bin/relearn-eval`. A dangling symlink, or a shebang whose interpreter
is missing, is the same ENOENT. These checks keep that shape out of both
Dockerfiles; the publish scoring job is what proves the *pushed digest*.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_contract_dockerfile_does_not_symlink_the_cli_into_usr_local():
    body = (REPO / "eval" / "Dockerfile").read_text(encoding="utf-8")
    assert "ln -sf /usr/local/bin/relearn-eval" not in body
    assert "install-cli.sh" in body
    assert "COPY eval/bin/relearn-eval /usr/bin/relearn-eval" in body
    assert "ENTRYPOINT" in body
    assert "/usr/bin/relearn-eval-entrypoint" in body


def test_the_scoring_dockerfile_is_cuda_and_installs_a_regular_file():
    body = (REPO / "eval" / "Dockerfile.scoring").read_text(encoding="utf-8")
    assert "nvidia/cuda" in body
    assert "ln -sf" not in body
    assert "ln -s" not in body
    assert "install-cli.sh" in body
    assert "COPY eval/bin/relearn-eval /usr/bin/relearn-eval" in body
    assert "/usr/bin/relearn-eval-entrypoint" in body
    assert "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help" in body
    # Live cbc4bbb8: Qwen3VLVideoProcessor requires torchvision, and auto
    # backend fell through to transformers because vllm was never installed.
    assert '".[runtime,vllm]"' in body
    assert "import torchvision; import vllm" in body


def test_the_runtime_extra_includes_torchvision():
    body = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "torchvision>=0.19" in body
    assert "pillow>=10.3" in body
    assert 'vllm = ["vllm>=0.6"]' in body


def test_the_committed_launcher_is_posix_sh():
    path = REPO / "eval" / "bin" / "relearn-eval"
    body = path.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/sh")
    assert "\r" not in body
    assert "-m relearn_eval" in body
    assert "ln -s" not in body
    assert path.read_bytes()[:2] != b"\x7fE"


def test_the_installer_writes_a_regular_file_and_proves_the_harvest_path():
    body = (REPO / "eval" / "install-cli.sh").read_text(encoding="utf-8")
    commands = [
        line
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    writes_file = any(
        "cat > /usr/bin/relearn-eval" in line or "install -m 0755" in line
        for line in commands
    )
    assert writes_file
    assert not any("ln -s" in line or "ln -sf" in line for line in commands)
    assert "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help" in body
    assert "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval score --help" in body
    assert any("-m relearn_eval" in line for line in commands)
    assert any("-L /usr/bin/relearn-eval" in line for line in commands)


def test_the_entrypoint_execs_the_usr_bin_binary():
    body = (REPO / "eval" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "exec /usr/bin/relearn-eval" in body
    assert "exec relearn-eval" not in body


def test_publish_scoring_job_pulls_the_digest_it_just_pushed():
    body = (REPO / ".github" / "workflows" / "publish-eval-image.yml").read_text(
        encoding="utf-8"
    )
    assert "file: eval/Dockerfile.scoring" in body
    assert (
        "test -f /usr/bin/relearn-eval && test -x /usr/bin/relearn-eval && "
        "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help"
    ) in body
    pull_at = body.index("pull the published digest and prove harvest PATH")
    report_at = body.index("report the digest to pin")
    assert pull_at < report_at
