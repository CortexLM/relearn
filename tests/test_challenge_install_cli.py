"""`/usr/bin/relearn-image-eval` and `/usr/bin/relearn-agent-eval` must be
real files the harvest PATH can run.

Live 127 on the LLM image (`sha256:00839671`, then `sha256:86240d76`) was
first read as a missing `/usr/bin/relearn-eval` under SSH `PATH=/usr/bin:/bin`.
A dangling symlink, or a shebang whose interpreter is missing, is the same
ENOENT. These checks keep that shape out of both challenge Dockerfiles so
the binaries exist once harvest is wired.

That PATH-clean image is **not** the live 127 fix by itself.
`LiumClient::provision` ignores `InstanceSpec.image_digest` and rents
`prism-recipe-v10`. A published digest is not live-ready until the
control plane template is `pin.image@digest`.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_contract_dockerfile_does_not_symlink_the_cli_into_usr_local() -> None:
    body = (REPO / "eval" / "Dockerfile.challenge").read_text(encoding="utf-8")
    assert "ln -sf" not in body
    assert "ln -s" not in body
    assert "install-cli.sh" in body
    assert "COPY eval/bin/relearn-image-eval /usr/bin/relearn-image-eval" in body
    assert "COPY eval/bin/relearn-agent-eval /usr/bin/relearn-agent-eval" in body
    assert "ENTRYPOINT" in body
    assert "/usr/bin/relearn-challenge-entrypoint" in body
    assert "env -i PATH=/usr/bin:/bin" in body


def test_the_scoring_dockerfile_is_cuda_and_installs_a_regular_file() -> None:
    body = (REPO / "eval" / "Dockerfile.scoring").read_text(encoding="utf-8")
    assert "nvidia/cuda" in body
    assert "ln -sf" not in body
    assert "ln -s" not in body
    assert "install-cli.sh" in body
    assert "COPY eval/bin/relearn-image-eval /usr/bin/relearn-image-eval" in body
    assert "COPY eval/bin/relearn-agent-eval /usr/bin/relearn-agent-eval" in body
    assert "/usr/bin/relearn-challenge-entrypoint" in body
    assert "env -i PATH=/usr/bin:/bin" in body
    assert "/usr/bin/relearn-image-eval" in body
    assert "/usr/bin/relearn-agent-eval" in body
    assert "/opt/relearn-venv" in body


def test_the_committed_image_launcher_is_posix_sh() -> None:
    path = REPO / "eval" / "bin" / "relearn-image-eval"
    body = path.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/sh")
    assert "\r" not in body
    assert "-m relearn_image_eval" in body
    assert "ln -s" not in body
    assert path.read_bytes()[:2] != b"\x7fE"


def test_the_committed_agent_launcher_is_posix_sh() -> None:
    path = REPO / "eval" / "bin" / "relearn-agent-eval"
    body = path.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/sh")
    assert "\r" not in body
    assert "-m relearn_agent_eval" in body
    assert "ln -s" not in body
    assert path.read_bytes()[:2] != b"\x7fE"


def test_the_installer_writes_a_regular_file_and_proves_the_harvest_path() -> None:
    body = (REPO / "eval" / "install-cli.sh").read_text(encoding="utf-8")
    commands = [
        line
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    writes_file = any(
        "cat > \"${target}\"" in line or "install -m 0755" in line for line in commands
    )
    assert writes_file
    assert not any("ln -s" in line or "ln -sf" in line for line in commands)
    assert 'env -i PATH=/usr/bin:/bin "${target}" --help' in body
    assert 'env -i PATH=/usr/bin:/bin "${target}" score --help' in body
    assert any("-m ${module}" in line for line in commands)
    assert any("-L \"${target}\"" in line or '-L "${target}"' in line for line in commands)


def test_the_entrypoint_execs_the_usr_bin_binary() -> None:
    body = (REPO / "eval" / "entrypoint-challenge.sh").read_text(encoding="utf-8")
    assert "/usr/bin/relearn-image-eval" in body
    assert "/usr/bin/relearn-agent-eval" in body
    assert "exec relearn-challenge-eval" not in body
    assert "exec relearn-image-eval" not in body
    assert "exec relearn-agent-eval" not in body
    assert "PATH=/usr/bin:/bin" in body


def test_publish_scoring_job_pulls_the_digest_it_just_pushed() -> None:
    body = (REPO / ".github" / "workflows" / "publish-challenge-images.yml").read_text(
        encoding="utf-8"
    )
    assert "file: eval/Dockerfile.scoring" in body
    assert "file: eval/Dockerfile.challenge" in body
    assert (
        "test -f /usr/bin/${BINARY} && test -x /usr/bin/${BINARY} && "
        "env -i PATH=/usr/bin:/bin /usr/bin/${BINARY} --help"
    ) in body
    assert 'test ! -L /usr/bin/${BINARY}' in body
    pull_at = body.index("pull the published digest and prove harvest PATH")
    report_at = body.index("report the digest to pin")
    assert pull_at < report_at


def test_ci_contract_job_proves_harvest_path_on_the_slim_image() -> None:
    body = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "harvest PATH finds a real" in body
    assert "PATH=/usr/bin:/bin" in body
    assert "! test -L /usr/bin/${{ matrix.binary }}" in body


def test_a_published_digest_is_not_called_live_ready() -> None:
    # Cortex harvest does not rent ghcr.io/cortexlm/relearn-*-eval today.
    # Claiming a digest is live-ready would hide the template gap.
    for rel in (
        "README.md",
        "docs/IMAGE-EVAL-IMAGE.md",
        "docs/AGENT-EVAL-IMAGE.md",
        "docs/EVAL-IMAGE.md",
        ".github/workflows/publish-challenge-images.yml",
        "eval/Dockerfile.scoring",
    ):
        body = (REPO / rel).read_text(encoding="utf-8")
        assert "not live-ready" in body, rel
        assert "prism-recipe-v10" in body, rel
        assert "pin.image@digest" in body, rel
