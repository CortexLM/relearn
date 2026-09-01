"""`/usr/bin/relearn-eval` must be a real file the harvest PATH can run.

Live 127 on `sha256:00839671`: the pod came RUNNING, SSH `PATH` was
`/usr/bin:/bin`, and `relearn-eval` was a dangling symlink to wherever pip
had put the console script on that base. These checks keep that shape out of
the Dockerfile; the image-contract job is what proves the built image.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_dockerfile_does_not_symlink_the_cli_into_usr_local():
    body = (REPO / "eval" / "Dockerfile").read_text(encoding="utf-8")
    assert "ln -sf /usr/local/bin/relearn-eval" not in body
    assert "install-cli.sh" in body
    assert "ENTRYPOINT" in body
    assert "/usr/bin/relearn-eval-entrypoint" in body


def test_the_installer_writes_a_regular_file_and_proves_the_harvest_path():
    body = (REPO / "eval" / "install-cli.sh").read_text(encoding="utf-8")
    commands = [
        line
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any("cat > /usr/bin/relearn-eval" in line for line in commands)
    assert not any("ln -s" in line or "ln -sf" in line for line in commands)
    assert "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help" in body
    assert "env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval score --help" in body
    assert any("-m relearn_eval" in line for line in commands)
    assert any("-L /usr/bin/relearn-eval" in line for line in commands)


def test_the_entrypoint_execs_the_usr_bin_binary():
    body = (REPO / "eval" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "exec /usr/bin/relearn-eval" in body
    assert "exec relearn-eval" not in body
