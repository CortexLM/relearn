#!/bin/sh
# Put a *real* executable at /usr/bin/relearn-eval.
#
# The harvest SSHes into the pod with a non-interactive PATH of typically
# `/usr/bin:/bin` and runs `relearn-eval score`. A symlink to wherever pip
# happened to drop the console script is a 127 when that path is not
# `/usr/local/bin` (CUDA / conda / venv bases) — and worse, `ln -sf
# /usr/local/bin/relearn-eval /usr/bin/relearn-eval` *overwrites* a working
# script pip already put in /usr/bin with a dangling link.
#
# This writes a regular file whose only job is to exec the interpreter that
# can import the package, by absolute path, so PATH does not matter after
# the process is started.
set -eu

python=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        resolved=$(command -v "$candidate")
        if "$resolved" -c "import relearn_eval" >/dev/null 2>&1; then
            python=$resolved
            break
        fi
    fi
done

if [ -z "$python" ]; then
    echo "install-cli: no python that can import relearn_eval" >&2
    echo "PATH=$PATH" >&2
    command -v python3 || true
    command -v python || true
    exit 1
fi

# Quote the interpreter so a path with spaces still works. Harvest never
# interpolates into this file; the path is baked at image build.
cat > /usr/bin/relearn-eval <<EOF
#!/bin/sh
exec '${python}' -m relearn_eval "\$@"
EOF
chmod 0755 /usr/bin/relearn-eval

# Refuse to ship a symlink. That is the live 127.
if [ -L /usr/bin/relearn-eval ]; then
    echo "install-cli: /usr/bin/relearn-eval must be a regular file, not a symlink" >&2
    ls -l /usr/bin/relearn-eval >&2
    exit 1
fi
if [ ! -f /usr/bin/relearn-eval ] || [ ! -x /usr/bin/relearn-eval ]; then
    echo "install-cli: /usr/bin/relearn-eval is not an executable file" >&2
    exit 1
fi

# The harvest's PATH, and nothing else. `--help` and `score --help` are the
# same argv shape the pod is asked for; if either is 127 here, it is 127 live.
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help >/dev/null
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval score --help >/dev/null

echo "install-cli: /usr/bin/relearn-eval -> ${python} -m relearn_eval"
