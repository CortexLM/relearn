#!/bin/sh
# Put a *regular file* at /usr/bin/relearn-eval.
#
# The harvest SSHes into the pod with a non-interactive PATH of typically
# `/usr/bin:/bin` and runs `/usr/bin/relearn-eval` under `timeout`. A
# symlink to wherever pip dropped the console script is a 127 when that
# path is not on PATH (CUDA / conda / venv) — and `ln -sf
# /usr/local/bin/relearn-eval /usr/bin/relearn-eval` overwrites a working
# script with a dangling link.
#
# This writes the same launcher as eval/bin/relearn-eval: a regular file
# whose shebang is /bin/sh (always present on the CUDA Ubuntu base and on
# slim) and which execs a python that can import the package. A shebang
# that names a missing interpreter is the same ENOENT timeout reports for
# a missing file.
set -eu

launcher=""
for candidate in \
    /tmp/relearn-eval-launcher \
    /opt/relearn/eval/bin/relearn-eval \
    /usr/bin/relearn-eval
do
    if [ -f "${candidate}" ] && [ ! -L "${candidate}" ]; then
        if head -n 1 "${candidate}" | grep -q '^#!/bin/sh'; then
            launcher=${candidate}
            break
        fi
    fi
done

if [ -n "${launcher}" ] && [ "${launcher}" != /usr/bin/relearn-eval ]; then
    install -m 0755 "${launcher}" /usr/bin/relearn-eval
elif [ -z "${launcher}" ]; then
    cat > /usr/bin/relearn-eval <<'EOF'
#!/bin/sh
set -eu
try() {
    py=$1
    shift
    [ -n "${py}" ] || return 1
    [ -x "${py}" ] || return 1
    "${py}" -c "import relearn_eval" >/dev/null 2>&1 || return 1
    exec "${py}" -m relearn_eval "$@"
}
try /opt/relearn-venv/bin/python "$@" \
    || try /usr/bin/python3 "$@" \
    || try /usr/local/bin/python3 "$@" \
    || try /opt/conda/bin/python "$@" \
    || try /opt/conda/bin/python3 "$@" \
    || try "$(command -v python3 2>/dev/null || true)" "$@" \
    || try "$(command -v python 2>/dev/null || true)" "$@" \
    || {
        echo "relearn-eval: no python that can import relearn_eval" >&2
        echo "PATH=${PATH-}" >&2
        exit 127
    }
EOF
    chmod 0755 /usr/bin/relearn-eval
else
    chmod 0755 /usr/bin/relearn-eval
fi

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

# The harvest's PATH, and nothing else. `--help` is the argv shape the
# publish job re-runs against the *pushed digest*. If this is 127 here,
# it is 127 live.
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help >/dev/null
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval score --help >/dev/null

echo "install-cli: /usr/bin/relearn-eval is a regular file (not a symlink)"
