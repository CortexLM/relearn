#!/bin/sh
# Put a *regular file* at /usr/bin/relearn-image-eval or
# /usr/bin/relearn-agent-eval.
#
# The harvest SSHes into the pod with a non-interactive PATH of typically
# `/usr/bin:/bin` and runs that path under `timeout`. A symlink to wherever
# pip dropped the console script is a 127 when that path is not on PATH
# (CUDA / conda / venv) — and `ln -sf /usr/local/bin/relearn-*-eval
# /usr/bin/relearn-*-eval` overwrites a working script with a dangling link.
#
# This writes the same launcher as eval/bin/relearn-*-eval: a regular file
# whose shebang is /bin/sh (always present on the CUDA Ubuntu base and on
# slim) and which execs a python that can import the package. A shebang
# that names a missing interpreter is the same ENOENT timeout reports for
# a missing file.
#
# CHALLENGE=image|agent selects which binary. The Dockerfiles set it.
set -eu

case "${CHALLENGE:-}" in
    image)
        binary=relearn-image-eval
        module=relearn_image_eval
        ;;
    agent)
        binary=relearn-agent-eval
        module=relearn_agent_eval
        ;;
    *)
        echo "install-cli: CHALLENGE must be 'image' or 'agent', got '${CHALLENGE:-}'" >&2
        exit 1
        ;;
esac

target="/usr/bin/${binary}"

launcher=""
for candidate in \
    "/tmp/${binary}-launcher" \
    "/opt/relearn/eval/bin/${binary}" \
    "${target}"
do
    if [ -f "${candidate}" ] && [ ! -L "${candidate}" ]; then
        if head -n 1 "${candidate}" | grep -q '^#!/bin/sh'; then
            launcher=${candidate}
            break
        fi
    fi
done

if [ -n "${launcher}" ] && [ "${launcher}" != "${target}" ]; then
    install -m 0755 "${launcher}" "${target}"
elif [ -z "${launcher}" ]; then
    cat > "${target}" <<EOF
#!/bin/sh
set -eu
try() {
    py=\$1
    shift
    [ -n "\${py}" ] || return 1
    [ -x "\${py}" ] || return 1
    "\${py}" -c "import ${module}" >/dev/null 2>&1 || return 1
    exec "\${py}" -m ${module} "\$@"
}
try /opt/relearn-venv/bin/python "\$@" \\
    || try /usr/bin/python3 "\$@" \\
    || try /usr/local/bin/python3 "\$@" \\
    || try /opt/conda/bin/python "\$@" \\
    || try /opt/conda/bin/python3 "\$@" \\
    || try "\$(command -v python3 2>/dev/null || true)" "\$@" \\
    || try "\$(command -v python 2>/dev/null || true)" "\$@" \\
    || {
        echo "${binary}: no python that can import ${module}" >&2
        echo "PATH=\${PATH-}" >&2
        exit 127
    }
EOF
    chmod 0755 "${target}"
else
    chmod 0755 "${target}"
fi

# Refuse to ship a symlink. That is the live 127.
if [ -L "${target}" ]; then
    echo "install-cli: ${target} must be a regular file, not a symlink" >&2
    ls -l "${target}" >&2
    exit 1
fi
if [ ! -f "${target}" ] || [ ! -x "${target}" ]; then
    echo "install-cli: ${target} is not an executable file" >&2
    exit 1
fi

# The harvest's PATH, and nothing else. `--help` is the argv shape the
# publish job re-runs against the *pushed digest*. If this is 127 here,
# it is 127 live.
env -i PATH=/usr/bin:/bin "${target}" --help >/dev/null
env -i PATH=/usr/bin:/bin "${target}" score --help >/dev/null

echo "install-cli: ${target} is a regular file (not a symlink)"
