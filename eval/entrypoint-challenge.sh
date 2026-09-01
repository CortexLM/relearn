#!/bin/sh
# Entrypoint for the Relearn Image and Relearn Agent eval images.
#
#   serve            keep the pod reachable over SSH so the harvest can stage
#                    `request.json` and run the scorer (the pod's default)
#   score … | verify … | --help
#                    run this image's scorer directly, which is how CI and a
#                    local operator drive it
#
# The scorer is whichever binary the build linked as `relearn-challenge-eval`:
# `relearn-image-eval` or `relearn-agent-eval`. A pod runs exactly one
# challenge, so there is nothing to choose at runtime.
#
# Public keys arrive as pod environment, never baked into the image. No private
# key, judge endpoint, or API token is written here.
set -eu

install_authorized_keys() {
    keys=""
    for name in PUBLIC_KEY SSH_PUBLIC_KEY SSH_PUBLIC_KEYS LIUM_SSH_PUBLIC_KEY; do
        value=$(printenv "$name" 2>/dev/null || true)
        if [ -n "$value" ]; then
            keys="${keys}${value}
"
        fi
    done
    if [ -z "$keys" ]; then
        return 0
    fi
    mkdir -p /root/.ssh
    printf '%s' "$keys" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
}

serve() {
    install_authorized_keys
    if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
        ssh-keygen -A
    fi
    mkdir -p /run/sshd
    # Keys only: a password-authenticated pod would be reachable by anyone who
    # can see the provider's address.
    /usr/sbin/sshd -D -e \
        -o PermitRootLogin=prohibit-password \
        -o PasswordAuthentication=no \
        -o KbdInteractiveAuthentication=no
}

case "${1:-serve}" in
    serve)
        serve
        ;;
    *)
        exec relearn-challenge-eval "$@"
        ;;
esac
