# syntax=docker/dockerfile:1.7
#
# Relearn LLM eval image. Cortex pins the *digest* of a build of this file in
# `config/relearn-pin.toml`; a floating tag is never the pin.
#
# Build context is the repository root:
#
#   docker build -f eval/Dockerfile -t relearn-eval:dev .
#
# The model runtime is a build argument rather than a hard-coded base, because
# the pod's accelerator decides it:
#
#   # contract-only build: fast, no CUDA wheels. Cannot score, and says so.
#   docker build -f eval/Dockerfile --build-arg WITH_RUNTIME=0 .
#
#   # scoring image: CUDA base, published and pin-checked by
#   # .github/workflows/publish-eval-image.yml
#   docker build -f eval/Dockerfile.scoring .
#
# No secret, teacher endpoint, holdout item, or model weight is baked in here.

ARG BASE_IMAGE=python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579
FROM ${BASE_IMAGE}

# `1` installs the torch / transformers runtime so the image can score.
ARG WITH_RUNTIME=1
# Wheel index for the accelerator wheels, e.g. the CUDA index.
ARG TORCH_INDEX_URL=""
# Provenance only: the CortexLM/relearn commit this image was built from.
ARG RELEARN_GIT_SHA="unknown"

# The harvest keeps only the last 8 KiB of the run log, so a library that logs
# a line per HTTP request or per shard would bury the one line that says why a
# run failed. Quiet by default; `RELEARN_LOG_LEVEL=DEBUG` puts it all back.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    RELEARN_GIT_SHA=${RELEARN_GIT_SHA}

# openssh-server: the provider lands on the pod over SSH to stage the request
# and run the scorer, so the image has to be reachable that way.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates openssh-server tini; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /run/sshd /root/.ssh; \
    chmod 700 /root/.ssh

WORKDIR /opt/relearn
COPY pyproject.toml README.md LICENSE ./
COPY eval/src ./eval/src
COPY eval/install-cli.sh /tmp/install-cli.sh
COPY eval/bin/relearn-eval /tmp/relearn-eval-launcher

RUN set -eux; \
    if command -v pip >/dev/null 2>&1; then \
        pip_cmd="pip"; \
    else \
        pip_cmd="python3 -m pip"; \
    fi; \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        $pip_cmd install --extra-index-url "${TORCH_INDEX_URL}" ".[runtime]"; \
    elif [ "${WITH_RUNTIME}" = "1" ]; then \
        $pip_cmd install ".[runtime]"; \
    else \
        $pip_cmd install "."; \
    fi; \
    sh /tmp/install-cli.sh; \
    rm -f /tmp/install-cli.sh /tmp/relearn-eval-launcher

# Regular files under /usr/bin so a harvest SSH with PATH=/usr/bin:/bin
# can run them. Do not symlink to /usr/local/bin: that is the live 127.
COPY eval/bin/relearn-eval /usr/bin/relearn-eval
COPY eval/entrypoint.sh /usr/bin/relearn-eval-entrypoint
RUN set -eux; \
    chmod 0755 /usr/bin/relearn-eval /usr/bin/relearn-eval-entrypoint; \
    test -f /usr/bin/relearn-eval; \
    test ! -L /usr/bin/relearn-eval; \
    test -x /usr/bin/relearn-eval; \
    env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help >/dev/null

# Where the control plane stages `request.json` over stdin. Never persisted:
# the request carries the private holdout for the length of the run only.
WORKDIR /tmp/relearn_eval

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/relearn-eval-entrypoint"]
# `serve` keeps the pod reachable so the harvest can stage and run. Any other
# argument list is passed straight to `relearn-eval`.
CMD ["serve"]
