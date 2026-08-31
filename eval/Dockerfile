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
#   # scoring build on a CUDA base (what the published image is)
#   docker build -f eval/Dockerfile \
#     --build-arg BASE_IMAGE=<cuda-or-vllm-base@sha256:…> .
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

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
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

RUN set -eux; \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        pip install --extra-index-url "${TORCH_INDEX_URL}" ".[runtime]"; \
    elif [ "${WITH_RUNTIME}" = "1" ]; then \
        pip install ".[runtime]"; \
    else \
        pip install "."; \
    fi; \
    ln -sf /usr/local/bin/relearn-eval /usr/bin/relearn-eval; \
    relearn-eval --help > /dev/null

COPY eval/entrypoint.sh /usr/local/bin/relearn-eval-entrypoint
RUN chmod 0755 /usr/local/bin/relearn-eval-entrypoint

# Where the control plane stages `request.json` over stdin. Never persisted:
# the request carries the private holdout for the length of the run only.
WORKDIR /tmp/relearn_eval

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/relearn-eval-entrypoint"]
# `serve` keeps the pod reachable so the harvest can stage and run. Any other
# argument list is passed straight to `relearn-eval`.
CMD ["serve"]
