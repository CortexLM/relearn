# The Relearn eval image

The live scorer for challenge `relearn`. Everything that turns an artifact into
numbers is here; the control plane cannot compute a score and has no simulated
fallback, so a pod that does not return a well-formed, correctly bound document
is a **503** rather than a verdict.

Normative contract: `docs/RELEARN.md` § Eval image contract in
[`CortexLM/cortex`](https://github.com/CortexLM/cortex). The client is
`crates/relearn-lium-harvest`. Where this document and those disagree, they win.

## One run

The control plane, per submission:

1. boots `eval_image@<digest>` on a Lium pod the **miner** pays for, with the
   master SSH public key, under the price / GPU / lifetime guardrails;
2. writes `request.json` into `/tmp/relearn_eval` **over stdin** — no run input
   is interpolated into the remote command;
3. runs `relearn-eval score --request request.json --out metrics.json`;
4. reads back `RELEARN_METRICS=<document>` and `RELEARN_EVAL_OK`;
5. scrubs the workdir, terminates the pod, and requires *verified* termination
   before accepting anything the run returned.

The image's side of that is one command and two markers:

```bash
relearn-eval score --request request.json --out metrics.json
# RELEARN_METRICS={"schema_version":1,…}
# RELEARN_EVAL_OK
```

Both markers go to the process's own stdout as well as into `metrics.json`, so
a harvest that captures only the process output still sees a completed run.

## Making the run fit the timeout

The harvest runs the scorer under `timeout --kill-after=60 <secs>`. A live run
generates for every holdout item twice, plus every slice the image owns — a
120-item holdout is around 350 generations and 280 judge calls — so a run that
does one thing at a time does not finish, and an unfinished run is a 503 with
nothing to read. What the image does about it:

* **Preflight.** The judge is probed, the scoring runtime is imported, the base
  weights are checked for presence, and the artifact is fetched and
  digest-verified *before* the model is loaded. A pod that cannot score says
  which dependency is missing in seconds.
* **Refuses to download the base model** unless `RELEARN_ALLOW_MODEL_DOWNLOAD=1`.
  Pulling tens of gibibytes inside the run timeout is how a pod spends its whole
  budget and returns nothing. Prime the pod with `RELEARN_BASE_MODEL_DIR` or a
  warm Hugging Face cache.
* **Batched generation** (`RELEARN_BATCH_SIZE`, default 8) and **concurrent
  judging** (`RELEARN_JUDGE_CONCURRENCY`, default 8). Results stay positional,
  so concurrency cannot put a score on the wrong item.
* **Per-slice decode widths.** Open-ended answers get
  `RELEARN_MAX_NEW_TOKENS` (default 256); a multiple-choice answer gets 8.
* **Phase logging**, so a run that is killed leaves a tail naming the slice it
  died in.
* **`SIGTERM` and `SIGINT` are handled**: a killed run prints one line saying
  which phase it was in and how long it had been running, then exits non-zero.
  It never prints a document or a completion marker.
* **`RELEARN_RUN_BUDGET_SECS`** (optional) stops the run between phases with a
  reason instead of being killed mid-phase.

To check a pod without scoring on it:

```bash
relearn-eval preflight --request request.json   # needs a request and a judge
relearn-eval selftest                           # needs neither
```

## The scoring runtime

The scoring image is a CUDA build of `eval/Dockerfile.scoring`, and it installs
`.[runtime,vllm]`. Two of those wheels are the difference between a pod that
scores and a pod that burns the rent:

| Import | Without it |
|--------|------------|
| `vllm` | `auto` falls back to transformers, which is the slow way to overrun the harvest's timeout |
| `torchvision` | the native VLM base dies building its processor: `ImportError: Qwen3VLVideoProcessor requires Torchvision` |
| `torch`, `transformers` | no backend at all |
| `pillow` | holdout images cannot be decoded |

That pair is exactly what live `sha256:cbc4bbb8` was missing: it came up
RUNNING, skipped vLLM because it was not installed, and exited 1 on the
torchvision import minutes into loading the 27B base, so no champion was
recorded. The build imports them, but the build is not what a pod boots, so
they are checked where it counts:

```bash
relearn-eval selftest   # every scoring dependency, or exit 2 naming what is missing
```

`selftest` takes no request and needs no judge, so it can run against a pulled
digest. The publish job runs it on the bytes it just pushed, through
`PATH=/usr/bin:/bin`, which answers in one command both "is
`/usr/bin/relearn-eval` runnable on the harvest's PATH" and "can the python it
picks import the runtime". A contract-only build fails it, which is intended: a
slim digest is not one the control plane may pin.

`metrics.json` is exactly one line with no trailing newline, because the
harvest reconstructs the marker line with `printf 'RELEARN_METRICS='; cat
metrics.json`. `RELEARN_EVAL_OK` is the last thing the process prints, and only
after the document has passed the control plane's own acceptance checks
(mirrored in `verify.py` and `harvest.py`). Any failure exits non-zero with no
marker and no sidecar.

## The request

`HarvestRequest`, mirroring the harvest client:

```json
{
  "schema_version": 1,
  "submission_digest": "<frozen run>",
  "artifact_digest": "<sha256, or base-relearn-champion>",
  "base_model": "Qwen/Qwen3.8-27B",
  "teacher_model": "glm-5.3",
  "eval_image_digest": "sha256:…",
  "holdout_commitment": "<64 hex>",
  "holdout": [{"id": 801, "prompt": "…", "dataset_id": "…", "task": "text", "image_hash": ""}]
}
```

Unknown fields are tolerated, so the control plane can grow the request without
a new image. Everything the image acts on is checked, and each of these is a
failed run rather than a scored one:

| Refusal | Why |
|---------|-----|
| `schema_version` is not 1 | the document would not deserialize |
| no `submission_digest` / `artifact_digest` / `base_model` / `teacher_model` | nothing to bind the run to |
| `eval_image_digest` is not a `sha256:` pin | the control plane only rents digest-pinned images |
| empty holdout, duplicate ids, empty prompt, unknown task | not a scorable split |
| a vision item with no 64-hex `image_hash` | it would be scored without its image |
| the items do not hash to `holdout_commitment` | the request was edited in flight |
| `RELEARN_EVAL_IMAGE_DIGEST` is set and disagrees | the request is for another image build |

`artifact_digest: "base-relearn-champion"` is the boot baseline: the base model
with no artifact, which is how a live host records a champion to compare
against.

**The request carries the private holdout.** The image never writes a holdout
prompt to stdout, to a log, or to a persisted path: the document's series keys
are `h<id>`, `x<id>`, `p<id>`, `c<id>`, `g<id>` — ids, never text — and the
verifier refuses a document whose keys are any other shape.

## The document

`RelearnEvalMetrics`: a `BaselineMeasurement` envelope (`#[serde(flatten)]`)
plus the run identity.

```json
{
  "schema_version": 1,
  "submission_digest": "<echo of the request>",
  "artifact_digest": "<echo of the request>",
  "eval_image_digest": "sha256:…",
  "holdout_commitment": "…",
  "holdout": {}, "public": {}, "perturbed": {},
  "canaries": {}, "general_canary": {},
  "agent_trace": 0.0,
  "vision_shuffle": {}
}
```

| Series | Items | Graded by |
|--------|-------|-----------|
| `holdout` | the request's private split | frozen teacher |
| `perturbed` | the same items, pinned rewrite | frozen teacher |
| `public` | the published split | frozen teacher |
| `canaries` | shipped known-answer items | reference match |
| `general_canary` | shipped MMLU / MMMU-style choices | choice letter |
| `agent_trace` | shipped ordered-plan tasks | rubric coverage and order |
| `vision_shuffle` | one entry per vision family in the holdout | teacher, real vs shuffled pixels |

The holdout and the public split are judged by the same judge on the same
scale, because the gap between them is itself a gate. Scores are rounded to six
decimals so two runs of the same model on the same items agree exactly.

The same document is what an operator installs as `RELEARN_BASE_CHAMPION_FILE`:
run the pinned image on the base model once and use the output as-is.

## Environment

Nothing below is baked into the image, and none of it is a secret this repo
knows. All of it is pod environment the operator sets.

| Variable | Role |
|----------|------|
| `RELEARN_TEACHER_API_URL` | OpenAI-compatible judge endpoint. **Required**: with no judge the run fails |
| `RELEARN_TEACHER_API_KEY` | Bearer for that endpoint. Never logged |
| `RELEARN_TEACHER_MODEL` | Wire id override (default `glm-5.3`) |
| `RELEARN_TEACHER_MAX_TOKENS` | Judge completion budget (default 1024). GLM-5.3 thinking is mandatory; 32 tokens is spent on the think block and `content` comes back null |
| `RELEARN_JUDGE_CONCURRENCY` | Judge calls in flight (default 8) |
| `RELEARN_JUDGE_ATTEMPTS`, `RELEARN_JUDGE_RETRY_SECS` | Retry budget per judge call |
| `RELEARN_BASE_MODEL_DIR` | Local base weights. **Strongly recommended**: without it the run needs a warm cache or the download opt-in |
| `RELEARN_ALLOW_MODEL_DOWNLOAD` | Permit pulling the base model inside the run timeout |
| `RELEARN_BATCH_SIZE` | Prompts per forward pass (default 8) |
| `RELEARN_RUN_BUDGET_SECS` | Stop between phases rather than be killed mid-phase |
| `RELEARN_ARTIFACT_DIR` | Content-addressed artifact store, checked first |
| `RELEARN_ARTIFACT_URL_TEMPLATE` | Fallback fetch, e.g. `.../{digest}.tar` |
| `RELEARN_IMAGE_STORE` | Content-addressed image store. Required when the holdout has vision items |
| `RELEARN_PUBLIC_FILE` | The live public split. Strongly recommended — see below |
| `RELEARN_CANARY_FILE`, `RELEARN_GENERAL_CANARY_FILE`, `RELEARN_AGENT_TRACE_FILE` | Replace a shipped slice |
| `RELEARN_EVAL_BACKEND` | `auto` (default), `vllm`, or `transformers` |
| `RELEARN_TENSOR_PARALLEL`, `RELEARN_MAX_NEW_TOKENS` | Runtime shape and decode width |
| `RELEARN_EVAL_IMAGE_DIGEST` | Pin the image's own digest so a request for another build is refused |
| `RELEARN_MAX_ARTIFACT_BYTES` | Ceiling on a fetched artifact |
| `RELEARN_LOG_LEVEL` | stderr verbosity |

The artifact is always verified against `artifact_digest` before it is loaded,
whichever source produced it. A store that serves different bytes fails the run.

### The shipped slices are the CI default, not the live seal

The request carries only the holdout, but the document must carry `public` and
`general_canary` or the control plane refuses the champion at boot. So the image
owns those slices, and ships a synthetic set — the same arrangement as the
committed `holdout_commitment` in the control plane, which is the CI seal rather
than the production one.

A live host should point `RELEARN_PUBLIC_FILE` at the real published records
(the ids the pin publishes as trainable). With the shipped default, the
public–holdout gap gate still runs, but against items no miner trained on, so it
is weaker than it looks. The shipped items are deliberately not drawn from any
official benchmark: a general-bench canary lifted from a bench cannot detect
regression on that bench.

## What the image will not do

* **No simulated numbers.** There is no offline harness and no fallback judge.
  If the model cannot be loaded, the judge cannot be reached, or a vision item's
  pixels are missing, the run ends without a document. A hole in a series would
  be read as a low score nobody measured.
* **Never silence.** Every failure exits non-zero, writes nothing that looks
  like a score, and prints one line naming the cause: `no judge: …`,
  `no model: …`, `artifact: … hashes to …`, `terminated by SIGTERM after Ns
  during <phase>`, or `run budget of Ns spent during <phase>`.
* **No second protocol.** `verify.py` and `harvest.py` are mirrors of the
  control plane's checks, used so a transcript this image produces is proven
  acceptable before it is printed.
* **No teacher weights, hostnames, or credentials in the repo**, and no Modal.
  The teacher API is judge-only: a payload that looks like weights (safetensors,
  GGUF, NVFP4, ckpt) or a model id that looks like an artifact digest is refused
  before any request is sent.
* **Never DFlash2** (CC BY-NC-ND) and never the Flash teacher variants.

## Building and pinning

```bash
# scoring image (CUDA base) — this is the digest the control plane pins
docker build -f eval/Dockerfile.scoring -t relearn-eval:dev .

# contract-only, for a fast local loop: slim, cannot score, and says so
docker build -f eval/Dockerfile --build-arg WITH_RUNTIME=0 -t relearn-eval:contract .
```

CI publishes `ghcr.io/cortexlm/relearn-eval` from `eval/Dockerfile.scoring`
and, **in the same job**, pulls that digest and runs

```bash
docker run --rm --entrypoint /bin/sh DIGEST -c \
  'test -f /usr/bin/relearn-eval && test -x /usr/bin/relearn-eval && env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help'
docker run --rm --entrypoint /opt/relearn-venv/bin/python DIGEST -c 'import vllm, torchvision'
docker run --rm --entrypoint /bin/sh DIGEST -c \
  'env -i PATH=/usr/bin:/bin HOME=/root /usr/bin/relearn-eval selftest'
```

If any of those fails, the job fails and no pin is reported. Put a passing digest —
never a tag, never a contract-only digest — in `eval_image_digest` in the
control plane's `config/relearn-pin.toml`, together with this repo's git SHA in
`relearn_git_sha`, and re-sign the trust root.

A package GHCR created from a workflow starts **private**, and a Lium pod boots
with no registry credential, so a one-time step is needed before the pin can be
used: set the package's visibility to public (or give the pod a pull
credential). The digest is what pins the image either way — visibility decides
whether the pod can fetch it, not which bytes it gets.

The pod's default command (`serve`) keeps the container reachable over SSH so
the harvest can stage the request and run the scorer; any other argument list is
passed straight to `relearn-eval`, which is how CI and a local operator drive
the image.

The harvest SSH is non-interactive: `PATH` is typically `/usr/bin:/bin` only.
`/usr/bin/relearn-eval` is therefore a **regular file** on the CUDA scoring
image (`eval/bin/relearn-eval` copied onto `/usr/bin/relearn-eval`), not a
symlink into `/usr/local/bin` or a conda prefix, and not only present on the
slim contract image. A dangling symlink — or a shebang whose interpreter is
missing — is the live 127 (`timeout: failed to run command '/usr/bin/relearn-eval':
No such file or directory`). The publish job proves that against the pushed
scoring digest:

```bash
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval --help
env -i PATH=/usr/bin:/bin /usr/bin/relearn-eval score --help
env -i PATH=/usr/bin:/bin HOME=/root /usr/bin/relearn-eval selftest
```

Both Dockerfiles pin their base by digest, so the bytes under a reviewed digest
cannot change without the review.

## Checking a run

```bash
# does this document belong to this run?
relearn-eval verify --request request.json --metrics metrics.json

# would the control plane accept this pod transcript?
relearn-eval verify --request request.json --transcript run.log
```

Both apply the control plane's acceptance rules: schema, run identity, image
digest, holdout commitment, one score per requested item, the series the gates
need, and a pixel-shuffle control for every vision family in the holdout.
