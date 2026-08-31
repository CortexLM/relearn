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
| `RELEARN_BASE_MODEL_DIR` | Local base weights. Preferred over pulling the pinned id per run |
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
# what CI publishes: contract plus the torch / transformers runtime
docker build -f eval/Dockerfile -t relearn-eval:dev .

# contract-only, for a fast local loop: cannot score, and says so
docker build -f eval/Dockerfile --build-arg WITH_RUNTIME=0 -t relearn-eval:contract .

# a vLLM or CUDA base of the operator's choosing
docker build -f eval/Dockerfile \
  --build-arg BASE_IMAGE=<base@sha256:…> \
  --build-arg TORCH_INDEX_URL=<accelerator wheel index> .
```

CI publishes `ghcr.io/cortexlm/relearn-eval` and prints the pushed
`sha256:` digest. Put that digest — never a tag — in `eval_image_digest` in the
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
