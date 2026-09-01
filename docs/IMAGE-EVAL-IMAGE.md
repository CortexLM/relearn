# The Relearn Image eval image

The live scorer for challenge `relearn-image` (the challenge cortex still calls
`relearn-t2i` in its crates and its pin). Everything that turns an artifact into
numbers is here; the control plane cannot compute a score and has no simulated
fallback, so a pod that does not return a well-formed, correctly bound document
is a **503** rather than a verdict.

Normative contract: `docs/RELEARN-T2I.md` in
[`CortexLM/cortex`](https://github.com/CortexLM/cortex), together with
`crates/relearn-t2i-task`, `crates/relearn-t2i-judge`, and
`crates/relearn-t2i-score`. Where this document and those disagree, they win.

## One run

The contracted run, per submission, once harvest's Lium template is
`pin.image@digest`:

1. boots this image at its pinned digest on a Lium pod the **miner** pays for;
2. writes `request.json` into `/tmp/relearn_image_eval` **over stdin** — no run
   input is interpolated into the remote command;
3. runs `/usr/bin/relearn-image-eval score --request request.json --out metrics.json`
   with harvest SSH `PATH=/usr/bin:/bin` (the binary is a regular file, not a
   symlink);
4. reads back `RELEARN_METRICS=<document>` and `RELEARN_EVAL_OK`;
5. scrubs the workdir, terminates the pod, and requires *verified* termination
   before accepting anything the run returned.

That is not what happens today. `crates/relearn-lium-harvest` puts
`pin.eval_image_digest` on `InstanceSpec.image_digest`, but
`LiumClient::provision` ignores that field and rents `prism-recipe-v10`.
A digest published from this repo is therefore **not live-ready** until the
control plane template is `pin.image@digest`. Republishing a PATH-clean
`/usr/bin/relearn-image-eval` does not fix a live 127 on `prism-recipe-v10`.
The regular file is still required: the moment harvest is wired, SSH
`PATH=/usr/bin:/bin` will exec it.

```bash
relearn-image-eval score --request request.json --out metrics.json
# RELEARN_METRICS={"schema_version":1,…}
# RELEARN_EVAL_OK
```

`metrics.json` is exactly one line with no trailing newline, because the
harvest reconstructs the marker line with `printf 'RELEARN_METRICS='; cat
metrics.json`. `RELEARN_EVAL_OK` is the last thing the process prints, and only
after the document has passed the control plane's own acceptance checks
(mirrored in `verify.py`). Any failure exits non-zero with no marker and no
sidecar.

### Why the markers are the language challenge's markers

They are the same strings, and the workdir is not. This challenge is meant to
be harvested by a copy of `crates/relearn-lium-harvest` with a different
document type, rather than by a second transport that has to be reviewed and
kept in step. What stops a digest pinned into the wrong challenge's config from
scoring something nobody asked for is the binding inside the document:

* every document carries `challenge_id`, and each image refuses a document that
  is not its own challenge;
* the three challenges' series keys do not overlap (`p<id>#v<n>` here,
  `h<id>` for the language challenge, `t<id>` for the agent challenge), so a
  document that survived the identity check would still fail the series check.

## The request

```json
{
  "schema_version": 1,
  "challenge_id": "relearn-image",
  "submission_digest": "<frozen run>",
  "artifact_digest": "<sha256, or base-relearn-image-champion>",
  "base_model": "nvidia/Cosmos3-Super-Text2Image",
  "judge_model": "Qwen/Qwen-Image-Bench",
  "eval_image_digest": "sha256:…",
  "holdout_commitment": "<64 hex>",
  "pin_salt": "cortex-t2i-v0",
  "variations_per_prompt": 4,
  "holdout": [{"id": 901, "text": "…"}],
  "public":  [{"id": 1,   "text": "…"}],
  "sampler": {"width": 1024, "height": 1024, "num_inference_steps": 50,
              "guidance_scale": 4.0, "flow_shift": 3.0, "negative_prompt": "",
              "num_frames": 1, "dtype": "bfloat16",
              "scheduler": "UniPCMultistepScheduler"},
  "manifest": {"base": "nvidia/Cosmos3-Super-Text2Image",
               "base_license": "OpenMDW-1.1",
               "train_prompt_ids": [],
               "claimed_output_hashes": {"p1#v0": "<sha256>"}}
}
```

`challenge_id` may also be `relearn-t2i`. cortex's pin still declares the
earlier name, and refusing it would mean the rename needs a new image digest
before the control plane can boot anything. Whichever name arrives is echoed
back in the document.

Unknown fields are tolerated, so the control plane can grow the request without
a new image. Everything the image acts on is checked, and each of these is a
failed run rather than a scored one:

| Refusal | Why |
|---------|-----|
| `schema_version` is not 1 | the document would not deserialize |
| `challenge_id` is another challenge | this image does not score it |
| no `submission_digest` / `artifact_digest` | nothing to bind the run to |
| `eval_image_digest` is not a `sha256:` pin | the control plane only rents digest-pinned images |
| `base_model` is a Flux family | non-commercial weights, refused as a product rule |
| `base_model` is not the pinned Cosmos3 | the pin is the comparison |
| `manifest.base` is Flux, or is not the pin, or the licence is not OpenMDW-1.1 | the miner's own attestation has to match |
| `judge_model` is not Q-Judger | a run judged by anything else is not comparable |
| empty `pin_salt`, or `variations_per_prompt` ≤ 0 | the seed lattice is not pinned |
| either split yields fewer than 100 cells | the paired test refuses a verdict below its floor, so the pin could never promote anything |
| a prompt id outside 1..=1000, a duplicate, or empty text | not a scorable split |
| a holdout id that is also in the public split | the gap gate would compare a split with itself |
| `num_frames` is not 1 | this challenge scores single images |
| the prompts do not hash to `holdout_commitment` | the request was edited in flight |
| `RELEARN_IMAGE_EVAL_IMAGE_DIGEST` is set and disagrees | the request is for another image build |

`artifact_digest: "base-relearn-image-champion"` is the boot baseline: the
pinned checkpoint with no artifact, which is how a live host records a champion
to compare against. It carries no manifest, because there is nothing to attest.

**The request carries the private holdout.** The image never writes a prompt to
stdout, to a log, or to a persisted path. Cell keys are
`p<prompt id>#v<variation index>` — ids, never text — and the verifier refuses a
document whose keys are any other shape.

## The document

```json
{
  "schema_version": 1,
  "challenge_id": "relearn-image",
  "submission_digest": "…", "artifact_digest": "…",
  "eval_image_digest": "sha256:…", "holdout_commitment": "…",
  "base_model": "nvidia/Cosmos3-Super-Text2Image",
  "judge_model": "Qwen/Qwen-Image-Bench",
  "holdout": {"p901#v0": 0.83},
  "public":  {"p1#v0": 0.81},
  "holdout_by_pillar": {"quality": {"p901#v0": 0.80}, "alignment": {}, …},
  "na_rate": 0.04,
  "replay": {"cells_checked": 3, "exact_hash_matches": 3, "max_embedding_drift": 0.0},
  "faithfulness": {"checks": 8, "agreements": 8},
  "contaminated_prompt_ids": []
}
```

The field names are `relearn_t2i_score::T2iSliceScores` plus the run identity,
and the pillar keys are that crate's serde spelling (`quality`, `aesthetics`,
`alignment`, `real_world_fidelity`, `creative_generation`).

| Series / evidence | Measured from |
|---|---|
| `holdout` | Q-Judger's total per private cell, normalized to `0..=1` |
| `public` | the same, on the published cells |
| `holdout_by_pillar` | the same score trees, split by level-1 pillar |
| `na_rate` | level-3 items the judge declined, over all holdout cells |
| `replay` | three published cells regenerated and compared |
| `faithfulness` | targeted Alignment spot checks against the scored Alignment |
| `contaminated_prompt_ids` | holdout ids the submission admits to training on |

Holdout and public are judged by the same judge on the same scale, because the
gap between them is itself a gate: measuring one strictly and the other loosely
would make that gap an artefact of the graders. Scores are rounded to six
decimals so two runs of the same model on the same cells agree exactly.

### Scoring follows the paper, including the `N/A` rule

Q-Judger returns a thinking trace followed by a JSON score tree over five level-1
pillars. Raw `0|1|2` map to `0|60|100`; `N/A` is **excluded, not zeroed**; level 3
averages into level 2, level 2 into level 1, and the five pillars into the total.

Zeroing `N/A` is not a rounding choice, it is a correctness bug. A prompt where a
criterion does not apply would be punished for it, and prompts differ in which
criteria apply, so the punishment would land unevenly across the split.

A reply that is *entirely* `N/A` is a failed judge run and ends the item, never a
score of zero. Above a 25% N/A rate over the holdout the whole run is void.

### The two pieces of evidence that are not series

**Seed replay.** Three published cells, chosen deterministically from the
submission digest, are regenerated after the scored pass and compared two ways.
Against the miner's `claimed_output_hashes`, which detects an artifact whose
claimed outputs came from weights other than the submitted ones; and against
this run's own first-pass image, which detects non-determinism in the generation
path. Exact hashes are the fast path and are not required — pixel determinism
does not survive a driver change — so a small descriptor distance is accepted,
exactly as `ReplayEvidence::passes` does on the control plane.

`max_embedding_drift` is `1 − cosine` between two **descriptors**, not two
learned embeddings: the image downsampled to 32×32 RGB and L2-normalized. The
eval image deliberately does not depend on a second model to decide whether two
of its own generations are the same picture.

**Prompt faithfulness.** Q-Judger is the only judge this challenge accepts, so a
spot check cannot be a second model. It is a second *question*: eight holdout
cells, chosen deterministically from the submission digest, are re-judged on a
narrower pass that scores only concrete checkable claims — object counts,
rendered text, spatial relations — and its verdict is compared with the
Alignment pillar from the scored pass at the same threshold. Disagreement
discards the run on the control plane rather than picking a winner between the
two passes.

## Environment

Nothing below is baked into the image, and none of it is a secret this repo
knows.

| Variable | Role |
|----------|------|
| `RELEARN_IMAGE_JUDGE_API_URL` | OpenAI-compatible Q-Judger endpoint. **Required**: with no judge the run fails |
| `RELEARN_IMAGE_JUDGE_API_KEY` | Bearer for that endpoint. Never logged |
| `RELEARN_IMAGE_JUDGE_TIMEOUT_SECS` | Per-call timeout (default 300) |
| `RELEARN_IMAGE_BASE_MODEL_DIR` | Local Cosmos3 weights. Strongly preferred over pulling a 65B checkpoint per run |
| `RELEARN_IMAGE_ARTIFACT_DIR` | Content-addressed artifact store, checked first |
| `RELEARN_IMAGE_ARTIFACT_URL_TEMPLATE` | Fallback fetch, e.g. `.../{digest}.tar` |
| `RELEARN_IMAGE_MAX_ARTIFACT_BYTES` | Ceiling on a fetched artifact |
| `RELEARN_IMAGE_EVAL_IMAGE_DIGEST` | Pin the image's own digest so a request for another build is refused |
| `RELEARN_IMAGE_LOG_LEVEL` | stderr verbosity |

`RELEARN_T2I_JUDGE_API_URL` and `RELEARN_T2I_JUDGE_API_KEY` are read as
fallbacks, so a host already configured under the earlier name keeps working.

The artifact is always verified against `artifact_digest` before it is loaded,
whichever source produced it. A store that serves different bytes fails the run.

## What the image will not do

* **No simulated numbers.** There is no offline harness and no fallback judge.
  If the pipeline cannot be loaded or the judge cannot be reached, the run ends
  without a document.
* **No Flux.** Refused at request validation, before any GPU time is spent —
  never scored low, never scored at all.
* **No second judge.** The judge client will only call Q-Judger, and a payload
  that looks like weights or a model id that looks like an artifact digest is
  refused before any request is sent.
* **No prompt upsampling.** NVIDIA recommends upsampling a prompt into a JSON
  document before generation. That is fine for a miner's own training and fatal
  for a benchmark, so the frozen string in the pin is replayed verbatim.
* **No Modal, no DFlash2, no Flash variants.**

## Building and pinning

```bash
# pin path: CUDA Ubuntu, venv at /opt/relearn-venv, regular file at
# /usr/bin/relearn-image-eval. Harvest SSH PATH is /usr/bin:/bin.
docker build -f eval/Dockerfile.scoring --build-arg CHALLENGE=image \
  -t relearn-image-eval:dev .

# contract-only, for a fast local loop: cannot score, and says so. Not the pin.
docker build -f eval/Dockerfile.challenge --build-arg CHALLENGE=image \
  --build-arg WITH_RUNTIME=0 -t relearn-image-eval:contract .
```

CI publishes the same manifest to `ghcr.io/cortexlm/relearn-image-eval` **and**
`ghcr.io/cortexlm/relearn-t2i-eval` — one build, one digest, two names — so
cortex's `config/relearn-t2i-pin.toml` can take the digest without its
`eval_image` string having to change first. Taking the digest is not enough:
harvest must rent a Lium template whose docker image is that
`repository@sha256:…`. Until it does, do not call the digest live-ready.

A package GHCR creates from a workflow starts **private**, and a Lium pod boots
with no registry credential, so a one-time step is needed before the pin can be
used: set the package's visibility to public, or give the pod a pull credential.
The digest pins the image either way; visibility decides whether the pod can
fetch it.

## Checking a run

```bash
relearn-image-eval verify --request request.json --metrics metrics.json
relearn-image-eval verify --request request.json --transcript run.log
```
