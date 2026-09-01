# The Relearn Agent eval image

The live scorer for challenge `relearn-agent`. Everything that turns an artifact
into numbers is here; the control plane cannot compute a score and has no
simulated fallback, so a pod that does not return a well-formed, correctly bound
document is a **503** rather than a verdict.

## Why this is not the Relearn LLM image renamed

Both challenges post-train `Qwen/Qwen3.8-27B`, and that is the whole of the
overlap.

| | `relearn` | `relearn-agent` |
|---|---|---|
| Holdout item | a prompt | a recorded tool-use episode |
| What is measured | the text that comes back | the action chosen at each step |
| Graded by | the frozen teacher | exact tool and argument match, deterministically |
| Judge's role | every item | the free-text final answer only |
| Control | pixel shuffle on vision items | observations withheld, and pixels shuffled |
| Image | `ghcr.io/cortexlm/relearn-eval` | `ghcr.io/cortexlm/relearn-agent-eval` |
| Entrypoint | `relearn-eval` | `relearn-agent-eval` |
| Workdir | `/tmp/relearn_eval` | `/tmp/relearn_agent_eval` |

The controls are why the two cannot be collapsed. A model that never looks at a
tool result can still write a plausible, confident answer, and prose grading
rewards it. Here the same model is measured against the recorded argument that
could only have come from the observation, and then measured again with that
observation taken away.

## One run

```bash
relearn-agent-eval score --request request.json --out metrics.json
# RELEARN_METRICS={"schema_version":1,…}
# RELEARN_EVAL_OK
```

The lifecycle is the language challenge's: boot the digest-pinned image, stage
`request.json` into `/tmp/relearn_agent_eval` over stdin, run the scorer, read
back the two markers, scrub, terminate with verification. `metrics.json` is
exactly one line with no trailing newline, because the harvest reconstructs the
marker line with `printf 'RELEARN_METRICS='; cat metrics.json`.

The markers are deliberately the same strings as the other challenges', so a
cortex harvest client for this one is `crates/relearn-lium-harvest` with a
different document type. What keeps the challenges apart is the `challenge_id`
in the document and the non-overlapping series keys, not a different marker —
see the same section in [`IMAGE-EVAL-IMAGE.md`](./IMAGE-EVAL-IMAGE.md).

## The request

```json
{
  "schema_version": 1,
  "challenge_id": "relearn-agent",
  "submission_digest": "<frozen run>",
  "artifact_digest": "<sha256, or base-relearn-agent-champion>",
  "base_model": "Qwen/Qwen3.8-27B",
  "teacher_model": "glm-5.3",
  "eval_image_digest": "sha256:…",
  "holdout_commitment": "<64 hex>",
  "holdout": [
    {
      "id": 4102,
      "goal": "Find who is on call for payments and open a ticket assigned to them.",
      "dataset_id": "ops-traces-v1",
      "tools": [
        {"name": "get_oncall", "description": "…", "parameters": {"service": {"type": "string"}}},
        {"name": "open_ticket", "description": "…", "parameters": {"title": {"type": "string"}, "assignee": {"type": "string"}}}
      ],
      "steps": [
        {"tool": "get_oncall", "arguments": {"service": "payments"},
         "observation": "{\"engineer\": \"rmoreau\"}"},
        {"tool": "open_ticket", "arguments": {"title": "…", "assignee": "rmoreau"},
         "observation": "{\"ticket\": \"OPS-8814\"}",
         "observation_image_hash": ""}
      ],
      "final_answer": "Opened OPS-8814 and assigned it to rmoreau."
    }
  ]
}
```

Unknown fields are tolerated. Everything the image acts on is checked, and each
of these is a failed run rather than a scored one:

| Refusal | Why |
|---------|-----|
| `schema_version` is not 1 | the document would not deserialize |
| `challenge_id` is not `relearn-agent` | this image does not score it |
| no `submission_digest` / `artifact_digest` / `teacher_model` | nothing to bind the run to |
| `eval_image_digest` is not a `sha256:` pin | the control plane only rents digest-pinned images |
| `base_model` is not `Qwen/Qwen3.8-27B` | the pin is the comparison |
| fewer than 100 episodes | the paired test refuses a verdict below its floor |
| a duplicate episode id | two scores would collide on one key |
| an episode with no goal, no tools, no steps, or no final answer | nothing to replay or to grade |
| a recorded step calling a tool outside its own episode's schema | the model is shown the schema, so it could never produce that action and every model would score zero there |
| a step with a malformed `observation_image_hash` | the pixels could not be verified |
| the episodes do not hash to `holdout_commitment` | the request was edited in flight |
| `RELEARN_AGENT_EVAL_IMAGE_DIGEST` is set and disagrees | the request is for another image build |

The commitment covers the goal, the dataset id, the tool names, the final
answer, and every step's tool, canonical arguments, observation, and image hash.
All of it, because every part of it decides what counts as correct: editing an
observation changes what the right next action is, and reordering steps changes
what counts as in order. Anything left out would be an editable knob on a
"verified" holdout. The domain tag is the agent challenge's own, so the same
records committed under the language challenge produce a different digest.

**The request carries the private holdout**, and an episode is more sensitive
than a prompt: it contains the goal, the schemas, every argument, and every
observation. Nothing in the image writes any part of one to stdout or to a
persisted path. Series keys are ids.

Pixels stay out of the request. A step whose observation is a screenshot names
it by `sha256`, and the operator mounts a content-addressed store on the pod.

## How an episode is replayed

Each step is presented on the **recorded** prefix: the goal, the tool schemas,
and every earlier step exactly as it happened, including what the tools
returned. The model is asked only for the next action.

Teacher-forcing rather than letting the model run free is the central design
choice, and it is not a convenience:

* Off-policy, the model would call tools this image cannot execute. There is no
  live environment on the pod — only a recording — so an unrecorded action has
  no observation to return, and inventing one would be a simulated number.
* One early mistake would otherwise zero an entire episode, which makes the
  score a measure of the first step and nothing else.
* Every model sees byte-identical context at every step, so a champion and a
  challenger are compared on decisions.

The reply is read as one JSON object. A bare `{"tool": …}`, the `{"name": …}`
spelling, and the OpenAI `{"function": {"name": …, "arguments": "…"}}` envelope
are all accepted, after any amount of thinking preamble: the challenge is about
choosing the right action, not about guessing which JSON dialect the harness
wanted.

## The document

```json
{
  "schema_version": 1,
  "challenge_id": "relearn-agent",
  "submission_digest": "…", "artifact_digest": "…",
  "eval_image_digest": "sha256:…", "holdout_commitment": "…",
  "base_model": "Qwen/Qwen3.8-27B", "teacher_model": "glm-5.3",
  "holdout":   {"t4102": 0.72},
  "tool_call": {"s4102": 0.85},
  "order":     {"o4102": 1.0},
  "public":    {"q1": 0.70},
  "canaries":  {"c1": 1.0},
  "invalid_action_rate": 0.01,
  "tool_blind": {"traces": 120, "score": 0.85, "degraded_score": 0.61},
  "observation_shuffle": {"image": {"traces": 30, "score": 0.81, "degraded_score": 0.64}}
}
```

| Series | Key | What it is |
|---|---|---|
| `holdout` | `t<id>` | the composite score for one private episode |
| `tool_call` | `s<id>` | mean action score across that episode's steps |
| `order` | `o<id>` | whether the episode was acted in the recorded order |
| `public` | `q<id>` | the same composite on the shipped published episodes |
| `canaries` | `c<id>` | shipped single-call episodes, matched not judged |

### How a step is scored

Deterministically, against the recording. No judge is involved: the episode was
recorded, so the tool that was called and the arguments it was called with are
known exactly, and asking a language model whether an action "looks reasonable"
would put judge variance back into a measurement that does not need it.

* Naming the recorded tool is worth `0.5`; the rest is argument agreement, so
  picking the right tool with the wrong arguments is worth something and not
  everything.
* Argument agreement is F1 over `(key, value)` pairs, so both halves of getting
  arguments wrong are measured: omitting a required one costs recall, inventing
  extras costs precision. Values compare past whitespace, case, and `3` versus
  `3.0`.
* Naming a tool the episode offers but did not use scores zero.
* Naming a tool **outside** the episode's schema, or replying with something
  that is not an action at all, scores zero *and* counts toward
  `invalid_action_rate`. Above 25% the run is void: a model that cannot emit a
  well-formed action is not being measured on tool use.

`order` counts a step as in order when the model names the tool recorded at that
index, and out of order when it names a tool the episode uses somewhere else —
filing the ticket before the lookup, or repeating a lookup it was already given
the answer to. A tool the episode never uses is neither: that is already scored
as a wrong action, and counting it twice would let one mistake sink two series.
A model that never names a recorded tool scores zero, not one, because absent
evidence of ordering is not credit for it.

The composite is `0.6 × action + 0.2 × order + 0.2 × answer`. Pinned rather than
configurable, because a challenger and the champion it displaces have to be
scored on the same composition. The action term dominates because this challenge
is about acting; the answer term exists so a model that acts correctly and then
reports nonsense does not score full marks.

### The controls

Every episode is replayed a second time with **every observation replaced by a
content-free placeholder**, and the episodes carrying screenshots are replayed a
third time with **the pixels shuffled**. Both report the same shape as
`relearn_mm_score::AgenticEvidence`: score, degraded score, and the gap.

A model that read the tool results loses ground on those replays. A model that
pattern-matched the goal does not move, and not moving is what fails it on the
control plane. A low score would not be enough — a uniformly mediocre model can
still beat a champion on noise — which is why the gate is on the *gap* rather
than on the level. The threshold matches `relearn_mm_score::MIN_SHUFFLE_DROP`
(0.10), because it is the same measurement applied to different evidence.

The controls compare the **action** score only, and involve no judge. The action
score is deterministic, so the difference between two passes is attributable to
the withheld information rather than to judge variance, and running the judge
three times over would triple the cost of a run to measure something the judge
is worse at.

The prompt keeps its shape across all three passes — same turn count, same tool
schemas, same goal — so a drop cannot be attributed to the context having become
a different length.

### The shipped slices are the CI default, not the live seal

The request carries only the holdout, but the document must carry `public` and
`canaries`. A champion with no public split has no memorization gap to measure,
and a champion with no canaries has nothing to catch the failure a relative
comparison cannot see: a model that has lost the ability to emit a tool call at
all scores badly against the champion, and so does the champion, so the
comparison says nothing.

A live host should point `RELEARN_AGENT_PUBLIC_FILE` at the real published
episodes. The shipped ones are synthetic and deliberately not lifted from any
published agent benchmark, for the same reason the language challenge does not
lift its canaries from MMLU.

## Environment

| Variable | Role |
|----------|------|
| `RELEARN_AGENT_TEACHER_API_URL` | OpenAI-compatible judge endpoint for the final answers. **Required** |
| `RELEARN_AGENT_TEACHER_API_KEY` | Bearer for that endpoint. Never logged |
| `RELEARN_AGENT_TEACHER_MODEL` | Wire id override (default `glm-5.3`) |
| `RELEARN_AGENT_BASE_MODEL_DIR` | Local base weights. Preferred over pulling the pinned id per run |
| `RELEARN_AGENT_ARTIFACT_DIR` | Content-addressed artifact store, checked first |
| `RELEARN_AGENT_ARTIFACT_URL_TEMPLATE` | Fallback fetch, e.g. `.../{digest}.tar` |
| `RELEARN_AGENT_IMAGE_STORE` | Content-addressed screenshot store. Required when an episode has image observations |
| `RELEARN_AGENT_PUBLIC_FILE`, `RELEARN_AGENT_CANARY_FILE` | Replace a shipped slice |
| `RELEARN_AGENT_EVAL_BACKEND` | `auto` (default), `vllm`, or `transformers` |
| `RELEARN_AGENT_TENSOR_PARALLEL`, `RELEARN_AGENT_MAX_NEW_TOKENS` | Runtime shape and decode width |
| `RELEARN_AGENT_EVAL_IMAGE_DIGEST` | Pin the image's own digest so a request for another build is refused |
| `RELEARN_AGENT_LOG_LEVEL` | stderr verbosity |

`RELEARN_TEACHER_API_URL`, `RELEARN_TEACHER_API_KEY`, and
`RELEARN_TEACHER_MODEL` are read as fallbacks, so a host already running the
language challenge against a judge deployment configures it once. Falling back
to nothing is still nothing: an unset judge is a refusal.

## What the image will not do

* **No simulated numbers.** No offline harness, no fallback judge, no
  passthrough when a screenshot is missing.
* **No off-policy rollout.** There is no environment on the pod to roll out
  against, and a synthesized observation would be a made-up number.
* **No judge in the action loop.** Actions are matched against the recording.
* **No Modal, no DFlash2, no Flash variants.**

## Building and pinning

```bash
docker build -f eval/Dockerfile.challenge --build-arg CHALLENGE=agent \
  -t relearn-agent-eval:dev .

docker build -f eval/Dockerfile.challenge --build-arg CHALLENGE=agent \
  --build-arg WITH_RUNTIME=0 -t relearn-agent-eval:contract .
```

CI publishes `ghcr.io/cortexlm/relearn-agent-eval` and prints the pushed
`sha256:` digest. That digest — never a tag — is what the control plane pins.

## Checking a run

```bash
relearn-agent-eval verify --request request.json --metrics metrics.json
relearn-agent-eval verify --request request.json --transcript run.log
```
