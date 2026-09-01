<!-- protocol_version: 1 -->

# Relearn

Public challenge repo for the Cortex **Relearn** subnet (challenge id
`relearn`).

**This repo is miner + eval only.** It does not contain the Cortex control
plane, gateway, or validator. Those live in [`CortexLM/cortex`](https://github.com/CortexLM/cortex).

| Pin | Value |
|-----|--------|
| Base model | `Qwen/Qwen3.8-27B` (Apache-2.0, native VLM) |
| Teacher / judge | HTTP API, wire id `glm-5.3` (operator sets `RELEARN_TEACHER_*`) |
| Teacher weights | `incoai/GLM-5.3-NVFP4`, served from `RELEARN_TEACHER_LOCAL_DIR` |
| Eval image | `ghcr.io/cortexlm/relearn-eval`, pinned by digest |
| Score | Displacement vs the previous champion on a private holdout |
| Trust | No TDX / no Phala CVM. Miner pays Lium. The holdout reaches the pod only for the length of a run |

The pin of record is [`config/relearn-pin.toml`](https://github.com/CortexLM/cortex/blob/main/config/relearn-pin.toml)
in the control plane: it carries this repo's git SHA and the eval image digest.
Deploy = bump the pin after this repo's CI publishes a green image.

While `eval_image_digest` is empty, live hosts answer **503** on
`POST /v1/submissions` rather than scoring — there is no simulated fallback.

## Eval images

The image is the whole live scorer. The control plane boots it on a digest pin,
stages the run request over stdin, and accepts a score only when the pod prints
`RELEARN_METRICS=<document>` and `RELEARN_EVAL_OK`.

This repo ships one such image per live Relearn challenge. They share that
transport — a cortex harvest client for any of them is
`crates/relearn-lium-harvest` with a different document type — and share nothing
else. Each document carries its own `challenge_id`, each image refuses a
document that is not its challenge, and the three challenges' series keys do not
overlap, so a digest pinned into the wrong config fails closed rather than
scoring something nobody asked for.

| Challenge | Image | Entrypoint | Scores |
|---|---|---|---|
| `relearn` | `ghcr.io/cortexlm/relearn-eval` | `relearn-eval` | post-trained `Qwen/Qwen3.8-27B` on a private prompt holdout, judged by the frozen teacher |
| `relearn-image` | `ghcr.io/cortexlm/relearn-image-eval` | `relearn-image-eval` | fine-tuned `nvidia/Cosmos3-Super-Text2Image` on the frozen Qwen-Image-Bench split, judged by Q-Judger |
| `relearn-agent` | `ghcr.io/cortexlm/relearn-agent-eval` | `relearn-agent-eval` | post-trained `Qwen/Qwen3.8-27B` replayed against recorded tool-use episodes |

`relearn-image` is the challenge cortex's crates still call `relearn-t2i`; the
image accepts a request under either id. `relearn-agent` shares a base
checkpoint with `relearn` and is otherwise a different challenge: it grades the
action taken at each step against what was recorded, and replays every episode
again with the tool results withheld, because a model that ignores its tools can
still write a confident answer that prose grading rewards.

Bounty has no GPU eval image, here or anywhere: it scores filed bug reports.

```bash
docker build -f eval/Dockerfile -t relearn-eval:dev .
relearn-eval score --request request.json --out metrics.json

docker build -f eval/Dockerfile.challenge --build-arg CHALLENGE=image \
  -t relearn-image-eval:dev .
docker build -f eval/Dockerfile.challenge --build-arg CHALLENGE=agent \
  -t relearn-agent-eval:dev .
```

Full contract, environment, and operator notes per challenge:
[`docs/EVAL-IMAGE.md`](./docs/EVAL-IMAGE.md),
[`docs/IMAGE-EVAL-IMAGE.md`](./docs/IMAGE-EVAL-IMAGE.md),
[`docs/AGENT-EVAL-IMAGE.md`](./docs/AGENT-EVAL-IMAGE.md).
Normative source: `docs/RELEARN.md` and `docs/RELEARN-T2I.md` in the control
plane.

## Miner path

See [docs/](./docs/) and the control-plane mirror
[`docs/external-miner/relearn.md`](https://github.com/CortexLM/cortex/blob/main/docs/external-miner/relearn.md).

```bash
curl -sS -X POST https://<gateway>/challenge/relearn/v1/submissions \
  -H 'content-type: application/json' \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  -d '{
    "miner_hotkey": "<64-hex>",
    "artifact_digest": "<sha256>",
    "manifest": {"train_item_ids": [1], "train_dataset_ids": ["my-sft-mix-v3"]}
  }'
```

`manifest` is required evidence: an undeclared one fails the contamination gate
rather than skipping it. Never commit `LIUM_API_KEY` or any secret.

## Layout

| Path | Role |
|------|------|
| `eval/Dockerfile`, `eval/entrypoint.sh` | The digest-pinned `relearn` eval image |
| `eval/Dockerfile.challenge`, `eval/entrypoint-challenge.sh` | The `relearn-image` and `relearn-agent` eval images |
| `eval/src/relearn_common/` | The marker protocol, run-identity binding, artifact resolution, judge transport |
| `eval/src/relearn_image_eval/` | `relearn-image`: Cosmos3 generation, Q-Judger scoring |
| `eval/src/relearn_agent_eval/` | `relearn-agent`: trace replay, action grading, the tool-blind control |
| `eval/src/relearn_eval/contract.py` | Markers, schema, and the metrics document |
| `eval/src/relearn_eval/request.py` | The harvest request, and what it refuses |
| `eval/src/relearn_eval/scoring.py` | Every series in the document, measured |
| `eval/src/relearn_eval/verify.py`, `harvest.py` | The control plane's acceptance checks, mirrored |
| `eval/src/relearn_eval/teacher.py` | Frozen judge over HTTP; never serves miner weights |
| `eval/src/relearn_eval/catalog/` | Public, canary, general-canary, agent-trace slices |
| `eval/src/relearn_eval/generators.py`, `decontam.py` | Miner-side disjoint factory and bench blocklist |
| `tests/` | Image contract tests |
| `docs/` | Miner-facing and operator-facing docs |

## Teacher serving

The teacher is judge-only, and the scored artifact is always the miner weights
loaded **inside** the eval image — never anything served through the teacher
API. Download the NVFP4 weights, then point vLLM at
`RELEARN_TEACHER_LOCAL_DIR`; never pass the Hugging Face repo id to vLLM. GPU
shape is the operator's, described in the control-plane pin.
