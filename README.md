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

## Eval image

The image is the whole live scorer. The control plane boots it on a digest pin,
stages the run request into `/tmp/relearn_eval`, and accepts a score only when
the pod prints `RELEARN_METRICS=<document>` and `RELEARN_EVAL_OK`.

```bash
docker build -f eval/Dockerfile -t relearn-eval:dev .
relearn-eval score --request request.json --out metrics.json
```

Full contract, environment, and operator notes: [`docs/EVAL-IMAGE.md`](./docs/EVAL-IMAGE.md).
Normative source: `docs/RELEARN.md` § Eval image contract in the control plane.

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
| `eval/Dockerfile`, `eval/entrypoint.sh` | The digest-pinned eval image |
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
