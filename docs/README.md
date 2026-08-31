# Relearn miner docs

Post-train `Qwen/Qwen3.8-27B` (Apache-2.0, native VLM). Score is paired
displacement against the current champion on a private holdout. Regressions are
never crowned.

- Submit over HTTP to the Cortex gateway (`/challenge/relearn/v1/submissions`).
- Pay Lium yourself (`LIUM_API_KEY` / `X-Lium-Api-Key`).
- Declare what you trained on in `manifest`. An undeclared manifest fails the
  contamination gate — it does not skip it.
- The holdout reaches the eval pod only after your submission digest freezes,
  and only for the length of that run.
- Official public benchmarks are out of bounds; the blocklist is
  `eval/src/relearn_eval/decontam.py`.
- Only the public ids on `GET /challenge/relearn/v1/status` are trainable.

Check `GET /challenge/relearn/v1/status` before submitting: while `can_score`
is `false`, submissions answer 503. `eval_backend` tells you whether a verdict
came from the pinned eval image (`lium`) or an operator's offline harness
(`sim`, CI and local only).

Eval image contract and how your artifact is loaded and scored:
[`EVAL-IMAGE.md`](./EVAL-IMAGE.md).

Control plane: <https://github.com/CortexLM/cortex>
