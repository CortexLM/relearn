"""What the Relearn Agent challenge is pinned to, and what it refuses.

This challenge is **not** the Relearn LLM challenge with a different name. Both
post-train `Qwen/Qwen3.8-27B`, and there the resemblance ends:

| | `relearn` | `relearn-agent` |
|---|---|---|
| Holdout item | a prompt | a recorded tool-use episode |
| What is measured | the text that comes back | the action chosen at each step |
| Graded by | the frozen teacher | exact tool and argument match, deterministically |
| Control | pixel shuffle on vision items | observations withheld, and pixels shuffled |

The controls are the reason the two cannot be collapsed. A model that never
looks at a tool result can still write a plausible answer, and the language
challenge would score it well. Here it is measured against what the tool
actually returned, and replayed a second time with those returns withheld: a
model that reads them loses ground, a model that pattern-matches the goal does
not move, and not moving is what fails.
"""

from __future__ import annotations

#: Challenge id. Distinct from `relearn`, so leaf digests never collide.
CHALLENGE_ID = "relearn-agent"

#: Challenge ids this image will score. Exactly one: unlike the Image
#: challenge, this one has no earlier name to bridge.
CHALLENGE_IDS = (CHALLENGE_ID,)

#: Environment prefix for every operator setting this image reads.
ENV_PREFIX = "RELEARN_AGENT_"

#: Where the control plane stages `request.json`, over stdin. Not a persisted
#: path: the request carries the private holdout for the length of the run.
POD_WORKDIR = "/tmp/relearn_agent_eval"  # noqa: S108 - the control plane's staging path

#: Must equal the control plane's metrics schema for this challenge.
SCHEMA_VERSION = 1

#: Artifact id the control plane asks for when it measures the un-post-trained
#: base, which is how a live host records a champion to compare against.
BASE_CHAMPION_ARTIFACT = "base-relearn-agent-champion"

#: The base miners post-train. Same checkpoint as the language challenge; the
#: eval is what differs.
BASE_MODEL_ID = "Qwen/Qwen3.8-27B"

#: Teacher wire id. Judge-only, and only for the free-text final answer: every
#: action is graded deterministically against what was recorded.
TEACHER_MODEL_ID = "glm-5.3"

#: Teacher wire ids an operator may configure.
CONFIGURED_TEACHER_MODELS = (
    TEACHER_MODEL_ID,
    "zai-org/GLM-5.3",
    "kimi-k3",
    "moonshotai/Kimi-K3",
)

#: Domain tag for the holdout trace commitment. Distinct from the language
#: challenge's `base-relearn-holdout-v1`, so the same records under the two
#: challenges never produce the same commitment.
HOLDOUT_DOMAIN = b"base-relearn-agent-holdout-v1"

#: Minimum holdout traces. The bootstrap paired test refuses a verdict below
#: 100 decided examples, so a request that cannot reach it can never promote
#: anything and is a configuration error rather than a permanent champion-hold.
MIN_HOLDOUT_TRACES = 100

#: How the three measured parts of a trace compose into its holdout score.
#: Pinned rather than configurable: a challenger and the champion it displaces
#: have to be scored on the same composition. The action term dominates because
#: this challenge is about acting; the answer term exists so a model that acts
#: correctly and then reports nonsense does not score full marks.
TRACE_ACTION_WEIGHT = 0.6
TRACE_ORDER_WEIGHT = 0.2
TRACE_ANSWER_WEIGHT = 0.2

#: Credit for naming the recorded tool, before any argument is checked. The
#: rest is argument agreement, so picking the right tool with the wrong
#: arguments is worth something and not everything.
TOOL_NAME_CREDIT = 0.5

#: How far the action score must fall when observations are withheld before the
#: model counts as having used them. Matches `relearn_mm_score::MIN_SHUFFLE_DROP`,
#: which is the same measurement applied to pixels.
MIN_BLIND_DROP = 0.10

#: Largest share of steps that may be unparsable or name a tool outside the
#: trace's own schema before the run means nothing. A model that cannot emit a
#: well-formed action is not being measured on tool use.
MAX_INVALID_ACTION_RATE = 0.25

#: What replaces an observation in the tool-blind replay. Content-free on
#: purpose: it must remove the information without removing the turn, so the
#: conversation shape the model sees is otherwise identical.
WITHHELD_OBSERVATION = "<observation withheld>"

#: Observation modalities that carry pixels and take the shuffle control.
IMAGE_MODALITY = "image"

#: Series key prefixes. Ids, never text.
HOLDOUT_KEY_PREFIX = "t"
TOOL_CALL_KEY_PREFIX = "s"
ORDER_KEY_PREFIX = "o"
PUBLIC_KEY_PREFIX = "q"
CANARY_KEY_PREFIX = "c"
