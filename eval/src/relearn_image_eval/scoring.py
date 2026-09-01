"""Scoring one artifact into one Relearn Image metrics document.

The loop is: generate every cell at its frozen seed, judge every generated
image with Q-Judger, fold the per-image score trees into the series the control
plane's gates read, and gather the evidence those gates need.

| Series / evidence | Measured from |
|---|---|
| `holdout` | Q-Judger total per private cell, normalized |
| `public` | Q-Judger total per published cell, normalized |
| `holdout_by_pillar` | the same trees, split by level-1 pillar |
| `na_rate` | level-3 items the judge declined, over all holdout cells |
| `replay` | three published cells regenerated and compared |
| `faithfulness` | targeted Alignment spot checks versus the scored Alignment |
| `contaminated_prompt_ids` | holdout ids the submission admits to training on |

Holdout and public are judged by the same judge on the same scale, because the
gap between them is itself a gate: measuring one strictly and the other loosely
would make that gap an artefact of the graders.

Nothing here has a path that produces a number without the generator and the
judge. A cell that could not be generated, or an image the judge declined
entirely, ends the run. A hole in a series would be read as a low score nobody
measured.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from relearn_common.errors import ContractError
from relearn_common.imagestore import cosine_distance, descriptor, sha256_bytes
from relearn_common.wire import mean

from .contract import (
    FaithfulnessEvidence,
    ImageDocument,
    ImageMeasurement,
    ReplayEvidence,
)
from .generator import CosmosGenerator
from .judge import ImageScore, QJudger
from .pillars import ALIGNMENT, ALL_PILLARS
from .pins import (
    FAITHFULNESS_ALIGNMENT_THRESHOLD,
    MAX_NA_RATE,
    MIN_FAITHFULNESS_CHECKS,
    REPLAY_CELLS,
)
from .request import ImageHarvestRequest
from .seeds import cell_key

log = logging.getLogger(__name__)

#: Domain tags for the two deterministic cell selections below. Separated so
#: the replay cells and the faithfulness cells are not the same three cells
#: shuffled.
_REPLAY_DOMAIN = b"relearn-image-replay-cells-v1"
_FAITHFULNESS_DOMAIN = b"relearn-image-faithfulness-cells-v1"


class ScoringError(ContractError):
    """The run could not be scored end to end."""


@dataclass(frozen=True)
class Cell:
    """One image to generate."""

    prompt_id: int
    variation_index: int
    seed: int

    @property
    def key(self) -> str:
        return cell_key(self.prompt_id, self.variation_index)


def cells_from(raw: Sequence[tuple[int, int, int]]) -> tuple[Cell, ...]:
    return tuple(Cell(*item) for item in raw)


def select_cells(
    candidates: Sequence[Cell], count: int, submission_digest: str, domain: bytes
) -> tuple[Cell, ...]:
    """Pick `count` cells, deterministically, and bound to this submission.

    Ordering by `sha256(domain ‖ submission_digest ‖ cell_key)` gives the same
    answer on every pod that scores this run — so the evidence is reproducible
    — while differing between submissions, so the choice is not a fixed target
    a miner could special-case.
    """

    def rank(cell: Cell) -> bytes:
        digest = hashlib.sha256()
        digest.update(domain)
        digest.update(b"\xff")
        digest.update(submission_digest.encode("utf-8"))
        digest.update(b"\xff")
        digest.update(cell.key.encode("utf-8"))
        return digest.digest()

    return tuple(sorted(candidates, key=rank)[:count])


def contaminated_ids(
    train_prompt_ids: Sequence[int], holdout_ids: Sequence[int]
) -> tuple[int, ...]:
    """Holdout prompt ids the submission admits to having trained on.

    The published split is trainable by design, so only the private ids count.
    A non-empty result rejects the submission on the control plane; it is
    recorded here rather than raised, because it is a verdict about the miner
    rather than a failure of the run.
    """
    return tuple(sorted(set(train_prompt_ids) & set(holdout_ids)))


@dataclass
class _SplitResult:
    """One generated and judged split."""

    totals: dict[str, float]
    per_pillar: dict[str, dict[str, float]]
    scored_items: int
    na_items: int
    kept_images: dict[str, bytes]
    hashes: dict[str, str]


def _judge_split(
    cells: Sequence[Cell],
    request: ImageHarvestRequest,
    generator: CosmosGenerator,
    judge: QJudger,
    what: str,
    keep: frozenset[str],
) -> _SplitResult:
    """Generate and judge every cell of one split."""
    totals: dict[str, float] = {}
    per_pillar: dict[str, dict[str, float]] = {pillar.wire: {} for pillar in ALL_PILLARS}
    scored_items = 0
    na_items = 0
    kept: dict[str, bytes] = {}
    hashes: dict[str, str] = {}

    log.info("generating and judging %d %s cells", len(cells), what)
    for cell in cells:
        prompt = request.prompt_text(cell.prompt_id)
        image = generator.generate(prompt, cell.seed)
        hashes[cell.key] = sha256_bytes(image)
        if cell.key in keep:
            kept[cell.key] = image
        score = judge.score_image(prompt, image)
        totals[cell.key] = score.normalized_total()
        for pillar in ALL_PILLARS:
            value = score.normalized_pillar(pillar)
            if value is not None:
                per_pillar[pillar.wire][cell.key] = value
        scored_items += score.scored_items
        na_items += score.na_items

    if not totals:
        raise ScoringError(f"{what} split produced no scored cells")
    return _SplitResult(
        totals=totals,
        per_pillar={name: values for name, values in per_pillar.items() if values},
        scored_items=scored_items,
        na_items=na_items,
        kept_images=kept,
        hashes=hashes,
    )


def _na_rate(scored_items: int, na_items: int) -> float:
    denominator = scored_items + na_items
    if denominator <= 0:
        # Nothing was scored at all. Reported as fully declined, which the
        # control plane voids — never as a clean run with a zero rate.
        return 1.0
    return na_items / denominator


def _replay_evidence(
    cells: Sequence[Cell],
    request: ImageHarvestRequest,
    generator: CosmosGenerator,
    first_pass: _SplitResult,
) -> ReplayEvidence:
    """Regenerate the pinned cells and compare them with what was scored.

    Two independent comparisons, because they catch different things:

    * against the miner's `claimed_output_hashes`, which detects an artifact
      whose claimed outputs were produced by weights other than the submitted
      ones;
    * against this run's own first-pass image, which detects non-determinism
      in the generation path itself.

    Exact hashes are not required. Pixel determinism does not survive a driver
    change, so the descriptor distance is the tolerant path, and the control
    plane accepts either.
    """
    claimed = request.manifest.claimed_output_hashes
    exact = 0
    worst_drift = 0.0
    for cell in cells:
        prompt = request.prompt_text(cell.prompt_id)
        again = generator.generate(prompt, cell.seed)
        again_hash = sha256_bytes(again)
        if claimed.get(cell.key, "") == again_hash:
            exact += 1
        original = first_pass.kept_images.get(cell.key)
        if original is None:
            raise ScoringError(f"replay cell {cell.key} was not retained from the first pass")
        drift = cosine_distance(descriptor(original), descriptor(again))
        worst_drift = max(worst_drift, drift)
    return ReplayEvidence(
        cells_checked=len(cells),
        exact_hash_matches=exact,
        max_embedding_drift=worst_drift,
    )


def _faithfulness_evidence(
    cells: Sequence[Cell],
    request: ImageHarvestRequest,
    judge: QJudger,
    holdout: _SplitResult,
) -> FaithfulnessEvidence:
    """Agree, or disagree, with the Alignment pillar that was scored.

    Q-Judger is the only judge this challenge accepts, so the spot check cannot
    be a second model. It is a second *question*: a narrower pass that scores
    only concrete, checkable claims — object counts, rendered text, spatial
    relations — instead of the full five-pillar tree. Its verdict is compared
    with the Alignment pillar from the scored pass at the same threshold.

    Disagreement discards the run on the control plane rather than picking a
    winner between the two passes.
    """
    scored_alignment = holdout.per_pillar.get(ALIGNMENT.wire, {})
    agreements = 0
    checks = 0
    for cell in cells:
        image = holdout.kept_images.get(cell.key)
        if image is None:
            raise ScoringError(f"faithfulness cell {cell.key} was not retained")
        if cell.key not in scored_alignment:
            # The scored pass returned no Alignment for this cell, so there is
            # nothing to agree with. Skipping it silently would inflate the
            # agreement rate, so the check simply does not count.
            continue
        prompt = request.prompt_text(cell.prompt_id)
        spot = judge.spot_check_alignment(prompt, image)
        spot_value = spot.normalized_pillar(ALIGNMENT)
        if spot_value is None:
            continue
        checks += 1
        threshold = FAITHFULNESS_ALIGNMENT_THRESHOLD / 100.0
        if (spot_value >= threshold) == (scored_alignment[cell.key] >= threshold):
            agreements += 1
    return FaithfulnessEvidence(checks=checks, agreements=agreements)


def score_request(
    request: ImageHarvestRequest, generator: CosmosGenerator, judge: QJudger
) -> ImageDocument:
    """Measure every series and bind them to the run that was requested.

    # Raises
    [`ScoringError`], [`GeneratorError`], or [`JudgeError`] — all of which end
    the run without a document.
    """
    holdout_cells = cells_from(request.holdout_cells())
    public_cells = cells_from(request.public_cells())

    # Decide up front which cells the evidence passes will need, so only those
    # images are retained. A full split of 1024x1024 PNGs held in memory would
    # be gigabytes for no reason.
    replay_cells = select_cells(
        public_cells, REPLAY_CELLS, request.submission_digest, _REPLAY_DOMAIN
    )
    faithfulness_cells = select_cells(
        holdout_cells, MIN_FAITHFULNESS_CHECKS, request.submission_digest, _FAITHFULNESS_DOMAIN
    )

    holdout = _judge_split(
        holdout_cells,
        request,
        generator,
        judge,
        "holdout",
        frozenset(cell.key for cell in faithfulness_cells),
    )
    public = _judge_split(
        public_cells,
        request,
        generator,
        judge,
        "public",
        frozenset(cell.key for cell in replay_cells),
    )

    na_rate = _na_rate(holdout.scored_items, holdout.na_items)
    if na_rate > MAX_NA_RATE:
        # The judge declined most of the split, so there is no comparable
        # score here. Publishing the survivors would be a verdict drawn from
        # whichever cells happened to be scorable.
        raise ScoringError(
            f"judge N/A rate {na_rate:.3f} above the {MAX_NA_RATE:.3f} ceiling"
        )

    replay = _replay_evidence(replay_cells, request, generator, public)
    faithfulness = _faithfulness_evidence(faithfulness_cells, request, judge, holdout)

    log.info(
        "measured holdout=%d public=%d pillars=%d na_rate=%.4f replay=%d/%d faithfulness=%d/%d",
        len(holdout.totals),
        len(public.totals),
        len(holdout.per_pillar),
        na_rate,
        replay.exact_hash_matches,
        replay.cells_checked,
        faithfulness.agreements,
        faithfulness.checks,
    )

    return ImageDocument(
        identity=request.identity,
        measurement=ImageMeasurement(
            base_model=request.base_model,
            judge_model=request.judge_model,
            holdout=holdout.totals,
            public=public.totals,
            holdout_by_pillar=holdout.per_pillar,
            na_rate=na_rate,
            replay=replay,
            faithfulness=faithfulness,
            contaminated_prompt_ids=contaminated_ids(
                request.manifest.train_prompt_ids, request.holdout_ids()
            ),
        ),
    )


def split_mean(scores: ImageScore | dict[str, float]) -> float | None:
    """Mean of a measured series, for logs and tests."""
    if isinstance(scores, dict):
        return mean(scores.values())
    return scores.normalized_total()
