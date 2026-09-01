"""Binding a document to the run that asked for it.

Every Relearn eval image measures something different, and every one of them
answers the same four questions the same way: which submission, which artifact,
which image, which holdout. A document that cannot answer all four is not a
score of anything — it is numbers with no provenance, and a replayed or
misdirected transcript would be indistinguishable from a real one.

The checks live here so all three challenges agree on them literally, rather
than each drifting its own way about, say, whether digest comparison is
case-sensitive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import ContractError


def is_hex(value: str, length: int) -> bool:
    """Whether `value` is exactly `length` hex characters."""
    trimmed = value.strip()
    return len(trimmed) == length and all(char in "0123456789abcdefABCDEF" for char in trimmed)


def is_sha256_pin(value: str) -> bool:
    """Whether `value` is a `sha256:<64 hex>` image pin.

    The control plane only ever rents a digest-pinned image (`can_rent` on
    every pin type checks exactly this), so a request that names the image any
    other way is a request the image must not honour.
    """
    trimmed = value.strip()
    return trimmed.startswith("sha256:") and is_hex(trimmed.removeprefix("sha256:"), 64)


def same(left: str, right: str) -> bool:
    """Identity comparison for run fields the control plane treats literally."""
    return left.strip() == right.strip()


def same_digest(left: str, right: str) -> bool:
    """Identity comparison for hex digests, which are case-insensitive."""
    return left.strip().lower() == right.strip().lower()


@dataclass(frozen=True)
class RunIdentity:
    """Who this run is, echoed from request to document unchanged."""

    challenge_id: str
    submission_digest: str
    artifact_digest: str
    eval_image_digest: str
    holdout_commitment: str

    def to_wire(self) -> dict[str, object]:
        return {
            "challenge_id": self.challenge_id,
            "submission_digest": self.submission_digest,
            "artifact_digest": self.artifact_digest,
            "eval_image_digest": self.eval_image_digest,
            "holdout_commitment": self.holdout_commitment,
        }

    @classmethod
    def from_wire(cls, body: Mapping[str, object]) -> RunIdentity:
        return cls(
            challenge_id=str(body.get("challenge_id", "") or ""),
            submission_digest=str(body.get("submission_digest", "") or ""),
            artifact_digest=str(body.get("artifact_digest", "") or ""),
            eval_image_digest=str(body.get("eval_image_digest", "") or ""),
            holdout_commitment=str(body.get("holdout_commitment", "") or ""),
        )

    def validate(self, *, challenge_ids: tuple[str, ...]) -> None:
        """Refuse an identity the image cannot bind a score to.

        # Raises
        [`ContractError`] when the challenge is not one this image scores, when
        any identity field is missing, when the image is not named by digest,
        or when the holdout commitment is not a 64-hex digest.
        """
        if self.challenge_id.strip() not in challenge_ids:
            raise ContractError(
                f"challenge_id {self.challenge_id!r} is not scored by this image "
                f"(expected one of {', '.join(challenge_ids)})"
            )
        if not self.submission_digest.strip():
            raise ContractError("no submission_digest: nothing to bind the run to")
        if not self.artifact_digest.strip():
            raise ContractError("no artifact_digest: nothing to bind the run to")
        if not is_sha256_pin(self.eval_image_digest):
            raise ContractError("eval_image_digest is not a sha256: pin")
        if not is_hex(self.holdout_commitment, 64):
            raise ContractError("holdout_commitment is not 64 hex chars")

    def check_document(self, other: RunIdentity) -> None:
        """Refuse a document that belongs to some other run.

        # Raises
        [`ContractError`] on any identity field the document did not echo back
        exactly. This is what makes another artifact's numbers, a replayed
        submission, and another image's output all failures rather than scores.
        """
        if not same(other.challenge_id, self.challenge_id):
            raise ContractError("document challenge_id is not this run's challenge")
        if not same(other.submission_digest, self.submission_digest):
            raise ContractError("document submission_digest is not the frozen run")
        if not same_digest(other.artifact_digest, self.artifact_digest):
            raise ContractError("document artifact_digest is not the scored artifact")
        if not same(other.eval_image_digest, self.eval_image_digest):
            raise ContractError("document eval_image_digest is not the pinned image")
        if not same_digest(other.holdout_commitment, self.holdout_commitment):
            raise ContractError("document holdout_commitment does not match the request")
