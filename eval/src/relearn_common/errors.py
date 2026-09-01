"""The one error every refusal in an eval image derives from."""

from __future__ import annotations


class ContractError(RuntimeError):
    """A request, document, or transcript the image refuses.

    Every failure path in an eval image raises this or a subclass, and the CLI
    turns it into a non-zero exit with no document and no completion marker.
    There is no path that downgrades a refusal into a low score: a number
    nobody measured is worse than a 503.
    """
