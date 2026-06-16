"""Unified model interface for the Modular Arithmetic Challenge.

All submissions implement ``ModularMultiplicationModel``.

Pipeline contract (per problem ``(a, b, p)``)::

    a_enc = model.preprocess_a(a)
    b_enc = model.preprocess_b(b)
    p_enc = model.preprocess_p(p)
    digits = model.predict_digits(a_enc, b_enc, p_enc)
    answer = pipeline_decoder(digits, base=manifest.output_base, prime=int(p))

The decoder is implemented by the harness, not by the submission. Contestants
declare the base they want their model to emit answers in via the
``output_base`` field in ``manifest.json``. The decoder reads the returned
digits MSB-first and reconstructs the integer answer.

Each ``preprocess_*`` function may only access its own argument. Crossing
arguments (e.g. reading ``b`` inside ``preprocess_a``) defeats the structural
anti-cheat design and is checked by the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModularMultiplicationModel(ABC):
    """Abstract base class that all submissions must implement.

    The model may use any architecture and any internal representation,
    subject to the rules in ``rules/evaluation.md``. The only fixed contract
    is the interface below. The one substantive constraint on what happens
    inside the model: the answer must be produced by *trained parameters*,
    not by an arithmetic algorithm hand-coded into the forward pass (see
    Prohibited Practices in ``rules/evaluation.md`` — that is a computational
    circuit, not a learned model).
    """

    # -- lifecycle ------------------------------------------------------

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Load model weights and config from ``model_dir``.

        Called exactly once before any ``preprocess_*`` or ``predict_digits``
        call. Time spent here is bounded separately from the inference budget.
        """

    # -- per-argument preprocessing (optional; defaults are identity) ---

    def preprocess_a(self, a: str) -> Any:
        """Convert the decimal string ``a`` into the model's input representation.

        The default returns ``a`` unchanged. Override to perform tokenisation,
        base conversion, embedding lookup, etc.

        **Must depend only on ``a``.** Reading the value of ``b`` or ``p``
        from instance state populated by a previous call is a rule violation
        and the pipeline runs a sanity check for the simplest forms of this.
        """
        return a

    def preprocess_b(self, b: str) -> Any:
        """Convert the decimal string ``b``. Per-argument; see ``preprocess_a``."""
        return b

    def preprocess_p(self, p: str) -> Any:
        """Convert the decimal string ``p``. Per-argument; see ``preprocess_a``."""
        return p

    # -- inference ------------------------------------------------------

    @abstractmethod
    def predict_digits(self, a_enc: Any, b_enc: Any, p_enc: Any) -> list[int]:
        """Return the answer ``(a * b) mod p`` as a list of base-B digits.

        Digits are **most-significant-first**; each must be ``int`` in
        ``[0, B - 1]`` where ``B`` is the value declared in the
        manifest's ``output_base`` field (or the current prime ``p`` if the
        manifest declares ``output_base = "p"``).

        For Tier 0 (pure multiplication, no modular reduction) the decoded
        value may exceed ``p``. On scored tiers, decoded ``value >= p`` is
        treated as malformed.

        Malformed outputs (wrong type, out-of-range digit, oversize value on
        a scored tier) score 0 for that problem; they do not abort the run.
        """

    def predict_digits_batch(
        self, inputs: list[tuple[Any, Any, Any]]
    ) -> list[list[int]]:
        """Batched form of :meth:`predict_digits`.

        Default implementation calls :meth:`predict_digits` one item at a
        time. Override for GPU batching.

        Returning a list whose length differs from ``len(inputs)`` is a
        batch-contract failure and marks the entire tier as incomplete
        (score 0%).
        """
        return [self.predict_digits(a, b, p) for a, b, p in inputs]

    def max_batch_size(self) -> int:
        """Maximum comfortable batch size. The pipeline will not exceed this.

        Default is 1 (sequential). Override for GPU-batched models.
        """
        return 1
