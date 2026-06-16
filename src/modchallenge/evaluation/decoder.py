"""Decoder for model outputs.

Contestants emit answers as lists of base-B digits via
``ModularMultiplicationModel.predict_digits``. The harness — not the
submission — converts those digits into the canonical decimal answer using
this module.

This is the only step that produces the final answer integer, so the
attacker has no opportunity to inject computation here.
"""

from __future__ import annotations

from typing import Union

from modchallenge.interface.submission_schema import (
    MAX_OUTPUT_BASE,
    OUTPUT_BASE_PRIME_SENTINEL,
)


class MalformedOutput(ValueError):
    """Raised when a model's emitted digit list cannot be decoded."""


def resolve_base(base: Union[int, str], prime: int) -> int:
    """Resolve a manifest ``output_base`` value to an integer base.

    The sentinel ``"p"`` means 'use the current prime as the base'.
    """
    if isinstance(base, str):
        if base == OUTPUT_BASE_PRIME_SENTINEL:
            if prime < 2:
                raise MalformedOutput(
                    f"prime={prime} is not a valid base (must be >= 2)"
                )
            return prime
        raise MalformedOutput(
            f"invalid output_base string: {base!r} "
            f"(only {OUTPUT_BASE_PRIME_SENTINEL!r} is allowed)"
        )
    if isinstance(base, int) and not isinstance(base, bool):
        if not (2 <= base <= MAX_OUTPUT_BASE):
            raise MalformedOutput(
                f"output_base {base} out of range [2, {MAX_OUTPUT_BASE}]"
            )
        return base
    raise MalformedOutput(
        f"output_base must be int or string {OUTPUT_BASE_PRIME_SENTINEL!r}; "
        f"got {type(base).__name__}"
    )


def decode_answer(
    digits: object,
    *,
    base: Union[int, str],
    prime: int,
    is_tier_zero: bool = False,
) -> int:
    """Decode a list of base-B digits (MSB-first) into the integer answer.

    Args:
        digits: The model's emitted output. Expected to be a ``list`` of
            ``int`` values, each in ``[0, base - 1]``.
        base: Manifest's ``output_base`` value (int or ``"p"`` sentinel).
        prime: The modulus ``p`` for this problem (decimal-string ``p`` parsed
            to int). Used for the ``"p"`` sentinel and to bound the decoded
            value on scored tiers.
        is_tier_zero: If True, the decoded value may exceed ``prime``
            (Tier 0 is pure multiplication with no modular reduction).

    Returns:
        The decoded integer answer.

    Raises:
        MalformedOutput: if ``digits`` is not a list, any digit is the wrong
            type or out of range, or (on scored tiers) the decoded value is
            ``>= prime``.
    """
    actual_base = resolve_base(base, prime)

    if not isinstance(digits, list):
        raise MalformedOutput(
            f"predict_digits must return list[int]; got {type(digits).__name__}"
        )

    value = 0
    for i, d in enumerate(digits):
        if isinstance(d, bool) or not isinstance(d, int):
            raise MalformedOutput(
                f"digit at position {i} has type {type(d).__name__}; "
                f"expected int"
            )
        if not (0 <= d < actual_base):
            raise MalformedOutput(
                f"digit at position {i} = {d} out of range [0, {actual_base})"
            )
        value = value * actual_base + d

    if not is_tier_zero and value >= prime:
        raise MalformedOutput(
            f"decoded value {value} >= prime {prime} on scored tier"
        )

    return value


def encode_answer(
    value: int,
    *,
    base: Union[int, str],
    prime: int,
    width: int | None = None,
) -> list[int]:
    """Inverse of :func:`decode_answer`: encode ``value`` as base-B digits.

    Provided as a helper for contestants and tests; not part of the
    evaluation pipeline. Returns digits MSB-first. If ``width`` is given,
    the result is left-padded with zeros to that length.
    """
    actual_base = resolve_base(base, prime)
    if value < 0:
        raise ValueError(f"cannot encode negative value {value}")
    if value == 0:
        digits = [0]
    else:
        digits = []
        v = value
        while v > 0:
            digits.append(v % actual_base)
            v //= actual_base
        digits.reverse()
    if width is not None:
        if len(digits) > width:
            raise ValueError(
                f"value {value} requires {len(digits)} digits in base "
                f"{actual_base}, exceeds requested width {width}"
            )
        digits = [0] * (width - len(digits)) + digits
    return digits
