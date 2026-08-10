# digit_transformer

Small decoder-only Transformer trained from scratch on synthetic `(a, b, p) → a*b mod p` examples. Per-argument preprocess hooks pass through; `predict_digits` feeds the **raw decimal strings** to the model, which must learn both the reduction of the full-width operands and the modular multiplication end-to-end.

## Raw inputs — no pre-reduction

The rules prohibit deterministically computing `a % p` / `b % p` in non-learned code before the model runs: reducing the full-width operands is itself a core part of the task (operands are generated much larger than `p` precisely so that genuine reduction is tested). An earlier revision of this example reduced the operands inside `predict_digits` and trained on reduced data; following the organizer clarification, that pattern is a prohibited practice, and this model was retrained on the raw distribution. This makes the learning problem genuinely harder — the honest cost of the rule.

Training data is generated in the same raw form (`train_digit_transformer.py` samples full-width operands per tier), so train and inference distributions match.

## Architecture

| | |
|---|---|
| Type | Decoder-only Transformer (causal LM) |
| Vocab | 15 tokens: digits 0-9 + `SEP`, `EQ`, `BOS`, `EOS`, `PAD` |
| d_model | 128 |
| Layers | 4 |
| Heads | 4 |
| Feedforward | 256 |
| Params | ~544K |
| Max seq len | 80 |

## Training

Mixed Tier 1-3 problem distribution (uniform 1/3 each) with **raw full-width operands**, 20000 steps, batch=64, AdamW lr=3e-4. ~9 min on Apple Silicon MPS. Sequence format: `BOS a_digits SEP b_digits SEP p_digits EQ answer_digits EOS`. Causal-LM loss masked to the answer-token positions only.

```bash
python examples/exploration/train_digit_transformer.py --steps 20000
```

The training script lives **outside** the submission directory (in `examples/exploration/train_digit_transformer.py`, which is gitignored and therefore not part of a clone) because it imports `sympy` for prime generation, which the static check rejects in submissions. `dlp_grokking` ships its trainers in-repo and is the example to copy if you want a runnable training loop. Contestants train locally, then push only the submission directory (`manifest.json`, `model.py`, `weights.pt`) to HuggingFace.

## Score (public benchmark, seed = deadbeef × 8)

| Total problems | overall_accuracy | highest_tier_above_90 | deterministic |
|---|---|---|---|
| 110 | 0.190 | 1 | True |
| **1100** | **0.094** | **-1** | True |

(`highest_tier_above_90` differs between the two rows because tier 1 sits right at the 90%
threshold: with 10 problems/tier it clears it, with 100 problems/tier it does not.)

Per-tier at total=1100:

| Tier | Accuracy | Notes |
|------|----------|-------|
| 0 | 0.070 | No modulus — Tier 0 is pure multiplication, model wasn't trained for it |
| 1 | **0.830** | 4 fixed primes {2, 3, 5, 7}, raw operands up to 32 bits. The model must learn the reduction itself: mod 2/5 (last digit) and mod 3 (digit sum) are learnable patterns; mod 7's positional-weight cycle over ~10 digits is harder — hence below the old pre-reduction version's 1.000 |
| 2 | 0.040 | Random primes in [16, 255] with 48-bit raw operands; reduction + multiplication both unlearned at this scale |
| 3 | 0.020 | ~6500 primes, 64-bit raw operands; edge cases only |
| 4-6 | 0.010-0.020 | Untrained; the raw prompt no longer fits the 80-token context, so the model answers from a clipped window and is right only on trivial `a`/`b ∈ {0,1}` cases |
| 7-10 | 0.000 | Nothing survives. An earlier revision returned `[]` for over-long prompts, which decodes to `0` and collected the same ~2% these tiers used to show; that was non-learned code emitting an answer, so it is gone |

## What the math-loop notes about this result

This example illustrates two of the methodology's principles:

- **Step 1 (understand the problem)**: a transformer of 544K params was never going to crack tier 2+ on raw inputs. The training-loss plateau at ~1.42 over 20000 steps was a clear signal of saturation; we acknowledged it and shipped the partial tier-1 result (0.83) as the genuine outcome rather than chasing tier 2 with the same shape.
- **Step 4 micro-iteration "change something each round"**: an early revision pivoted to reducing the operands inside `predict_digits` and training on reduced data, which locked in tier 1 immediately — but the organizer clarification later ruled that deterministic pre-reduction is a prohibited practice (the model must receive the raw `(a, b, p)`). The example was retrained end-to-end on raw operands: the model now has to learn the reduction itself, which is the honest, harder version of the task. The lesson stands in inverted form — when a "one-line interface change" makes the task dramatically easier, check whether it moved work across the learned/deterministic boundary.

For pushing higher, the natural next steps are a bigger model (a few M params), more capable tokenisation (Charton-style), and dedicated tier 2 / tier 3 fine-tuning. Out of scope for this example, on purpose: this is the smallest honest neural baseline.

## Status under the rules

Honest baseline:

- Per-argument preprocess functions are pass-through identities. No cross-argument leakage.
- `predict_digits` feeds the **raw** decimal strings to the model — no deterministic pre-reduction (`a % p` / `b % p` in non-learned code is a prohibited practice).
- The model's emitted digit list materially determines the answer — the trained weights are doing the reduction *and* the modular multiplication.
- **No non-learned code emits an answer.** Every problem is answered by the network; there is no fallback value and no branch on `a`, `b` or `p` that short-circuits it. Prompts longer than the trained context are clipped to the rightmost window — a tokenisation limit imposed by the finite positional-embedding table, not a decision about the answer. (The rules do not forbid an out-of-range fallback; these reference models simply hold to the stricter line so that the boundary is unambiguous.)
- Passes the `modchallenge check` static analysis.
- Deterministic (`eval()` mode, no dropout, no sampling).
