# dlp_grokking

A compliant neural model for `a*b mod p` built on the **discrete-log (DLP) grokking** idea
that came out of the design discussion. It is the most detailed worked example of how to turn
a mathematical insight into an *inductive bias* without crossing into a hand-coded algorithm —
including learning the reduction of the raw operands, which is now the accuracy bottleneck.

## The idea

Over the prime field, `a*b mod p` is the multiplicative group operation. Pick a generator
`g`; then every nonzero residue is `g^k`, and

```
a*b mod p   =   g^((log_a + log_b) mod (p-1))
```

so multiplication becomes **addition in log space**. The grokking literature (Power et al.;
Nanda et al.'s "Progress measures for grokking") has shown that small Transformers *can*
learn modular **addition** by discovering a Fourier / discrete-log representation on their own.

We give the network exactly one structural nudge toward that solution and let it learn
everything else from data:

```
r_a = Reduce(a, p)             # LEARNED streaming reduction (recurrent cell)
r_b = Reduce(b, p)             # same weights
e_a = Enc(r_a, p)              # shared residue encoder
e_b = Enc(r_b, p)              # same weights
z   = e_a + e_b                # ADDITIVE bottleneck  <- the only DLP bias
ans = Dec(z, p)                # decoder emits answer digits
```

The additive combination is the whole inductive bias. The embedding that turns a residue
into something log-like, and the decoder that turns a sum-in-log-space back into a residue,
are **all trained parameters**. There is no precomputed discrete-log table, no generator
search, no `(log_a + log_b) % (p-1)` written in Python. Perturb the weights and the accuracy
collapses — the operational test for "the answer came from learning, not a hand-coded
circuit" (rules/evaluation.md, Principle 2).

## Architecture

| | |
|---|---|
| Residue encoder | Transformer encoder, shared across the a- and b-branch |
| d_model | 256 |
| Layers / heads | 3 / 8 |
| Feedforward | 768 |
| Bottleneck | `z = e_a + e_b` (additive, log-space) |
| Decoder | re-encodes `p` as context, MLP → 5 digit heads (base-10) |
| Answer width | 5 digits, MSB-first, zero-padded |
| Params | ~6M |

Reduction of the raw operands is **learned**, not hand-coded: the rules prohibit computing
`a % p` / `b % p` in non-learned code, so a `StreamingDigitReducer` — a trained recurrent cell —
consumes each raw operand's decimal digits one at a time (a fixed, feedback-free feeding
schedule: the deterministic-encoder pattern the rules explicitly allow) and emits the residue.
Its trained weights carry out the DFA update `r' = (10*r + d) mod p`; randomise them and the
residues, and hence the answers, collapse. For `p >= 10^5` (out of the small-prime regime the
model can learn) the model emits `0` without invoking the network — the honest fallback rather
than a guess.

## Training

Two stages (the second merges into the same `weights.pt`):

```bash
.venv312/bin/python examples/dlp_grokking/train.py --minutes 8    # 1. DLP core (residue space)
.venv312/bin/python examples/dlp_grokking/train_reducer.py        # 2. learned streaming reducer
```

**Stage 1 — DLP core.** Synthetic `(a mod p, b mod p, p) -> (a*b mod p)` sampled across
**random** primes spanning the tier 1-3 bit ranges, with a curriculum bias toward the small
primes the network can actually generalise over. (Using `%` to *synthesise training data* is
fine — the prohibition on deterministic reduction applies at inference time, to the path that
produces answers.) Primes are split into a train pool and a **held-out val pool**, so the val
metric measures generalisation to *unseen primes* — the thing that separates "learned the field
structure" from "memorised one prime". Cross-entropy on the fixed-width answer digits, AdamW with
cosine decay, Apple MPS. Best-by-val checkpoint is saved to `weights.pt`.

**Stage 2 — streaming reducer.** Trained on raw digit streams with **per-prefix residue
supervision** (the ground-truth trajectory of `r' = (10*r + d) mod p` after every digit), over
the same prime buckets. Per-prefix supervision is what makes the learned transition track the
exact residue instead of drifting over long operands.

Evaluate through the real pipeline (manifest → static check → load → determinism → inference →
decode → score):

```bash
.venv312/bin/python examples/dlp_grokking/eval_tiers.py examples/dlp_grokking
```

## Score (public benchmark, seed = deadbeef × 8)

| Total problems | overall_accuracy | deterministic |
|---|---|---|
| **1100** | **0.093** | True |

Per-tier at total=1100:

| Tier | Accuracy | Notes |
|------|----------|-------|
| 1 | **0.740** | 4 fixed primes {2,3,5,7}. The DLP core groks this regime perfectly *in residue space*; the 26% loss is entirely the learned reducer mis-reducing some raw 32-bit operands |
| 2 | 0.050 | random primes in [16,255], 48-bit raw operands; reducer errors compound with the core's own tier-2 ceiling |
| 3 | 0.000 | ~6500 primes, 64-bit raw operands; neither stage survives |
| 4-10 | 0.020 | `p >= 10^5` → honest 0 fallback; scores come from a=0 / b=0 edge cases |

This now trails `digit_transformer` (0.103) overall: forcing the reduction into learned weights
costs this two-stage design more than it costs the end-to-end Transformer. An earlier revision
that hand-reduced the operands in Python scored 0.127 with a perfect tier 1 — that pattern is a
prohibited practice (deterministic pre-reduction), and the delta is the honest price of the rule.

## What we learned (the honest ceiling)

A decisive A/B test (`exploration/_dlp_grokking_dev/experiment_ab.py`) trained the **same**
network twice, changing only the bottleneck:

- **A — additive** `z = e_a + e_b` (the DLP bias): tier-2 val 0.075
- **B — concat** `z = [e_a; e_b]` (generic learned interaction): tier-2 val 0.057

They land in the same place. So the additive DLP bottleneck is **not** what caps tier 2 — the
real wall is that learning modular multiplication that *generalises across many unseen random
primes* is intrinsically hard at this scale. In residue space, tier 1 groks perfectly because
it is the exact single-/few-prime regime the grokking papers solve; tier 2+ asks the network to
generalise the field operation across thousands of primes it has limited samples for, and it
plateaus.

The learned reducer adds a second, independent ceiling. Its training loss flattens at ~0.20
(spot exact-match on long operands: ~0.56 for {2,3,5,7}, ~0.12 for 8-bit primes), and every
mis-reduced operand poisons an otherwise-solved tier-1 problem — that is the entire gap between
tier 1 at 1.000 (residue-space core alone) and 0.740 (end-to-end). A GRU cell asked to emulate
the DFA `r' = (10*r + d) mod p` for *arbitrary* `p` drifts over long digit streams; per-prefix
supervision slows the drift but does not eliminate it. Ideas that might close the gap — a wider
recurrent state, prime-conditioned transition matrices, curriculum over operand length — are
left open deliberately: this is a reference example, not the ceiling.

That outcome is expected and honest. More compute would mostly buy overfitting to the *public*
seed (which would not transfer to the secret official seed), not real tier-2+ capability. We
ship the genuine result rather than chase a gamed one.

## Status under the rules

Compliant:

- Per-argument preprocess hooks are pass-through identities — no cross-argument leakage.
- `predict_digits` receives the **raw** operands; reduction is performed by the trained
  `StreamingDigitReducer` (deterministic pre-reduction in non-learned code is a prohibited
  practice). The digit-feeding loop is fixed and feedback-free — the deterministic-encoder
  pattern the rules allow; the arithmetic of the update is entirely in trained weights.
- No discrete-log table, generator search, or hand-coded modular arithmetic — the reduction and
  the multiplication are both in trained weights. Perturbing them degrades accuracy.
- Passes `modchallenge check` static analysis.
- Deterministic (`eval()` mode, no dropout, no sampling).
