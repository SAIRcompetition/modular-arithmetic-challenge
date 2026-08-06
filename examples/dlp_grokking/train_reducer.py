"""Train the StreamingDigitReducer (learned modular reduction) and merge it
into the dlp_grokking checkpoint.

The competition rules prohibit deterministically pre-reducing the operands
(``a % p`` / ``b % p`` in non-learned code): the model must receive the raw
``(a, b, p)`` and the reduction itself must come from trained parameters.
This script trains the recurrent reducer cell that replaces the old
hand-coded reduction: it consumes raw decimal digits one at a time and is
supervised on the residue of *every prefix*, which is exactly the DFA
transition ``r' = (10*r + d) mod p`` — learned, not hard-coded.

The trained reducer is merged into the existing ``weights.pt`` (the DLP core
encoder/decoder weights are unchanged — they already operate on residues).

Usage:
    .venv312/bin/python examples/dlp_grokking/train_reducer.py [--steps 12000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import StreamingDigitReducer, WIDTH, _digits_fixed  # noqa: E402
from train import sieve_primes, PRIME_LIMIT  # noqa: E402

# Tier prime buckets (see train.py) and the raw operand digit lengths that
# co-occur with them at eval time (tier operand_bits -> max decimal digits):
#   tier 1: p in {2,3,5,7},    operands up to 32 bits (~10 digits)
#   tier 2: p in [8, 256),     operands up to 48 bits (~15 digits)
#   tier 3: p in [256, 65536), operands up to 64 bits (~20 digits)
# Weights favour the regimes where the DLP core can actually use the
# residues; tier 3 gets token exposure (honestly not learnable to high
# accuracy — ~6500 distinct automata for a small cell).
BUCKET_SPEC = (
    # (fixed_primes_or_range, max_operand_digits, weight)
    ((2, 3, 5, 7), 10, 0.40),
    ((8, 256), 15, 0.45),
    ((256, PRIME_LIMIT), 20, 0.15),
)


def build_pools(seed: int = 0):
    primes = sieve_primes(PRIME_LIMIT)
    pools = []
    for spec, max_digits, weight in BUCKET_SPEC:
        if len(spec) == 4:  # fixed tier-1 primes
            pool = list(spec)
        else:
            lo, hi = spec
            pool = [p for p in primes if lo <= p < hi]
        pools.append((pool, max_digits, weight))
    return pools


def sample_row(pools, rng: random.Random, max_len: int):
    """Sample (digit_row, prime) — raw decimal digits, MSB-first."""
    pool, max_digits, _ = rng.choices(pools, weights=[w for _, _, w in pools])[0]
    p = pool[rng.randrange(len(pool))]
    if rng.random() < 0.05:
        n = rng.choice((0, 1))  # edge cases, as in the test set
    else:
        ndigits = rng.randint(1, max_digits)
        n = rng.randrange(0, 10**ndigits)
    row = [int(c) for c in str(n)]
    # Occasionally prepend explicit leading zeros so batched left-padding is
    # in-distribution (value-neutral).
    if rng.random() < 0.2:
        row = [0] * rng.randint(1, 3) + row
    return row[:max_len], p


def make_batch(pools, batch_size: int, rng: random.Random, device):
    rows, primes = [], []
    for _ in range(batch_size):
        row, p = sample_row(pools, rng, max_len=24)
        rows.append(row)
        primes.append(p)
    max_len = max(len(r) for r in rows)
    ops, targets, p_rows = [], [], []
    for row, p in zip(rows, primes):
        padded = [0] * (max_len - len(row)) + row
        ops.append(padded)
        # Supervise EVERY prefix: r_t = int(d_1..d_t) mod p. This is the
        # ground-truth DFA trajectory the cell must learn to track.
        r = 0
        traj = []
        for d in padded:
            r = (10 * r + d) % p
            traj.append(_digits_fixed(r))
        targets.append(traj)
        p_rows.append(_digits_fixed(p))
    return (
        torch.tensor(ops, dtype=torch.long, device=device),
        torch.tensor(p_rows, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),  # (B, L, WIDTH)
    )


@torch.no_grad()
def spot_check(reducer, pools, device, n: int = 300, seed: int = 123):
    """Exact-match accuracy of the final residue, per bucket."""
    rng = random.Random(seed)
    results = []
    for pool, max_digits, _ in pools:
        correct = total = 0
        for _ in range(n):
            p = pool[rng.randrange(len(pool))]
            ndigits = rng.randint(1, max_digits)
            v = rng.randrange(0, 10**ndigits)
            ops = torch.tensor(
                [[int(c) for c in str(v)]], dtype=torch.long, device=device
            )
            p_t = torch.tensor([_digits_fixed(p)], dtype=torch.long, device=device)
            pred = reducer(ops, p_t).argmax(dim=-1)[0].tolist()
            want = _digits_fixed(v % p)
            correct += int(pred == want)
            total += 1
        results.append(correct / total)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--ckpt", type=Path, default=HERE / "weights.pt")
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training reducer on {device}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pools = build_pools(args.seed)

    reducer = StreamingDigitReducer(d_model=args.d_model).to(device)
    n_params = sum(p.numel() for p in reducer.parameters())
    print(f"Reducer params: {n_params:,}")

    opt = torch.optim.AdamW(reducer.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    for step in range(args.steps):
        ops, p_t, targets = make_batch(pools, args.batch_size, rng, device)
        logits = reducer(ops, p_t, all_steps=True)  # (B, L, WIDTH, 10)
        loss = ce(logits.reshape(-1, 10), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reducer.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            print(
                f"step {step+1:6d}  loss={loss.item():.4f}  "
                f"elapsed={time.time()-t0:.1f}s"
            )
        if (step + 1) % 2000 == 0:
            reducer.eval()
            accs = spot_check(reducer, pools, device)
            reducer.train()
            print(f"  spot exact-match by bucket (t1/t2/t3): {accs}")

    reducer.eval()
    accs = spot_check(reducer, pools, device, n=500)
    print(f"FINAL spot exact-match by bucket (t1/t2/t3): {accs}")

    # Merge into the existing checkpoint (core weights untouched).
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ckpt["reducer_state_dict"] = {
        k: v.cpu() for k, v in reducer.state_dict().items()
    }
    ckpt["reducer_config"] = reducer.config
    torch.save(ckpt, args.ckpt)
    print(f"Merged reducer into {args.ckpt}")


if __name__ == "__main__":
    main()
