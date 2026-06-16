# Modular Arithmetic Challenge

Can a neural network learn modular multiplication efficiently?

Given two large integers `a`, `b` and a prime `p`, compute `(a * b) mod p`. Operands can be hundreds of digits long — far beyond what fits in a 64-bit integer. Your model must **learn** to compute the answer — the answer must come from trained parameters, not from symbolic-math libraries, Python's built-in arbitrary-precision arithmetic, or an arithmetic algorithm hand-coded over the inputs (in Python **or** tensor operations). Submissions implement a narrow interface (three per-argument preprocessing hooks plus `predict_digits` returning the answer as base-`B` digits); the harness-provided decoder does the rest. See [rules/overview.md](rules/overview.md#prohibited-practices) for the precise rules.

See **[rules/overview.md](rules/overview.md)** for full competition rules, background, scoring, and submission workflow. For evaluation details (sandbox, test generation, time and resource budgets) see **[rules/evaluation.md](rules/evaluation.md)**, and for related prior work see **[rules/literature.md](rules/literature.md)**.

This repository contains the open-source evaluation system used for both local testing and official evaluation.

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch  # required for example models

# Run example models from examples/examples.json (private models via HF token)
modchallenge evaluate-example --group private --total 110

# Evaluate a single HuggingFace submission
modchallenge evaluate-hf user/my-model <commit_hash> --token hf_xxx

# Quick-test any HuggingFace LLM (exploratory, not ranked)
modchallenge evaluate-llm Qwen/Qwen2.5-0.5B-Instruct

# Run tests
pytest
```

## Project Structure

```
src/modchallenge/
  interface/        Model ABC + manifest schema (what contestants implement)
  testgen/          Test case generation (CSPRNG, 11 difficulty tiers)
  evaluation/       Pipeline, scorer, decoder, LLM wrapper, model loader
  security/         AST-based static-analysis scanner (pre-load anti-cheat)
  leaderboard/      JSON-based ranking store
  cli.py            CLI entry point

examples/
  examples.json       Public HF baseline registry (Qwen)
  README.md           Reference models + step-by-step submission guide
  always_zero/        Compliant baseline: emits [0]
  digit_transformer/  Compliant baseline: ~544K-param Transformer
  dlp_grokking/       Compliant baseline: ~6M-param discrete-log model

rules/
  overview.md       Competition rules, background, scoring, anti-cheating policy
  evaluation.md     Evaluation setup, sandbox, test generation, scoring details
  literature.md     Bibliography of related prior work
public_benchmark/   1100 test cases (fixed seed, answers included)
tests/              pytest test suite
```

## Difficulty Tiers

11 tiers. Tier 0 is a diagnostic (pure multiplication, unscored). Tiers 1-10 are scored.

Each tier independently controls both the prime size and the operand size. Operands grow progressively across tiers, so every tier tests a distinct combination of multiplication scale and modular reduction complexity.

| Tier | Prime p (bits) | Operands a, b (bits) | ~Decimal digits of a, b |
|------|---------------|----------------------|-------------------------|
| 0    | diagnostic    | 1-4096               | 1-1233 (10 sub-levels)  |
| 1    | 1-3           | up to 32             | up to 10                |
| 2    | 4-8           | up to 48             | up to 15                |
| 3    | 9-16          | up to 64             | up to 19                |
| 4    | 17-32         | up to 96             | up to 29                |
| 5    | 33-64         | up to 128            | up to 39                |
| 6    | 65-128        | up to 256            | up to 77                |
| 7    | 129-256       | up to 512            | up to 154               |
| 8    | 257-512       | up to 1024           | up to 309               |
| 9    | 513-1024      | up to 2048           | up to 617               |
| 10   | 1025-2048     | up to 4096           | up to 1233              |

Tier 0 covers the full operand range (10 sub-levels from 1-digit to 1233-digit) as pure multiplication without modular reduction. This allows diagnosing whether a model fails at high tiers due to multiplication capacity or modular reduction ability.

## CLI Commands

| Command | Description |
|---------|-------------|
| `modchallenge evaluate <dir>` | Evaluate a local submission directory |
| `modchallenge evaluate-hf <repo> <hash>` | Evaluate a HuggingFace submission (trusted-use only, no sandbox) |
| `modchallenge evaluate-sandboxed <dir>` | Evaluate a local submission inside the sandbox Docker image |
| `modchallenge evaluate-example` | Run all models from examples/examples.json |
| `modchallenge evaluate-llm <model_id>` | Quick-test any HF LLM (exploratory, not ranked) |
| `modchallenge check <dir>` | Static-analyze a submission for prohibited code patterns |
| `modchallenge build-sandbox` | Build the sandbox Docker image |
| `modchallenge leaderboard` | Display the leaderboard |
| `modchallenge generate-public <dir>` | Generate the public benchmark test set |

## Test Generation

- Master seed: `secrets.token_bytes(32)` (secret) or fixed (public benchmark)
- Per-tier seeds derived via `HMAC-SHA256(master_seed, tier_id)`
- 5 different primes per tier to prevent overfitting
- Edge cases: a=0, b=0, a=1, b=1 in every tier
- Public benchmark uses a fixed seed so results are reproducible

## Organizers

Alberto Alfarano, François Charton, Yongzheng Jia, Kristin Lauter, Cathy Li, Terence Tao, Emily Wenger

## Literature

Note: this bibliography is not intended to be comprehensive. Some of the papers study the "grokking" phenomenon that can occur for very small moduli, but this phenomenon may not necessarily be applicable to the medium-sized moduli considered in this competition.

- David Demitri Africa, Sara M. Kapoor, Theo Simon Sorg, [Learning modular exponentiation with transformers](https://arxiv.org/html/2506.23679) (2025)

- François Charton, Julia Kempe. [Emergent Properties with Repeated Examples](https://arxiv.org/pdf/2410.07041).

- Darshil Doshi, Tianyu He, Aritra Das, Andrey Gromov. [Grokking modular polynomials](https://arxiv.org/pdf/2406.03495). 

- Andrey Gromov. [Grokking modular arithmetic](https://arxiv.org/pdf/2301.02679).

- Tianyu He, Darshil Doshi, Aritra Das, Andrey Gromov. Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks. Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS), 2024.

- Kristin Lauter, Cathy Y. Li, Krystal Maughan, Rachel Newton, Megha Srivastava. Machine Learning for Modular Multiplication. Research Directions in Number Theory (Springer 978-3-032-11182-1).

- Chenyang Li, Yingyu Liang, Zhenmei Shi, Zhao Song, Tianyi Zhou, [Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs](https://arxiv.org/abs/2402.09469)  (2024)

- Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love, [Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks](https://arxiv.org/abs/2505.18266) (2025)

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/pdf/2201.02177)

- Eshika Saxena, Alberto Alfarano, François Charton, Zeyuan Allen-Zhu, Emily Wenger, Kristin Lauter. [Making Hard Problems Easier with Custom Data Distributions and Loss Regularization: A Case Study in Modular Arithmetic](https://arxiv.org/pdf/2410.03569).

- Emily Wenger, Mingjie Chen, François Charton, Kristin Lauter. SALSA: Attacking Lattice Cryptography with Transformers. Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

- Nikolay Yudin, [Mitigating Position-Shift Failures in Text-Based Modular Arithmetic via Position Curriculum and Template Diversity](https://arxiv.org/abs/2601.04283) (2026)

- Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas, [The clock and the pizza: Two stories in mechanistic interpretation of neural networks](https://arxiv.org/abs/2306.17844) (2023)
