# Modular Arithmetic Challenge - Evaluation Setup

This page specifies how submissions to the Modular Arithmetic Challenge are evaluated: submission format, model interface, sandbox, evaluation pipeline, scoring, and prohibited practices.

For the high-level task description, key dates, eligibility, and team / anti-cheating policy, see **[overview.md](overview.md)**.

## Submission Format

A submission is a **single HuggingFace repository** identified by `repo_id` + an immutable `commit_hash`.

| File | Purpose |
|------|---------|
| `manifest.json` | Declares the entry class, the output base, and short descriptions of the model and its training. Required fields: `entry_class` (dotted Python path, e.g. `"model.MyModel"`), `output_base` (see below), `model_description` (non-empty free-text: architecture, approximate parameter count, input/output representation, and any key design choices — what a reviewer needs to understand the submission at a glance), and `training_description` (non-empty free-text: how the weights were obtained — used for the provenance check in review; see **Prohibited Practices**). Optional field: `framework`. Validated against a Pydantic schema. |
| `model.py` | The model implementation. Must define the entry class declared in `manifest.json`. |
| Weight files | The trained weights (`.pt`, `.pth`, `.safetensors`, `.bin`, `.gguf`, `.onnx`, etc.). |
| Other files | Any additional code, configuration, or assets needed at inference time. |

**`manifest.json` example:**

```json
{
  "entry_class": "model.MyModel",
  "output_base": 10,
  "framework": "pytorch",
  "model_description": "small decoder-only transformer, digit-level tokens",
  "training_description": "trained from random init on 5M synthetic (a, b, p) -> a*b mod p examples, digit-level tokens, AdamW"
}
```

The `output_base` field tells the harness's decoder how to interpret the digits your model emits (see **Model Interface** below). Allowed values:

- any integer in `[2, 2^32]` — model emits answers in that base
- the string `"p"` — model emits answers in base equal to the current problem's prime (so the answer is always a single digit in `[0, p)`)

| Limit | Reference value |
|-------|-----------------|
| Total artifact size (weights + code + all files) | **20 GB** |
| Total inference time for the full evaluation set | **5 minutes** for 1100 problems |

The repository may be private during development and must be made **public** before the official submission deadline. Private repositories are not ranked on the official leaderboard.

## Model Interface

The entry class must subclass `ModularMultiplicationModel` from `modchallenge.interface.base_model`. The interface is split into three per-argument preprocessing hooks and a `predict_digits` method that emits the answer as base-`B` digits:

```python
from modchallenge.interface.base_model import ModularMultiplicationModel

class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load weights, initialize the model. Called once before any
        # preprocess / predict_digits call.
        ...

    # Per-argument preprocessing. Each hook MAY ONLY ACCESS ITS OWN ARGUMENT.
    # Defaults return the input unchanged; override to tokenise, embed, etc.
    def preprocess_a(self, a: str): return a
    def preprocess_b(self, b: str): return b
    def preprocess_p(self, p: str): return p

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        # Run the model on the encoded inputs. Return the answer
        # (a * b mod p) as a list of base-B digits, MOST-SIGNIFICANT-FIRST.
        # B is the value declared in manifest.json's output_base field.
        ...

    def predict_digits_batch(self, inputs) -> list[list[int]]:
        # Optional. Override to enable GPU batching.
        ...

    def max_batch_size(self) -> int:
        return 64
```

The pipeline runs, for each problem `(a, b, p)`:

```
a_enc  = model.preprocess_a(a)        # operates on a only
b_enc  = model.preprocess_b(b)        # operates on b only
p_enc  = model.preprocess_p(p)        # operates on p only
digits = model.predict_digits(a_enc, b_enc, p_enc)
answer = pipeline_decoder(digits, base=manifest.output_base, prime=int(p))
```

The pipeline-provided decoder reads `digits` as base-`B` digits (MSB-first) and produces the canonical integer answer. **Contestants do not implement the decoder, and there is no contestant-side decoding or post-processing step** — that closes the post-processing attack surface. Your model's only output is the base-`B` digit list; converting it back to a decimal answer is done entirely by the harness.

Per-argument preprocessing means no single point in your code has access to `a`, `b`, and `p` together. (The pipeline runs a sanity check that catches the obvious workarounds, like stashing previous-call inputs in instance state; see **Prohibited Practices** below.)

## Output Format

`predict_digits` (and each element of `predict_digits_batch`) must return a **list of `int`** representing the answer as base-`B` digits, where `B` is the value declared by `manifest.output_base` (or the current prime if `output_base == "p"`).

Format requirements (enforced by the decoder):

- type must be `list`; each entry must be a plain `int` (not `bool`, not `numpy.int64` — convert to `int` before returning)
- digits are most-significant-first
- each digit must be in `[0, base - 1]`
- on scored tiers (Tier 1–10): the decoded integer must be in `[0, p)`. A value `>= p` is malformed.
- on Tier 0 (pure multiplication, no modular reduction): the decoded value may exceed `p`.

Outputs that violate any of the above are scored as **incorrect** for that problem. They do not raise an error and do not abort the run.

**Batch contract:** if `predict_digits_batch(inputs)` returns a list whose length differs from `len(inputs)`, the entire tier is marked incomplete and scores **0%**. The pipeline does not attempt to re-align partial results.

## Submission Workflow

**Phase 1 — Development (private):**

```
Train model locally
  -> Push to a private HuggingFace repo
  -> Create a fine-grained read-only access token for that repo
  -> Test:  modchallenge evaluate-hf <repo_id> <commit_hash> --token hf_xxx
  -> Iterate
```

Local-only testing is also supported, without a HuggingFace round trip:

```
modchallenge evaluate ./my-local-model --total 110
```

**Phase 2 — Final submission (public):**

1. Make the HuggingFace repo **public**.
2. Submit `repo_id` + `commit_hash` to organizers.
3. Organizers verify the repo is public, accessible, and passes the manifest schema.
4. Organizers run official evaluation in the sandboxed environment with a **secret random seed**.
5. Results are posted to the leaderboard.

Multiple submissions per team are allowed; each submission is locked by its commit hash. The leaderboard keeps the best result **per team** (not per repo); see [overview.md](overview.md#team-participation-and-anti-cheating-policy) for how team identity is bound to HuggingFace accounts at registration time.

## Solver Environment

The submitted model runs in an isolated sandbox during evaluation:

- **No network access** — outbound and inbound networking is blocked at the sandbox boundary.
- **No secrets** — the HuggingFace token used to download the repository is not propagated to the model process; no other API keys or environment variables are exposed beyond a minimal allowlist (`PATH`, `HOME`, `LANG`, etc.).
- **Restricted filesystem** — read access is limited to the submission directory and standard system paths required by the runtime; write access is limited to a scratch directory. Reading other locations on the host is blocked.
- **No subprocess / no dynamic code execution** — the model may not spawn subprocesses, invoke `eval`/`exec`, or otherwise execute code outside the loaded module.
- **Determinism is the model's responsibility** — the pipeline does not seed your RNG; if your model uses any internal randomness, seed it inside `load()` or the preprocess / `predict_digits` methods. The pipeline runs an automated determinism check (see below) and excludes non-deterministic submissions from the ranked leaderboard.

```
HuggingFace repo --(loader, manifest validation)--> Sandbox --(preprocess_* / predict_digits_batch)--> Decoder --> Scorer
```

The sandbox is a Docker container at official evaluation time. Local testing via `modchallenge evaluate-hf` currently runs **without** the sandbox (trusted-use only); contestants are expected to test against the sandbox before final submission once the Docker image is published.

## Evaluation Pipeline

For each submission the pipeline runs the following stages in order:

1. **Manifest validation** — parse and schema-check `manifest.json` (`entry_class`, `output_base`, etc.).
2. **Artifact-size check** — reject submissions whose total file size exceeds the artifact-size limit.
3. **Static analysis** — AST-scan the submission's `.py` files for disallowed imports and the obvious modular-product shortcut patterns. Submissions with findings are rejected before the model is loaded.
4. **Test generation** — generate the test set for this run from the master seed (see below).
5. **Model load** — import the entry class and call `load(model_dir)` once.
6. **Preprocess-isolation sanity check** — call each preprocess hook twice on the same input (with calls to the other hooks interleaved) and verify the outputs match. This flags the simplest forms of cross-argument leakage.
7. **Determinism check** — run a sampled subset of problems through the full preprocess + `predict_digits_batch` + decode pipeline twice and compare. Non-deterministic submissions are flagged and excluded from the ranked leaderboard.
8. **Inference** — run the model over all 1100 problems, per tier, batched up to `max_batch_size()` where applicable; decode each batch's emitted digits into canonical integers via the harness decoder.
9. **Scoring** — compare decoded answers against ground truth; produce `overall_accuracy` and `highest_tier_above_90`.

The pipeline is implemented in `src/modchallenge/evaluation/pipeline.py` and is shared between local testing and official evaluation. Official evaluation additionally executes the model inside the sandbox described in **Solver Environment**; local testing via `modchallenge evaluate-hf` currently runs the model in the host process (trusted-use only). Apart from the sandbox boundary and the package allowlist that comes with it, every stage above is the same code path in both settings.

## Test Generation

- **Master seed** — `secrets.token_bytes(32)` for the official run (secret, not disclosed); a fixed seed for the public benchmark and reproducible local testing.
- **Per-tier seeds** — derived via `HMAC-SHA256(master_seed, tier_id)` so that tiers are independent and reproducible from the master seed.
- **Primes per tier** — 5 different primes per scored tier, to prevent overfitting to a single modulus. Tier 0 uses pre-selected Mersenne primes for the higher sub-tiers (where `sympy.nextprime` is prohibitively slow).
- **Edge cases** — `a = 0`, `b = 0`, `a = 1`, `b = 1` are included in every scored tier.
- **Default size** — 1100 test cases (100 per tier).

Implementation: `src/modchallenge/testgen/generator.py`.

## Difficulty Tiers

Evaluation uses 11 tiers. Tier 0 is a **diagnostic** (pure multiplication, unscored). Tiers 1-10 are **scored**.

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

Each scored tier independently controls **both** the prime size and the operand size. Operands grow progressively across tiers, so each tier tests a distinct combination of multiplication scale and modular reduction complexity. Operands `a, b` may be much larger than `p`, requiring genuine modular reduction (not just memorization of `a * b mod p` for `a, b < p`).

Tier 0 covers the full operand range as **pure multiplication without modular reduction**. It is unscored and exists to diagnose whether a model fails at higher tiers due to multiplication capacity or due to modular reduction ability.

The single source of truth for tier geometry is `src/modchallenge/config.py`.

## Scoring

**Reported metrics** (computed for every submission):

- `overall_accuracy` — average accuracy across Tiers 1-10, with **equal weight per tier**.
- `highest_tier_above_90` — the highest tier index at which accuracy `>= 90%`.

**Leaderboard ranking keys** (sort order):

1. `highest_tier_above_90` (descending)
2. `overall_accuracy` (descending) as tiebreaker

**Tier 0** is diagnostic only and is not counted toward either metric.
**Incomplete tiers** (timeout, error, crash, batch-contract failure) score 0% for that tier.

Final scoring rules and tiebreakers are subject to community feedback.

## Determinism Requirement

Submissions must be **deterministic**: the same `(a, b, p)` input must produce the same output across runs.

The pipeline performs an automated determinism check on a sampled subset of problems by running them twice and comparing outputs. Submissions that fail the check are flagged and excluded from the ranked leaderboard.

If a model uses any source of randomness internally (sampling, dropout at inference, MCMC, etc.), it is the contestant's responsibility to seed it deterministically inside `load()` or inside the preprocess / `predict_digits` methods.

## Time and Resource Budget

| Resource | Reference value | Notes |
|----------|-----------------|-------|
| Total inference wall-clock | 5 minutes for 1100 problems | May be tuned before the official runs. |
| Per-problem soft target | ~273 ms | Use `predict_digits_batch()` and GPU batching to amortize. |
| Total artifact size | 20 GB | Weights + code + all files. |

`load()` and the determinism check are bounded separately and are not counted against the inference wall-clock.

**Wall-clock measurement (mechanical policy):**

- The inference timer **starts** at the entry to `run_inference`, immediately after `load()`, the static check, the preprocess-isolation check, and the determinism check have all completed.
- The timer **covers** every per-problem call inside the inference loop: each `preprocess_a` / `preprocess_b` / `preprocess_p`, each `predict_digits_batch`, and the harness decoder.
- The timer **excludes** `load()` (separate bounded budget), the determinism check (separate bounded budget on a small sample), and any organizer-side overhead.
- Tiers run in order: Tier 0 first, then Tiers 1, 2, ..., 10.
- **On timeout:**
  - The tier currently being processed scores **0%**; partial results within that tier are discarded.
  - All subsequent tiers score **0%**.
  - Tiers fully completed before the timeout retain their actual scores.

## Prohibited Practices

**The principle: the model must *learn* to compute `(a * b) mod p` — it may not delegate, look up, or hard-code the computation.** The interface already closes the obvious paths: there is no contestant post-processor, and each preprocess hook sees only its own argument, so no single point in your code witnesses `a`, `b`, and `p` together.

**Not allowed at inference time:**

- Computing the answer with symbolic-math libraries (`sympy`, `gmpy2`, `mpmath`, `flint`, …) or with built-in big-integer arithmetic on the original `(a, b, p)` (e.g. stashing them across the preprocess hooks and recombining inside `predict_digits`).
- **Hand-coding the arithmetic** — schoolbook multiplication, long division, Barrett/Montgomery reduction, CRT recombination over the actual `(a, b, p)` — **whether in Python integers or in tensor / array operations**. Such code is correct for *any* weights, so the weights aren't what solve the problem: that is a computational circuit, not a learned model.
- Hard-coding answers, lookup tables indexed by the inputs, or hashes / fingerprints of the evaluation set into the weights or code.
- **Cross-argument leakage**: a `preprocess_a` call depending on a previously-seen `b` or `p` (and likewise for the others).
- `eval` / `exec` / `compile` / `__import__` of arbitrary modules / `ctypes`, network access, reading outside the submission directory, or spawning subprocesses.

**Allowed:** any per-argument representation work inside a single hook (`int()`, base conversion, modular arithmetic on small intermediates, p-adic decomposition, CRT splitting against fixed small moduli, byte encoding, …), and any *learned* computation inside the model. The constraint is not what the internal computation looks like — it is where the answer comes from.

**Two principles decide whether the model is really computing the answer:**

1. **The emitted digits must materially determine the answer.** If `predict_digits` could return garbage and the answer would still come out right (e.g. precomputed in preprocessing), you are computing it outside the model.
2. **The capability must reside in the trained parameters.** Randomizing the weights should collapse a legitimate model's accuracy, whereas a hand-coded solver keeps working — that gap is the operational test.

This is about **provenance, not architecture.** There is no architecture whitelist: Turing-complete, recurrent, or looped models are fine, and a model that *learns* an algorithm-like circuit is exactly what this competition is looking for. A model trained to internally implement an algorithm is permitted; the same algorithm hand-coded into the forward pass is not. Genuinely borderline cases — e.g. weights hand-initialized to nearly solve the task and then barely trained — are resolved by examining *how the weights were obtained*, which is why `training_description` is required. A submission with no trained parameters is by definition a circuit, not a model.

**Enforcement** (layers marked *planned* are under development, to be published before the official evaluation):

1. **Sandbox allowlist** *(planned)* — a published Docker image (PyTorch-centric) without `sympy` / `gmpy2` / `mpmath` / `flint` or networking / subprocess libraries; any unlisted `import` fails at load time.
2. **Static analysis** — every submission is AST-scanned before load for disallowed imports, `eval` / `exec` / `compile` / `__import__` / `ctypes`, and the modular-product shortcut `int(_) * int(_) % int(_)` (and equivalents like `pow(int(_), int(_), int(_))`). Narrow by design; subtler attempts fall to Layer 4. (`src/modchallenge/security/static_check.py`)
3. **Behavioral signals** *(planned)* — weight-perturbation sweep (the primary test for principle 2), distribution-shift re-evaluation, and latency-vs-operand-size profile. These produce **signals for review**, not automatic disqualification.
4. **Manual review** — for top-ranked entries and anything flagged by Layers 2–3 (it does not scale to every submission). Starting from the required `model_description` / `training_description`, the reviewer reads the code to decide trained-parameters (permitted) vs hand-coded algorithm (prohibited), and may request training code for top-ranked or flagged entries.

Any submission found to violate the rules is disqualified and removed from the leaderboard.

## Evaluation Problem Sets

- **Public benchmark** (in repository, `public_benchmark/`): 1100 test cases, fixed seed, answers included. Reproducible bit-for-bit. Use this for development and local testing.
- **Private evaluation set**: generated at official evaluation time from a secret master seed unknown to contestants. **Separate from the public benchmark.** Used to compute the official leaderboard score.

The public benchmark and the private evaluation set share the same tier geometry and the same generation algorithm; only the master seed differs.

## Official Repository

The official GitHub repository for the Modular Arithmetic Challenge:

- [https://github.com/SAIRcompetition/modular-arithmetic-challenge](https://github.com/SAIRcompetition/modular-arithmetic-challenge)

The repository includes:

- the evaluation pipeline (`src/modchallenge/evaluation/`)
- the test-case generator (`src/modchallenge/testgen/`)
- the model interface and submission schema (`src/modchallenge/interface/`)
- the leaderboard store (`src/modchallenge/leaderboard/`)
- the public benchmark (`public_benchmark/`)
- demo / baseline submissions (`examples/` reference models + `examples/examples.json`)
- a step-by-step contestant guide (`examples/README.md`)

## Reference Example Models

The repository ships three small, **compliant** reference models under `examples/`, each a complete runnable submission (`manifest.json` + `model.py`, plus a trained `weights.pt` for the neural ones). They serve as a pipeline smoke test, a set of honest baselines, and worked examples of the prohibited-practices boundary above:

| Model | What it is | overall_accuracy (public, 1100) |
|-------|------------|---------------------------------|
| `always_zero` | Emits `[0]` for every problem — pipeline smoke test and leaderboard floor. | 0.079 |
| `digit_transformer` | ~544K-param decoder-only Transformer; reduces operands `mod p` inside `predict_digits`, then learns small-number modular multiplication. | 0.121 |
| `dlp_grokking` | ~6M-param discrete-log "grokking" model: a shared residue encoder, an additive (log-space) bottleneck, and a learned decoder. | 0.127 |

All three are deterministic and pass the static check. None hand-codes the arithmetic — each produces its answer through the trained model — so they sit on the compliant side of the line: the two neural models genuinely learn Tier 1 (the fixed primes `{2,3,5,7}`) and degrade honestly on the higher tiers rather than faking them. `dlp_grokking` is the most detailed worked example of turning a mathematical insight (`a*b mod p` as addition in discrete-log space) into a *learned* inductive bias.

Run any of them locally:

```bash
modchallenge check    examples/dlp_grokking
modchallenge evaluate examples/dlp_grokking --total 110
```

See `examples/README.md` for the full line-up, per-tier results, weight-reproduction steps, and the end-to-end submission guide.

## Local Testing

A typical local-testing workflow:

```bash
# One-time setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch                 # required for example models

# Run tests against your own model directory
modchallenge evaluate ./my-local-model --total 110

# Run against a HuggingFace submission (private repo + read-only token)
modchallenge evaluate-hf user/my-model <commit_hash> --token hf_xxx

# Quick-test any HuggingFace LLM (exploratory; not used for ranking)
modchallenge evaluate-llm Qwen/Qwen2.5-0.5B-Instruct

# Run all reference example models from examples/examples.json
modchallenge evaluate-example --group public --total 110

# Run the full pipeline test suite
pytest
```

A recommended workflow:

1. Implement and unit-test your model locally on a few problems per tier (`--total 11`).
2. Once your model passes the determinism check on a small sample, run the full public benchmark (`--total 1100`) and read off `overall_accuracy` and `highest_tier_above_90`.
3. Push to a private HuggingFace repo and re-test with `evaluate-hf` to verify the manifest, the loader path, and the artifact size.
4. Once stable, make the repo public and submit `repo_id` + `commit_hash`.

## Changes and Versioning

The evaluation pipeline (test generation, scoring, sandbox configuration) is versioned in the official repository. Material changes to the rules, the scoring formula, the tier geometry, or the prohibited-practices list will be announced through the official repository and the SAIR Foundation Zulip community before they take effect.
