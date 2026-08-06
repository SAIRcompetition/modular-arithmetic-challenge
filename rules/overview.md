# Modular Arithmetic Challenge

## What is This Competition?

The Modular Arithmetic Challenge asks a simple question: **can a neural network learn to do modular multiplication efficiently?**

Given two large integers `a`, `b` and a prime `p`, compute `(a * b) mod p`. All values are decimal strings. The operands can be hundreds of digits long — far beyond what fits in a 64-bit integer.

This sounds trivial for a calculator. For a neural network, it is an open problem.

## Organizers

Alberto Alfarano, François Charton, Yongzheng Jia, Kristin Lauter, Cathy Li, Terence Tao, Emily Wenger

## Background

Modular arithmetic is the foundation of modern cryptography, number theory, and large parts of algebra. Humans learn it through algorithms — long multiplication, then division for the remainder. But neural networks don't execute algorithms. They learn patterns from data and generalize.

Recent work has shown that small Transformers can learn modular addition (`a + b mod p`) for small primes, and even discover internal representations resembling Fourier analysis on cyclic groups. But multiplication is fundamentally harder: it requires carrying, multi-digit coordination, and the output length scales with the input length. Modular reduction on top of that adds another layer of difficulty.

**The key questions this competition explores:**

- Can neural networks learn exact multi-digit multiplication, not just approximate it?
- Can they generalize to operands much larger than the prime — requiring genuine modular reduction, not just memorization?
- What architectures, tokenizations, and training strategies work best?
- Where does the scaling break? At 10 digits? 100? 1000?

The difficulty tiers are designed to probe exactly this: at which point does your model stop being able to compute the correct answer?

A non-comprehensive bibliography of related prior work is available in [literature.md](literature.md).

## What We Expect to Learn

This is as much a research challenge as it is a competition. We expect:

- **Baselines from existing LLMs** will fail beyond small numbers, since they were not trained for exact arithmetic at scale.
- **Custom small models** with the right tokenization and training data may significantly outperform much larger general-purpose models on this task.
- **Novel approaches** — p-adic representations, CRT decomposition, number-theoretic tokenizations — could unlock capabilities that standard digit-level approaches cannot.
- **A clear picture of the frontier**: at what operand size and prime size do current neural approaches break down?

The results, techniques, and insights from all participants will be compiled and shared through the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network). All contributions are recognized and cited.

## The Task

Your model receives three decimal strings `a`, `b`, `p` and must compute `(a * b) mod p`.

**Input:**
- `a >= 0`, `b >= 0` (can be up to ~1233 decimal digits)
- `p >= 2` is prime (up to ~617 decimal digits)
- `a` and `b` can be much larger than `p`

**Output:** the integer `(a * b) mod p`. Your model emits it as a list of base-`B` digits through the interface (see **Model Requirements**); the harness converts those digits into the canonical answer. There is no contestant-side decoding step.

**Example:**
```
a = "123456789", b = "987654321", p = "97"
answer = "52"   # because (123456789 * 987654321) % 97 == 52
```

This is a pure mathematical reasoning challenge. The official contract is fixed: your model receives the three decimal strings `(a, b, p)` (after a per-argument preprocessing pass you control) and emits the answer as a list of base-`B` digits. The harness decodes those digits into the canonical integer answer — there is no contestant-side decoding step. **Internally your model may use any representation** — digit-level tokens, p-adic, CRT decomposition, other bases, custom embeddings — as long as the answer is genuinely produced by your model's *trained parameters*, not by an arithmetic algorithm hand-coded into the model or any other shortcut. See **Prohibited Practices** below for the precise boundary.

For the breakdown of difficulty tiers, the test-generation procedure, and the scoring rules used by the official evaluation, see [evaluation.md](evaluation.md).

## Official Repository

All competition code — the evaluation pipeline, test-case generator, model interface, public benchmark, and runnable reference models — is open source:

- [https://github.com/SAIRcompetition/modular-arithmetic-challenge](https://github.com/SAIRcompetition/modular-arithmetic-challenge)

New contestants should start from `examples/`: three small, compliant reference models — a trivial `always_zero` baseline plus two trained neural models (`digit_transformer`, `dlp_grokking`) — that run end-to-end through the same pipeline used for official evaluation. `examples/README.md` is a clone-and-run walkthrough and a step-by-step submission guide. See [evaluation.md](evaluation.md#reference-example-models) for what each model does and how they illustrate the compliance boundary.

## Timeline

| Event | Date |
|-------|------|
| Competition opens | June 8, 2026 |
| Submission deadline | August 12, 2026, 23:59 AoE |

## Model Requirements

- Submissions must implement the `ModularMultiplicationModel` Python interface. The interface is split into three per-argument preprocessing hooks (`preprocess_a`, `preprocess_b`, `preprocess_p`) and a single `predict_digits` method that returns the answer as a list of base-`B` digits, MSB-first.
- The model declares the base `B` it uses in `manifest.json` via the `output_base` field. Allowed values: any integer in `[2, 2^32]`, or the string `"p"` (meaning "use the current prime as the base").
- The pipeline-provided decoder — not the contestant's code — converts the model's emitted digit list into the canonical integer answer and compares it against the ground truth. Contestants do not write post-processing code.
- Submissions must be deterministic.
- Any architecture **implementable within the supported sandbox runtime** is allowed: Transformer, RNN, CNN, hybrid, or novel approaches. There is no architecture whitelist and Turing-complete / recurrent models are not prohibited as such; what matters is that the answer is produced by *trained parameters*, not by an arithmetic procedure hand-coded into the model (see **Prohibited Practices** and the model/circuit boundary in [evaluation.md](evaluation.md)). The runtime contract (initially centered on PyTorch; broader runtime support such as JAX, TensorFlow, ONNX may be added based on contestant demand) is documented in [evaluation.md](evaluation.md).
- Any internal representation is allowed: digit-level tokens, p-adic, CRT decomposition, other bases, learned embeddings, etc. The pipeline contract only fixes the I/O shape; the internal computation is up to you, subject to the one boundary that the answer must come from trained parameters rather than a hand-coded arithmetic algorithm (see **Prohibited Practices**).

The full interface signature, required file layout, artifact size limit, output format, and inference time budget are specified in [evaluation.md](evaluation.md).

## Submission Workflow

A submission is a **public HuggingFace repository** containing your model, identified by `repo_id` + an immutable `commit_hash`.

1. Push your model (`manifest.json` + `model.py` + weight files) to a public HuggingFace repo.
2. Submit `repo_id` + `commit_hash` to organizers.
3. Organizers verify the repo is accessible and run the official evaluation with a secret random seed.
4. Results are posted to the leaderboard.

### Submission Rules

- The HuggingFace repo must be **public** at submission time — private repos are not ranked.
- The commit hash locks the submission version immutably.
- Multiple submissions are allowed; the leaderboard keeps your best result per team.
- Official evaluation uses a secret random seed unknown to contestants.
- The test-generation, scoring, and model-interface code is open source and shared by both local testing and official evaluation. Official evaluation additionally runs inside the sandboxed environment described in [evaluation.md](evaluation.md), so the two setups differ in sandbox enforcement and in the available package surface — contestants should test their submission against the published sandbox image before final submission.

For local testing during development (including the option of using a private HF repo with a read-only token) and full evaluation details, see [evaluation.md](evaluation.md).

## Prohibited Practices

The principle: **the model must learn to compute `(a * b) mod p`; it may not delegate, look up, or hard-code the computation.**

The interface is designed so that the most obvious attack paths are structurally impossible: the pipeline-provided decoder is the only code that converts the model's emitted digits into the answer, and the three preprocessing hooks each see only their own argument, so no single point in the contestant's code has simultaneous access to `a`, `b`, and `p`. The rules below close the remaining gaps.

The following are **not allowed at inference time**:

- **Computing the final answer** `(a * b) mod p` using symbolic-math libraries (`sympy`, `gmpy2`, `mpmath`, `flint`, etc.) or Python's built-in arbitrary-precision integer arithmetic on the original `(a, b, p)` arguments (e.g. by stashing them in instance state across preprocessing hooks and recombining inside `predict_digits`).
- **Deterministically pre-reducing the operands**: computing `a mod p`, `b mod p`, or any equivalent reduction of the full-width operands against the actual `p` in non-learned code before the model runs. The model must receive the raw `(a, b, p)` — reducing the full-width operands is itself a core part of the task.
- **Hard-coded answers** or lookup tables indexed by the evaluation inputs (or by hashes / fingerprints of them).
- Dynamic code execution: `eval`, `exec`, `compile`, `__import__`, `ctypes`.
- Network access of any kind.
- Reading files outside the submission directory.
- Subprocess calls or system commands.
- Cross-argument leakage in preprocessing: a `preprocess_a` call must not depend on previously seen `b` or `p` values (and similarly for the others). The pipeline runs a sanity check that flags the simplest forms of this.

**What is explicitly allowed:** using `int()`, base conversion, modular arithmetic on small intermediate values, or any other standard operation **inside a single per-argument preprocessing hook** (operating only on its own argument) or **inside the model itself** (provided the answer is produced by the model's trained parameters, not by a hard-coded shortcut or a by-construction arithmetic algorithm; see the two principles and the model/circuit boundary in [evaluation.md](evaluation.md)).

In particular, a loop that feeds the model its input tokens one at a time is a valid deterministic encoder, so long as the encoder receives **no feedback** from the model to help it choose the next token.

Submissions found to violate these rules will be disqualified. Sandbox configuration, static-analysis checks, and the planned behavioral signals (weight perturbation, distribution shift, latency profile) are documented in [evaluation.md](evaluation.md).

## Team Participation and Anti-Cheating Policy

**Team registration:**

- Each individual or organization can participate in only one team.
- Teams must register members and sponsors with organizers in advance.
- At registration, a team declares one or more HuggingFace accounts or organizations as their submission identities.

**Team identity binding:**

- A submission is attributed to a team if its `repo_id` is owned by one of that team's declared HuggingFace accounts or organizations.
- A team may submit from multiple repositories across any of their declared accounts.
- Leaderboard deduplication is **per team**, not per repo: the leaderboard keeps each team's best result across all their submissions, regardless of source repo or account.
- Repo ownership is verified against the team's declared accounts at submission time.

**Anti-cheating:**

- If coordinated cheating is detected — including sockpuppet teams, weight laundering across accounts, or hard-coding of evaluation answers — all related teams will be disqualified.

## SAIR Contributor Network

This competition is integrated with the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network).

During the competition, you can share your model through the Contributor Network:

1. Make your HF repo public (or create a separate public repo for sharing)
2. Submit `repo_id` + `commit_hash` via the Contributor Network
3. Organizers will evaluate your shared model and publish the results
4. Community members can review and give feedback on your work
5. All contributions are recognized and cited in the competition report

Sharing is **optional** but encouraged. It does not replace the final submission — you still need to submit before the deadline to be ranked on the official leaderboard.

## Community Feedback

Rules, scoring details, and evaluation procedures are still being refined and will be shaped by community input. Community contributions are welcome.

Join the SAIR Foundation Zulip community for discussion and collaboration:

- [https://zulip.sair.foundation/](https://zulip.sair.foundation/)

## Experimental Status

This challenge is currently in an experimental phase. Rules, scoring details, and evaluation procedures may be adjusted based on implementation experience and community feedback.
