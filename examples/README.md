# Example models & submission guide

This folder holds three small **compliant** reference models plus the end-to-end workflow for
writing, testing, and submitting your own. The models run start-to-finish through the official
pipeline (`manifest validation → static check → load → preprocess isolation → determinism →
inference → decode → score`). They exist to (a) smoke-test a fresh setup, (b) give a floor and a
couple of honest baselines for the leaderboard, and (c) serve as worked examples of the one rule
that matters: **the answer must come from trained parameters, not hand-coded arithmetic**
(see [../rules/evaluation.md](../rules/evaluation.md#prohibited-practices)).

Each model directory holds `manifest.json`, `model.py`, a `README.md`, and (for the two neural
models) the trained `weights.pt` — they ship in-repo so the examples are **clone-and-run**, no
separate download or training step required. `always_zero` needs no weights.

| Model | What it is | overall_accuracy (1100) | Compliant |
|---|---|---|---|
| [`always_zero/`](always_zero/README.md) | Emits `[0]` for everything. Pipeline smoke test + leaderboard floor. | 0.079 | trivially |
| [`digit_transformer/`](digit_transformer/README.md) | ~544K-param decoder-only Transformer fed the raw `(a, b, p)`; learns full-width reduction + mod-multiplication end-to-end. | **0.094** | yes |
| [`dlp_grokking/`](dlp_grokking/README.md) | ~6M-param discrete-log "grokking" model: **learned** streaming reducer (raw digits, feedback-free schedule) + shared residue encoder + **additive** (log-space) bottleneck + learned decoder. | 0.085 | yes |

All three are deterministic and pass `modchallenge check`. Tier 1 (the 4 fixed primes
`{2,3,5,7}`) is where the neural models score best (0.83 / 0.74) — now that they must learn the
reduction of the raw operands themselves, even tier 1 is no longer a free grok; tier 2+ is the
honest ceiling — see each model's README for the per-tier breakdown and why. The `dlp_grokking` README is the most detailed
worked example of turning a mathematical insight into a compliant inductive bias.

Note how close the overall numbers sit to `always_zero`'s 0.079. That is not a bug: the neural
models earn a real tier 1 and then score ~0 everywhere else, while `always_zero` picks up ~8%
across all tiers from problems whose answer happens to be 0. Neither neural example emits a
fallback value or branches on `a`, `b` or `p` — every problem is answered by the network, so
nothing catches them when they fall. Emitting `0` when your model is out of its depth is *not*
against the rules, and it would raise these numbers; the reference models forgo it so that the
learned/non-learned boundary in them is unambiguous. `overall_accuracy` alone is a poor read on a
model this early — look at the per-tier tables.

## Run the in-repo examples (no HuggingFace token)

```bash
pip install -e ".[dev]"
pip install torch

# Static compliance check
modchallenge check examples/dlp_grokking

# Evaluate through the full pipeline (10 problems/tier)
modchallenge evaluate examples/always_zero       --total 110
modchallenge evaluate examples/digit_transformer --total 110
modchallenge evaluate examples/dlp_grokking      --total 110
```

`--total 110` = 10 per tier (11 tiers). Use `--total 1100` for a full 100-per-tier run.

The weights ship in-repo, so the commands above run as-is. To **reproduce** them from scratch:
`dlp_grokking` has a self-contained, sympy-free trainer in its own directory; `digit_transformer`'s
trainer needs `sympy` for prime generation (blocklisted inside a submission, so it lives in the
gitignored `exploration/`):

```bash
.venv312/bin/python examples/dlp_grokking/train.py --minutes 8          # DLP core (residue space)
.venv312/bin/python examples/dlp_grokking/train_reducer.py              # learned streaming reducer
.venv312/bin/python examples/exploration/train_digit_transformer.py --steps 20000   # local dev only
```

To reproduce the **public benchmark** numbers in the table (100 problems/tier, public seed):

```bash
.venv312/bin/python examples/dlp_grokking/eval_tiers.py examples/dlp_grokking
```

## Run the public HuggingFace baselines (optional, exploratory)

`examples.json` registers two public LLM baselines (Qwen). They are evaluated via a generic LLM
wrapper with a fixed prompt — an **exploratory smoke test only**, not the official ranking
pipeline — and need no token:

```bash
modchallenge evaluate-example --group public --total 110
modchallenge evaluate-example qwen2.5-math-1.5b --total 110   # one by name
```

## Build your own model

The three examples above are minimal, honest baselines — not the ceiling. Here is the full
workflow.

### 1. Implement the interface

The interface is intentionally narrow: three per-argument preprocessing hooks (each only ever
sees its own argument) and one `predict_digits` method that emits the answer as a list of
base-`B` digits. The harness — not your code — decodes those digits into the final answer.

```python
# model.py
from modchallenge.interface.base_model import ModularMultiplicationModel


class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load your trained weights
        ...

    # Per-argument preprocessing. Each hook only sees its own argument.
    # Defaults are identity (pass the string through unchanged); override
    # to tokenise, embed, do base conversion, etc.
    def preprocess_a(self, a: str):
        return [int(c) for c in a]   # e.g. digit-level tokens

    def preprocess_b(self, b: str):
        return [int(c) for c in b]

    def preprocess_p(self, p: str):
        return [int(c) for c in p]

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        # Run the model on the encoded inputs. Return the answer
        # (a * b mod p) as base-B digits, MOST-SIGNIFICANT-FIRST.
        # B = the value declared in manifest.json's output_base field.
        # For example, with output_base = 10 and answer = 52, return [5, 2].
        ...

    def predict_digits_batch(self, inputs) -> list[list[int]]:
        # Optional: override for GPU batching (default loops over predict_digits).
        ...

    def max_batch_size(self) -> int:
        return 64
```

Create a `manifest.json` declaring your entry class and the base you emit answers in:

```json
{
  "entry_class": "model.MyModel",
  "output_base": 10,
  "model_description": "4-layer decoder-only Transformer, ~1.2M params, digit-level tokens for a/b/p, autoregressive decode of answer digits in base 10",
  "training_description": "trained from random init on 1M synthetic (a, b, p, a*b mod p) examples; digit-level tokenisation; AdamW, ~10k steps"
}
```

Allowed `output_base` values:

- any integer in `[2, 2^32]` — for example `2` (binary), `10` (decimal digits), `256` (bytes), `65536` (word-level), etc.
- the string `"p"` — answers are emitted in base equal to the current prime, so each answer is a single digit in `[0, p)`

Your submission directory:

```
my-model/
├── manifest.json
├── model.py
└── weights.pt (or any model files)
```

### The one rule that matters

Your model must **learn** to compute `(a * b) mod p` — the answer must come from trained
parameters. You may use any architecture and any internal representation, but you may **not**
hand-code the arithmetic (schoolbook long multiplication, long division, Montgomery/Barrett
reduction, CRT recombination) over the input values — **whether in Python integers or in tensor
operations**. Such code returns the right answer regardless of the weights, which makes it a
computational circuit, not a learned model. There is also no contestant-side decoding step: your
only output is the base-`B` digit list, and the harness converts it to the answer.

The full boundary, the two governing principles, and the enforcement (static analysis,
weight-perturbation signal, manual review) are in
[../rules/evaluation.md](../rules/evaluation.md#prohibited-practices). Every `manifest.json` must
include two **required** non-empty fields that reviewers read first: `model_description` (what the
model is — architecture, approximate parameter count, input/output representation, key design
choices) and `training_description` (how the weights were obtained — training or fine-tuning
procedure, data, starting point). Routine submissions don't go through manual code review; it's
reserved for top-ranked entries and submissions flagged by static analysis or the behavioral
signals.

### 2. Test locally

```bash
modchallenge evaluate ./my-model --total 110
```

### 3. Upload to a private HuggingFace repo

```bash
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('your-username/my-model', private=True)
api.upload_folder(folder_path='./my-model', repo_id='your-username/my-model')
"
```

Create a **fine-grained read-only token** for your repo at
https://huggingface.co/settings/tokens — this lets you test without exposing write access.

### 4. Test via HuggingFace (private)

Get your commit hash, then run the same pipeline organizers will use:

```bash
python -c "
from huggingface_hub import HfApi
print(HfApi().repo_info('your-username/my-model').sha)
"

modchallenge evaluate-hf your-username/my-model <commit_hash> --token hf_xxx --total 110
```

If it works here, it will work in official evaluation.

### 5. Iterate

```
Improve model -> push to HF -> get new commit hash -> re-test
```

Each push creates a new commit hash. The old hash still points to the old version — nothing is
overwritten.

### 6. Share via Contributor Network (optional)

During the competition, share your model through the
[SAIR Contributor Network](https://competition.sair.foundation/contributor-network):

- Make your HF repo public (or create a new public repo for sharing)
- Submit repo_id + commit_hash via the Contributor Network
- Organizers evaluate your model and publish results; the community can give feedback

### 7. Final submission

Before the deadline:

1. Make your HF repo **public**
2. Submit `repo_id` + `commit_hash` to organizers
3. Organizers run official evaluation with a secret random seed
4. Results posted to the leaderboard

## Tips

- Start with `--total 110` (10 per tier) for fast iteration, use `--total 1100` for thorough testing
- Use `--seed <hex>` to get reproducible results across runs
- Check Tier 0 results to diagnose whether failures come from multiplication or modular reduction
- The preprocessing hooks each see decimal strings — `a` and `b` can be hundreds of digits long (much larger than `p`); `p` is up to ~617 digits
- If your model uses GPU batching, override `predict_digits_batch()` and `max_batch_size()`
- The pipeline runs a sanity check that catches obviously stateful preprocessing (e.g. caching `a` in instance state so a later hook can read it) — design your preprocessing as pure per-argument functions

## Not in this folder

Red-team probes and dev tooling live under `examples/exploration/` (gitignored). They are
deliberately **non-compliant** stress tests for the static check and reviewer process, kept out of
the distributed examples so nobody mistakes them for a sanctioned approach.
