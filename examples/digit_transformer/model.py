"""Small decoder-only Transformer for modular multiplication.

Tokenisation: digits 0-9 plus a small vocab of separator / control tokens.
Sequence: BOS A_digits SEP B_digits SEP P_digits EQ <answer_digits> EOS
At inference, the model is fed up to EQ inclusive and autoregressively
generates the answer digits until EOS or the max-output budget.

The model receives the RAW decimal strings (a, b, p) — no deterministic
pre-reduction. Reducing the full-width operands mod p is itself part of the
task, and the rules prohibit computing `a % p` / `b % p` in non-learned code
before the model runs; the network has to learn the reduction from data.
Problems whose raw prompt exceeds the trained context length are honestly
out of range and return [] (scored wrong).

Per-argument preprocessing passes the strings through; predict_digits
assembles the prompt, runs greedy generation, and returns the emitted
base-10 digits.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from modchallenge.interface.base_model import ModularMultiplicationModel

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# 0..9  : decimal digits
# 10    : SEP   (between a / b / p slots)
# 11    : EQ    (between input and output)
# 12    : BOS
# 13    : EOS
# 14    : PAD
DIGIT_TOKENS = list(range(10))
SEP, EQ, BOS, EOS, PAD = 10, 11, 12, 13, 14
VOCAB_SIZE = 15

# Maximum sequence length used during both training and inference.
MAX_LEN = 80

# Maximum number of output digits we'll generate during inference.
# Tier 3 answers are at most 5 digits; allow some slack.
MAX_OUTPUT_DIGITS = 20


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TinyTransformerLM(nn.Module):
    """Decoder-only Transformer with sinusoidal-free learned positions."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T). returns logits (B, T, V)."""
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        # Causal mask
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1
        )
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Helpers (used by both training and inference)
# ---------------------------------------------------------------------------

def encode_decimal(s: str) -> list[int]:
    """Decimal string -> list of digit-token ids."""
    return [int(c) for c in s]


def build_prompt(a: str, b: str, p: str) -> list[int]:
    """Construct the full input prefix: BOS A SEP B SEP P EQ."""
    return [BOS] + encode_decimal(a) + [SEP] + encode_decimal(b) + [SEP] + encode_decimal(p) + [EQ]


def build_full_sequence(a: str, b: str, p: str, answer: str) -> list[int]:
    return build_prompt(a, b, p) + encode_decimal(answer) + [EOS]


# ---------------------------------------------------------------------------
# Submission entry point
# ---------------------------------------------------------------------------

class DigitTransformer(ModularMultiplicationModel):
    def __init__(self):
        self.model: TinyTransformerLM | None = None
        self.device: torch.device | None = None

    def load(self, model_dir: str) -> None:
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ckpt_path = Path(model_dir) / "weights.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        config = ckpt.get("config", {})
        self.model = TinyTransformerLM(**config)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_a(self, a):
        return a

    def preprocess_b(self, b):
        return b

    def preprocess_p(self, p):
        return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        assert self.model is not None
        # Feed the RAW decimal strings. No deterministic pre-reduction:
        # computing a % p / b % p in non-learned code before the model runs
        # is prohibited — reducing the full-width operands is part of what
        # the network must learn (see rules/evaluation.md, Prohibited
        # Practices).
        prompt = build_prompt(str(a_enc), str(b_enc), str(p_enc))
        if len(prompt) > self.model.max_len - 1:
            # Prompt itself doesn't fit; we can't usefully predict.
            return []

        # Greedy autoregressive generation
        tokens = torch.tensor([prompt], dtype=torch.long, device=self.device)
        out_digits: list[int] = []
        for _ in range(MAX_OUTPUT_DIGITS):
            if tokens.shape[1] >= self.model.max_len:
                break
            logits = self.model(tokens)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == EOS:
                break
            if 0 <= next_id <= 9:
                out_digits.append(next_id)
            # Append the predicted token regardless (model may emit a sep
            # or other token in error; we still want to advance position).
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], device=self.device)], dim=1
            )

        # Strip any leading zeros (canonical decoded value is the integer; the
        # harness decoder ignores leading zeros, so this is purely cosmetic).
        return out_digits if out_digits else [0]
