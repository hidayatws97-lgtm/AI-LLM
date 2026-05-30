from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from attention import MultiHeadSelfAttention, resolve_attention_max_seq_len


# ============================================================
# Feed Forward Network (MLP)
# ============================================================

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")  # Fast GELU
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ============================================================
# Transformer Block (Decoder-Only)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)

        # Pre-Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=attention_dropout,
            bias=bias,
            max_seq_len=max_seq_len,
        )

        self.mlp = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, D)
        """

        # ---- Self Attention ----
        residual = x
        x = self.ln1(x)

        if use_cache:
            attn_out, present = self.attn(
                x,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=True,
            )
        else:
            attn_out = self.attn(
                x,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=False,
            )
            present = None

        x = residual + self.dropout(attn_out)

        # ---- Feed Forward ----
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        if use_cache:
            return x, present
        return x

    @classmethod
    def from_config(
        cls,
        cfg=CONFIG,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        resolved_d_model = int(
            d_model if d_model is not None else getattr(cfg, "EMBEDDING_DIM", 256)
        )
        resolved_n_head = int(getattr(cfg, "ATTENTION_NUM_HEADS", 8))

        if resolved_d_model % resolved_n_head != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({resolved_d_model}) must be divisible by "
                f"ATTENTION_NUM_HEADS ({resolved_n_head})."
            )

        resolved_max_seq_len = resolve_attention_max_seq_len(cfg=cfg, override=max_seq_len)

        return cls(
            d_model=resolved_d_model,
            n_head=resolved_n_head,
            mlp_ratio=float(
                getattr(cfg, "TRANSFORMER_MLP_RATIO", getattr(cfg, "MLP_RATIO", 4.0))
            ),
            dropout=float(
                getattr(cfg, "TRANSFORMER_DROPOUT", getattr(cfg, "DROPOUT", 0.1))
            ),
            attention_dropout=float(getattr(cfg, "ATTENTION_DROPOUT", 0.1)),
            bias=bool(
                getattr(
                    cfg,
                    "TRANSFORMER_USE_BIAS",
                    getattr(cfg, "USE_BIAS", getattr(cfg, "ATTENTION_USE_BIAS", False)),
                )
            ),
            max_seq_len=resolved_max_seq_len,
        )
