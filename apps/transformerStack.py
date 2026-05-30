from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config import CONFIG
from embedding import (
    build_tokenizer_and_embedding,
    resolve_embedding_max_seq_len,
)
from transformerBlock import TransformerBlock


class TransformerStack(nn.Module):
    """
    Stack of N Transformer blocks (decoder-only).
    Production-ready for training & inference.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_head=n_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm (GPT-style)
        self.final_ln = nn.LayerNorm(d_model)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, D)

        past_key_values:
            list of length num_layers
            each item: (k, v) with shape (B, n_head, T_cache, head_dim)

        Returns:
            if use_cache:
                hidden_states, new_past_key_values
            else:
                hidden_states
        """

        if past_key_values is not None:
            if len(past_key_values) != self.num_layers:
                raise ValueError(
                    f"past_key_values length {len(past_key_values)} "
                    f"!= num_layers {self.num_layers}"
                )

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[i]

            if use_cache:
                x, present = layer(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                new_past_key_values.append(present)
            else:
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=False,
                )

        x = self.final_ln(x)

        if use_cache:
            return x, new_past_key_values

        return x

    def forward_batch(
        self,
        batch: dict,
        embedding_layer: nn.Module,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
    ):
        """
        End-to-end helper from dataset batch:
        batch['input_ids'] -> embedding -> transformer stack output.

        position_offset controls positional embeddings during cached generation.
        If omitted with a cache, it defaults to the cached key length.
        """
        if "input_ids" not in batch:
            raise KeyError("Batch must contain 'input_ids'.")

        if position_offset is None:
            position_offset = (
                0
                if past_key_values is None
                else int(past_key_values[0][0].size(-2))
            )

        hidden = embedding_layer.forward_batch(batch, position_offset=position_offset)
        attention_mask = batch.get("attention_mask")
        return self.forward(
            hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    # ============================================================
    # Factory from config
    # ============================================================

    @classmethod
    def from_config(cls, cfg=CONFIG, d_model: Optional[int] = None):
        resolved_d_model = int(
            d_model if d_model is not None else getattr(cfg, "EMBEDDING_DIM", 256)
        )

        resolved_n_head = int(getattr(cfg, "ATTENTION_NUM_HEADS", 8))
        resolved_num_layers = int(
            getattr(cfg, "TRANSFORMER_NUM_LAYERS", getattr(cfg, "NUM_LAYERS", 6))
        )

        if resolved_d_model % resolved_n_head != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({resolved_d_model}) must be divisible by "
                f"ATTENTION_NUM_HEADS ({resolved_n_head})."
            )

        return cls(
            num_layers=resolved_num_layers,
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
            max_seq_len=resolve_transformer_max_seq_len(cfg=cfg, override=None),
        )


def resolve_transformer_max_seq_len(cfg=CONFIG, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)

    configured = getattr(cfg, "ATTENTION_MAX_SEQ_LEN", None)
    if configured is not None:
        return int(configured)

    return resolve_embedding_max_seq_len(cfg=cfg, override=None)


def build_transformer_stack_from_config(
    cfg=CONFIG,
    d_model: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> TransformerStack:
    stack = TransformerStack.from_config(cfg=cfg, d_model=d_model)
    if max_seq_len is not None:
        for layer in stack.layers:
            layer.attn.max_seq_len = int(max_seq_len)
    return stack


def build_tokenizer_embedding_transformer_stack(cfg=CONFIG, load_tokenizer: bool = True):
    tokenizer, embedding = build_tokenizer_and_embedding(cfg=cfg, load_tokenizer=load_tokenizer)
    stack = build_transformer_stack_from_config(
        cfg=cfg,
        d_model=embedding.token_embedding.embedding_dim,
        max_seq_len=embedding.max_seq_len,
    )
    return tokenizer, embedding, stack
