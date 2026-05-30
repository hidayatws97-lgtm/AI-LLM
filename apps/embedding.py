import math
from typing import Optional

import torch
import torch.nn as nn

from config import CONFIG
from tokenizer2 import HybridTokenizerLLM, build_tokenizer_from_config


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x, position_offset: int = 0):
        seq_len = x.size(1)
        position_offset = int(position_offset)
        return self.pe[:, position_offset:position_offset + seq_len]


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pos_type: str = "learned",
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.max_seq_len = int(max_seq_len)
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

        self.pos_type = pos_type

        if pos_type == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        elif pos_type == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(
                d_model,
                max_seq_len
            )
        else:
            raise ValueError("pos_type must be 'learned' or 'sinusoidal'")

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.padding_idx].zero_()

        if self.pos_type == "learned":
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, position_offset: int = 0):
        """
        input_ids: (B, T)
        position_offset: starting position for cached generation.
        """

        B, T = input_ids.size()
        position_offset = int(position_offset)
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative.")
        if position_offset + T > self.max_seq_len:
            raise ValueError(
                f"Input positions [{position_offset}, {position_offset + T}) exceed "
                f"max_seq_len={self.max_seq_len}. "
                "Increase EMBEDDING_MAX_SEQ_LEN / DATASET_BLOCK_SIZE in config."
            )

        token_emb = self.token_embedding(input_ids)  # (B, T, D)

        if self.pos_type == "learned":
            positions = torch.arange(position_offset, position_offset + T, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(B, T)
            pos_emb = self.position_embedding(positions)
        else:
            pos_emb = self.position_embedding(token_emb, position_offset=position_offset)

        x = token_emb + pos_emb
        x = self.dropout(x)

        return x

    @classmethod
    def from_config(
        cls,
        cfg=CONFIG,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        padding_idx: Optional[int] = None,
    ):
        resolved_vocab_size = int(vocab_size if vocab_size is not None else getattr(cfg, "VOCAB_SIZE"))
        resolved_max_seq_len = resolve_embedding_max_seq_len(cfg, override=max_seq_len)
        return cls(
            vocab_size=resolved_vocab_size,
            d_model=int(getattr(cfg, "EMBEDDING_DIM", 256)),
            max_seq_len=resolved_max_seq_len,
            dropout=float(getattr(cfg, "EMBEDDING_DROPOUT", 0.1)),
            pos_type=str(getattr(cfg, "POSITIONAL_EMBEDDING_TYPE", "learned")),
            padding_idx=padding_idx,
        )

    def forward_batch(self, batch: dict, position_offset: int = 0) -> torch.Tensor:
        """
        Helper for dataset2 output: expects batch dict with key input_ids.
        """
        if "input_ids" not in batch:
            raise KeyError("Batch must contain input_ids.")
        return self.forward(batch["input_ids"], position_offset=position_offset)


def resolve_embedding_max_seq_len(cfg=CONFIG, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)

    configured = getattr(cfg, "EMBEDDING_MAX_SEQ_LEN", None)
    if configured is not None:
        return int(configured)

    block_size = int(getattr(cfg, "DATASET_BLOCK_SIZE", 128))
    return block_size


def build_embedding_from_tokenizer(
    tokenizer: HybridTokenizerLLM,
    cfg=CONFIG,
    max_seq_len: Optional[int] = None,
) -> EmbeddingLayer:
    return EmbeddingLayer.from_config(
        cfg=cfg,
        vocab_size=len(tokenizer.token2id),
        max_seq_len=max_seq_len,
        padding_idx=tokenizer.token2id.get(tokenizer.pad_token),
    )


def build_tokenizer_and_embedding(cfg=CONFIG, load_tokenizer: bool = True):
    tokenizer = build_tokenizer_from_config(cfg)
    if load_tokenizer:
        tokenizer.load_all(
            str(cfg.VOCAB_PATH),
            str(cfg.TOKEN2ID_PATH),
            str(cfg.ID2TOKEN_PATH),
        )
    embedding = build_embedding_from_tokenizer(tokenizer, cfg=cfg)
    return tokenizer, embedding
