from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from embedding import build_tokenizer_and_embedding, resolve_embedding_max_seq_len


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.max_seq_len = int(max_seq_len)
        self.scale = self.head_dim ** -0.5
        self.mask_fill_value = -1e4

        # Fused QKV projection (faster than separate)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _mask_fill_value_for_dtype(self, dtype: torch.dtype) -> float:
        if dtype.is_floating_point:
            return torch.finfo(dtype).min
        return self.mask_fill_value

    def _compute_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, Hd)
        k = k.transpose(1, 2)  # (B, H, T, Hd)
        v = v.transpose(1, 2)  # (B, H, T, Hd)
        return q, k, v

    def _make_causal_allow_mask(self, q_len: int, k_len: int, past_len: int, device) -> torch.Tensor:
        query_positions = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(-1)
        key_positions = torch.arange(k_len, device=device).unsqueeze(0)
        return key_positions <= query_positions  # (q_len, k_len)

    def _resolve_key_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        key_len: int,
        device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            key_mask = attention_mask.to(device=device)
            if key_mask.size(0) != batch_size:
                raise ValueError(
                    f"attention_mask batch mismatch: got {key_mask.size(0)}, expected {batch_size}."
                )
            if key_mask.size(1) < key_len:
                raise ValueError(
                    f"attention_mask length {key_mask.size(1)} is smaller than key length {key_len}."
                )
            if key_mask.size(1) > key_len:
                key_mask = key_mask[:, -key_len:]
            return key_mask != 0

        if attention_mask.dim() == 4:
            return None

        raise ValueError(
            "attention_mask must be 2D (B, T) or 4D broadcastable tensor (B, 1, Q, K)."
        )

    def _normalize_4d_attention_mask(
        self,
        attention_mask: torch.Tensor,
        q: torch.Tensor,
        q_len: int,
        k_len: int,
        past_len: int,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=q.device)
        expected_shape = (q.size(0), self.n_head, q_len, k_len)
        try:
            torch.broadcast_shapes(tuple(mask.shape), expected_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"4D attention_mask shape {tuple(mask.shape)} is not broadcastable to {expected_shape}."
            ) from exc

        causal_allow = self._make_causal_allow_mask(
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            device=q.device,
        ).unsqueeze(0).unsqueeze(0)

        if mask.dtype == torch.bool or not mask.dtype.is_floating_point:
            allow = mask if mask.dtype == torch.bool else mask != 0
            return allow & causal_allow

        causal_mask = torch.zeros((1, 1, q_len, k_len), device=q.device, dtype=q.dtype)
        causal_mask = causal_mask.masked_fill(
            ~causal_allow,
            self._mask_fill_value_for_dtype(q.dtype),
        )
        return mask.to(dtype=q.dtype) + causal_mask

    def _build_attn_mask(
        self,
        q: torch.Tensor,
        q_len: int,
        k_len: int,
        past_len: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], bool, Optional[torch.Tensor]]:
        B = q.size(0)
        device = q.device

        if attention_mask is not None and attention_mask.dim() == 4:
            attn_mask = self._normalize_4d_attention_mask(
                attention_mask=attention_mask,
                q=q,
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
            )
            return attn_mask, False, None

        key_mask = self._resolve_key_mask(
            attention_mask=attention_mask,
            batch_size=B,
            key_len=k_len,
            device=device,
        )

        if past_len == 0 and key_mask is None:
            # Fast path: let backend generate causal mask (flash/mem-efficient kernels).
            return None, True, None

        causal_allow = self._make_causal_allow_mask(q_len=q_len, k_len=k_len, past_len=past_len, device=device)
        allow = causal_allow.unsqueeze(0).unsqueeze(0).expand(B, 1, q_len, k_len)

        if key_mask is not None:
            allow = allow & key_mask[:, None, None, :]

        attn_mask = torch.zeros((B, 1, q_len, k_len), device=device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allow, self._mask_fill_value_for_dtype(q.dtype))

        query_mask = None
        if key_mask is not None:
            query_mask = key_mask[:, -q_len:]

        return attn_mask, False, query_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, D)
        attention_mask: optional
          - (B, T) with 1 for valid tokens, 0 for padded tokens
          - or 4D broadcastable to (B, n_head, Q, K). Bool/int masks use
            True/1 for allowed attention; float masks are additive. Causal
            masking is always applied on top of 4D masks.
        past_key_value: optional tuple (k, v), each shape (B, n_head, T_cache, head_dim)
        use_cache:
          - False: return output tensor (B, T, D)
          - True: return tuple (output, (k, v))
        """

        B, T, D = x.size()
        if D != self.d_model:
            raise ValueError(f"Input hidden dim {D} does not match d_model={self.d_model}.")
        if T > self.max_seq_len and past_key_value is None:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                "Increase ATTENTION_MAX_SEQ_LEN / EMBEDDING_MAX_SEQ_LEN in config."
            )

        q, k, v = self._compute_qkv(x)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.dim() != 4 or past_v.dim() != 4:
                raise ValueError("past_key_value tensors must be rank-4: (B, n_head, T, head_dim).")
            if past_k.shape[:2] != (B, self.n_head) or past_v.shape[:2] != (B, self.n_head):
                raise ValueError("past_key_value batch/head dimensions do not match current input.")
            if past_k.shape[-1] != self.head_dim or past_v.shape[-1] != self.head_dim:
                raise ValueError("past_key_value head_dim does not match current attention head_dim.")
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        if k.size(-2) > self.max_seq_len:
            k = k[:, :, -self.max_seq_len :, :]
            v = v[:, :, -self.max_seq_len :, :]
            past_len = max(0, k.size(-2) - T)

        q_len = q.size(-2)
        k_len = k.size(-2)

        attn_mask, is_causal, query_mask = self._build_attn_mask(
            q=q,
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            attention_mask=attention_mask,
        )

        try:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        except TypeError:
            # Fallback for torch versions that do not expose `scale` argument.
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        # (B, n_head, T, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, D)

        # Final projection
        output = self.out_proj(attn_output)

        if query_mask is not None:
            output = output * query_mask[:, :, None].to(dtype=output.dtype)

        if use_cache:
            present = (k, v)
            return output, present
        return output

    @classmethod
    def from_config(
        cls,
        cfg=CONFIG,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        resolved_d_model = int(d_model if d_model is not None else getattr(cfg, "EMBEDDING_DIM", 256))
        resolved_n_head = int(getattr(cfg, "ATTENTION_NUM_HEADS", 8))
        if resolved_d_model % resolved_n_head != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({resolved_d_model}) must be divisible by ATTENTION_NUM_HEADS ({resolved_n_head})."
            )
        resolved_max_seq_len = resolve_attention_max_seq_len(cfg=cfg, override=max_seq_len)
        return cls(
            d_model=resolved_d_model,
            n_head=resolved_n_head,
            dropout=float(getattr(cfg, "ATTENTION_DROPOUT", 0.1)),
            bias=bool(getattr(cfg, "ATTENTION_USE_BIAS", False)),
            max_seq_len=resolved_max_seq_len,
        )

    def forward_batch(
        self,
        batch: dict,
        embedding_layer: nn.Module,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
    ):
        """
        End-to-end helper from dataset batch: input_ids -> embedding -> attention output.

        position_offset controls positional embeddings during cached generation.
        If omitted with a cache, it defaults to the cached key length.
        """
        if "input_ids" not in batch:
            raise KeyError("Batch must contain input_ids.")

        if position_offset is None:
            position_offset = 0 if past_key_value is None else int(past_key_value[0].size(-2))

        hidden = embedding_layer.forward_batch(batch, position_offset=position_offset)
        attention_mask = batch.get("attention_mask")
        return self.forward(
            hidden,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )


def resolve_attention_max_seq_len(cfg=CONFIG, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)

    configured = getattr(cfg, "ATTENTION_MAX_SEQ_LEN", None)
    if configured is not None:
        return int(configured)

    # Fallback to embedding sequence limit, then dataset block size.
    return resolve_embedding_max_seq_len(cfg=cfg, override=None)


def build_attention_from_config(
    cfg=CONFIG,
    d_model: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> MultiHeadSelfAttention:
    return MultiHeadSelfAttention.from_config(
        cfg=cfg,
        d_model=d_model,
        max_seq_len=max_seq_len,
    )


def build_tokenizer_embedding_attention(cfg=CONFIG, load_tokenizer: bool = True):
    tokenizer, embedding = build_tokenizer_and_embedding(cfg=cfg, load_tokenizer=load_tokenizer)
    attention = build_attention_from_config(
        cfg=cfg,
        d_model=embedding.token_embedding.embedding_dim,
        max_seq_len=embedding.max_seq_len,
    )
    return tokenizer, embedding, attention
