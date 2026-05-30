import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value))


def tokenizer_metadata(tokenizer, cfg=None) -> Dict[str, Any]:
    token2id = {str(tok): int(idx) for tok, idx in tokenizer.token2id.items()}
    special_token_ids = {
        "pad": token2id.get(str(tokenizer.pad_token)),
        "unk": token2id.get(str(tokenizer.unk_token)),
        "bos": token2id.get(str(tokenizer.bos_token)),
        "eos": token2id.get(str(tokenizer.eos_token)),
    }

    metadata = {
        "vocab_size": len(token2id),
        "token2id_sha256": _json_sha256(token2id),
        "special_tokens": {
            "pad": str(tokenizer.pad_token),
            "unk": str(tokenizer.unk_token),
            "bos": str(tokenizer.bos_token),
            "eos": str(tokenizer.eos_token),
        },
        "special_token_ids": special_token_ids,
        "word_start": str(getattr(tokenizer, "word_start", "")),
        "normalize_whitespace": bool(getattr(tokenizer, "normalize_whitespace", True)),
    }

    if cfg is not None:
        metadata["paths"] = {
            "vocab": _path_text(getattr(cfg, "VOCAB_PATH", None)),
            "token2id": _path_text(getattr(cfg, "TOKEN2ID_PATH", None)),
            "id2token": _path_text(getattr(cfg, "ID2TOKEN_PATH", None)),
        }

    return metadata


def model_metadata(model, cfg=None) -> Dict[str, Any]:
    metadata = {
        "vocab_size": int(model.vocab_size),
        "d_model": int(model.d_model),
        "num_layers": int(model.transformer.num_layers),
        "max_seq_len": int(model.embedding.max_seq_len),
        "positional_embedding_type": str(getattr(model.embedding, "pos_type", "")),
        "tie_weights": bool(getattr(model, "tie_weights", True)),
    }

    if cfg is not None:
        metadata.update(
            {
                "attention_num_heads": int(getattr(cfg, "ATTENTION_NUM_HEADS", 0)),
                "transformer_mlp_ratio": float(getattr(cfg, "TRANSFORMER_MLP_RATIO", 0.0)),
                "transformer_use_bias": bool(getattr(cfg, "TRANSFORMER_USE_BIAS", False)),
            }
        )

    return metadata


def build_checkpoint_metadata(model, tokenizer, cfg, env: str) -> Dict[str, Any]:
    return {
        "metadata_version": 1,
        "env": str(env),
        "model": model_metadata(model, cfg=cfg),
        "tokenizer": tokenizer_metadata(tokenizer, cfg=cfg),
    }


def validate_checkpoint_metadata(
    payload: Dict[str, Any],
    model,
    tokenizer,
    cfg=None,
    strict: bool = True,
) -> List[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        message = (
            "Checkpoint has no metadata fingerprint. It may be an older checkpoint; "
            "retrain or resave to enable production-safe validation."
        )
        if strict:
            raise ValueError(message)
        return [message]

    expected_model = model_metadata(model, cfg=cfg)
    actual_model = metadata.get("model")
    if not isinstance(actual_model, dict):
        raise ValueError("Checkpoint metadata missing model section.")

    for key, expected in expected_model.items():
        actual = actual_model.get(key)
        if actual != expected:
            raise ValueError(
                "Checkpoint model metadata mismatch: "
                f"{key} expected {expected!r}, found {actual!r}."
            )

    expected_tokenizer = tokenizer_metadata(tokenizer, cfg=cfg)
    actual_tokenizer = metadata.get("tokenizer")
    if not isinstance(actual_tokenizer, dict):
        raise ValueError("Checkpoint metadata missing tokenizer section.")

    for key in (
        "vocab_size",
        "token2id_sha256",
        "special_tokens",
        "special_token_ids",
        "word_start",
        "normalize_whitespace",
    ):
        expected = expected_tokenizer.get(key)
        actual = actual_tokenizer.get(key)
        if actual != expected:
            raise ValueError(
                "Checkpoint tokenizer metadata mismatch: "
                f"{key} expected {expected!r}, found {actual!r}."
            )

    return []
