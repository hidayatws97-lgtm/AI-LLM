import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from checkpoint_meta import validate_checkpoint_metadata
from gptModel import GPTModel


def _resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(req)


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_text(tokenizer, text: str):
    return tokenizer.encode_ids(text, add_bos=True)


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or generated_ids.numel() == 0:
        return logits

    adjusted = logits.clone()
    for idx in torch.unique(generated_ids):
        token_id = int(idx.item())
        value = adjusted[..., token_id]
        adjusted[..., token_id] = torch.where(value < 0, value * penalty, value / penalty)
    return adjusted


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")

    next_logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1)

    next_logits = next_logits / temperature

    if top_k > 0:
        k = min(top_k, next_logits.size(-1))
        values, _ = torch.topk(next_logits, k=k, dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_remove = cumulative_probs > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False

        remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(1, sorted_indices, sorted_remove)
        next_logits = next_logits.masked_fill(remove_mask, float("-inf"))

    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _resolve_model_state(payload: dict) -> Dict[str, torch.Tensor]:
    model_state = payload.get("model_state_dict")
    if isinstance(model_state, dict):
        return model_state

    state = payload.get("state_dict")
    if isinstance(state, dict):
        return state

    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload

    raise ValueError("Checkpoint missing compatible model state (model_state_dict/state_dict).")


def _load_checkpoint_payload(checkpoint: Path) -> dict:
    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    return payload


def _checkpoint_env(payload: dict) -> Optional[str]:
    env = payload.get("env")
    if env is None:
        return None
    return str(env)


def _load_model(
    checkpoint: Path,
    cfg,
    device: torch.device,
    payload: Optional[dict] = None,
    strict_metadata: bool = True,
):
    model, tokenizer = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    if payload is None:
        payload = _load_checkpoint_payload(checkpoint)

    metadata_warnings = validate_checkpoint_metadata(
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        strict=strict_metadata,
    )
    for warning in metadata_warnings:
        print(f"[warn] {warning}")

    model_state = _resolve_model_state(payload)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    ids = _encode_text(tokenizer, prompt)
    if not ids:
        raise ValueError("Prompt produced zero tokens.")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    max_seq_len = int(model.embedding.max_seq_len)
    eos_token_id = tokenizer.eos_token_id
    
    # Tambahkan deteksi UNK untuk debugging
    unk_token_id = getattr(tokenizer, 'unk_token_id', None)

    for _ in range(max_new_tokens):
        x = input_ids[:, -max_seq_len:]
        with torch.inference_mode():
            logits = model(x)
        next_token = _sample_next_token(
            logits=logits[:, -1, :],
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            generated_ids=input_ids[0],
            repetition_penalty=float(repetition_penalty),
        )
        
        if unk_token_id is not None and int(next_token.item()) == unk_token_id:
            # Opsional: beri penalti lebih berat atau log jika model terus pilih UNK
            pass
            
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break

    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained GPT checkpoint."
    )
    parser.add_argument("--env", default=None, help="Config environment: development/production/testing.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: <TRAIN_SAVE_DIR>/gpt_last.pt).",
    )
    parser.add_argument("--prompt", default=None, help="Prompt text. If omitted, starts interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; <=0 uses greedy.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 disables).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling in (0,1]; 1 disables.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty > 1.0 reduces repeats.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda (default from config).")
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="Allow checkpoints without metadata fingerprint (legacy/experimental only).",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise SystemExit("--top-p must be in (0, 1].")
    if args.repetition_penalty < 1.0:
        raise SystemExit("--repetition-penalty must be >= 1.0.")

    _set_seed(args.seed)

    requested_env = args.env or os.getenv("TOKENIZER_ENV")
    env = requested_env or "production"
    cfg = get_config(env)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else (
        Path(str(getattr(cfg, "TRAIN_SAVE_DIR", "reports/checkpoints"))) / "gpt_last.pt"
    )
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    try:
        payload = _load_checkpoint_payload(checkpoint_path)
    except Exception as exc:
        raise SystemExit(f"Failed to load checkpoint: {exc}")

    checkpoint_env = _checkpoint_env(payload)
    if checkpoint_env and checkpoint_env != env:
        if requested_env is None and args.checkpoint is not None:
            env = checkpoint_env
            cfg = get_config(env)
        else:
            raise SystemExit(
                "Checkpoint env mismatch.\n"
                f"  requested env: {env}\n"
                f"  checkpoint env: {checkpoint_env}\n"
                "Use --env to match the checkpoint or pass a matching checkpoint."
            )

    requested_device = args.device or str(getattr(cfg, "TRAIN_DEVICE", "auto"))
    device = _resolve_device(requested_device)

    try:
        model, tokenizer = _load_model(
            checkpoint=checkpoint_path,
            cfg=cfg,
            device=device,
            payload=payload,
            strict_metadata=not args.allow_unsafe_checkpoint,
        )
    except FileNotFoundError:
        raise SystemExit(
            "Tokenizer artifacts not found. Train/save tokenizer first:\n"
            f"  vocab: {cfg.VOCAB_PATH}\n"
            f"  token2id: {cfg.TOKEN2ID_PATH}\n"
            f"  id2token: {cfg.ID2TOKEN_PATH}"
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load model/tokenizer: {exc}")

    print("=" * 80)
    print("GPT Generate")
    print("=" * 80)
    print(f"env={env}")
    if checkpoint_env:
        print(f"checkpoint_env={checkpoint_env}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"temperature={args.temperature}")
    print(f"top_k={args.top_k}")
    print(f"top_p={args.top_p}")
    print(f"repetition_penalty={args.repetition_penalty}")
    print(f"seed={args.seed if args.seed is not None else 'none'}")

    validation_mode = "unsafe-allowed" if args.allow_unsafe_checkpoint else "strict"
    print(f"metadata_validation={validation_mode}")

    if args.prompt is not None:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            device=device,
        )
        print("-" * 80)
        print(f"Prompt : {args.prompt}")
        print(f"Output : {output}")
        print("-" * 80)
        return

    print("Interactive mode. Type ':q' to quit.\n")
    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if not prompt:
            continue
        if prompt.lower() == ":q":
            print("Exit.")
            break

        try:
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                device=device,
            )
            print(f"Output > {output}\n")
        except Exception as exc:
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    main()
