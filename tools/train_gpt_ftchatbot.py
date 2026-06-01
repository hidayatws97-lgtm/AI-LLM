import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from checkpoint_meta import build_checkpoint_metadata, validate_checkpoint_metadata
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


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[key] = value
    return moved


def _save_checkpoint(
    path: Path,
    model: GPTModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    env: str,
    cfg,
    finetune_metadata: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "env": env,
        "metadata": build_checkpoint_metadata(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            env=env,
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if finetune_metadata:
        payload["finetune"] = finetune_metadata
    torch.save(payload, str(path))


def _log_save_event(
    epoch: int,
    total_epochs: int,
    step: int,
    run_step: int,
    batch: str | int,
    loss: float,
    ma_loss: float | None,
    lr: float,
    start_time: float,
    cfg,
) -> None:
    elapsed = time.time() - start_time
    ppl = math.exp(min(loss, 20.0))
    ma_text = f"{ma_loss:.6f}" if ma_loss is not None else "0.000000"
    log_entry = (
        f"epoch={epoch}/{total_epochs} "
        f"step={step} "
        f"run_step={run_step} "
        f"batch={batch} "
        f"loss={loss:.6f} "
        f"ma_loss={ma_text} "
        f"lr={lr:.6e} "
        f"ppl~={ppl:.3f} "
        f"elapsed={elapsed:.1f}s"
    )

    ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
    base_log_path = Path(getattr(cfg, "TRAIN_SAVE_LOG_PATH", "reports/logs/train_save.log"))
    log_path = PROJECT_ROOT / base_log_path.parent / f"{base_log_path.stem}_{ts_str}{base_log_path.suffix}"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + '\n')

def _load_model_checkpoint(path: Path, model: GPTModel, tokenizer, cfg) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")

    model_state = payload.get("model_state_dict") or payload.get("model") or payload.get("state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint missing compatible model state.")

    warnings = validate_checkpoint_metadata(
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        strict=False,
    )
    for warning in warnings:
        print(f"[finetuneV3][warn] {warning}")

    model.load_state_dict(model_state, strict=True)
    return {
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
    }


def _load_resume_checkpoint(
    path: Path,
    model: GPTModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    cfg,
) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint payload must be a dictionary.")

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Resume checkpoint missing 'model_state_dict'.")
    warnings = validate_checkpoint_metadata(
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        strict=False,
    )
    for warning in warnings:
        print(f"[finetuneV3][warn] {warning}")
    model.load_state_dict(model_state, strict=True)

    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    return {
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
    }


def _encode_ids(
    tokenizer,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    return tokenizer.encode_ids(text, add_bos=add_bos, add_eos=add_eos)


def _split_records(text: str) -> List[str]:
    chunks = []
    current = []
    for raw_line in text.splitlines():
        if raw_line.strip():
            current.append(raw_line.rstrip())
            continue
        if current:
            chunks.append("\n".join(current).strip())
            current = []
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


class InstructionFineTuneDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        block_size: int,
        assistant_marker: str = ".assistant.",
        user_marker: str = ".user.",
        target_assistant: str = "last",
        window_stride: Optional[int] = None,
        min_response_tokens: int = 1,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Fine-tune data file not found: {self.data_path}")

        self.pad_id = int(tokenizer.token2id[tokenizer.pad_token])
        self.block_size = int(block_size)
        self.window_stride = int(window_stride if window_stride is not None else max(1, self.block_size // 2))
        self.min_response_tokens = int(min_response_tokens)
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2.")
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1.")

        self.assistant_marker = assistant_marker
        self.user_marker = user_marker
        self.target_assistant = target_assistant.lower().strip()
        if self.target_assistant not in {"last", "all"}:
            raise ValueError("target_assistant must be one of: last, all.")

        text = self.data_path.read_text(encoding="utf-8")
        records = _split_records(text)
        if not records:
            raise ValueError(f"No fine-tune records found in {self.data_path}")

        self.samples = []
        skipped_no_marker = 0
        skipped_short = 0

        for record_idx, record in enumerate(records):
            if self.assistant_marker not in record:
                skipped_no_marker += 1
                continue

            token_ids, supervised_targets = self._build_record_training_sequence(tokenizer, record)
            supervised_indices = [idx for idx, flag in enumerate(supervised_targets) if flag]
            if len(token_ids) < 2 or len(supervised_indices) < self.min_response_tokens:
                skipped_short += 1
                continue

            max_start = max(0, len(token_ids) - (self.block_size + 1))
            first_supervised = supervised_indices[0]
            first_start = min(max(0, first_supervised - 1), max_start)
            starts = list(range(first_start, max_start + 1, self.window_stride)) or [first_start]
            if starts[-1] != max_start:
                starts.append(max_start)

            seen = set()
            for start in starts:
                if start in seen:
                    continue
                seen.add(start)

                end = min(len(token_ids), start + self.block_size + 1)
                chunk = token_ids[start:end]
                if len(chunk) < 2:
                    continue

                input_ids = chunk[:-1]
                labels = chunk[1:]
                masked_labels = [-100] * len(labels)
                attention_mask = [1] * len(input_ids)

                for idx, label in enumerate(labels):
                    target_pos = start + idx + 1
                    if target_pos < len(supervised_targets) and supervised_targets[target_pos]:
                        masked_labels[idx] = label

                supervised_tokens = sum(1 for value in masked_labels if value != -100)
                if supervised_tokens < self.min_response_tokens:
                    continue

                if len(input_ids) < self.block_size:
                    pad_len = self.block_size - len(input_ids)
                    input_ids = input_ids + ([self.pad_id] * pad_len)
                    masked_labels = masked_labels + ([-100] * pad_len)
                    attention_mask = attention_mask + ([0] * pad_len)
                elif len(input_ids) > self.block_size:
                    input_ids = input_ids[: self.block_size]
                    masked_labels = masked_labels[: self.block_size]
                    attention_mask = attention_mask[: self.block_size]

                self.samples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(masked_labels, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "record_idx": record_idx,
                    }
                )

        if not self.samples:
            raise ValueError(
                "No fine-tune samples could be built. "
                "Check data format, assistant marker, tokenizer artifacts, and block size."
            )

        self.total_records = len(records)
        self.skipped_no_marker = skipped_no_marker
        self.skipped_short = skipped_short

    def _split_instruction_record(self, record: str) -> tuple[str, str]:
        marker_index = record.rfind(self.assistant_marker)
        if marker_index < 0:
            return "", ""

        prompt_text = record[: marker_index + len(self.assistant_marker)].strip()
        response_text = record[marker_index + len(self.assistant_marker):].strip()
        return prompt_text, response_text

    def _append_tokenized_piece(
        self,
        token_ids: List[int],
        supervised_targets: List[bool],
        tokenizer,
        text: str,
        supervised: bool,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> None:
        if not text and not add_bos and not add_eos:
            return
        piece_ids = _encode_ids(tokenizer, text, add_bos=add_bos, add_eos=add_eos)
        token_ids.extend(piece_ids)
        supervised_targets.extend([bool(supervised)] * len(piece_ids))

    def _build_record_training_sequence(self, tokenizer, record: str) -> tuple[List[int], List[bool]]:
        if self.target_assistant == "last":
            prompt_text, response_text = self._split_instruction_record(record)
            if not prompt_text or not response_text:
                return [], []
            prompt_ids = _encode_ids(tokenizer, prompt_text, add_bos=True)
            response_ids = _encode_ids(tokenizer, response_text, add_eos=True)
            return prompt_ids + response_ids, ([False] * len(prompt_ids)) + ([True] * len(response_ids))

        token_ids: List[int] = []
        supervised_targets: List[bool] = []
        cursor = 0
        add_bos = True

        while cursor < len(record):
            marker_index = record.find(self.assistant_marker, cursor)
            if marker_index < 0:
                self._append_tokenized_piece(
                    token_ids,
                    supervised_targets,
                    tokenizer,
                    record[cursor:].strip(),
                    supervised=False,
                    add_bos=add_bos,
                )
                break

            prefix = record[cursor: marker_index + len(self.assistant_marker)].strip()
            self._append_tokenized_piece(
                token_ids,
                supervised_targets,
                tokenizer,
                prefix,
                supervised=False,
                add_bos=add_bos,
            )
            add_bos = False

            response_start = marker_index + len(self.assistant_marker)
            next_user = record.find(self.user_marker, response_start)
            next_assistant = record.find(self.assistant_marker, response_start)
            candidates = [idx for idx in (next_user, next_assistant) if idx >= 0]
            response_end = min(candidates) if candidates else len(record)
            response_text = record[response_start:response_end].strip()
            self._append_tokenized_piece(
                token_ids,
                supervised_targets,
                tokenizer,
                response_text,
                supervised=True,
                add_eos=(response_end == len(record)),
            )
            cursor = response_end

        return token_ids, supervised_targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "attention_mask": sample["attention_mask"],
        }


def create_finetune_dataloader(
    data_path: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    window_stride: Optional[int] = None,
    assistant_marker: str = ".assistant.",
    user_marker: str = ".user.",
    target_assistant: str = "last",
    min_response_tokens: int = 1,
) -> DataLoader:
    dataset = InstructionFineTuneDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        assistant_marker=assistant_marker,
        user_marker=user_marker,
        target_assistant=target_assistant,
        window_stride=window_stride,
        min_response_tokens=min_response_tokens,
    )

    if num_workers == 0:
        persistent_workers = False

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
    )
    return loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GPTModel on .user./.assistant. instruction data, aligned with tokenizer2/dataset2/train_gptV2."
    )
    parser.add_argument("--env", default=None, help="Config environment: development/production/testing.")
    parser.add_argument("--data-path", default="data/corpus_ft.txt", help="Instruction fine-tune text file path.")
    parser.add_argument("--base-checkpoint", default="reports/checkpoints/gpt_last.pt", help="Base pretrained checkpoint.")
    parser.add_argument("--resume", default=None, help="Resume an in-progress fine-tune checkpoint.")
    parser.add_argument("--output", default="reports/checkpoints/gpt_finetuneV3.pt", help="Final checkpoint output path.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda (default from config).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default from config).")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default from config).")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default from config).")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm (default from config). <=0 disables.")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N optimizer steps.")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N optimizer steps. 0=disabled.")
    parser.add_argument("--loss-window", type=int, default=None, help="Moving-average loss window size.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is available.")
    parser.add_argument("--plateau", action="store_true", help="Enable ReduceLROnPlateau scheduler.")
    parser.add_argument("--no-plateau", action="store_true", help="Disable ReduceLROnPlateau scheduler.")
    parser.add_argument("--plateau-factor", type=float, default=None)
    parser.add_argument("--plateau-patience", type=int, default=None)
    parser.add_argument("--plateau-threshold", type=float, default=None)
    parser.add_argument("--plateau-min-lr", type=float, default=None)
    parser.add_argument("--plateau-cooldown", type=int, default=None)
    parser.add_argument("--early-stop", action="store_true", help="Enable early stop on moving-average loss.")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stop.")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Force enable shuffle.")
    parser.add_argument("--no-shuffle", action="store_true", help="Force disable shuffle.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--window-stride", type=int, default=None, help="Stride for slicing long assistant responses into windows.")
    parser.add_argument("--assistant-marker", default=".assistant.", help="Marker used to identify assistant turns.")
    parser.add_argument("--user-marker", default=".user.", help="Marker used to identify user turns in multi-turn records.")
    parser.add_argument("--target-assistant", choices=("last", "all"), default="last", help="Train only the last assistant turn or all assistant turns in each record.")
    parser.add_argument("--min-response-tokens", type=int, default=1, help="Discard windows with too few supervised tokens.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    if args.shuffle and args.no_shuffle:
        raise SystemExit("Choose only one of --shuffle or --no-shuffle.")
    if args.pin_memory and args.no_pin_memory:
        raise SystemExit("Choose only one of --pin-memory or --no-pin-memory.")
    if args.persistent_workers and args.no_persistent_workers:
        raise SystemExit("Choose only one of --persistent-workers or --no-persistent-workers.")
    if args.plateau and args.no_plateau:
        raise SystemExit("Choose only one of --plateau or --no-plateau.")
    if args.early_stop and args.no_early_stop:
        raise SystemExit("Choose only one of --early-stop or --no-early-stop.")

    env = args.env or os.getenv("TOKENIZER_ENV", "production")
    cfg = get_config(env)
    _set_seed(args.seed)

    train_device = str(args.device if args.device is not None else getattr(cfg, "TRAIN_DEVICE", "auto"))
    train_epochs = int(args.epochs if args.epochs is not None else getattr(cfg, "TRAIN_EPOCHS", 1))
    if train_epochs < 1:
        raise SystemExit("--epochs must be >= 1.")
    train_max_steps = args.max_steps if args.max_steps is not None else getattr(cfg, "TRAIN_MAX_STEPS", None)
    if train_max_steps is not None:
        train_max_steps = int(train_max_steps)
    if train_max_steps is not None and train_max_steps < 1:
        raise SystemExit("--max-steps must be >= 1 when provided.")
    train_lr = float(args.lr if args.lr is not None else getattr(cfg, "TRAIN_LR", 3e-4))
    train_weight_decay = float(
        args.weight_decay if args.weight_decay is not None else getattr(cfg, "TRAIN_WEIGHT_DECAY", 0.1)
    )
    train_grad_clip = float(
        args.grad_clip if args.grad_clip is not None else getattr(cfg, "TRAIN_GRAD_CLIP", 1.0)
    )
    train_log_every = int(
        args.log_every if args.log_every is not None else getattr(cfg, "TRAIN_LOG_EVERY", 20)
    )
    train_save_every = int(
        args.save_every if args.save_every is not None else getattr(cfg, "TRAIN_SAVE_EVERY", 0)
    )
    train_loss_window = int(
        args.loss_window if args.loss_window is not None else getattr(cfg, "TRAIN_LOSS_WINDOW", 50)
    )
    if train_loss_window <= 0:
        raise SystemExit("--loss-window must be > 0.")
    train_grad_accum = int(
        args.grad_accum if args.grad_accum is not None else getattr(cfg, "TRAIN_GRAD_ACCUM", 1)
    )
    if train_grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1.")

    plateau_enabled = (
        bool(args.plateau)
        if (args.plateau or args.no_plateau)
        else bool(getattr(cfg, "TRAIN_PLATEAU_ENABLED", True))
    )
    if args.no_plateau:
        plateau_enabled = False
    plateau_factor = float(
        args.plateau_factor if args.plateau_factor is not None else getattr(cfg, "TRAIN_PLATEAU_FACTOR", 0.5)
    )
    plateau_patience = int(
        args.plateau_patience if args.plateau_patience is not None else getattr(cfg, "TRAIN_PLATEAU_PATIENCE", 3)
    )
    plateau_threshold = float(
        args.plateau_threshold if args.plateau_threshold is not None else getattr(cfg, "TRAIN_PLATEAU_THRESHOLD", 1e-4)
    )
    plateau_min_lr = float(
        args.plateau_min_lr if args.plateau_min_lr is not None else getattr(cfg, "TRAIN_PLATEAU_MIN_LR", 1e-6)
    )
    plateau_cooldown = int(
        args.plateau_cooldown if args.plateau_cooldown is not None else getattr(cfg, "TRAIN_PLATEAU_COOLDOWN", 0)
    )
    early_stop_enabled = (
        bool(args.early_stop)
        if (args.early_stop or args.no_early_stop)
        else bool(getattr(cfg, "TRAIN_EARLY_STOP_ENABLED", False))
    )
    if args.no_early_stop:
        early_stop_enabled = False
    early_stop_patience = int(
        args.early_stop_patience if args.early_stop_patience is not None else getattr(cfg, "TRAIN_EARLY_STOP_PATIENCE", 8)
    )
    early_stop_min_delta = float(
        args.early_stop_min_delta if args.early_stop_min_delta is not None else getattr(cfg, "TRAIN_EARLY_STOP_MIN_DELTA", 1e-4)
    )

    batch_size = int(args.batch_size if args.batch_size is not None else getattr(cfg, "DATASET_BATCH_SIZE", 4))
    block_size = int(args.block_size if args.block_size is not None else getattr(cfg, "DATASET_BLOCK_SIZE", 128))
    shuffle = bool(args.shuffle) if (args.shuffle or args.no_shuffle) else bool(getattr(cfg, "DATASET_SHUFFLE", True))
    if args.no_shuffle:
        shuffle = False
    num_workers = int(args.num_workers if args.num_workers is not None else getattr(cfg, "DATASET_NUM_WORKERS", 0))
    pin_memory = bool(args.pin_memory) if (args.pin_memory or args.no_pin_memory) else bool(getattr(cfg, "DATASET_PIN_MEMORY", True))
    if args.no_pin_memory:
        pin_memory = False
    persistent_workers = bool(args.persistent_workers) if (args.persistent_workers or args.no_persistent_workers) else bool(getattr(cfg, "DATASET_PERSISTENT_WORKERS", False))
    if args.no_persistent_workers:
        persistent_workers = False

    device = _resolve_device(train_device)

    if os.name == "nt" and num_workers > 0:
        print(
            f"[finetuneV3] Windows detected: override num_workers {num_workers} -> 0 "
            "to avoid multiprocessing spawn/pickle errors."
        )
        num_workers = 0
        persistent_workers = False

    if device.type != "cuda" and pin_memory:
        print("[finetuneV3] CPU device detected: override pin_memory True -> False.")
        pin_memory = False

    try:
        model, tokenizer = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    except FileNotFoundError:
        raise SystemExit(
            "Tokenizer artifacts not found.\n"
            f"Expected vocab: {cfg.VOCAB_PATH}\n"
            f"Expected token2id: {cfg.TOKEN2ID_PATH}\n"
            f"Expected id2token: {cfg.ID2TOKEN_PATH}"
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_lr,
        weight_decay=train_weight_decay,
    )

    resume_epoch = 0
    global_step = 0
    if args.resume:
        resume_info = _load_resume_checkpoint(
            path=Path(args.resume),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            cfg=cfg,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = train_lr
        resume_epoch = int(resume_info["epoch"])
        global_step = int(resume_info["step"])
        print(
            f"[finetuneV3] resumed from {args.resume} "
            f"(checkpoint_epoch={resume_epoch}, checkpoint_step={global_step})"
        )
        print(f"[finetuneV3] effective_resume_lr={optimizer.param_groups[0]['lr']:.6e}")
    else:
        base_info = _load_model_checkpoint(
            Path(args.base_checkpoint),
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        print(
            f"[finetuneV3] loaded base checkpoint {args.base_checkpoint} "
            f"(checkpoint_epoch={base_info['epoch']}, checkpoint_step={base_info['step']})"
        )

    dataloader = create_finetune_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        window_stride=args.window_stride,
        assistant_marker=args.assistant_marker,
        user_marker=args.user_marker,
        target_assistant=args.target_assistant,
        min_response_tokens=args.min_response_tokens,
    )

    dataset = dataloader.dataset
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = None
    if plateau_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
        )

    output_path = Path(args.output)
    save_dir = output_path.parent
    run_step = 0
    skipped_non_finite = 0
    start_time = time.time()
    last_heartbeat = start_time
    recent_losses = deque(maxlen=train_loss_window)
    moving_avg_loss = None
    best_moving_avg = float("inf")
    early_bad_windows = 0
    stop_training = False
    model.train()

    print("=" * 80)
    print("GPT Fine-tuning V3")
    print("=" * 80)
    print(f"env={env}")
    print(f"device={device}")
    print(f"data_path={args.data_path}")
    print(f"base_checkpoint={args.base_checkpoint}")
    print(f"resume={args.resume if args.resume else 'none'}")
    print(f"output={output_path}")
    print(f"vocab_size={model.vocab_size}")
    print(f"d_model={model.d_model}")
    print(f"num_layers={model.transformer.num_layers}")
    print(f"dataset_records={dataset.total_records}")
    print(f"dataset_samples={len(dataset)}")
    print(f"skipped_no_marker={dataset.skipped_no_marker}")
    print(f"skipped_short={dataset.skipped_short}")
    print(f"assistant_marker={args.assistant_marker}")
    print(f"user_marker={args.user_marker}")
    print(f"target_assistant={args.target_assistant}")
    print(f"batch_size={batch_size}")
    print(f"block_size={block_size}")
    print(f"window_stride={dataset.window_stride}")
    print(f"epochs={train_epochs}")
    print(f"max_steps={train_max_steps}")
    print(f"lr={train_lr}")
    print(f"log_every={train_log_every}")
    print(f"loss_window={train_loss_window}")
    print(f"plateau_enabled={plateau_enabled}")
    print(f"early_stop_enabled={early_stop_enabled}")
    print(f"mixed_precision={use_amp}")
    print(f"grad_accum={train_grad_accum}")
    print(f"effective_batch={train_grad_accum * batch_size}")
    print("fine-tuning started... waiting for first batch/step")

    for epoch in range(resume_epoch, resume_epoch + train_epochs):
        optimizer.zero_grad(set_to_none=True)
        pending_accum = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model.forward_batch(batch)
            if not (isinstance(out, tuple) and len(out) >= 2):
                raise RuntimeError("Model must return (loss, logits) during training.")

            loss = out[0]
            if not torch.is_tensor(loss):
                raise RuntimeError("Loss output is not a tensor.")

            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                skipped_non_finite += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[warn] non-finite loss detected (loss={loss_value}) at "
                    f"epoch={epoch + 1} batch={batch_idx + 1}; step skipped. "
                    f"total_skipped={skipped_non_finite}"
                )
                continue

            loss = loss / train_grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            pending_accum += 1

            stepped = False
            if (batch_idx + 1) % train_grad_accum == 0:
                if train_grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pending_accum = 0
                stepped = True

            run_step += 1
            recent_losses.append(loss_value)

            if len(recent_losses) == train_loss_window:
                moving_avg_loss = sum(recent_losses) / float(train_loss_window)
                if scheduler is not None and stepped and global_step % train_loss_window == 0:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(moving_avg_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < prev_lr:
                        print(
                            f"[scheduler] lr reduced: {prev_lr:.6e} -> {new_lr:.6e} "
                            f"(ma_loss={moving_avg_loss:.6f})"
                        )

                if early_stop_enabled and stepped and global_step % train_loss_window == 0:
                    if moving_avg_loss < (best_moving_avg - early_stop_min_delta):
                        best_moving_avg = moving_avg_loss
                        early_bad_windows = 0
                    else:
                        early_bad_windows += 1
                        if early_bad_windows >= early_stop_patience:
                            print(
                                f"[early-stop] triggered at step={global_step} "
                                f"(ma_loss={moving_avg_loss:.6f}, best={best_moving_avg:.6f}, "
                                f"bad_windows={early_bad_windows})"
                            )
                            stop_training = True

            if stepped and (global_step == 1 or global_step % max(1, train_log_every) == 0):
                ppl = math.exp(min(20.0, loss_value))
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                ma_text = f"{moving_avg_loss:.6f}" if moving_avg_loss is not None else "n/a"
                print(
                    f"epoch={epoch + 1}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_step={run_step} "
                    f"batch={batch_idx + 1} "
                    f"loss={loss_value:.6f} "
                    f"ma_loss={ma_text} "
                    f"lr={current_lr:.6e} "
                    f"ppl~={ppl:.3f} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_heartbeat = time.time()

            now = time.time()
            if now - last_heartbeat >= 30.0:
                elapsed = now - start_time
                print(
                    f"[heartbeat] epoch={epoch + 1}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_step={run_step} "
                    f"batch={batch_idx + 1} "
                    f"elapsed={elapsed:.1f}s "
                    f"skipped_non_finite={skipped_non_finite}"
                )
                last_heartbeat = now

            if train_save_every > 0 and stepped and global_step % train_save_every == 0:
                ckpt = save_dir / f"{output_path.stem}_step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                    finetune_metadata={"base_checkpoint": str(args.base_checkpoint), "data_path": str(args.data_path)},
                )
                _log_save_event(
                    epoch=epoch + 1,
                    total_epochs=resume_epoch + train_epochs,
                    step=global_step,
                    run_step=run_step,
                    batch=batch_idx + 1,
                    loss=loss_value,
                    ma_loss=moving_avg_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    start_time=start_time,
                    cfg=cfg,
                )
                print(f"saved checkpoint: {ckpt}")

            if train_max_steps is not None and global_step >= train_max_steps:
                stop_training = True
                break
            if stop_training:
                break

        if pending_accum > 0 and not stop_training:
            if train_grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(
                f"[finetuneV3] applied final accumulated gradients for epoch={epoch + 1} "
                f"(micro_batches={pending_accum}, step={global_step})"
            )

        if stop_training:
            break

    _save_checkpoint(
        path=output_path,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=epoch + 1,
        step=global_step,
        env=env,
        cfg=cfg,
        finetune_metadata={"base_checkpoint": str(args.base_checkpoint), "data_path": str(args.data_path)},
    )
    _log_save_event(
        epoch=epoch + 1,
        total_epochs=resume_epoch + train_epochs,
        step=global_step,
        run_step=run_step,
        batch="end",
        loss=loss_value,
        ma_loss=moving_avg_loss,
        lr=optimizer.param_groups[0]["lr"],
        start_time=start_time,
        cfg=cfg,
    )
    print(f"fine-tuning complete: run_steps={run_step}, global_steps={global_step}")
    print(f"skipped_non_finite={skipped_non_finite}")
    print(f"final checkpoint: {output_path}")


if __name__ == "__main__":
    main()
