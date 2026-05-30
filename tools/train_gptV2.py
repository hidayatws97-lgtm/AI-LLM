import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict

import torch

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


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[k] = v
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
        },
        str(path),
    )


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
    
    # Buat timestamp berdasarkan start_time proses (misal: 20260527_1012)
    ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
    
    # Ambil base path dari config dan sisipkan timestamp ke nama filenya
    base_log_path = Path(getattr(cfg, "TRAIN_SAVE_LOG_PATH", "reports/logs/train_save.log"))
    log_path = PROJECT_ROOT / base_log_path.parent / f"{base_log_path.stem}_{ts_str}{base_log_path.suffix}"
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


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
    validate_checkpoint_metadata(
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        strict=False,
    )
    model.load_state_dict(model_state, strict=True)

    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    return {
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
    }


def _optimizer_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> None:
    if grad_clip > 0:
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GPTModel using integrated build_training_components() pipeline."
    )
    parser.add_argument("--env", default=None, help="Config environment: development/production/testing.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda (default from config).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default from config).")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (default from config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default from config).")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default from config).")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm (default from config). <=0 disables.")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N steps (default from config).")
    parser.add_argument("--save-dir", default=None, help="Checkpoint output directory (default from config).")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N steps (default from config). 0=disabled.")
    parser.add_argument("--loss-window", type=int, default=None, help="Moving-average loss window size (default from config).")
    parser.add_argument("--plateau", action="store_true", help="Enable ReduceLROnPlateau scheduler.")
    parser.add_argument("--no-plateau", action="store_true", help="Disable ReduceLROnPlateau scheduler.")
    parser.add_argument("--plateau-factor", type=float, default=None, help="LR reduction factor for plateau scheduler.")
    parser.add_argument("--plateau-patience", type=int, default=None, help="Plateau patience in scheduler evaluation windows.")
    parser.add_argument("--plateau-threshold", type=float, default=None, help="Plateau threshold for improvement.")
    parser.add_argument("--plateau-min-lr", type=float, default=None, help="Minimum LR for scheduler.")
    parser.add_argument("--plateau-cooldown", type=int, default=None, help="Cooldown windows for scheduler.")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stop on non-improving moving-average loss.")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stop.")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Early stop patience in eval windows.")
    parser.add_argument("--early-stop-min-delta", type=float, default=None, help="Minimum delta to count as improvement.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from (e.g. reports/checkpoints/gpt_last.pt).")
    parser.add_argument("--token-file", default=None, help="Override token file path.")

    # Tambahan Fitur
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")    
    
    # Dataloader overrides
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset-type", default=None, help="packed/sliding")
    parser.add_argument("--stride", type=int, default=None, help="Only for sliding dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Force enable shuffle.")
    parser.add_argument("--no-shuffle", action="store_true", help="Force disable shuffle.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
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

    train_device = str(args.device if args.device is not None else getattr(cfg, "TRAIN_DEVICE", "auto"))
    train_epochs = int(args.epochs if args.epochs is not None else getattr(cfg, "TRAIN_EPOCHS", 1))
    train_max_steps = args.max_steps if args.max_steps is not None else getattr(cfg, "TRAIN_MAX_STEPS", None)
    if train_max_steps is not None:
        train_max_steps = int(train_max_steps)
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
    train_save_dir = str(
        args.save_dir if args.save_dir is not None else getattr(cfg, "TRAIN_SAVE_DIR", "reports/checkpoints")
    )
    train_loss_window = int(
        args.loss_window if args.loss_window is not None else getattr(cfg, "TRAIN_LOSS_WINDOW", 50)
    )
    if train_epochs < 1:
        raise SystemExit("--epochs must be >= 1.")
    if train_max_steps is not None and train_max_steps < 1:
        raise SystemExit("--max-steps must be >= 1 when provided.")
    if train_loss_window <= 0:
        raise SystemExit("--loss-window must be > 0.")
    if train_log_every <= 0:
        raise SystemExit("--log-every must be > 0.")
    if train_save_every < 0:
        raise SystemExit("--save-every must be >= 0.")
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
    train_grad_accum = int(
        args.grad_accum if args.grad_accum is not None else getattr(cfg, "TRAIN_GRAD_ACCUM", 1)
    )

    if train_grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")

    device = _resolve_device(train_device)

    dataloader_kwargs = {}
    if args.batch_size is not None:
        dataloader_kwargs["batch_size"] = int(args.batch_size)
    if args.block_size is not None:
        dataloader_kwargs["block_size"] = int(args.block_size)
    if args.dataset_type is not None:
        dataloader_kwargs["dataset_type"] = str(args.dataset_type)
    if args.stride is not None:
        dataloader_kwargs["stride"] = int(args.stride)
    if args.num_workers is not None:
        dataloader_kwargs["num_workers"] = int(args.num_workers)
    if args.shuffle:
        dataloader_kwargs["shuffle"] = True
    if args.no_shuffle:
        dataloader_kwargs["shuffle"] = False
    if args.pin_memory:
        dataloader_kwargs["pin_memory"] = True
    if args.no_pin_memory:
        dataloader_kwargs["pin_memory"] = False
    if args.persistent_workers:
        dataloader_kwargs["persistent_workers"] = True
    if args.no_persistent_workers:
        dataloader_kwargs["persistent_workers"] = False

    # Windows spawn multiprocessing can fail for some dataset/file setups.
    # Force single-process loading on Windows for robust default behavior.
    if os.name == "nt":
        requested_workers = int(
            dataloader_kwargs.get("num_workers", getattr(cfg, "DATASET_NUM_WORKERS", 0))
        )
        if requested_workers > 0:
            dataloader_kwargs["num_workers"] = 0
            dataloader_kwargs["persistent_workers"] = False
            print(
                f"[train_gpt] Windows detected: override num_workers {requested_workers} -> 0 "
                "to avoid multiprocessing spawn/pickle errors."
            )

    # pin_memory is only useful when using CUDA.
    if device.type != "cuda" and "pin_memory" not in dataloader_kwargs:
        if bool(getattr(cfg, "DATASET_PIN_MEMORY", False)):
            dataloader_kwargs["pin_memory"] = False
            print("[train_gpt] CPU device detected: override pin_memory True -> False.")

    try:
        components = GPTModel.build_training_components(
            cfg=cfg,
            load_tokenizer=True,
            token_file=args.token_file,
            **dataloader_kwargs,
        )
    except FileNotFoundError:
        raise SystemExit(
            "Tokenizer artifacts or token file not found.\n"
            f"Expected vocab: {cfg.VOCAB_PATH}\n"
            f"Expected token2id: {cfg.TOKEN2ID_PATH}\n"
            f"Expected id2token: {cfg.ID2TOKEN_PATH}\n"
            f"Expected token file: {args.token_file or cfg.DATASET_TOKEN_FILE}"
        )

    model: GPTModel = components["model"].to(device)
    tokenizer = components["tokenizer"]
    dataloader = components["dataloader"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_lr,
        weight_decay=train_weight_decay,
    )
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

    save_dir = Path(train_save_dir)
    global_step = 0
    resume_epoch = 0
    run_step = 0
    run_optimizer_steps = 0
    skipped_non_finite = 0
    start_time = time.time()
    last_heartbeat = start_time
    recent_losses = deque(maxlen=train_loss_window)
    moving_avg_loss = None
    best_moving_avg = float("inf")
    early_bad_windows = 0
    stop_training = False
    model.train()

    if args.resume:
        resume_info = _load_resume_checkpoint(
            path=Path(args.resume),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            cfg=cfg,
        )
        # Keep the CLI/config learning rate when resuming; checkpoint optimizer
        # state should restore moments/step buffers, not silently override LR.
        for param_group in optimizer.param_groups:
            param_group["lr"] = train_lr
        resume_epoch = int(resume_info["epoch"])
        global_step = int(resume_info["step"])
        print(
            f"[train_gpt] resumed from {args.resume} "
            f"(checkpoint_epoch={resume_epoch}, checkpoint_step={global_step})"
        )
        print(f"[train_gpt] effective_resume_lr={optimizer.param_groups[0]['lr']:.6e}")

    print("=" * 80)
    print("GPT Training")
    print("=" * 80)
    print(f"env={env}")
    print(f"device={device}")
    print(f"vocab_size={model.vocab_size}")
    print(f"d_model={model.d_model}")
    print(f"num_layers={model.transformer.num_layers}")
    print(f"epochs={train_epochs}")
    print(f"max_optimizer_steps={train_max_steps}")
    print(f"lr={train_lr}")
    print(f"log_every={train_log_every}")
    print(f"loss_window={train_loss_window}")
    print(f"plateau_enabled={plateau_enabled}")
    print(f"early_stop_enabled={early_stop_enabled}")
    print(f"resume={args.resume if args.resume else 'none'}")
    print(f"mixed_precision={use_amp}")
    print(f"grad_accum={train_grad_accum}")
    print(f"effective_batch={train_grad_accum * dataloader.batch_size}")
    print("training started... waiting for first batch/optimizer step")

    last_epoch_number = resume_epoch
    for local_epoch in range(train_epochs):
        epoch_number = resume_epoch + local_epoch + 1
        last_epoch_number = epoch_number
        accum_batches = 0
        optimizer.zero_grad(set_to_none=True)

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
                accum_batches = 0
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[warn] non-finite loss detected (loss={loss_value}) at "
                    f"epoch={epoch_number} batch={batch_idx + 1}; accumulated step dropped. "
                    f"total_skipped={skipped_non_finite}"
                )
                continue

            loss = loss / train_grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            run_step += 1
            accum_batches += 1
            recent_losses.append(loss_value)
            optimizer_stepped = False

            if accum_batches >= train_grad_accum:
                _optimizer_step(model, optimizer, scaler, use_amp, train_grad_clip)
                global_step += 1
                run_optimizer_steps += 1
                accum_batches = 0
                optimizer_stepped = True

            if optimizer_stepped and len(recent_losses) == train_loss_window:
                moving_avg_loss = sum(recent_losses) / float(train_loss_window)
                if scheduler is not None and global_step % train_loss_window == 0:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(moving_avg_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < prev_lr:
                        print(
                            f"[scheduler] lr reduced: {prev_lr:.6e} -> {new_lr:.6e} "
                            f"(ma_loss={moving_avg_loss:.6f})"
                        )

                if early_stop_enabled and global_step % train_loss_window == 0:
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

            if optimizer_stepped and (run_optimizer_steps == 1 or global_step % train_log_every == 0):
                ppl = math.exp(min(20.0, loss_value))
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                ma_text = f"{moving_avg_loss:.6f}" if moving_avg_loss is not None else "n/a"
                print(
                    f"epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
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
                    f"[heartbeat] epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
                    f"batch={batch_idx + 1} "
                    f"elapsed={elapsed:.1f}s "
                    f"skipped_non_finite={skipped_non_finite}"
                )
                last_heartbeat = now

            if optimizer_stepped and train_save_every > 0 and global_step % train_save_every == 0:
                ckpt = save_dir / f"gpt_step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                )
                _log_save_event(
                    epoch=epoch_number,
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

            if train_max_steps is not None and run_optimizer_steps >= train_max_steps:
                stop_training = True
            if stop_training:
                break

        if accum_batches > 0 and not stop_training:
            _optimizer_step(model, optimizer, scaler, use_amp, train_grad_clip)
            global_step += 1
            run_optimizer_steps += 1
            print(
                f"[train_gpt] flushed {accum_batches} accumulated batch(es) at "
                f"epoch={epoch_number}; step={global_step}"
            )

            if len(recent_losses) == train_loss_window:
                moving_avg_loss = sum(recent_losses) / float(train_loss_window)
                if scheduler is not None and global_step % train_loss_window == 0:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(moving_avg_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < prev_lr:
                        print(
                            f"[scheduler] lr reduced: {prev_lr:.6e} -> {new_lr:.6e} "
                            f"(ma_loss={moving_avg_loss:.6f})"
                        )

                if early_stop_enabled and global_step % train_loss_window == 0:
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

            if run_optimizer_steps == 1 or global_step % train_log_every == 0:
                ppl = math.exp(min(20.0, loss_value))
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                ma_text = f"{moving_avg_loss:.6f}" if moving_avg_loss is not None else "n/a"
                print(
                    f"epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
                    f"batch=end "
                    f"loss={loss_value:.6f} "
                    f"ma_loss={ma_text} "
                    f"lr={current_lr:.6e} "
                    f"ppl~={ppl:.3f} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_heartbeat = time.time()

            if train_save_every > 0 and global_step % train_save_every == 0:
                ckpt = save_dir / f"gpt_step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                )
                _log_save_event(
                    epoch=epoch_number,
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
                print(f"saved checkpoint: {ckpt}")

        if train_max_steps is not None and run_optimizer_steps >= train_max_steps:
            break
        if stop_training:
            break

    final_ckpt = save_dir / "gpt_last.pt"
    _save_checkpoint(
        path=final_ckpt,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=last_epoch_number,
        step=global_step,
        env=env,
        cfg=cfg,
    )
    _log_save_event(
        epoch=last_epoch_number,
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
    print(f"training complete: micro_steps={run_step}, optimizer_steps={run_optimizer_steps}, global_steps={global_step}")
    print(f"skipped_non_finite={skipped_non_finite}")
    print(f"final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
