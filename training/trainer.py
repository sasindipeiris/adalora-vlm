import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from models.utils import get_rank_allocation
from training.utils import (
    append_jsonl,
    compute_total_steps,
    epoch_order,
    is_adalora,
    load_checkpoint,
    resolve_precision,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    """Built from the `train:` section of configs/config.yaml plus the run's output_dir and seed."""

    output_dir: str
    seed: int
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    precision: str  # auto | bf16 | fp16 | fp32
    num_workers: int
    log_every: int  # optimizer steps
    save_every: int  # optimizer steps
    alloc_log_every: int  # optimizer steps; must be a multiple of AdaLoRA deltaT


def train(model, train_dataset, collator, cfg):
    """
    Train a LoRA or AdaLoRA PEFT model with one loop.

    Writes to cfg.output_dir:
      checkpoint/state.pt      resumable state (adapter, optimizer, scheduler, scaler, AdaLoRA scores)
      train_log.jsonl          loss / lr per log_every steps
      rank_history.jsonl       AdaLoRA only: per-module ranks every alloc_log_every steps
      adapter/                 final PEFT adapter (AdaLoRA config includes rank_pattern)
      rank_allocation.json     final per-module ranks (both methods)
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, amp_dtype = resolve_precision(cfg.precision)
    adalora = is_adalora(model)

    model.to(device)
    model.config.use_cache = False

    batches_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
    total_steps = compute_total_steps(
        len(train_dataset), cfg.batch_size, cfg.grad_accum, cfg.epochs
    )
    if adalora:
        peft_cfg = model.peft_config["default"]
        if peft_cfg.total_step != total_steps:
            raise ValueError(
                f"AdaLoRA total_step={peft_cfg.total_step} but trainer runs {total_steps}"
            )
        if cfg.alloc_log_every % peft_cfg.deltaT != 0:
            raise ValueError(
                "alloc_log_every must be a multiple of deltaT (ranks are only masked then)"
            )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoint")
    state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    if os.path.exists(os.path.join(ckpt_dir, "state.pt")):
        state = load_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, device)
        print(f"Resumed from step {state['global_step']}")
    else:
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
            json.dump(
                {
                    **asdict(cfg),
                    "precision": precision,
                    "total_steps": total_steps,
                    "method": "adalora" if adalora else "lora",
                },
                f,
                indent=2,
            )
        if adalora:
            names = [row["name"] for row in get_rank_allocation(model)]
            with open(os.path.join(cfg.output_dir, "rank_modules.json"), "w") as f:
                json.dump(names, f)

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    history_path = os.path.join(cfg.output_dir, "rank_history.jsonl")

    model.train()
    progress = tqdm(total=total_steps, initial=state["global_step"], desc="train")
    running_loss, running_steps, start_time = 0.0, 0, time.time()

    for epoch in range(state["epoch"], cfg.epochs):
        first_batch = state["batch_in_epoch"] if epoch == state["epoch"] else 0
        order = epoch_order(len(train_dataset), cfg.seed, epoch)[
            first_batch * cfg.batch_size :
        ]
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=order,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        batch_idx = first_batch
        for batch in loader:
            # Size of the current accumulation group (the last one in an epoch may be short).
            group_start = (batch_idx // cfg.grad_accum) * cfg.grad_accum
            group_size = min(cfg.grad_accum, batches_per_epoch - group_start)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                # For AdaLoRA, PEFT adds the orthogonality regulariser to this loss.
                loss = model(**batch).loss / group_size
            scaler.scale(loss).backward()
            running_loss += loss.item()
            batch_idx += 1

            if batch_idx - group_start < group_size:
                continue

            # ---- optimizer step ----
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step = state["global_step"]
            if adalora:
                if scaler.get_scale() < scale_before:
                    # fp16 overflow: the step was skipped and grads are inf/nan. Zero them so
                    # the importance EMA isn't poisoned, but still call the allocator so the
                    # budget schedule (incl. fixing rank_pattern) never misses a step.
                    for p in params:
                        if p.grad is not None:
                            p.grad.zero_()
                # Uses the current grads, so it must run before zero_grad.
                model.base_model.update_and_allocate(step)
            optimizer.zero_grad(set_to_none=True)

            if adalora and step % cfg.alloc_log_every == 0:
                ranks = [row["rank"] for row in get_rank_allocation(model)]
                append_jsonl(history_path, {"step": step, "ranks": ranks})

            state["global_step"] = step + 1
            state["epoch"], state["batch_in_epoch"] = epoch, batch_idx
            running_steps += 1
            progress.update(1)

            if state["global_step"] % cfg.log_every == 0:
                record = {
                    "step": state["global_step"],
                    "epoch": epoch,
                    "loss": running_loss / running_steps,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - start_time, 1),
                }
                append_jsonl(log_path, record)
                progress.set_postfix(loss=f"{record['loss']:.4f}")
                running_loss, running_steps = 0.0, 0

            if state["global_step"] % cfg.save_every == 0:
                save_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, state)

        state["epoch"], state["batch_in_epoch"] = epoch + 1, 0

    progress.close()

    allocation = get_rank_allocation(model)
    if adalora:
        append_jsonl(
            history_path,
            {"step": total_steps, "ranks": [r["rank"] for r in allocation]},
        )
    with open(os.path.join(cfg.output_dir, "rank_allocation.json"), "w") as f:
        json.dump(allocation, f, indent=2)
    model.save_pretrained(os.path.join(cfg.output_dir, "adapter"))
    save_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, state)

    return model, allocation
