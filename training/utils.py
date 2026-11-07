import json
import math
import os

import torch


def compute_total_steps(n_examples, batch_size, grad_accum, epochs):
    """Optimizer steps for the whole run. AdaLoRA needs this before the model is built."""
    batches_per_epoch = math.ceil(n_examples / batch_size)
    return math.ceil(batches_per_epoch / grad_accum) * epochs


def resolve_precision(precision):
    """'auto' -> bf16 if supported (3090) else fp16 (T4). Returns (name, autocast dtype or None)."""
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    return precision, amp_dtype


def is_adalora(model):
    return hasattr(model.base_model, "update_and_allocate")


def epoch_order(n, seed, epoch):
    """Deterministic shuffle per epoch, so a resumed run sees the same batches."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator=generator).tolist()


def save_checkpoint(path, model, optimizer, scheduler, scaler, state):
    os.makedirs(path, exist_ok=True)
    payload = {
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "state": state,
    }
    if is_adalora(model):
        # Importance EMAs drive the allocation; without them a resume restarts the scoring.
        allocator = model.base_model.rankallocator
        payload["rankallocator"] = {k: getattr(allocator, k)
                                    for k in ("ipt", "exp_avg_ipt", "exp_avg_unc")}
        payload["rank_pattern"] = model.peft_config["default"].rank_pattern
    tmp = os.path.join(path, "state.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(path, "state.pt"))  # atomic: a disconnect never leaves a half file


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    payload = torch.load(os.path.join(path, "state.pt"), map_location=device, weights_only=False)
    model.load_state_dict(payload["trainable"], strict=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    if "rankallocator" in payload:
        allocator = model.base_model.rankallocator
        for k, v in payload["rankallocator"].items():
            setattr(allocator, k, v)
        model.peft_config["default"].rank_pattern = payload["rank_pattern"]
    return payload["state"]


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
