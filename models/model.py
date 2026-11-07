import os

STORAGE_ROOT = "/mnt/ext_drive/sasindi_rs/adalora_vlm"
os.environ.setdefault("HF_HOME", os.path.join(STORAGE_ROOT, "hf_cache"))

import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # older transformers
    from transformers import AutoModelForVision2Seq as AutoVLM

from models.utils import find_target_modules

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_base_model(model_id=MODEL_ID, dtype=torch.float32, attn_implementation="sdpa"):
    """fp32 master weights keep fp16 autocast + GradScaler safe on a T4 (bf16 on a 3090)."""
    return AutoVLM.from_pretrained(model_id, torch_dtype=dtype,
                                   attn_implementation=attn_implementation)


def build_lora_model(base_model, r=8, lora_alpha=16, lora_dropout=0.05):
    """Experiment 1: standard LoRA, uniform rank r on every target module."""
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=find_target_modules(base_model),
        bias="none",
    )
    return get_peft_model(base_model, config)


def build_adalora_model(base_model, total_step, init_r=12, target_r=8,
                        tinit_frac=0.1, tfinal_frac=0.2, deltaT=10, beta1=0.85, beta2=0.85,
                        orth_reg_weight=0.1, lora_alpha=16, lora_dropout=0.05):
    """
    Experiment 2: AdaLoRA. Every module starts at rank `init_r`; the total budget
    is pruned to an average of `target_r` per module (same budget as LoRA r=target_r).

    Schedule over `total_step` optimizer steps:
      [0, tinit)                     warm-up, no pruning
      [tinit, total_step - tfinal)   budget shrinks, re-allocated every deltaT steps
      [total_step - tfinal, end)     allocation frozen, fine-tune only

    The training loop must call model.base_model.update_and_allocate(global_step)
    after loss.backward() and before optimizer.zero_grad().
    """
    tinit = int(tinit_frac * total_step)
    tfinal = int(tfinal_frac * total_step)
    if tinit + tfinal >= total_step:
        raise ValueError("tinit + tfinal must leave a budgeting phase before total_step")

    config = AdaLoraConfig(
        init_r=init_r,
        target_r=target_r,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        beta1=beta1,
        beta2=beta2,
        orth_reg_weight=orth_reg_weight,
        total_step=total_step,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=find_target_modules(base_model),
        bias="none",
    )
    return get_peft_model(base_model, config)
