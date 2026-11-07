import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # older transformers
    from transformers import AutoModelForVision2Seq as AutoVLM

from models.utils import find_target_modules

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_base_model(model_id, dtype, attn_implementation):
    """`dtype` is a name from DTYPES. fp32 master weights keep fp16 autocast + GradScaler safe on a T4."""
    if dtype not in DTYPES:
        raise ValueError(f"model.dtype must be one of {list(DTYPES)}, got {dtype!r}")
    return AutoVLM.from_pretrained(model_id, torch_dtype=DTYPES[dtype],
                                   attn_implementation=attn_implementation)


def build_lora_model(base_model, r, alpha, dropout):
    """Experiment 1: standard LoRA, uniform rank r on every target module."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=find_target_modules(base_model),
        bias="none",
    )
    return get_peft_model(base_model, config)


def build_adalora_model(base_model, total_step, init_r, target_r, tinit_frac, tfinal_frac,
                        deltaT, beta1, beta2, orth_reg_weight, alpha, dropout):
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
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=find_target_modules(base_model),
        bias="none",
    )
    return get_peft_model(base_model, config)
