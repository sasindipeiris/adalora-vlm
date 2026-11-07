import re

import torch

# SmolVLM (Idefics3) layout:
#   vision_model.encoder.layers.{i}     SigLIP, 12 layers  -> q/k/v/out_proj, fc1/fc2
#   connector.modality_projection.proj  pixel-shuffle projection into the LM
#   text_model.layers.{i}               SmolLM2, 30 layers -> q/k/v/o_proj, gate/up/down_proj
# lm_head is never adapted.
_PATTERNS = {
    "vision": re.compile(
        r"vision_model\.encoder\.layers\.(\d+)\."
        r"(self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))$"
    ),
    "text": re.compile(
        r"text_model\.layers\.(\d+)\."
        r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    ),
    "connector": re.compile(r"connector\.modality_projection\.(proj)$"),
}

# Shared matrix types so vision and text sit on the same axis
# (AdaLoRA paper: attention Wq/Wk/Wv/Wo vs FFN Wf1/Wf2).
_MATRIX_TYPE = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.out_proj": "attn_o",
    "self_attn.o_proj": "attn_o",
    "mlp.fc1": "ffn_in",
    "mlp.gate_proj": "ffn_in",
    "mlp.up_proj": "ffn_in",
    "mlp.fc2": "ffn_out",
    "mlp.down_proj": "ffn_out",
}


def classify_module(name):
    """Return {tower, layer, matrix, matrix_type} for an adaptable module name, else None."""
    for tower, pattern in _PATTERNS.items():
        m = pattern.search(name)
        if m is None:
            continue
        if tower == "connector":
            return {
                "tower": tower,
                "layer": 0,
                "matrix": "proj",
                "matrix_type": "connector",
            }
        return {
            "tower": tower,
            "layer": int(m.group(1)),
            "matrix": m.group(2),
            "matrix_type": _MATRIX_TYPE[m.group(2)],
        }
    return None


def find_target_modules(model):
    """Full names of every nn.Linear in the vision encoder, connector and text decoder."""
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and classify_module(name) is not None
    ]
    if not targets:
        raise ValueError(
            "No target modules found; check module names for this transformers version"
        )
    return targets


@torch.no_grad()
def get_rank_allocation(peft_model, adapter_name="default"):
    """
    One row per adapted module with its tower, layer, matrix type and rank.

    AdaLoRA: rank = non-zero entries of lora_E (pruned singular values are set to
    exactly 0), so call this after training. LoRA: rank = r.
    """
    rows = []
    for name, module in peft_model.named_modules():
        info = classify_module(name)
        if info is None or adapter_name not in getattr(module, "lora_A", {}):
            continue

        if adapter_name in getattr(module, "lora_E", {}):
            lora_e = module.lora_E[adapter_name]
            init_rank = lora_e.shape[0]
            rank = int((lora_e != 0).sum().item())
        else:
            init_rank = rank = module.lora_A[adapter_name].weight.shape[0]

        base = module.get_base_layer()
        params_per_rank = base.in_features + base.out_features
        rows.append(
            {
                "name": name,
                **info,
                "rank": rank,
                "init_rank": init_rank,
                "params": rank * params_per_rank,
                "init_params": init_rank * params_per_rank,
            }
        )
    return rows


def count_parameters(model):
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
