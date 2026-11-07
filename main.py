"""
  python main.py --method zeroshot          # untuned baseline: evaluate only
  python main.py --method lora              # experiment 1
  python main.py --method adalora           # experiment 2
  python main.py --compare                  # table + figures from the three runs

All settings live in configs/config.yaml (pass --config to use another file).
"""
import argparse
import json
import os
import time

import yaml

METHODS = ("zeroshot", "lora", "adalora")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "configs", "config.yaml")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to the YAML config")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if not args.compare and args.method is None:
        p.error("give --method or --compare")
    return args


def _resolve(path):
    """Expand ~ and $VARS; relative paths are taken from the project directory."""
    path = os.path.expandvars(os.path.expanduser(path))
    return os.path.normpath(os.path.join(PROJECT_DIR, path))


def load_config(path):
    """
    Read the YAML and fill unset paths from the storage root
    (default ./adalora_vlm in the project root: a plain directory or a symlink).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    root = _resolve(paths.get("storage_root") or "adalora_vlm")
    paths["storage_root"] = root
    defaults = {"hf_home": "hf_cache", "output_root": "runs",
                "subset_dir": os.path.join("data", "docvqa_subset")}
    for key, sub in defaults.items():
        paths[key] = _resolve(paths[key]) if paths.get(key) else os.path.join(root, sub)
    return cfg


def run_method(method, cfg):
    # Imported here so HF_HOME / PYTORCH_CUDA_ALLOC_CONF from the config are set first.
    import torch

    from dataset.dataset import prepare_docvqa_dataset
    from evaluation.compare import run_dir
    from evaluation.evaluation import evaluate_anls
    from evaluation.utils import allocation_summary
    from models.model import build_adalora_model, build_lora_model, load_base_model
    from models.utils import count_parameters
    from training.trainer import TrainConfig, train
    from training.utils import compute_total_steps, resolve_precision

    seed, paths, train_cfg = cfg["seed"], cfg["paths"], cfg["train"]
    out_dir = run_dir(paths["output_root"], method, seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"method": method, **cfg}, f, indent=2)

    torch.manual_seed(seed)
    precision, amp_dtype = resolve_precision(train_cfg["precision"])

    train_ds, val_ds, processor, train_collator, eval_collator = prepare_docvqa_dataset(
        model_id=cfg["model"]["model_id"], subset_dir=paths["subset_dir"], seed=seed, **cfg["data"])

    model = load_base_model(**cfg["model"])

    results = {"method": method, "seed": seed, "n_train": len(train_ds),
               "n_val": len(val_ds), "precision": precision}
    start = time.time()

    if method != "zeroshot":
        total_steps = compute_total_steps(len(train_ds), train_cfg["batch_size"],
                                          train_cfg["grad_accum"], train_cfg["epochs"])
        if method == "lora":
            model = build_lora_model(model, **cfg["lora"])
        else:
            model = build_adalora_model(model, total_step=total_steps, **cfg["adalora"])
        trainable, total = count_parameters(model)
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        results.update(trainable_params=trainable, total_params=total, total_steps=total_steps)

        model, allocation = train(model, train_ds, train_collator,
                                  TrainConfig(output_dir=out_dir, seed=seed, **train_cfg))
        results["allocation_summary"] = allocation_summary(allocation)
        results["train_time_s"] = round(time.time() - start, 1)
    else:
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.config.use_cache = True
    mean_anls, predictions = evaluate_anls(
        model, val_ds, eval_collator, processor, amp_dtype=amp_dtype,
        num_workers=train_cfg["num_workers"], **cfg["eval"])
    results["anls"] = mean_anls
    results["total_time_s"] = round(time.time() - start, 1)

    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{method}] ANLS = {mean_anls:.4f}  -> {out_dir}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_home"])
    # Less fragmentation of GPU memory between steps (must be set before torch is imported).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.compare:
        from evaluation.compare import compare
        compare(cfg["paths"]["output_root"], cfg["seed"], METHODS)
    else:
        run_method(args.method, cfg)


if __name__ == "__main__":
    main()
