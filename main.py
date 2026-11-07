"""
Usage (run from the AdaLoRA/ directory):
  python main.py --method zeroshot          # untuned baseline: evaluate only
  python main.py --method lora              # experiment 1
  python main.py --method adalora           # experiment 2
  python main.py --compare                  # table + figures from the three runs
  python main.py --method adalora --debug   # tiny smoke test before a full run
"""
import argparse
import json
import os
import time

# Less fragmentation of GPU memory between steps (must be set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Imported first: sets HF_HOME before transformers is imported anywhere.
from data.dataset import STORAGE_ROOT, prepare_docvqa_dataset

import torch

from models.model import build_adalora_model, build_lora_model, load_base_model
from models.utils import classify_module, count_parameters, find_target_modules
from training.trainer import TrainConfig, train
from training.utils import compute_total_steps, resolve_precision
from evaluation.compare import METHODS, compare, run_dir
from evaluation.evaluation import evaluate_anls
from evaluation.utils import allocation_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--debug", action="store_true", help="16 train / 8 val examples, a few steps")
    p.add_argument("--output_root", default=os.path.join(STORAGE_ROOT, "runs"))
    p.add_argument("--seed", type=int, default=42)
    # data
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--image_longest_edge", type=int, default=None)
    # training (identical for LoRA and AdaLoRA)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)   # effective batch 16
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--num_workers", type=int, default=4)
    # adapters: LoRA r == AdaLoRA target_r -> same final rank budget
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=8)
    p.add_argument("--deltaT", type=int, default=10)
    p.add_argument("--alloc_log_every", type=int, default=20)
    # evaluation
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=32)
    args = p.parse_args()

    if not args.compare and args.method is None:
        p.error("give --method or --compare")
    if args.debug:
        args.n_train, args.n_val, args.epochs = 16, 8, 1
        args.batch_size, args.grad_accum, args.eval_batch_size = 2, 2, 4
        args.deltaT, args.alloc_log_every = 1, 1
    return args


def debug_checks(model, train_dataset, train_collator, processor):
    """Sanity checks worth seeing before spending a full run."""
    counts = {}
    for name in find_target_modules(model):
        tower = classify_module(name)["tower"]
        counts[tower] = counts.get(tower, 0) + 1
    print(f"[debug] target modules per tower: {counts}  (expected vision 72, connector 1, text 210)")

    batch = train_collator([train_dataset[0], train_dataset[1]])
    for i in range(2):
        answer_ids = batch["labels"][i][batch["labels"][i] != -100]
        print(f"[debug] label tokens decode to: {processor.tokenizer.decode(answer_ids)!r}"
              f"  | reference: {train_dataset[i]['answers'][0]!r}")
    print(f"[debug] sequence length: {batch['input_ids'].shape[1]}, "
          f"pixel_values: {tuple(batch['pixel_values'].shape)}")


def run_method(args):
    out_dir = run_dir(args.output_root, args.method, args.seed, args.debug)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    precision, amp_dtype = resolve_precision(args.precision)

    train_ds, val_ds, processor, train_collator, eval_collator = prepare_docvqa_dataset(
        n_train=args.n_train, n_val=args.n_val, seed=args.seed,
        image_longest_edge=args.image_longest_edge)

    model = load_base_model()
    if args.debug:
        debug_checks(model, train_ds, train_collator, processor)

    results = {"method": args.method, "seed": args.seed, "n_train": len(train_ds),
               "n_val": len(val_ds), "precision": precision}
    start = time.time()

    if args.method != "zeroshot":
        total_steps = compute_total_steps(len(train_ds), args.batch_size, args.grad_accum, args.epochs)
        if args.method == "lora":
            model = build_lora_model(model, r=args.lora_r)
        else:
            model = build_adalora_model(model, total_step=total_steps, init_r=args.init_r,
                                        target_r=args.target_r, deltaT=args.deltaT)
        trainable, total = count_parameters(model)
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        results.update(trainable_params=trainable, total_params=total, total_steps=total_steps)

        cfg = TrainConfig(output_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size,
                          grad_accum=args.grad_accum, lr=args.lr, precision=args.precision,
                          num_workers=args.num_workers, seed=args.seed,
                          alloc_log_every=args.alloc_log_every,
                          log_every=1 if args.debug else 10)
        model, allocation = train(model, train_ds, train_collator, cfg)
        results["allocation_summary"] = allocation_summary(allocation)
        results["train_time_s"] = round(time.time() - start, 1)
    else:
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.config.use_cache = True
    mean_anls, predictions = evaluate_anls(
        model, val_ds, eval_collator, processor, batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens, amp_dtype=amp_dtype, num_workers=args.num_workers)
    results["anls"] = mean_anls
    results["total_time_s"] = round(time.time() - start, 1)

    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.method}] ANLS = {mean_anls:.4f}  -> {out_dir}")

    if args.debug and args.method == "adalora":
        b = results["allocation_summary"]["budget"]
        print("[debug] AdaLoRA mean rank per tower:",
              {t: round(v["mean_rank"], 2) for t, v in b.items()},
              f"(should be below init_r={args.init_r} if pruning ran)")


if __name__ == "__main__":
    args = parse_args()
    if args.compare:
        compare(args.output_root, args.seed, args.debug)
    else:
        run_method(args)
