import os
import random
import shutil
from collections import defaultdict

import pyarrow.parquet as pq
import torch
from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk
from huggingface_hub import HfFileSystem
from tqdm.auto import tqdm
from transformers import AutoProcessor

KEEP_COLUMNS = ["questionId", "question", "answers", "image"]
SUBSET_FEATURES = Features({
    "questionId": Value("int32"),
    "question": Value("string"),
    "answers": Sequence(Value("string")),
    "image": Image(),
})


def _plan_subset(dataset_id, split, n, seed, rows_per_group):
    """
    Decide which rows to take, from parquet metadata only (no images downloaded).

    The shards are stored in row groups of ~100 rows. Every footer (a few KB) is
    read to list all row groups; a seeded random set of groups is chosen and
    `rows_per_group` random rows are picked from each, until `n` rows are planned.
    Returns [(file, row_group, [row indices]), ...] grouped by file.
    """
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{dataset_id}/data/{split}-*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found for split {split!r}")

    groups = []  # (file, row_group_index, num_rows)
    for path in tqdm(files, desc=f"{split}: reading footers"):
        # Small block size: only the footer is read here
        with fs.open(path, "rb", block_size=1 << 16) as fh:
            metadata = pq.ParquetFile(fh).metadata
            groups += [(path, g, metadata.row_group(g).num_rows) for g in range(metadata.num_row_groups)]

    rng = random.Random(seed)
    rng.shuffle(groups)
    plan, remaining = [], n
    for path, g, num_rows in groups:
        if remaining == 0:
            break
        k = min(rows_per_group, num_rows, remaining)
        plan.append((path, g, sorted(rng.sample(range(num_rows), k))))
        remaining -= k
    if remaining:
        raise RuntimeError(f"Split {split!r} has fewer than {n} rows")
    return sorted(plan)


def _download_rows(plan, split):
    """Yield the planned rows, downloading only the chosen row groups."""
    fs = HfFileSystem()
    by_file = defaultdict(list)
    for path, g, indices in plan:
        by_file[path].append((g, indices))

    with tqdm(total=len(plan), desc=f"{split}: downloading row groups", unit="group") as progress:
        for path, file_groups in by_file.items():
            with fs.open(path, "rb") as fh:
                parquet = pq.ParquetFile(fh)
                for g, indices in file_groups:
                    yield from parquet.read_row_group(g, columns=KEEP_COLUMNS).take(indices).to_pylist()
                    progress.update(1)


def _build_subset(dataset_id, split, n, seed, rows_per_group, out_dir):
    """
    Stream the planned rows straight to disk.    """
    plan = _plan_subset(dataset_id, split, n, seed, rows_per_group)
    tmp_cache, partial = out_dir + ".cache", out_dir + ".partial"
    for leftover in (tmp_cache, partial):
        shutil.rmtree(leftover, ignore_errors=True)

    dataset = Dataset.from_generator(_download_rows, features=SUBSET_FEATURES,
                                     gen_kwargs={"plan": plan, "split": split},
                                     cache_dir=tmp_cache)
    dataset.save_to_disk(partial)
    print(f"{split}: saved {len(dataset)} rows from {len(plan)} row groups "
          f"({dataset.data.nbytes / 1e9:.2f} GB) -> {out_dir}")
    del dataset
    os.rename(partial, out_dir)
    shutil.rmtree(tmp_cache, ignore_errors=True)


def build_docvqa_subset(dataset_id, n_train, n_val, seed, rows_per_group, subset_dir):
    """Create (once) and load the train/val subsets from `subset_dir`."""
    train_dir = os.path.join(subset_dir, f"train_{n_train}_seed{seed}")
    val_dir = os.path.join(subset_dir, f"validation_{n_val}_seed{seed}")

    if not os.path.isdir(train_dir):
        _build_subset(dataset_id, "train", n_train, seed, rows_per_group, train_dir)
    if not os.path.isdir(val_dir):
        _build_subset(dataset_id, "validation", n_val, seed, rows_per_group, val_dir)

    return load_from_disk(train_dir), load_from_disk(val_dir)


class DocVQACollator:
    """
    Turns raw subset rows into model inputs at batch time.

    train=True : prompt + answer, right-padded, `labels` cover only the answer
                 tokens (+ <end_of_utterance>); everything else is -100.
    train=False: prompt only (generation prompt), left-padded for `generate`,
                 plus the list of all reference answers for ANLS.
    """

    def __init__(self, processor, answer_instruction, train=True, image_longest_edge=None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # Instruction appended to every question; identical in training and evaluation.
        self.answer_instruction = answer_instruction
        self.train = train
        self.image_kwargs = {}
        if image_longest_edge is not None:
            self.image_kwargs["size"] = {"longest_edge": image_longest_edge}

    def _prompt(self, question):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": f"{question}\n{self.answer_instruction}"}],
        }]
        # Ends with "Assistant:"
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, examples):
        images = [[ex["image"].convert("RGB")] for ex in examples]
        prompts = [self._prompt(ex["question"]) for ex in examples]

        if not self.train:
            self.tokenizer.padding_side = "left"
            batch = self.processor(text=prompts, images=images, padding=True,
                                   return_tensors="pt", **self.image_kwargs)
            batch["answers"] = [ex["answers"] for ex in examples]
            batch["question_ids"] = [ex["questionId"] for ex in examples]
            return batch

        # Answer is appended after "Assistant:" exactly as the chat template would.
        answer_suffixes = [f" {ex['answers'][0]}<end_of_utterance>" for ex in examples]
        texts = [p + a for p, a in zip(prompts, answer_suffixes)]

        self.tokenizer.padding_side = "right"
        batch = self.processor(text=texts, images=images, padding=True,
                               return_tensors="pt", **self.image_kwargs)

        labels = torch.full_like(batch["input_ids"], -100)
        seq_lens = batch["attention_mask"].sum(dim=1)
        for i, suffix in enumerate(answer_suffixes):
            n_ans = len(self.tokenizer(suffix, add_special_tokens=False)["input_ids"])
            end = seq_lens[i].item()
            labels[i, end - n_ans:end] = batch["input_ids"][i, end - n_ans:end]
        batch["labels"] = labels
        return batch


def prepare_docvqa_dataset(model_id, dataset_id, n_train, n_val, seed, rows_per_group,
                           subset_dir, image_longest_edge, answer_instruction):
    """
    Returns (train_dataset, val_dataset, processor, train_collator, eval_collator).
    Datasets hold raw rows; pass the collators to DataLoader(collate_fn=...).
    """
    processor = AutoProcessor.from_pretrained(model_id)
    train_dataset, val_dataset = build_docvqa_subset(dataset_id, n_train, n_val, seed,
                                                     rows_per_group, subset_dir)

    train_collator = DocVQACollator(processor, answer_instruction, train=True,
                                    image_longest_edge=image_longest_edge)
    eval_collator = DocVQACollator(processor, answer_instruction, train=False,
                                   image_longest_edge=image_longest_edge)
    return train_dataset, val_dataset, processor, train_collator, eval_collator
