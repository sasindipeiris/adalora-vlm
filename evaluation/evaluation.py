import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Accuracy: ANLS (official DocVQA metric)
# --------------------------------------------------------------------------- #

def _normalize(text):
    return " ".join(text.lower().strip().split())


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction, answers, threshold=0.5):
    """Best 1 - normalised edit distance over all references; 0 if above `threshold`."""
    pred = _normalize(prediction)
    best = 0.0
    for answer in answers:
        ref = _normalize(answer)
        if not pred and not ref:
            return 1.0
        nl = _levenshtein(pred, ref) / max(len(pred), len(ref))
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


@torch.no_grad()
def evaluate_anls(model, dataset, collator, processor, batch_size=8, max_new_tokens=32,
                  amp_dtype=None, num_workers=4):
    """
    Greedy-decode every validation example and score it with ANLS.
    Returns (mean_anls, predictions) where predictions keep question ids for failure analysis.
    """
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    eos_ids = [tokenizer.convert_tokens_to_ids("<end_of_utterance>"), tokenizer.eos_token_id]

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=num_workers)
    predictions = []
    for batch in tqdm(loader, desc="eval"):
        answers = batch.pop("answers")
        question_ids = batch.pop("question_ids")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                       use_cache=True, eos_token_id=eos_ids,
                                       pad_token_id=tokenizer.pad_token_id)
        # Left-padded prompts: everything after the prompt length is the answer.
        texts = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        for qid, text, refs in zip(question_ids, texts, answers):
            text = text.strip()
            predictions.append({"question_id": int(qid), "prediction": text,
                                "answers": list(refs), "anls": anls(text, refs)})

    mean = float(np.mean([p["anls"] for p in predictions])) if predictions else 0.0
    return mean, predictions
