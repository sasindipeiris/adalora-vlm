import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from models.utils import classify_module

TOWERS = ("vision", "connector", "text")


# --------------------------------------------------------------------------- #
# Claim-tied metrics: where did the rank budget go?
# --------------------------------------------------------------------------- #

def _depth_correlation(rows, tower):
    """Spearman rho between layer index and total rank of that layer (one point per layer)."""
    per_layer = defaultdict(int)
    for r in rows:
        if r["tower"] == tower:
            per_layer[r["layer"]] += r["rank"]
    layers = sorted(per_layer)
    ranks = [per_layer[l] for l in layers]
    if len(layers) < 3 or len(set(ranks)) == 1:  # uniform (e.g. LoRA): undefined
        return {"rho": None, "p": None}
    rho, p = spearmanr(layers, ranks)
    return {"rho": float(rho), "p": float(p)}


def _thirds(rows, tower):
    """Mean kept-rank fraction in the bottom / middle / top third of a tower's layers."""
    tower_rows = [r for r in rows if r["tower"] == tower]
    if not tower_rows:
        return None
    n_layers = max(r["layer"] for r in tower_rows) + 1
    buckets = {"bottom": [], "middle": [], "top": []}
    for r in tower_rows:
        third = min(3 * r["layer"] // n_layers, 2)
        buckets[("bottom", "middle", "top")[third]].append(r["rank"] / r["init_rank"])
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def allocation_summary(rows):
    """
    Statistics that test the AdaLoRA claim on the VLM.

    budget[tower]:
      rank_share         fraction of all final rank held by the tower
      init_rank_share    fraction it started with (its share of modules)
      rank_ratio         rank_share / init_rank_share   (>1: tower gained budget)
      param_share / init_param_share / param_ratio: same, in adapter parameters
    depth[tower]:        Spearman rho(layer index, layer rank) and bottom/middle/top thirds
    matrix_type[tower]:  kept-rank fraction per matrix type (attn_q ... ffn_out)
    text_over_vision:    text rank_ratio / vision rank_ratio  (H3: > 1)
    """
    total_rank = sum(r["rank"] for r in rows)
    total_init_rank = sum(r["init_rank"] for r in rows)
    total_params = sum(r["params"] for r in rows)
    total_init_params = sum(r["init_params"] for r in rows)

    budget, depth, matrix_type = {}, {}, {}
    for tower in TOWERS:
        tower_rows = [r for r in rows if r["tower"] == tower]
        if not tower_rows:
            continue
        rank = sum(r["rank"] for r in tower_rows)
        init_rank = sum(r["init_rank"] for r in tower_rows)
        params = sum(r["params"] for r in tower_rows)
        init_params = sum(r["init_params"] for r in tower_rows)

        rank_share = rank / total_rank if total_rank else 0.0
        init_rank_share = init_rank / total_init_rank
        param_share = params / total_params if total_params else 0.0
        init_param_share = init_params / total_init_params
        budget[tower] = {
            "n_modules": len(tower_rows),
            "rank": rank, "init_rank": init_rank,
            "mean_rank": rank / len(tower_rows),
            "rank_share": rank_share, "init_rank_share": init_rank_share,
            "rank_ratio": rank_share / init_rank_share,
            "param_share": param_share, "init_param_share": init_param_share,
            "param_ratio": param_share / init_param_share,
        }

        if tower != "connector":
            depth[tower] = {**_depth_correlation(rows, tower), "thirds": _thirds(rows, tower)}
            by_type = defaultdict(list)
            for r in tower_rows:
                by_type[r["matrix_type"]].append(r["rank"] / r["init_rank"])
            matrix_type[tower] = {k: float(np.mean(v)) for k, v in sorted(by_type.items())}

    text_over_vision = None
    if "text" in budget and "vision" in budget and budget["vision"]["rank_ratio"] > 0:
        text_over_vision = budget["text"]["rank_ratio"] / budget["vision"]["rank_ratio"]

    return {"total_rank": total_rank, "total_init_rank": total_init_rank,
            "budget": budget, "depth": depth, "matrix_type": matrix_type,
            "text_over_vision": text_over_vision}


def tower_share_history(history_path, module_names):
    """
    From rank_history.jsonl: per step, each tower's rank_ratio
    (share of current total rank / share of initial rank).
    """
    towers = [classify_module(n)["tower"] for n in module_names]
    init_counts = {t: towers.count(t) for t in set(towers)}
    n_modules = len(towers)

    records = {}
    with open(history_path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["step"]] = rec["ranks"]  # resumed runs may repeat a step; keep the last

    # Step 0 is all zeros (lora_E starts at 0 and the warm-up lr is 0), so skip empty snapshots.
    steps, ratios = sorted(s for s in records if sum(records[s]) > 0), defaultdict(list)
    for step in steps:
        ranks = records[step]
        total = sum(ranks)
        for tower, count in init_counts.items():
            share = sum(r for r, t in zip(ranks, towers) if t == tower) / total
            ratios[tower].append(share / (count / n_modules))
    return steps, dict(ratios)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

# Reference palette (light mode): categorical slots in fixed order, one blue ramp
# for magnitude, recessive ink/grid.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

TOWER_COLOR = {"vision": SERIES[0], "connector": SERIES[1], "text": SERIES[2]}
METHOD_COLOR = {"zeroshot": SERIES[0], "lora": SERIES[1], "adalora": SERIES[2]}
METHOD_LABEL = {"zeroshot": "Untuned", "lora": "LoRA", "adalora": "AdaLoRA"}

VISION_ROWS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.out_proj", "mlp.fc1", "mlp.fc2"]
TEXT_ROWS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _grid(rows, tower, row_names):
    n_layers = max(r["layer"] for r in rows if r["tower"] == tower) + 1
    grid = np.full((len(row_names), n_layers), np.nan)
    for r in rows:
        if r["tower"] == tower and r["matrix"] in row_names:
            grid[row_names.index(r["matrix"]), r["layer"]] = r["rank"]
    return grid


def plot_rank_heatmap(rows, path, title):
    """Final rank per (matrix, layer) for the vision encoder and text decoder."""
    init_r = max(r["init_rank"] for r in rows)
    connector = [r["rank"] for r in rows if r["tower"] == "connector"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2),
                             gridspec_kw={"height_ratios": [len(VISION_ROWS), len(TEXT_ROWS)]})
    for ax, tower, row_names in ((axes[0], "vision", VISION_ROWS), (axes[1], "text", TEXT_ROWS)):
        grid = _grid(rows, tower, row_names)
        im = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=init_r, aspect="auto")
        ax.grid(False)
        ax.set_yticks(range(len(row_names)), [n.split(".")[-1] for n in row_names])
        ax.set_xticks(range(grid.shape[1]))
        ax.set_ylabel(f"{tower} encoder" if tower == "vision" else "text decoder")
        for (i, j), value in np.ndenumerate(grid):
            if not np.isnan(value):
                ax.text(j, i, int(value), ha="center", va="center", fontsize=6,
                        color=SURFACE if value > 0.6 * init_r else INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[1].set_xlabel("layer (0 = closest to input)")
    suffix = f"   ·   connector rank: {connector[0]}/{init_r}" if connector else ""
    axes[0].set_title(f"{title}{suffix}", loc="left")
    fig.colorbar(im, ax=axes, shrink=0.8, label="final rank")
    _save(fig, path)


def plot_budget_ratio(summaries, path):
    """Rank share / initial share per tower, one bar per method. 1.0 = no change."""
    methods = list(summaries)
    towers = [t for t in ("vision", "connector", "text")
              if all(t in s["budget"] for s in summaries.values())]
    x = np.arange(len(towers))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, method in enumerate(methods):
        values = [summaries[method]["budget"][t]["rank_ratio"] for t in towers]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width * 0.92,
                      color=METHOD_COLOR[method], label=METHOD_LABEL[method],
                      edgecolor=SURFACE, linewidth=1)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_2, padding=2)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(x, towers)
    ax.set_ylabel("rank share ÷ initial share")
    ax.set_title("Budget re-allocation by tower", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_depth_profile(rows, path, title):
    """Kept-rank fraction per layer vs relative depth, vision and text on one axis."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for tower in ("vision", "text"):
        tower_rows = [r for r in rows if r["tower"] == tower]
        n_layers = max(r["layer"] for r in tower_rows) + 1
        kept = np.zeros(n_layers)
        init = np.zeros(n_layers)
        for r in tower_rows:
            kept[r["layer"]] += r["rank"]
            init[r["layer"]] += r["init_rank"]
        depth = np.arange(n_layers) / max(n_layers - 1, 1)
        ax.plot(depth, kept / init, color=TOWER_COLOR[tower], marker="o", markersize=3,
                label=f"{tower} ({n_layers} layers)")
        ax.annotate(tower, (depth[-1], (kept / init)[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_xlabel("relative depth (0 = first layer, 1 = last layer)")
    ax.set_ylabel("kept rank / initial rank")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_matrix_types(summary, path, title):
    """Kept-rank fraction per matrix type, vision vs text."""
    types = ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_in", "ffn_out"]
    x = np.arange(len(types))
    width = 0.4

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, tower in enumerate(("vision", "text")):
        values = [summary["matrix_type"][tower].get(t, np.nan) for t in types]
        ax.bar(x + (i - 0.5) * width, values, width * 0.92, color=TOWER_COLOR[tower],
               label=tower, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, types)
    ax.set_ylabel("kept rank / initial rank")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_ratio_history(steps, ratios, path, title):
    """Each tower's rank share / initial share over AdaLoRA training steps."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for tower in ("vision", "connector", "text"):
        if tower in ratios:
            ax.plot(steps, ratios[tower], color=TOWER_COLOR[tower], label=tower)
            ax.annotate(tower, (steps[-1], ratios[tower][-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("rank share ÷ initial share")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_anls(results, path):
    """Validation ANLS per method (single series, values labelled on the bars)."""
    methods = list(results)
    values = [results[m] for m in methods]

    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar([METHOD_LABEL[m] for m in methods], values, 0.55, color=SERIES[0],
                  edgecolor=SURFACE, linewidth=1)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, color=INK_2, padding=2)
    ax.set_ylabel("ANLS (validation)")
    ax.set_ylim(0, max(values + [0.1]) * 1.2)
    ax.set_title("DocVQA validation ANLS", loc="left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
