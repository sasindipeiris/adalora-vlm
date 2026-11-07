import json
import os

from evaluation.utils import (
    METHOD_LABEL,
    plot_anls,
    plot_budget_ratio,
    plot_depth_profile,
    plot_matrix_types,
    plot_rank_heatmap,
    plot_ratio_history,
    tower_share_history,
)


def run_dir(output_root, method, seed):
    return os.path.join(output_root, f"{method}_seed{seed}")


def _fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def _table(results):
    """Markdown table for the report."""
    lines = [
        "| Method | ANLS | Trainable params | Vision ratio | Connector ratio | Text ratio "
        "| Text / vision | ρ depth (vision) | ρ depth (text) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method, r in results.items():
        s = r.get("allocation_summary")
        b = s["budget"] if s else {}
        d = s["depth"] if s else {}
        params = f"{r['trainable_params']:,}" if "trainable_params" in r else "0"
        lines.append(
            " | ".join(
                [
                    f"| {METHOD_LABEL[method]}",
                    _fmt(r["anls"]),
                    params,
                    _fmt(b.get("vision", {}).get("rank_ratio"), 2),
                    _fmt(b.get("connector", {}).get("rank_ratio"), 2),
                    _fmt(b.get("text", {}).get("rank_ratio"), 2),
                    _fmt(s["text_over_vision"] if s else None, 2),
                    _fmt(d.get("vision", {}).get("rho"), 2),
                    _fmt(d.get("text", {}).get("rho"), 2) + " |",
                ]
            )
        )
    return "\n".join(lines)


def compare(output_root, seed, methods):
    """Build the comparison table and figures from whichever runs exist."""
    results = {}
    for method in methods:
        path = os.path.join(run_dir(output_root, method, seed), "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[method] = json.load(f)
    if not results:
        raise SystemExit(f"No results.json found under {output_root}")

    out_dir = os.path.join(output_root, f"compare_seed{seed}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_anls(
        {m: r["anls"] for m, r in results.items()}, os.path.join(fig_dir, "anls.png")
    )

    summaries = {
        m: r["allocation_summary"]
        for m, r in results.items()
        if "allocation_summary" in r
    }
    if summaries:
        plot_budget_ratio(summaries, os.path.join(fig_dir, "budget_ratio.png"))

    if "adalora" in results:
        ada_dir = run_dir(output_root, "adalora", seed)
        with open(os.path.join(ada_dir, "rank_allocation.json")) as f:
            rows = json.load(f)
        plot_rank_heatmap(
            rows,
            os.path.join(fig_dir, "adalora_rank_heatmap.png"),
            "AdaLoRA final rank allocation",
        )
        plot_depth_profile(
            rows,
            os.path.join(fig_dir, "adalora_depth_profile.png"),
            "AdaLoRA: kept rank by depth",
        )
        plot_matrix_types(
            summaries["adalora"],
            os.path.join(fig_dir, "adalora_matrix_types.png"),
            "AdaLoRA: kept rank by matrix type",
        )
        with open(os.path.join(ada_dir, "rank_modules.json")) as f:
            module_names = json.load(f)
        steps, ratios = tower_share_history(
            os.path.join(ada_dir, "rank_history.jsonl"), module_names
        )
        plot_ratio_history(
            steps,
            ratios,
            os.path.join(fig_dir, "adalora_budget_history.png"),
            "AdaLoRA: budget share over training",
        )

    table = _table(results)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(table)
    print(f"\nFigures and summary -> {out_dir}")
