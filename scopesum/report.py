"""Tables and figures for the paper from ``03_eval`` outputs.

Input: ``<eval out_dir>/<variant>.json`` files (one per variant).
Output: pandas tables (main results, probes, memory), a LaTeX version of the
main table, a B-vs-C bar chart and a memory chart.
"""

from __future__ import annotations

import json
import os

VARIANT_NAMES = {
    "A": "A baseline",
    "B": "B fixed-chunk <sum>",
    "C": "C scope <sum>",
    "D": "D C + scope windows",
    "E": "E D + call edges",
    "F": "F C, parts <= W (KV-matched)",
    "G": "G fixed-chunk, K matched to F",
}


def _headline_probe(p: dict) -> bool:
    """The probe row shown in the main table: <sum> features, all functions."""
    return p.get("feature", "sum") == "sum" and p.get("subset", "all") == "all"


def collect_results(out_dir: str) -> dict:
    res = {}
    for v in "ABCDEFG":
        p = os.path.join(out_dir, f"{v}.json")
        if os.path.exists(p):
            with open(p) as f:
                res[v] = json.load(f)
    return res


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def results_tables(allres: dict) -> dict:
    import pandas as pd

    main, probes, memory = [], [], []
    for v, r in allres.items():
        row = {
            "variant": VARIANT_NAMES.get(v, v),
            "loss_all": _get(r, "longrange", "loss_all"),
            "loss_long_range": _get(r, "longrange", "loss_long_range"),
            "n_long_range": _get(r, "longrange", "n_tokens_long_range"),
            "loc_top1": _get(r, "localize", "top1"),
            "loc_top5": _get(r, "localize", "top5"),
            "loc_tasks": _get(r, "localize", "n_tasks"),
            "kv_live_mean": _get(r, "memory_analytic", "live_mean"),
            "kv_mb_mean": _get(r, "memory_analytic", "kv_mb_mean"),
            "peak_gb": _get(r, "memory_measured", "peak_gb"),
        }
        for p in r.get("probe", []) or []:
            probes.append(p)
            if p.get("layer") in (-1, "-1") and p.get("accuracy") is not None and _headline_probe(p):
                row[f"probe_{p['task']}"] = p["accuracy"]
        main.append(row)
        ma = r.get("memory_analytic") or {}
        mm = r.get("memory_measured") or {}
        if ma:
            memory.append({"variant": v, **{k: ma.get(k) for k in ("length_mean", "live_mean", "live_max", "visible_mean", "kv_mb_mean", "kv_mb_max")},
                           "peak_gb": mm.get("peak_gb"), "seconds": mm.get("seconds")})
    return {
        "main_results": pd.DataFrame(main),
        "probe_results": pd.DataFrame(probes),
        "memory_results": pd.DataFrame(memory),
    }


def to_latex(df, floatfmt: str = "{:.3f}") -> str:
    """Minimal LaTeX tabular (no jinja2 dependency)."""
    cols = list(df.columns)
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            x = row[c]
            if isinstance(x, float):
                cells.append("--" if x != x else floatfmt.format(x))
            elif x is None:
                cells.append("--")
            else:
                cells.append(str(x).replace("_", "\\_").replace("<", "$<$").replace(">", "$>$"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def plot_b_vs_c(allres: dict, path: str | None = None):
    """Bar chart: B (fixed chunks) vs C (scopes) on every headline metric."""
    import matplotlib.pyplot as plt

    metrics = [
        ("long-range loss ↓", ("longrange", "loss_long_range")),
        ("all-token loss ↓", ("longrange", "loss_all")),
        ("localization top-1 ↑", ("localize", "top1")),
        ("localization top-5 ↑", ("localize", "top5")),
    ]
    probe_tasks = sorted({p["task"] for v in ("B", "C") for p in (allres.get(v, {}).get("probe") or [])})
    fig, axes = plt.subplots(1, len(metrics) + len(probe_tasks), figsize=(3 * (len(metrics) + len(probe_tasks)), 3.2))
    axes = list(axes) if hasattr(axes, "__len__") else [axes]
    colors = {"B": "#8a8f98", "C": "#2f6fdf"}
    for ax, (title, keys) in zip(axes, metrics):
        vals = [_get(allres.get(v, {}), *keys) for v in ("B", "C")]
        ax.bar(["B", "C"], [x if x is not None else 0 for x in vals], color=[colors["B"], colors["C"]])
        for i, x in enumerate(vals):
            ax.text(i, (x or 0), "n/a" if x is None else f"{x:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9)
    for ax, task in zip(axes[len(metrics):], probe_tasks):
        vals = []
        for v in ("B", "C"):
            acc = [p["accuracy"] for p in (allres.get(v, {}).get("probe") or [])
                   if p["task"] == task and p.get("accuracy") is not None and _headline_probe(p)]
            vals.append(acc[-1] if acc else 0)
        ax.bar(["B", "C"], vals, color=[colors["B"], colors["C"]])
        for i, x in enumerate(vals):
            ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"probe: {task} ↑", fontsize=9)
    fig.suptitle("Fixed-chunk (B) vs scope-aligned (C) summaries — same number of <sum> tokens", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_memory(allres: dict, path: str | None = None):
    """Left: live KV entries per local layer over position. Right: mean KV MB."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    names, mbs = [], []
    for v, r in allres.items():
        ma = r.get("memory_analytic") or {}
        curve = ma.get("live_curve_mean")
        if curve:
            ax1.plot(range(len(curve)), curve, label=v)
        if ma.get("kv_mb_mean") is not None:
            names.append(v)
            mbs.append(ma["kv_mb_mean"])
    ax1.set_xlabel("position")
    ax1.set_ylabel("live KV entries (per local layer)")
    ax1.set_title("KV a streaming local layer must keep", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.bar(names, mbs, color="#2f6fdf")
    for i, x in enumerate(mbs):
        ax2.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_title("mean local-layer KV memory (MB, analytic)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
