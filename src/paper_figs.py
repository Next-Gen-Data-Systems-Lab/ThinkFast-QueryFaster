"""The short paper's main figure: recall against seconds per query.

Each scan model is a curve: it classifies documents in embedding order at its
measured documents per second, so after T seconds it has read k = T * rate
documents of each query's corpus (t = 0.5). Each agent variant is a point at its
mean seconds and recall. One panel per dataset; arXiv (100K by default, see
--arxiv-n) appears once its rows exist, and its curve ends where the job's
classified top 20K runs out. Everything is replayed from saved scores.

Usage: python -m src.paper_figs --outdir paper/figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import access_paths as AP
from . import agents_compare as AC
from . import synergy_compare as SC
from .crossover import RATE

INK, MUTED = "#16202E", "#5A6675"
COLOR = {"Qwen 7B": "#1F6FC4", "Kev-9B": "#1B8A5A", "JEV-9B": "#8E44AD", "agent": "#C2410C"}
GRID = np.logspace(-0.5, 2.6, 60)        # seconds per query
AGENT_STYLE = {   # marker, filled
    "7B, embeddings": ("o", True), "7B, embeddings + BM25": ("o", False),
    "32B, embeddings": ("D", True), "32B, embeddings + BM25": ("D", False),
    "7B, persistent": ("s", True),
}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9CED6")
    ax.grid(True, color="#ECEEF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4A5564", labelsize=8, length=0)


def curve(queries: list, rates: list) -> np.ndarray:
    """Mean recall after T seconds, each query at its own corpus's rate."""
    return np.array([np.mean([AP.quality(q, min(q.n, int(t * r)), 0.5)[0]
                              for q, r in zip(queries, rates)]) for t in GRID])


def agent_points(paths: dict, n: int | None = None, dataset: str | None = None) -> dict:
    out = {}
    for name, path in paths.items():
        if not Path(path).exists():
            continue
        with open(path) as fh:
            a = pd.DataFrame(json.loads(line) for line in fh if line.strip())
        a = a[a.arm == "agent"]
        if n is not None:
            a = a[a.corpus_size == n]
        if dataset is not None:
            a = a[a.dataset.str.startswith(dataset)]
        a = a.drop_duplicates(["dataset", "query_id"], keep="last")
        if len(a):
            out[name] = (float(a.wall_s.mean()), float(a.recall.mean()))
    return out


def panel_20ng() -> tuple[str, dict, dict]:
    n = 10000
    q = AP.load("20newsgroups", n, "results/20newsgroups_sweep_bs32.scores.jsonl")
    kev = AP.load("20newsgroups", n, "results/20newsgroups_kev9bfull_packed.scores.jsonl")
    jev = AP.load("20newsgroups", n, "results/20newsgroups_jev9bfull_separate.scores.jsonl")
    rates = {s: AC.rate(rows, n) for s, (_, rows) in AC.SCANS.items()}
    curves = {"Qwen 7B": curve(q, [RATE] * len(q)),
              "Kev-9B": curve(kev, [rates["kev"]] * len(kev)),
              "Kev-9B, packed": curve(kev, [rates["kev packed"]] * len(kev)),
              "JEV-9B": curve(jev, [rates["jev"]] * len(jev))}
    return "20 Newsgroups, 10K", curves, agent_points(AC.AGENTS, n=n)


def panel_synergy() -> tuple[str, dict, dict]:
    c = SC.collect()
    names = {"qwen": "Qwen 7B", "kev": "Kev-9B", "jev": "JEV-9B"}
    curves = {names[m]: curve(c["queries"][m], c["rate"][m]) for m in names}
    return "SYNERGY, 26 reviews", curves, agent_points(SC.AGENTS, dataset="synergy_")


def panel_arxiv(n: int = 1_000_000) -> tuple[str, dict, dict] | None:
    rows, scores = "results/arxiv_scale.jsonl", "results/arxiv_scale.scores.jsonl"
    if not (Path(rows).exists() and Path(scores).exists()):
        return None
    with open(rows) as fh:
        r = pd.DataFrame(json.loads(line) for line in fh if line.strip())
    h = r[(r.arm == "hybrid") & (r.corpus_size == n)]
    if h.empty:
        return None
    rate = float(h.n_docs_scored.sum() / h.t_classify.sum())
    q = AP.load("arxiv", n, scores, arm="hybrid")
    y = curve(q, [rate] * len(q))
    # Only each query's top k was classified, so the curve stops at that depth
    # rather than going flat as if the scan had nothing more to find.
    y[GRID * rate > int(h.k.iloc[0])] = np.nan
    curves = {"Qwen 7B": y}
    size = f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}K"
    return f"arXiv, {size}", curves, agent_points(
        {"7B, embeddings": rows, "7B, persistent": "results/arxiv_agentdry.jsonl"}, n=n)


def figure(out: Path, arxiv_n: int = 100_000, panel_width: float = 3.4,
           height: float = 3.0, dpi: int = 200, dashed_note: bool = True) -> None:
    panels = [p for p in (panel_20ng(), panel_synergy(), panel_arxiv(arxiv_n)) if p]
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_width * len(panels), height), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, curves, agents) in zip(axes, panels):
        for name, y in curves.items():
            base = name.replace(", packed", "")
            ax.plot(GRID, y, color=COLOR[base], linewidth=1.8,
                    linestyle="--" if "packed" in name else "-")
            if "packed" in name and dashed_note:
                # Ends where the separate curve ends, so label it in the empty
                # lower right, below the plateaus and right of the agents.
                ax.text(GRID[-1], 0.04, "dashed: Kev, one\npass for all queries",
                        ha="right", va="bottom", fontsize=7.5, color=INK)
            elif "packed" not in name:
                last = np.flatnonzero(np.isfinite(y))[-1]
                ax.annotate(name.replace("-9B", "").replace(" 7B", ""), (GRID[last], y[last]),
                            xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK)
        for name, (secs, rec) in agents.items():
            marker, filled = AGENT_STYLE[name]
            ax.scatter([secs], [rec], s=40, marker=marker, zorder=4, linewidth=1.4,
                       facecolor=COLOR["agent"] if filled else "white",
                       edgecolor=COLOR["agent"],
                       label="Agent: " + name.replace(", embeddings", "").replace(" + BM25", " + BM25"))
        ax.set_xscale("log")
        ax.set_xlim(GRID[0], GRID[-1] * 3)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Seconds per query", fontsize=8.5, color="#4A5564")
        style(ax)
    axes[0].set_ylabel("Recall", fontsize=8.5, color="#4A5564")
    axes[0].set_ylim(0, 0.85)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "recall_time.pdf")
    fig.savefig(out / "recall_time.png", dpi=dpi)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="paper/figs")
    p.add_argument("--arxiv-n", type=int, default=100_000,
                   help="arXiv corpus size for the third panel (10000, 100000 or 1000000)")
    p.add_argument("--slide", action="store_true",
                   help="narrower panels at higher resolution, so the text is larger on a slide")
    args = p.parse_args(argv)
    if args.slide:
        figure(Path(args.outdir), args.arxiv_n, panel_width=2.45, height=2.55, dpi=320,
               dashed_note=False)
    else:
        figure(Path(args.outdir), args.arxiv_n)
    print("wrote", Path(args.outdir) / "recall_time.pdf")


if __name__ == "__main__":
    main()
