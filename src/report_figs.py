"""Figures for the final report, from results/access_paths.json and crossover.json.

Usage: python -m src.report_figs   (writes report/figs/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCAN, AGENT, MUTED, INK = "#1F6FC4", "#D9730D", "#8A94A3", "#16202E"
SIZES = (1000, 5000, 10000)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9CED6")
    ax.grid(True, color="#ECEEF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4A5564", labelsize=9, length=0)


def frontier(paths: dict, cross: dict, out: Path) -> None:
    """Recall against documents classified, with the agent at its equal-time depth."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, n in zip(axes, SIZES):
        entry = paths[f"20newsgroups_{n}"]["thresholds"]["0.5"]
        fixed = entry["fixed"]
        ax.plot([p["cost"] for p in entry["oracle"]], [p["recall"] for p in entry["oracle"]],
                color=MUTED, linewidth=1.5, linestyle="--", label="Oracle k per query")
        ax.plot([p["k"] for p in fixed], [p["recall"] for p in fixed],
                color=SCAN, linewidth=2, label="Fixed k")
        ax.scatter([r["cost"] for r in entry["plans"]["yield"]],
                   [r["recall"] for r in entry["plans"]["yield"]], s=26, marker="o",
                   facecolor="white", edgecolor=INK, linewidth=1.2, zorder=3, label="Yield rule")
        a = next(r for r in cross["equal_time"] if r["n"] == n)
        ax.scatter([a["k"]], [a["agent_recall"]], s=80, marker="D", color=AGENT,
                   edgecolor="white", linewidth=1.5, zorder=4, label="Agent, same time")
        ax.set_xscale("log")
        ax.set_xlim(10, n)
        ax.set_title(f"N = {n:,}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Documents classified per query", fontsize=9, color="#4A5564")
        style(ax)
    axes[0].set_ylabel("Recall", fontsize=9, color="#4A5564")
    axes[0].set_ylim(0, 0.34)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "frontier.png", dpi=200)
    plt.close(fig)


def exchange(cross: dict, out: Path) -> None:
    """Decisions one agent call costs, as queries in flight grow, against break-even."""
    rows = sorted((r for r in cross["exchange"] if r["n"] == 10000), key=lambda r: r["in_flight"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = [r["in_flight"] for r in rows]
    y = [r["exchange"] for r in rows]
    ax.plot(x, y, color=AGENT, linewidth=2, marker="o", markersize=6, label="Measured, N = 10,000")
    even = next(r for r in cross["equal_time"] if r["n"] == 10000)
    for key, label, style_ in (("half_breakeven", "Break-even, t = 0.5", "-"),
                               ("cal_breakeven", "Break-even, calibrated t", "--")):
        ax.axhline(even[key], color=SCAN, linewidth=1.5, linestyle=style_)
        ax.text(x[0], even[key] * 0.86, f"{label}: {even[key]:.1f}", ha="left", fontsize=8.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [v for v in (1, 2, 4, 8, 16, 32, 64, 128) if v <= max(x) * 1.2]
    ax.set_xticks(ticks, labels=[str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Mean queries in flight", fontsize=9, color="#4A5564")
    ax.set_ylabel("Scan decisions per agent call", fontsize=9, color="#4A5564")
    ax.set_yticks([3, 5, 10, 20, 50, 100], labels=["3", "5", "10", "20", "50", "100"])
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "exchange.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", default="results/access_paths.json")
    p.add_argument("--crossover", default="results/crossover.json")
    p.add_argument("--outdir", default="report/figs")
    args = p.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = json.loads(Path(args.paths).read_text())
    cross = json.loads(Path(args.crossover).read_text())
    frontier(paths, cross, out)
    exchange(cross, out)
    print(f"[report_figs] wrote {out}")


if __name__ == "__main__":
    main()
