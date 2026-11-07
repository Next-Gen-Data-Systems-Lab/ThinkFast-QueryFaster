"""Turn results JSONL into the figures and tables for the report.

    python -m src.analyze --results results/20newsgroups_sweep_bs32.jsonl --outdir figs

Produces a markdown summary (paste into the report) and four figures:

  throughput_vs_batch   docs/sec against effective batch size      -- H3
  latency_vs_corpus     wall seconds per query against corpus size -- H2
  quality_by_arm        recall / precision / F1 per arm            -- H1, H4
  hybrid_ceiling        hybrid recall against its retrieval cap    -- H4 caveat

Palette: slots 1-3 of the reference categorical theme, which validate on the
all-pairs list in both light and dark. Aqua sits under 3:1 on a light surface,
so every figure ships direct labels and the markdown table provides the
relief view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARM_COLOR = {"classify": "#2a78d6", "ann": "#eb6834", "hybrid": "#1baf7a",
             "agent": "#7a4fd0"}
ARM_LABEL = {"classify": "Classify (full scan)", "ann": "ANN only",
             "hybrid": "Hybrid (ANN + classify)", "agent": "Search agent"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def load(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return pd.DataFrame(rows)


def _style(ax, title, xlabel, ylabel, logx=False, logy=False):
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_axisbelow(True)


def fig_throughput(df: pd.DataFrame, out: Path):
    """H3: does effective batch size buy throughput, and where does it flatten?"""
    d = df[df.arm == "classify"]
    if d.empty or d.batch_size.nunique() < 2:
        return None
    g = (d.groupby(["corpus_size", "batch_size"])
           .apply(lambda x: x.n_docs_scored.sum() / x.wall_s.sum(),
                  include_groups=False)
           .rename("docs_per_sec").reset_index())

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sizes = sorted(g.corpus_size.unique())
    shades = ["#9cc4ef", "#5a9ce2", "#2a78d6", "#17508f"]
    for i, n in enumerate(sizes):
        s = g[g.corpus_size == n].sort_values("batch_size")
        c = shades[min(i, len(shades) - 1)]
        ax.plot(s.batch_size, s.docs_per_sec, marker="o", color=c, label=f"N={n:,}")
        last = s.iloc[-1]
        ax.annotate(f"N={n:,}", (last.batch_size, last.docs_per_sec),
                    textcoords="offset points", xytext=(6, 0),
                    color=c, fontsize=8, va="center")
    _style(ax, "Classification throughput scales with batch size",
           "vLLM max_num_seqs (effective batch)", "documents / second", logx=True)
    ax.legend(title="corpus size", loc="upper left")
    fig.savefig(out / "throughput_vs_batch.png")
    plt.close(fig)
    return "throughput_vs_batch.png"


def fig_latency(df: pd.DataFrame, out: Path):
    """H2: per-query wall time against corpus size, one line per arm."""
    if df.corpus_size.nunique() < 2:
        return None
    bs = df.batch_size.max()
    d = df[df.batch_size == bs]
    g = d.groupby(["arm", "corpus_size"]).wall_s.mean().rename("s").reset_index()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for arm in ["classify", "ann", "hybrid"]:
        s = g[g.arm == arm].sort_values("corpus_size")
        if s.empty:
            continue
        ax.plot(s.corpus_size, s.s, marker="o", color=ARM_COLOR[arm],
                label=ARM_LABEL[arm])
        last = s.iloc[-1]
        ax.annotate(ARM_LABEL[arm], (last.corpus_size, last.s),
                    textcoords="offset points", xytext=(6, 0),
                    color=ARM_COLOR[arm], fontsize=8, va="center")
    _style(ax, f"Query latency against corpus size (batch {bs})",
           "corpus size (documents)", "mean seconds / query",
           logx=True, logy=True)
    ax.legend(loc="upper left")
    fig.savefig(out / "latency_vs_corpus.png")
    plt.close(fig)
    return "latency_vs_corpus.png"


def fig_quality(df: pd.DataFrame, out: Path):
    """H1 and H4: exhaustiveness is the classifier's advantage."""
    n = df.corpus_size.max()
    bs = df.batch_size.max()
    d = df[(df.corpus_size == n) & (df.batch_size == bs)]
    arms = [a for a in ["classify", "ann", "hybrid", "agent"] if a in set(d.arm)]
    if not arms:
        return None
    metrics = ["recall", "precision", "f1"]
    g = d.groupby("arm")[metrics].mean()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(metrics))]
        vals = [g.loc[arm, m] for m in metrics]
        ax.bar(xs, vals, width * 0.88, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.08)
    _style(ax, f"Retrieval quality by arm (N={n:,})", "", "macro-average")
    ax.legend(loc="lower right", ncol=1)
    fig.savefig(out / "quality_by_arm.png")
    plt.close(fig)
    return "quality_by_arm.png"


def fig_ceiling(df: pd.DataFrame, out: Path):
    """The caveat on 'hybrid dominates': k caps what retrieval can reach."""
    d = df[(df.arm == "hybrid") & df.get("ann_recall_ceiling").notna()] \
        if "ann_recall_ceiling" in df.columns else pd.DataFrame()
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(d.ann_recall_ceiling, d.recall, s=22, alpha=0.75,
               color=ARM_COLOR["hybrid"], edgecolor="white", linewidth=0.5,
               label="one query")
    lim = max(0.05, float(max(d.ann_recall_ceiling.max(), d.recall.max())) * 1.05)
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--",
            label="ceiling (y = x)")
    _style(ax, "Hybrid recall is capped by what retrieval reaches",
           "fraction of gold inside top-k", "hybrid recall")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    fig.savefig(out / "hybrid_ceiling.png")
    plt.close(fig)
    return "hybrid_ceiling.png"


def markdown(df: pd.DataFrame) -> str:
    bs = df.batch_size.max()
    d = df[df.batch_size == bs]
    rows = (d.groupby(["arm", "corpus_size"])
              .agg(recall=("recall", "mean"), precision=("precision", "mean"),
                   f1=("f1", "mean"),
                   best_f1=("best_f1", "mean") if "best_f1" in d else ("f1", "mean"),
                   s_per_query=("wall_s", "mean"),
                   docs_scored=("n_docs_scored", "mean"))
              .reset_index().sort_values(["corpus_size", "arm"]))

    out = [f"### Results (batch size {bs})", "",
           "| Arm | N | Recall | Precision | F1 | Best F1 | s/query | docs scored |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in rows.iterrows():
        out.append(
            f"| {ARM_LABEL.get(r.arm, r.arm)} | {int(r.corpus_size):,} | "
            f"{r.recall:.3f} | {r.precision:.3f} | {r.f1:.3f} | "
            f"{r.best_f1:.3f} | {r.s_per_query:.2f} | {r.docs_scored:,.0f} |")

    cls = df[df.arm == "classify"]
    if not cls.empty and cls.batch_size.nunique() > 1:
        out += ["", "### Throughput against batch size (classify)", "",
                "| Batch | docs/sec | prefill tok/sec |", "|---:|---:|---:|"]
        for b, grp in cls.groupby("batch_size"):
            dps = grp.n_docs_scored.sum() / grp.wall_s.sum()
            tps = (grp.n_prompt_tokens.sum() / grp.wall_s.sum()
                   if "n_prompt_tokens" in grp else float("nan"))
            out.append(f"| {b} | {dps:,.1f} | {tps:,.0f} |")

    if "ann_recall_ceiling" in df.columns:
        h = df[(df.arm == "hybrid") & df.ann_recall_ceiling.notna()]
        if not h.empty:
            out += ["", "### Hybrid retrieval ceiling", "",
                    f"- mean fraction of gold inside top-k: "
                    f"**{h.ann_recall_ceiling.mean():.3f}**",
                    f"- mean hybrid recall: **{h.recall.mean():.3f}**",
                    f"- queries where gold exceeds k: "
                    f"**{(h.k_vs_gold_ratio < 1).mean():.0%}**"
                    if "k_vs_gold_ratio" in h else ""]
    return "\n".join(x for x in out if x is not None)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/20newsgroups_sweep_bs32.jsonl")
    p.add_argument("--outdir", default="figs")
    a = p.parse_args(argv)

    df = load(Path(a.results))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    made = [f for f in (fig_throughput(df, out), fig_latency(df, out),
                        fig_quality(df, out), fig_ceiling(df, out)) if f]
    md = markdown(df)
    (out / "summary.md").write_text(md + "\n")

    print(md)
    print(f"\n[analyze] {len(df)} rows -> {out}/")
    for f in made:
        print(f"  {f}")
    if not made:
        print("  (no figures: need >1 batch size or >1 corpus size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
