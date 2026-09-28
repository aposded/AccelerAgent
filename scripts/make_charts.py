#!/usr/bin/env python3
"""Draw the README charts from results/*.csv.

    python3 scripts/make_charts.py
"""
import csv, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
LABEL = {"heuristic": "no-LLM run", "gemini": "Gemini run"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = RES / f"{name}.csv"
    return {r["benchmark"]: r for r in csv.DictReader(p.open())} if p.exists() else None


def short(b):
    return b.replace("mlsys-2026-", "#")


def speedup_chart(runs, base):
    mode = "gemini" if "gemini" in runs else "heuristic"
    rows = [(b, float(base[b]["baseline_latency"]) / float(r["latency"]))
            for b, r in runs[mode].items() if r["accepted"] == "True"]
    rows.sort(key=lambda x: x[1])
    geo = math.exp(sum(math.log(s) for _, s in rows) / len(rows))
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.barh([short(b) for b, _ in rows], [s for _, s in rows], color=BLUE, height=0.7)
    ax.set_xscale("log")  # one benchmark reaches 17x; a linear axis hides the rest
    ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20]); ax.set_xticklabels(["1x", "1.5x", "2x", "3x", "5x", "10x", "20x"])
    ax.set_xlim(1, 25)
    ax.axvline(geo, color=INK, lw=1, ls="--")
    ax.text(geo, len(rows) - 0.4, f" geometric mean {geo:.2f}x", color=INK, va="bottom")
    for i, (_, s) in enumerate(rows):
        ax.text(s, i, f" {s:.2f}x", va="center", color=MUTED, fontsize=8)
    ax.set_xlabel("Latency reduction vs one-op-per-step baseline (higher is better)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"Speedup over a one-op-per-step schedule ({LABEL[mode]})", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(RES / "speedup_vs_baseline.png", dpi=160); plt.close(fig)
    return geo


def walltime_chart(runs):
    mode = "gemini" if "gemini" in runs else "heuristic"
    items = sorted(runs[mode].items(), key=lambda kv: int(kv[0].split("-")[-1]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([short(b) for b, _ in items], [float(r["wall_s"]) for _, r in items], width=0.7,
           color=[BLUE if r["accepted"] == "True" else MUTED for _, r in items])
    bad = [short(b) for b, r in items if r["accepted"] != "True"]
    if bad:
        ax.text(0, 560, f"grey: invalid schedule ({', '.join(bad)})", color=MUTED, fontsize=8)
    ax.axhline(600, color=ORANGE, lw=1.2)
    ax.text(len(items) - 0.5, 600, "10-minute limit", color=ORANGE, ha="right", va="bottom", fontsize=9)
    ax.set_ylabel("Wall time (s)"); ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"Wall time per benchmark ({LABEL[mode]})", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(RES / "walltime.png", dpi=160); plt.close(fig)


def llm_gain_chart(runs):
    """Only when both runs exist: what Gemini adds over the deterministic fallback."""
    h, g = runs["heuristic"], runs["gemini"]
    rows = [(b, 100 * (1 - float(g[b]["latency"]) / float(h[b]["latency"])))
            for b in h if h[b]["accepted"] == "True" and g.get(b, {}).get("accepted") == "True"]
    rows.sort(key=lambda kv: int(kv[0].split("-")[-1]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([short(b) for b, _ in rows], [v for _, v in rows], width=0.7,
           color=[BLUE if v >= 0 else ORANGE for _, v in rows])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Latency cut vs no-LLM (%)"); ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("What Gemini adds over the deterministic fallback", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(RES / "llm_gain.png", dpi=160); plt.close(fig)


def main():
    base = load("baseline")
    runs = {m: r for m in ("heuristic", "gemini") if (r := load(m))}
    geo = speedup_chart(runs, base)
    walltime_chart(runs)
    if len(runs) == 2:
        llm_gain_chart(runs)
    print(f"charts written to results/ (geomean speedup {geo:.2f}x)")


if __name__ == "__main__":
    main()
