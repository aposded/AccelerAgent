#!/usr/bin/env python3
"""One-op-per-step baseline: every op runs alone, with its best tile size and traversal,
and nothing is kept in fast memory between steps. No fusion, no retention, no
recomputation. It isolates what the agent's scheduling decisions add.

    python3 scripts/baseline.py   # writes results/baseline.csv
"""
import csv, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import _best_traversal, _gen_grans, find_boundary_tensors, load_problem, topo_sort  # noqa: E402


def baseline_latency(path: Path) -> float:
    prob, ops = load_problem(str(path))
    total = 0.0
    for oid in topo_sort(ops):
        step = [ops[oid]]
        b_in, b_out = find_boundary_tensors(step, ops)
        best = math.inf
        for g, _ in _gen_grans(step, b_in, b_out, prob, top_k=20):
            best = min(best, _best_traversal(step, b_in, b_out, *g, prob)[1])
        total += best
    return total


def main() -> None:
    benches = sorted((ROOT / "benchmarks").glob("mlsys-2026-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    rows = [{"benchmark": b.stem, "baseline_latency": round(baseline_latency(b), 1)} for b in benches]
    assert all(math.isfinite(r["baseline_latency"]) for r in rows), rows  # every op fits alone
    with (ROOT / "results" / "baseline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"baseline: {len(rows)} benchmarks -> results/baseline.csv")


if __name__ == "__main__":
    main()
