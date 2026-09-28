#!/usr/bin/env python3
"""Run the agent on every benchmark and record what the README reports.

    python3 scripts/evaluate.py --mode heuristic          # no API calls
    python3 scripts/evaluate.py --mode gemini --attempts 3  # needs GOOGLE_API_KEY

Each finished attempt is recorded in results/outputs/<mode>/, so an interrupted run
resumes where it stopped. Delete that folder to start over.

Each run is a fresh `python3 agent.py in.json out.json` subprocess under the contest's
10-minute limit. The output is checked with the agent's validator and its latency is
the sum of `subgraph_latencies` (the roofline model in PROBLEM.md).
"""
import argparse, csv, json, os, re, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import load_problem, validate_solution  # noqa: E402

DONE = re.compile(r"\[Done\] (\d+) steps, lat=([\d.]+), wall=([\d.]+)s \(LLM=(\d+), AUTO=(\d+), LA=(\d+), strategy=(\w+)\)")


def run_once(bench: Path, mode: str, timeout: int, out_dir: Path, attempt: int) -> dict:
    record = out_dir / f"{bench.stem}.{attempt}.record.json"
    if record.exists():  # resume: this attempt already finished in an earlier run
        return json.loads(record.read_text())
    row = _run(bench, mode, timeout, out_dir, attempt)
    record.write_text(json.dumps(row))
    return row


def _run(bench: Path, mode: str, timeout: int, out_dir: Path, attempt: int) -> dict:
    env = dict(os.environ)
    if mode == "heuristic":
        env["GOOGLE_API_KEY"] = ""  # agent falls back to its deterministic heuristic
    out = out_dir / f"{bench.stem}.{attempt}.json"
    out.unlink(missing_ok=True)  # never score a stale file from an interrupted run
    t0 = time.time()
    row = {"benchmark": bench.stem, "attempt": attempt, "accepted": False, "error": ""}
    try:
        p = subprocess.run([sys.executable, str(ROOT / "agent.py"), str(bench), str(out)],
                           env=env, capture_output=True, text=True, timeout=timeout)
        log = p.stderr
    except subprocess.TimeoutExpired as e:
        log, row["error"] = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""), "timeout"
    row["wall_s"] = round(time.time() - t0, 1)
    (out_dir / f"{bench.stem}.{attempt}.log").write_text(log)
    m = DONE.search(log)
    if m:
        row.update(steps=int(m[1]), llm_calls=int(m[4]), auto_decisions=int(m[5]),
                   lookahead=int(m[6]), strategy=m[7])
    if out.exists():
        try:
            sol = json.loads(out.read_text())
            prob, ops = load_problem(str(bench))
            validate_solution(sol, prob, ops)
            row.update(accepted=True, ops=len(ops), latency=round(sum(sol["subgraph_latencies"]), 1))
        except Exception as e:  # invalid schedule = 0 points under contest rules
            row["error"] = row["error"] or f"invalid: {e}"[:120]
    elif not row["error"]:
        row["error"] = "no output"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heuristic", "gemini"], required=True)
    ap.add_argument("--attempts", type=int, default=1, help="contest scoring used the median of 3")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", nargs="*", help="benchmark names, e.g. mlsys-2026-5")
    a = ap.parse_args()
    if a.mode == "gemini" and not os.environ.get("GOOGLE_API_KEY") and not (ROOT / ".env").exists():
        sys.exit("gemini mode needs GOOGLE_API_KEY (env or .env)")
    benches = sorted((ROOT / "benchmarks").glob("mlsys-2026-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    if a.only:
        benches = [b for b in benches if b.stem in a.only]
    out_dir = ROOT / "results" / "outputs" / a.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(b, k) for b in benches for k in range(a.attempts)]
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda j: run_once(j[0], a.mode, a.timeout, out_dir, j[1]), jobs))

    # one row per benchmark: median latency over accepted attempts, as in the contest
    fields = ["benchmark", "ops", "accepted", "attempts_ok", "latency", "steps", "wall_s",
              "llm_calls", "auto_decisions", "lookahead", "strategy", "error"]
    summary = []
    for b in benches:
        rs = [r for r in rows if r["benchmark"] == b.stem]
        ok = [r for r in rs if r["accepted"]]
        best = min(ok, key=lambda r: r["latency"]) if ok else rs[0]
        s = {k: best.get(k, "") for k in fields}
        s.update(ops=len(load_problem(str(b))[1]), accepted=bool(ok), attempts_ok=f"{len(ok)}/{len(rs)}",
                 latency=statistics.median(r["latency"] for r in ok) if ok else "")
        summary.append(s)
    # a subset run must not overwrite the full results the README is built from
    dest = ROOT / "results" / (f"{a.mode}_subset.csv" if a.only else f"{a.mode}.csv")
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    n_ok = sum(s["accepted"] for s in summary)
    print(f"{a.mode}: {n_ok}/{len(summary)} benchmarks accepted -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
