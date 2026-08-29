"""Turn runs/<name>/iterations.jsonl into the submission deliverables:
results table, delta-over-baseline, oracle-headroom, and the autonomy summary.

    python scripts/analyze_runs.py                 # latest run
    python scripts/analyze_runs.py runs/demo/iterations.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 0.6016
ORACLE = 0.8484


def latest() -> Path:
    runs = sorted((ROOT / "runs").glob("*/iterations.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit("no runs found — run: python run_agent.py")
    return runs[-1]


def main(path: Path) -> None:
    iters, events, summary, meta = [], [], None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o["type"] == "iteration":
            iters.append(o)
        elif o["type"] == "event":
            events.append(o)
        elif o["type"] == "summary":
            summary = o
        elif o["type"] == "meta":
            meta = o

    base = meta.get("baseline_primary", BASELINE)
    oracle = meta.get("oracle_primary", ORACLE)

    print(f"\nRun: {path.parent.name}    (baseline {base:.4f} · oracle {oracle:.4f})\n")
    print(f"{'it':>3}  {'method':<11} {'primary':>8} {'gauc':>8} {'ndcg@5':>8} "
          f"{'Δbase':>8} {'%headroom':>10}  decision")
    print("-" * 92)
    for it in iters:
        m = it.get("metrics")
        if m:
            hr = (m["primary"] - base) / (oracle - base) * 100
            print(f"{it['iter']:>3}  {it['method']:<11} {m['primary']:>8.4f} "
                  f"{m['gauc']:>8.4f} {m['ndcg5']:>8.4f} "
                  f"{m['primary'] - base:>+8.4f} {hr:>9.1f}%  {it.get('decision','')}")
        else:
            print(f"{it['iter']:>3}  {it['method']:<11} {'FAILED':>8} "
                  f"{'':>8} {'':>8} {'':>8} {'':>10}  "
                  f"{(it.get('error') or '').splitlines()[0][:40]}")

    if events:
        print("\nRecovery / lifecycle events (Robustness evidence):")
        for e in events:
            print(f"  iter {e['iter']:>2}  {e['event']:<26} {e.get('detail','')}")

    if summary:
        b = summary["best_val_primary"]
        print("\n" + "=" * 58)
        print("FINAL SUMMARY  (Devpost results table)")
        print("=" * 58)
        print(f"  best val primary      {b:.4f}")
        print(f"  Δ over FM baseline     {summary['delta_over_baseline']:+.4f}")
        print(f"  oracle headroom used   {summary['pct_of_oracle_headroom']*100:.1f}%"
              f"   (baseline→oracle span = {oracle-base:.4f})")
        print(f"  best iteration         #{summary['best_iter']} "
              f"({summary.get('best_method','?')})")
        print(f"  iterations used        {summary['iterations_used']} / "
              f"{meta.get('max_iters','?')}")
        print(f"  failures recovered     {summary['failed_iterations_recovered']}")
        print(f"  manual interventions   {summary['manual_interventions']}")
        print(f"  wall-clock             {summary['wall_clock_seconds']/60:.1f} min")
        t = summary["llm_tokens"]
        print(f"  LLM tokens             {t.get('total',0):,} "
              f"(in {t.get('input',0):,} / out {t.get('output',0):,})")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else latest()
    main(p)
