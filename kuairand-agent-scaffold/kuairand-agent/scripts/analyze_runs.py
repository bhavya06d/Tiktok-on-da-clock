"""Person 5 — turn iterations.jsonl into the deliverables:
results table, iteration trajectory, intervention/token/wall-clock summary.

Usage: python scripts/analyze_runs.py runs/<run_name>/iterations.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE = {"gauc": 0.6674, "ndcg5": 0.5357, "primary": 0.6016}  # validation


def main(path: Path):
    iters, summary, events = [], None, []
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        if obj["type"] == "iteration":
            iters.append(obj)
        elif obj["type"] == "summary":
            summary = obj
        else:
            events.append(obj)

    print(f"{'iter':>4} {'primary':>8} {'gauc':>8} {'ndcg5':>8} "
          f"{'Δbaseline':>10}  hypothesis")
    for it in iters:
        m = it["metrics"]
        if m:
            print(f"{it['iter']:>4} {m['primary']:>8.4f} {m['gauc']:>8.4f} "
                  f"{m['ndcg5']:>8.4f} {m['primary']-BASELINE['primary']:>+10.4f}  "
                  f"{it['hypothesis'][:70]}")
        else:
            print(f"{it['iter']:>4} {'FAILED':>8} {'':>8} {'':>8} {'':>10}  "
                  f"{(it['error'] or '')[:70]}")

    print("\nRecovery events (evidence for Robustness scoring):")
    for e in events:
        print(f"  iter {e['iter']}: {e['event']} {e.get('detail','')}")

    if summary:
        print("\n=== FINAL SUMMARY (goes into the Devpost results table) ===")
        print(f"Best val primary : {summary['best_val_primary']:.4f} "
              f"(Δ {summary['best_val_primary']-BASELINE['primary']:+.4f} vs baseline)")
        print(f"Iterations used  : {summary['iterations_used']} / 50")
        print(f"Failures recovered: {summary['failed_iterations_recovered']}")
        print(f"Manual interventions: {summary['manual_interventions']}")
        print(f"Wall-clock       : {summary['wall_clock_seconds']/60:.1f} min")
        print(f"LLM tokens       : {summary['llm_tokens']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
