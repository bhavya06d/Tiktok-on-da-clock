"""The autonomous research loop.

Auto-discovers every experiment in experiments/*.py (see experiments/README.md
for the contract each one must follow), runs them in priority order, tracks
the best validation `primary` score seen so far (the "champion"), and keeps
or discards each new result using the convergence rule from README.md:
eps=0.002, N=3 consecutive non-improving attempts. Once that first triggers,
the officially *scored* checkpoint is locked in (see `converged_at` in the
summary) - but the loop keeps running any remaining experiment files rather
than stopping dead, so a later session's new ideas still get tried and
logged instead of being silently skipped by file-ordering alone.

Usage:
    python3 agent.py [--data_dir ./KuaiRand-Pure/data] [--eps 0.002] [--patience 3]

Every attempt is appended to agent_log.jsonl (one JSON object per line) and
the final result is written to agent_summary.json, for Person 5's tracking
dashboard to consume.
"""
import argparse
import importlib
import json
import os
import time

from data import load

EXPERIMENTS_DIR = 'experiments'

# Experiments committed before the AUTHOR convention existed. We do not edit
# teammates' files to retrofit it (merge hygiene) - authorship for those is
# recorded here instead. New experiments should just set AUTHOR themselves.
LEGACY_AUTHOR = {
    'bpr_loss': 'human',  # Person 1, hand-written and hand-tuned before this convention existed
}


def _jsonable(obj):
    """Recursively convert numpy scalars (float32 etc., which json can't
    serialize) to native Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


def _read_source(name):
    """Full source of experiments/<name>.py, captured as the 'code diff'
    for the run log (each experiment is a new file, so its whole content
    *is* the diff against the baseline)."""
    try:
        with open(os.path.join(EXPERIMENTS_DIR, f'{name}.py')) as fh:
            return fh.read()
    except OSError:
        return None


def discover_experiments():
    """Import every experiments/*.py that isn't private (_-prefixed) and
    exposes a run() function. Returns a list of (priority, name, module)
    sorted by priority then name."""
    found = []
    for fname in sorted(os.listdir(EXPERIMENTS_DIR)):
        if not fname.endswith('.py') or fname.startswith('_'):
            continue
        name = fname[:-3]
        mod = importlib.import_module(f'{EXPERIMENTS_DIR}.{name}')
        if not hasattr(mod, 'run'):
            continue
        author = getattr(mod, 'AUTHOR', None) or LEGACY_AUTHOR.get(name, 'human')
        found.append((getattr(mod, 'PRIORITY', 100), name, mod, author))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--eps', type=float, default=0.002,
                     help='min valid-primary improvement over champion to count as progress')
    ap.add_argument('--patience', type=int, default=3,
                     help='consecutive non-improving attempts before declaring convergence')
    ap.add_argument('--log', default='agent_log.jsonl')
    ap.add_argument('--summary', default='agent_summary.json')
    ap.add_argument('--reveal-test-live', action='store_true',
                     help="print test-split scores during iteration, not just at the end. "
                          "Off by default: the challenge rules say development should use "
                          "train+validation only - decisions here are already valid-only, "
                          "this flag controls what a human sees on the console while iterating.")
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k: len(v) for k, v in splits.items()})

    experiments = discover_experiments()
    print(f"\ndiscovered {len(experiments)} experiment(s): {[n for _, n, _, _ in experiments]}\n")

    open(a.log, 'w').close()  # fresh log each run

    # Two champions tracked in parallel: `champion` is the best result seen
    # regardless of who wrote the code (what score is actually achievable);
    # `auto_champion` only considers AUTHOR == 'agent' entries - this is the
    # one that answers the hackathon's actual question (what did the agent,
    # autonomously, manage to find on its own). Convergence (the eps/N stop
    # rule) is tracked against the overall stream.
    champion_name, champion_valid_primary, champion_test = None, None, None
    auto_champion_name, auto_champion_valid_primary, auto_champion_test = None, None, None
    author_counts = {}
    no_improve = 0
    history = []
    converged = False
    # Index (1-based) and champion identity at the moment convergence first
    # triggers. Kept separate from `converged`/`champion_name` because the
    # loop no longer stops there (see below) - later experiment files may
    # exist (e.g. added in a follow-up session) and still deserve to run and
    # be logged, but the officially *scored* checkpoint per the eps/N rule
    # is whichever champion stood at this point, not whatever comes after.
    converged_at = None
    champion_at_convergence, champion_at_convergence_test = None, None

    for priority, name, mod, author in experiments:
        author_counts[author] = author_counts.get(author, 0) + 1
        desc = getattr(mod, 'DESCRIPTION', '')
        print(f"--- [{priority}] {name} (author={author})" + (f" — {desc}" if desc else ''))
        t0 = time.time()
        try:
            res = mod.run(splits)
            status = 'ok'
            error = None
        except Exception as e:
            status = 'error'
            error = repr(e)
            res = None
        dt = time.time() - t0

        entry = {
            'timestamp': time.time(), 'experiment': name, 'priority': priority,
            'author': author, 'description': desc, 'seconds': round(dt, 2),
            'status': status, 'error': error, 'code': _read_source(name),
        }

        if status == 'error':
            print(f"    FAILED: {error}")
            entry['decision'] = 'ERROR'
            history.append(entry)
            with open(a.log, 'a') as fh:
                fh.write(json.dumps(_jsonable(entry)) + '\n')
            continue

        vp = res['valid']['primary']
        first = champion_valid_primary is None
        improvement = float('inf') if first else vp - champion_valid_primary
        if first or improvement > a.eps:
            decision = 'KEPT (new champion)'
            champion_name, champion_valid_primary, champion_test = name, vp, res['test']
            no_improve = 0
        else:
            decision = 'DISCARDED'
            no_improve += 1

        # Separate autonomous-only champion, evaluated the same way but only
        # among AUTHOR == 'agent' entries - does not affect the convergence
        # counter above, which tracks the overall stream.
        auto_decision = None
        if author == 'agent':
            auto_first = auto_champion_valid_primary is None
            auto_improvement = float('inf') if auto_first else vp - auto_champion_valid_primary
            if auto_first or auto_improvement > a.eps:
                auto_decision = 'KEPT (new autonomous champion)'
                auto_champion_name, auto_champion_valid_primary, auto_champion_test = name, vp, res['test']
            else:
                auto_decision = 'DISCARDED (autonomous track)'

        imp_str = 'n/a (first)' if first else f"{improvement:+.4f}"
        test_str = f"  test primary={res['test']['primary']:.4f}" if a.reveal_test_live else ""
        print(f"    valid primary={vp:.4f}{test_str}  "
              f"improvement={imp_str}  -> {decision}  ({dt:.1f}s)")

        entry.update({
            'valid': res['valid'], 'test': res['test'],
            'improvement_over_champion': None if first else round(improvement, 4),
            'decision': decision, 'champion_after': champion_name,
            'auto_decision': auto_decision, 'auto_champion_after': auto_champion_name,
        })
        history.append(entry)
        with open(a.log, 'a') as fh:
            fh.write(json.dumps(_jsonable(entry)) + '\n')

        if no_improve >= a.patience and not converged:
            print(f"\nCONVERGED at attempt {len(history)}: {a.patience} consecutive attempts "
                  f"with improvement <= {a.eps}. Scored checkpoint is locked in as of here; "
                  f"still running any remaining experiment files so nothing already-discovered "
                  f"gets silently dropped if more ideas were added after this point.")
            converged = True
            converged_at = len(history)
            champion_at_convergence, champion_at_convergence_test = champion_name, champion_test

    print(f"\n=== OVERALL BEST: {champion_name} | valid primary={champion_valid_primary:.4f} "
          f"(any author - best score actually achievable) ===")
    if champion_test:
        print(f"    test  GAUC {champion_test['GAUC']:.4f} | nDCG@5 {champion_test['nDCG@5']:.4f} "
              f"| primary {champion_test['primary']:.4f}")
    if converged_at is not None and champion_name != champion_at_convergence:
        print(f"    (officially converged at attempt {converged_at} on '{champion_at_convergence}' - "
              f"'{champion_name}' is a later, better result from experiments added after that point)")

    print(f"\n=== AUTONOMOUS CHAMPION: {auto_champion_name} "
          f"(what the agent found on its own, no human-authored ideas) ===")
    if auto_champion_test:
        print(f"    valid primary={auto_champion_valid_primary:.4f}")
        print(f"    test  GAUC {auto_champion_test['GAUC']:.4f} | nDCG@5 {auto_champion_test['nDCG@5']:.4f} "
              f"| primary {auto_champion_test['primary']:.4f}")
    else:
        print("    (no agent-authored experiment has run yet)")

    manual_interventions = author_counts.get('human', 0)
    print(f"\nauthor breakdown: {author_counts}  "
          f"({manual_interventions} manual intervention(s) out of {len(history)} attempts)")

    total_wall_clock_seconds = round(sum(e['seconds'] for e in history), 2)
    print(f"total wall-clock (sum of experiment run times): {total_wall_clock_seconds:.1f}s "
          f"of the 6h (21600s) cap; {len(history)} of the 50-iteration cap")

    summary = {
        'champion': champion_name,
        'champion_valid_primary': champion_valid_primary,
        'champion_test': champion_test,
        'autonomous_champion': auto_champion_name,
        'autonomous_champion_valid_primary': auto_champion_valid_primary,
        'autonomous_champion_test': auto_champion_test,
        'author_counts': author_counts,
        'manual_interventions': manual_interventions,
        'converged': converged,
        # Attempt index (1-based) and champion at the moment convergence first
        # triggered - the officially *scored* checkpoint per the eps/N rule.
        # `champion`/`champion_test` above may differ if experiments after
        # this point (a later session's new ideas) went on to do better -
        # those are real, honest results too, just not what the convergence
        # rule itself locks in as the scored submission.
        'converged_at': converged_at,
        'champion_at_convergence': champion_at_convergence,
        'champion_at_convergence_test': champion_at_convergence_test,
        'attempts': len(history),
        'eps': a.eps, 'patience': a.patience,
        # Resource usage required for the Feasibility & Practicality submission section:
        'total_wall_clock_seconds': total_wall_clock_seconds,
        'wall_clock_cap_seconds': 21600,       # 6h
        'iteration_cap': 50,
        'gpu_hours': 0,                        # CPU-only (numpy), no GPU used
        'total_llm_tokens': None,              # NOT auto-tracked - this script has no
        # visibility into its own LLM usage. Whoever compiles the final report needs to
        # pull input+output token counts for the session(s) that proposed/wrote the
        # AUTHOR == 'agent' experiments (Claude Code usage panel / API dashboard) and
        # fill this in by hand before submission.
        'history': history,
    }
    with open(a.summary, 'w') as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(f"\nlog: {a.log}\nsummary: {a.summary}")


if __name__ == '__main__':
    main()
