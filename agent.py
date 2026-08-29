"""The autonomous research loop.

Auto-discovers every experiment in experiments/*.py (see experiments/README.md
for the contract each one must follow), runs them in priority order, tracks
the best validation `primary` score seen so far (the "champion"), and keeps
or discards each new result using the convergence rule from README.md:
eps=0.002, N=3 consecutive non-improving attempts stops the run.

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
        found.append((getattr(mod, 'PRIORITY', 100), name, mod))
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
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k: len(v) for k, v in splits.items()})

    experiments = discover_experiments()
    print(f"\ndiscovered {len(experiments)} experiment(s): {[n for _, n, _ in experiments]}\n")

    open(a.log, 'w').close()  # fresh log each run

    champion_name, champion_valid_primary, champion_test = None, None, None
    no_improve = 0
    history = []
    converged = False

    for priority, name, mod in experiments:
        desc = getattr(mod, 'DESCRIPTION', '')
        print(f"--- [{priority}] {name}" + (f" — {desc}" if desc else ''))
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
            'description': desc, 'seconds': round(dt, 2), 'status': status, 'error': error,
            'code': _read_source(name),
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

        imp_str = 'n/a (first)' if first else f"{improvement:+.4f}"
        print(f"    valid primary={vp:.4f}  test primary={res['test']['primary']:.4f}  "
              f"improvement={imp_str}  -> {decision}  ({dt:.1f}s)")

        entry.update({
            'valid': res['valid'], 'test': res['test'],
            'improvement_over_champion': None if first else round(improvement, 4),
            'decision': decision, 'champion_after': champion_name,
        })
        history.append(entry)
        with open(a.log, 'a') as fh:
            fh.write(json.dumps(_jsonable(entry)) + '\n')

        if no_improve >= a.patience:
            print(f"\nCONVERGED: {a.patience} consecutive attempts with improvement <= {a.eps}")
            converged = True
            break

    print(f"\n=== FINAL CHAMPION: {champion_name} | valid primary={champion_valid_primary:.4f} ===")
    if champion_test:
        print(f"    test  GAUC {champion_test['GAUC']:.4f} | nDCG@5 {champion_test['nDCG@5']:.4f} "
              f"| primary {champion_test['primary']:.4f}")

    summary = {
        'champion': champion_name,
        'champion_valid_primary': champion_valid_primary,
        'champion_test': champion_test,
        'converged': converged,
        'attempts': len(history),
        'eps': a.eps, 'patience': a.patience,
        'history': history,
    }
    with open(a.summary, 'w') as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(f"\nlog: {a.log}\nsummary: {a.summary}")


if __name__ == '__main__':
    main()
