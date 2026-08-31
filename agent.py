"""The autonomous research loop.

Auto-discovers every experiment in experiments/*.py (see experiments/README.md
for the contract each one must follow) and, for each, runs it, keeps or discards
the result against the running champion, and logs the attempt. Convergence uses
the official rule from README.md: eps=0.002, N=3 consecutive non-improving
attempts. Once that first triggers, the officially *scored* checkpoint is locked
in (see `converged_at` in the summary) - but the loop keeps running any
remaining experiment files rather than stopping dead, so a later session's new
ideas still get tried and logged instead of being silently skipped by
file-ordering alone.

RELIABILITY: each experiment runs in its own subprocess (experiments/_runner.py)
with a hard timeout. That gives us (a) a kill switch for an infinite loop /
deadlock, (b) isolation - an OOM kill or segfault takes down only the worker,
not the orchestrator, and (c) a real stderr traceback to log. There is no
`solution_backup.py` to restore because nothing is mutated: every experiment is
its own immutable file, the append-only log is never rewritten, and the champion
state only ever advances on a successful KEEP - so any failure leaves the last
good champion exactly where it was, recoverable from agent_log.jsonl.

Usage:
    python agent.py [--data_dir ./KuaiRand-Pure/data] [--eps 0.002] [--patience 3]
                    [--step-timeout 1800] [--llm-tokens-in N --llm-tokens-out N]

Every attempt is appended to agent_log.jsonl (one JSON object per line) and the
final result is written to agent_summary.json, for Person 5's tracking dashboard.
"""
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time

EXPERIMENTS_DIR = 'experiments'
STEP_TIMEOUT_SECONDS = 30 * 60          # per-experiment hard ceiling
WALL_CLOCK_CAP_SECONDS = 6 * 3600       # official run cap
ITERATION_CAP = 50                     # official run cap

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


CHECKPOINT_PATH = os.path.join('results', 'champion_checkpoint.json')


def _write_checkpoint(name, valid_primary, test, attempt):
    """Persist the champion the moment it changes. This is the state-backup
    half of 'backup & rollback': if the run is killed / crashes / times out
    mid-way, the last good champion is still on disk here (not just half-written
    in the log). The rollback half is structural - a failed experiment never
    mutates anything and the champion only advances on a KEEP, so there is
    nothing to *restore*, only something to *recover*: this file + its winning
    experiment source in agent_log.jsonl fully reconstruct it."""
    try:
        os.makedirs('results', exist_ok=True)
        with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as fh:
            json.dump({'champion': name, 'valid_primary': valid_primary,
                       'test': test, 'kept_at_attempt': attempt,
                       'source_file': f'experiments/{name}.py',
                       'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}, fh, indent=2)
    except Exception:  # noqa: BLE001 - never let bookkeeping crash the loop
        pass


def _read_source(name):
    """Full source of experiments/<name>.py, captured as the 'code diff' for the
    run log (each experiment is a new file, so its whole content *is* the diff
    against the baseline)."""
    try:
        with open(os.path.join(EXPERIMENTS_DIR, f'{name}.py'),
                  encoding='utf-8', errors='replace') as fh:
            return fh.read()
    except OSError:
        return None


def discover_experiments():
    """Import every experiments/*.py that isn't private (_-prefixed) and exposes
    a run() function. Returns a list of (priority, name, module, author) sorted
    by priority then name."""
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


def run_experiment(name, data_dir, timeout):
    """Run one experiment in an isolated subprocess. Returns a dict:
        {status: 'ok'|'timeout'|'error', result: {...}|None,
         error: <str>|None, seconds: <float>}
    `error` on failure is the full stderr (python traceback)."""
    cmd = [sys.executable, '-m', 'experiments._runner', name,
           '--data_dir', data_dir]
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {'status': 'timeout', 'result': None, 'seconds': time.time() - t0,
                'error': f'killed after {timeout}s hard timeout\n'
                         f'{(e.stderr or "")[-2000:]}'}
    except Exception as e:  # noqa: BLE001 - the loop must survive anything
        return {'status': 'error', 'result': None, 'seconds': time.time() - t0,
                'error': f'orchestrator could not launch worker: {e!r}'}

    dt = time.time() - t0
    if proc.returncode != 0:
        return {'status': 'error', 'result': None, 'seconds': dt,
                'error': (proc.stderr or proc.stdout or 'non-zero exit, no output'
                          ).strip()[-4000:]}
    m = re.search(r'RESULT_JSON:\s*(\{.*\})', proc.stdout)
    if not m:
        return {'status': 'error', 'result': None, 'seconds': dt,
                'error': 'worker exited 0 but printed no RESULT_JSON line\n'
                         f'stdout tail:\n{proc.stdout[-1500:]}'}
    try:
        result = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return {'status': 'error', 'result': None, 'seconds': dt,
                'error': f'RESULT_JSON did not parse: {e}'}
    return {'status': 'ok', 'result': result, 'seconds': dt, 'error': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--eps', type=float, default=0.002,
                    help='min valid-primary improvement over champion to count as progress')
    ap.add_argument('--patience', type=int, default=3,
                    help='consecutive non-improving attempts before declaring convergence')
    ap.add_argument('--step-timeout', type=int, default=STEP_TIMEOUT_SECONDS,
                    help='per-experiment hard timeout in seconds')
    ap.add_argument('--log', default='agent_log.jsonl')
    ap.add_argument('--summary', default='agent_summary.json')
    ap.add_argument('--llm-tokens-in', type=int, default=None,
                    help='cumulative LLM input tokens for the sessions that wrote '
                         'the AUTHOR=agent experiments (Feasibility scoring); the '
                         'agent has no visibility into its own usage, so pass it here')
    ap.add_argument('--llm-tokens-out', type=int, default=None,
                    help='cumulative LLM output tokens (see --llm-tokens-in)')
    ap.add_argument('--reveal-test-live', action='store_true',
                    help="print test-split scores during iteration, not just at the "
                         "end. Off by default: decisions are valid-only either way; "
                         "this only controls what a human sees on the console.")
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(a.data_dir, 'log_standard_4_08_to_4_21_pure.csv')):
        sys.exit(f"dataset not found under {a.data_dir} - see README.md")

    experiments = discover_experiments()
    print(f"discovered {len(experiments)} experiment(s): "
          f"{[n for _, n, _, _ in experiments]}\n")

    open(a.log, 'w').close()  # fresh log each run

    # Two champions in parallel: `champion` is the best result from any author
    # (what score is actually achievable); `auto_champion` only considers
    # AUTHOR == 'agent' entries (what the agent found on its own). Convergence
    # tracks the overall stream.
    champion_name = champion_valid_primary = champion_test = None
    auto_champion_name = auto_champion_valid_primary = auto_champion_test = None
    author_counts = {}
    no_improve = 0
    history = []
    converged = False
    converged_at = None
    champion_at_convergence = champion_at_convergence_test = None

    # The rule's two mechanisms are separate and must be tracked separately:
    #   - `no_improve`/`champion_*` (eps-gated) decide WHEN to stop searching.
    #   - `running_best_*` (plain argmax, no eps) is "the validation-best
    #     checkpoint" the rule actually asks to submit - snapshotted, bounded
    #     to attempts up through the convergence trigger ("at that point"),
    #     the moment convergence first fires. A later, larger raw score found
    #     by an experiment the loop chose to run *after* convergence (for
    #     logging completeness) is real and worth reporting, but is not the
    #     officially scored checkpoint under a plain reading of "at that point".
    running_best_name = running_best_valid_primary = running_best_test = None
    official_checkpoint_name = official_checkpoint_valid_primary = None
    official_checkpoint_test = None
    auto_running_best_name = auto_running_best_valid_primary = auto_running_best_test = None
    official_auto_checkpoint_name = official_auto_checkpoint_valid_primary = None
    official_auto_checkpoint_test = None

    run_t0 = time.time()

    for priority, name, mod, author in experiments:
        if time.time() - run_t0 > WALL_CLOCK_CAP_SECONDS:
            print("\n6-hour wall-clock cap reached; stopping.")
            break
        if len(history) >= ITERATION_CAP:
            print(f"\n{ITERATION_CAP}-iteration cap reached; stopping.")
            break

        author_counts[author] = author_counts.get(author, 0) + 1
        desc = getattr(mod, 'DESCRIPTION', '')
        print(f"--- [{priority}] {name} (author={author})"
              + (f" - {desc}" if desc else ''))

        out = run_experiment(name, a.data_dir, a.step_timeout)
        status, res, dt = out['status'], out['result'], round(out['seconds'], 2)

        entry = {
            'timestamp': time.time(), 'experiment': name, 'priority': priority,
            'author': author, 'description': desc, 'seconds': dt,
            'status': status, 'error': out['error'], 'code': _read_source(name),
        }

        if status != 'ok':
            tag = 'TIMEOUT' if status == 'timeout' else 'ERROR'
            print(f"    {tag} ({dt:.1f}s) - recovered, champion unchanged")
            print(f"      {(out['error'] or '').splitlines()[0][:120]}")
            entry['decision'] = tag
            history.append(entry)
            with open(a.log, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(_jsonable(entry)) + '\n')
            no_improve += 1
            if no_improve >= a.patience and not converged:
                converged = True
                converged_at = len(history)
                champion_at_convergence = champion_name
                champion_at_convergence_test = champion_test
                official_checkpoint_name = running_best_name
                official_checkpoint_valid_primary = running_best_valid_primary
                official_checkpoint_test = running_best_test
                official_auto_checkpoint_name = auto_running_best_name
                official_auto_checkpoint_valid_primary = auto_running_best_valid_primary
                official_auto_checkpoint_test = auto_running_best_test
                print(f"\nCONVERGED at attempt {len(history)} (incl. this failure): "
                      f"{a.patience} consecutive non-improving attempts.")
            continue

        vp = res['valid']['primary']
        first = champion_valid_primary is None
        improvement = float('inf') if first else vp - champion_valid_primary
        if first or improvement > a.eps:
            decision = 'KEPT (new champion)'
            champion_name, champion_valid_primary, champion_test = name, vp, res['test']
            no_improve = 0
            _write_checkpoint(champion_name, champion_valid_primary,
                              champion_test, len(history) + 1)
        else:
            decision = 'DISCARDED'
            no_improve += 1

        # Plain argmax, no eps gate - this is "the validation-best checkpoint",
        # tracked independently of the eps-gated stopping decision above.
        if running_best_valid_primary is None or vp > running_best_valid_primary:
            running_best_name, running_best_valid_primary, running_best_test = name, vp, res['test']

        auto_decision = None
        if author == 'agent':
            auto_first = auto_champion_valid_primary is None
            auto_imp = float('inf') if auto_first else vp - auto_champion_valid_primary
            if auto_first or auto_imp > a.eps:
                auto_decision = 'KEPT (new autonomous champion)'
                auto_champion_name = name
                auto_champion_valid_primary = vp
                auto_champion_test = res['test']
            else:
                auto_decision = 'DISCARDED (autonomous track)'
            if auto_running_best_valid_primary is None or vp > auto_running_best_valid_primary:
                auto_running_best_name = name
                auto_running_best_valid_primary = vp
                auto_running_best_test = res['test']

        imp_str = 'n/a (first)' if first else f"{improvement:+.4f}"
        test_str = f"  test primary={res['test']['primary']:.4f}" if a.reveal_test_live else ""
        print(f"    valid primary={vp:.4f}{test_str}  improvement={imp_str}  "
              f"-> {decision}  ({dt:.1f}s)")

        entry.update({
            'valid': res['valid'], 'test': res['test'],
            'improvement_over_champion': None if first else round(improvement, 4),
            'decision': decision, 'champion_after': champion_name,
            'auto_decision': auto_decision, 'auto_champion_after': auto_champion_name,
        })
        history.append(entry)
        with open(a.log, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(_jsonable(entry)) + '\n')

        if no_improve >= a.patience and not converged:
            converged = True
            converged_at = len(history)
            champion_at_convergence = champion_name
            champion_at_convergence_test = champion_test
            official_checkpoint_name = running_best_name
            official_checkpoint_valid_primary = running_best_valid_primary
            official_checkpoint_test = running_best_test
            official_auto_checkpoint_name = auto_running_best_name
            official_auto_checkpoint_valid_primary = auto_running_best_valid_primary
            official_auto_checkpoint_test = auto_running_best_test
            print(f"\nCONVERGED at attempt {len(history)}: {a.patience} consecutive "
                  f"attempts with improvement <= {a.eps}. Official checkpoint "
                  f"('validation-best at that point', plain argmax, no eps gate) "
                  f"locked in here; still running any remaining experiment files "
                  f"so nothing already-discovered gets silently dropped from the "
                  f"log, but nothing found after this point is eligible to be "
                  f"the scored checkpoint.")

    # The rule also treats hitting the iteration/wall-clock cap as convergence
    # ("whichever comes first"). If the loop ended that way rather than via
    # the eps/N streak, the official checkpoint was never snapshotted above -
    # do it now, bounded to the full run since a cap ends the run outright.
    if official_checkpoint_name is None and running_best_name is not None:
        converged = True
        converged_at = converged_at if converged_at is not None else len(history)
        official_checkpoint_name = running_best_name
        official_checkpoint_valid_primary = running_best_valid_primary
        official_checkpoint_test = running_best_test
        official_auto_checkpoint_name = auto_running_best_name
        official_auto_checkpoint_valid_primary = auto_running_best_valid_primary
        official_auto_checkpoint_test = auto_running_best_test

    # ---------------- summary ----------------
    print(f"\n=== OFFICIAL CHECKPOINT (validation-best at convergence, no eps "
          f"gate - this is what 'the submission scored for ranking' means): "
          f"{official_checkpoint_name} | valid primary="
          f"{(official_checkpoint_valid_primary or 0):.4f} ===")
    if official_checkpoint_test:
        print(f"    test  GAUC {official_checkpoint_test['GAUC']:.4f} | nDCG@5 "
              f"{official_checkpoint_test['nDCG@5']:.4f} | primary "
              f"{official_checkpoint_test['primary']:.4f}")

    print(f"\n=== OVERALL BEST: {champion_name} | valid primary="
          f"{(champion_valid_primary or 0):.4f} (any author) ===")
    if champion_test:
        print(f"    test  GAUC {champion_test['GAUC']:.4f} | nDCG@5 "
              f"{champion_test['nDCG@5']:.4f} | primary {champion_test['primary']:.4f}")
    if converged_at is not None and champion_name != champion_at_convergence:
        print(f"    (officially converged at attempt {converged_at} on "
              f"'{champion_at_convergence}' - '{champion_name}' is a later, "
              f"better result from experiments added after that point)")

    print(f"\n=== AUTONOMOUS CHAMPION: {auto_champion_name} "
          f"(agent-authored only) ===")
    if auto_champion_test:
        print(f"    valid primary={auto_champion_valid_primary:.4f}")
        print(f"    test  GAUC {auto_champion_test['GAUC']:.4f} | nDCG@5 "
              f"{auto_champion_test['nDCG@5']:.4f} | primary "
              f"{auto_champion_test['primary']:.4f}")
    else:
        print("    (no agent-authored experiment succeeded)")

    manual_interventions = author_counts.get('human', 0)
    total_wall_clock_seconds = round(sum(e['seconds'] for e in history), 2)
    n_failures = sum(1 for e in history if e['status'] != 'ok')
    print(f"\nauthor breakdown: {author_counts}  "
          f"({manual_interventions} human-authored idea(s), "
          f"{n_failures} failure(s) recovered, {len(history)} attempts)")
    print(f"total wall-clock: {total_wall_clock_seconds:.1f}s of the "
          f"{WALL_CLOCK_CAP_SECONDS}s cap; {len(history)} of the {ITERATION_CAP} cap")

    tok_in, tok_out = a.llm_tokens_in, a.llm_tokens_out
    total_llm_tokens = (tok_in + tok_out) if (tok_in is not None and tok_out is not None) else None

    # Manual interventions bounded to the officially scored window (attempts
    # up through converged_at), matching what the official checkpoint is
    # itself bounded to - a human-authored idea tried only after convergence
    # doesn't count against the score that was actually submitted.
    scored_window = history[:converged_at] if converged_at is not None else history
    manual_interventions_in_scored_window = sum(
        1 for e in scored_window if e.get('author') == 'human')

    summary = {
        'official_checkpoint': official_checkpoint_name,
        'official_checkpoint_valid_primary': official_checkpoint_valid_primary,
        'official_checkpoint_test': official_checkpoint_test,
        'official_auto_checkpoint': official_auto_checkpoint_name,
        'official_auto_checkpoint_valid_primary': official_auto_checkpoint_valid_primary,
        'official_auto_checkpoint_test': official_auto_checkpoint_test,
        'manual_interventions_in_scored_window': manual_interventions_in_scored_window,
        'champion': champion_name,
        'champion_valid_primary': champion_valid_primary,
        'champion_test': champion_test,
        'autonomous_champion': auto_champion_name,
        'autonomous_champion_valid_primary': auto_champion_valid_primary,
        'autonomous_champion_test': auto_champion_test,
        'author_counts': author_counts,
        'manual_interventions': manual_interventions,
        'failures_recovered': n_failures,
        'converged': converged,
        'converged_at': converged_at,
        'champion_at_convergence': champion_at_convergence,
        'champion_at_convergence_test': champion_at_convergence_test,
        'attempts': len(history),
        'eps': a.eps, 'patience': a.patience,
        'step_timeout_seconds': a.step_timeout,
        # --- Feasibility & Practicality ---
        'total_wall_clock_seconds': total_wall_clock_seconds,
        'wall_clock_cap_seconds': WALL_CLOCK_CAP_SECONDS,
        'iteration_cap': ITERATION_CAP,
        'gpu_hours': 0,                        # CPU-only (numpy), no GPU
        'llm_tokens_input': tok_in,
        'llm_tokens_output': tok_out,
        'total_llm_tokens': total_llm_tokens,  # None until --llm-tokens-in/out passed;
        # see results/RESOURCES.md for how to fill it in from the Claude Code usage panel
        'history': history,
    }
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    print(f"\nlog: {a.log}\nsummary: {a.summary}")
    if total_llm_tokens is None:
        print("NOTE: total_llm_tokens is null - pass --llm-tokens-in/--llm-tokens-out "
              "or edit agent_summary.json + results/RESOURCES.md before submitting.")


if __name__ == '__main__':
    main()
