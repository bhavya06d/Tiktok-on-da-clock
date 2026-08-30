"""Bridge run_agent.py's log format to dashboard.py's expected input.

The repo ended up with two independent autonomous-loop implementations with
two different log formats:
    agent.py        writes agent_log.jsonl + agent_summary.json
    run_agent.py     writes runs/<name>/iterations.jsonl

dashboard.py (Person 5) only reads the first format. Rather than rewrite
either system, this converts the second into the first's shape, so the same
dashboard renders either run.

Test-split scores: run_agent.py's solution.py contract only ever prints
val metrics for --split test (the agent has no test labels, by design - see
agent/prompts.py's TASK_BRIEF). KNOWN_TEST_PRIMARY below holds the few
variants that have been separately, directly confirmed by running
`solutions.runner.run_variant(variant, 'test', ...)` by hand (see AGENT.md's
Results section) - not something the agent itself ever saw or used to decide
anything. Any other variant's test score is left null rather than guessed.

Usage:
    python3 runs_to_dashboard_log.py runs/demo/iterations.jsonl
    python3 dashboard.py --log runs_agent_log.jsonl --summary runs_agent_summary.json --out runs_dashboard.html
"""
import argparse
import json
import sys

KNOWN_TEST_PRIMARY = {
    'bpr_numpy': 0.5985,
    'listwise_numpy': 0.5973,
}


def convert(path):
    meta, iterations, summary = None, [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d['type'] == 'meta':
                meta = d
            elif d['type'] == 'iteration':
                iterations.append(d)
            elif d['type'] == 'summary':
                summary = d
    if meta is None or summary is None:
        sys.exit(f"{path}: missing meta or summary line - is this a complete run log?")

    is_live_llm = meta.get('llm') not in (None, 'OfflinePlanner')
    author = 'agent' if is_live_llm else 'agent_scripted'
    # 'agent_scripted' is an honest label, not 'agent': the OfflinePlanner walks
    # a fixed, pre-written idea queue - real for demo/robustness purposes, but
    # not a live model proposing ideas. See PLAN.md / experiments/README.md's
    # AUTHOR convention for why this distinction is tracked at all.

    log = []
    running_best_name, running_best = None, float('-inf')
    for it in iterations:
        m = it.get('metrics')
        valid = {'GAUC': m['gauc'], 'nDCG@5': m['ndcg5'], 'primary': m['primary']} if m else None
        if it.get('accepted'):
            running_best_name, running_best = it['method'], m['primary']
        test = None
        if valid and it['method'] in KNOWN_TEST_PRIMARY:
            test = {'primary': KNOWN_TEST_PRIMARY[it['method']]}  # GAUC/nDCG@5 not re-derived here
        log.append({
            'timestamp': it.get('ts'), 'experiment': it['method'], 'priority': it['iter'],
            'author': author, 'description': it.get('hypothesis', ''),
            'seconds': it.get('duration_s'), 'status': 'ok' if m else 'error',
            'error': it.get('error'), 'code': it.get('code'),
            'valid': valid, 'test': test,
            'decision': it.get('decision'), 'champion_after': running_best_name,
        })

    author_counts = {}
    for e in log:
        author_counts[e['author']] = author_counts.get(e['author'], 0) + 1

    champion_test = ({'primary': KNOWN_TEST_PRIMARY[summary['best_method']]}
                     if summary.get('best_method') in KNOWN_TEST_PRIMARY else None)
    out_summary = {
        'champion': summary.get('best_method'),
        'champion_valid_primary': summary.get('best_val_primary'),
        'champion_test': champion_test,
        'autonomous_champion': summary.get('best_method') if is_live_llm else None,
        'autonomous_champion_valid_primary': summary.get('best_val_primary') if is_live_llm else None,
        'autonomous_champion_test': champion_test if is_live_llm else None,
        'author_counts': author_counts,
        'manual_interventions': summary.get('manual_interventions', 0),
        'converged': True,
        'attempts': summary.get('iterations_used', len(log)),
        'eps': meta.get('epsilon'), 'patience': meta.get('patience'),
        'total_wall_clock_seconds': summary.get('wall_clock_seconds'),
        'wall_clock_cap_seconds': 21600, 'iteration_cap': 50,
        'gpu_hours': 0,
        'total_llm_tokens': summary.get('llm_tokens', {}).get('total') if is_live_llm else 0,
        'history': log,
    }
    return log, out_summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_log', help='e.g. runs/demo/iterations.jsonl')
    ap.add_argument('--log-out', default='runs_agent_log.jsonl')
    ap.add_argument('--summary-out', default='runs_agent_summary.json')
    a = ap.parse_args()

    log, summary = convert(a.run_log)
    with open(a.log_out, 'w') as fh:
        for e in log:
            fh.write(json.dumps(e) + '\n')
    with open(a.summary_out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {a.log_out} ({len(log)} entries) and {a.summary_out}")
    print(f"then: python3 dashboard.py --log {a.log_out} --summary {a.summary_out} --out runs_dashboard.html")
