"""Render agent_log.jsonl + agent_summary.json into a static, self-contained
HTML report (dashboard.html) - Person 5's "receipts" for the judges.

Reads:
    agent_log.jsonl        - one JSON object per attempt (from agent.py)
    agent_summary.json     - final champions + resource totals (from agent.py)
    baseline_scores.json   - official random/popularity/FM/oracle numbers,
                              so the page never hardcodes a number that could
                              drift from the organizers' own published values.

Writes:
    dashboard.html - open it directly (file://) or serve it locally with
    `python3 -m http.server`. Not committed to git (see .gitignore) - like
    submission.csv, it's a derived artifact. Regenerate it any time
    agent_log.jsonl changes: re-run agent.py, then re-run this script.

Usage:
    python3 dashboard.py [--log agent_log.jsonl] [--summary agent_summary.json]
                          [--baseline-scores baseline_scores.json] [--out dashboard.html]
"""
import argparse
import datetime
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='agent_log.jsonl')
    ap.add_argument('--summary', default='agent_summary.json')
    ap.add_argument('--baseline-scores', default='baseline_scores.json')
    ap.add_argument('--template', default='dashboard_template.html')
    ap.add_argument('--out', default='dashboard.html')
    a = ap.parse_args()

    if not os.path.exists(a.log) or not os.path.exists(a.summary):
        sys.exit(f"missing {a.log} / {a.summary} - run `python3 agent.py` first.")

    with open(a.log) as fh:
        log = [json.loads(line) for line in fh if line.strip()]
    with open(a.summary) as fh:
        summary = json.load(fh)
    with open(a.baseline_scores) as fh:
        baseline = json.load(fh)
    with open(a.template) as fh:
        template = fh.read()

    payload = {
        'log': log,
        'summary': summary,
        'baseline': baseline,
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    data_json = json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')

    if '__RUN_DATA__' not in template:
        sys.exit(f"{a.template} is missing the __RUN_DATA__ placeholder - did it get edited?")
    out_html = template.replace('__RUN_DATA__', data_json)

    with open(a.out, 'w') as fh:
        fh.write(out_html)
    print(f"wrote {a.out} ({len(out_html):,} bytes) from {len(log)} logged attempt(s)")
    print(f"open it directly, or run: python3 -m http.server  (then visit http://localhost:8000/{a.out})")


if __name__ == '__main__':
    main()
