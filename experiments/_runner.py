"""Private subprocess worker — runs ONE experiment in an isolated process.

`agent.py` invokes this as:

    python -m experiments._runner <experiment_name> [--data_dir ...]

It loads the dataset, calls `experiments.<name>.run(splits)`, and prints the
result as a single JSON line prefixed `RESULT_JSON:` to stdout. Any exception
is printed as a full traceback to stderr and the process exits non-zero.

Running each experiment here (instead of in-process inside `agent.py`) is what
gives the orchestrator: a hard per-experiment timeout (kill a hang), process
isolation (an OOM kill or segfault takes down only the worker, not the loop),
and a real stderr traceback to log. The `_` prefix keeps this file out of
`agent.py`'s experiment auto-discovery.
"""
import argparse
import importlib
import json
import os
import sys
import traceback

# tolerate `python experiments/_runner.py NAME` as well as `-m experiments._runner`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_splits(data_dir):
    """`data.load()` re-parses ~1.4M CSV rows (~40s) every call. Since each
    experiment is its own process, cache the parsed splits once to
    .cache/splits_<mtime>.pkl and reuse it across the run."""
    import glob
    import pickle
    from data import load

    log_a = os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")
    key = str(int(os.path.getmtime(log_a)))
    cache_dir = os.path.join(os.getcwd(), ".cache")
    cache = os.path.join(cache_dir, f"splits_{key}.pkl")
    if os.path.isfile(cache):
        try:
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001
            pass
    splits = load(data_dir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        for stale in glob.glob(os.path.join(cache_dir, "splits_*.pkl")):
            os.remove(stale)
        with open(cache, "wb") as fh:
            pickle.dump(splits, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001
        pass
    return splits


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    a = ap.parse_args()

    try:
        splits = _load_splits(a.data_dir)
        mod = importlib.import_module(f"experiments.{a.name}")
        if not hasattr(mod, "run"):
            raise AttributeError(f"experiments/{a.name}.py has no run(splits)")
        res = mod.run(splits)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("RESULT_JSON: " + json.dumps(_jsonable(res)))


if __name__ == "__main__":
    main()
