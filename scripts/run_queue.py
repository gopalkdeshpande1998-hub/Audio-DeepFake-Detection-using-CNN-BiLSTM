"""Run experiment steps one at a time (only one GPU job at once on a 16 GB machine).

Usage:
    python scripts/run_queue.py configs/queues/phase2_features.yaml

A queue file is a YAML list of steps:
    - train:    {config: configs/e2a_lfcc.yaml, seed: 42[, stop_after: 22]}
    - evaluate: {run: e2a_lfcc_s42}                              # default sets
    - evaluate: {run: e2a_lfcc_s42, tag: trimmed, trim_silence: on, sets: [asvspoof2019_la:eval]}
    - script:   [scripts/aggregate_runs.py, --name, e2a_lfcc, --glob, "e2a_lfcc_s*"]

Finished steps are skipped (train: runs/<name>/train_summary.json exists;
evaluate: experiments/<run>[__tag]/metrics.json exists), so a queue can be
re-run after a crash. An unfinished run folder is removed before retraining.
Progress goes to stdout and runs/queue.log.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from adfd.config import load_config
from adfd.paths import REPO_ROOT, load_paths

PY = sys.executable


def log(msg: str, logfile: Path) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], logfile: Path) -> int:
    log("$ " + " ".join(cmd), logfile)
    t0 = time.time()
    with open(logfile, "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f"  exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min", logfile)
    return proc.returncode


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    steps = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    paths = load_paths()
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    logfile = paths.runs_dir / "queue.log"
    log(f"queue {sys.argv[1]}: {len(steps)} steps", logfile)
    failures = 0
    for i, step in enumerate(steps, 1):
        kind, spec = next(iter(step.items()))
        prefix = f"step {i}/{len(steps)} {kind}"
        if kind == "train":
            cfg = load_config(REPO_ROOT / spec["config"])
            name = spec.get("run_name") or f"{cfg['name']}_s{spec['seed']}"
            run_dir = paths.runs_dir / name
            if (run_dir / "train_summary.json").exists():
                log(f"{prefix} {name}: already trained, skipping", logfile)
                continue
            if run_dir.exists():
                log(f"{prefix} {name}: removing unfinished run folder", logfile)
                shutil.rmtree(run_dir)
            cmd = [PY, "-u", "-m", "adfd.train", "--config", spec["config"], "--seed", str(spec["seed"]),
                   "--run-name", name]
            if "workers" in spec:
                cmd += ["--workers", str(spec["workers"])]
            if "stop_after" in spec:  # keeps the config's LR schedule, unlike --max-epochs
                cmd += ["--stop-after", str(spec["stop_after"])]
        elif kind == "evaluate":
            name = spec["run"]
            tag = spec.get("tag")
            out = REPO_ROOT / "experiments" / (f"{name}__{tag}" if tag else name) / "metrics.json"
            if out.exists():
                log(f"{prefix} {name} {tag or ''}: already evaluated, skipping", logfile)
                continue
            if not (paths.runs_dir / name / "best.pt").exists():
                log(f"{prefix} {name}: no checkpoint, skipping", logfile)
                failures += 1
                continue
            cmd = [PY, "-m", "adfd.evaluate", "--run", str(paths.runs_dir / name)]
            if tag:
                cmd += ["--tag", tag]
            if "trim_silence" in spec:
                cmd += ["--trim-silence", {True: "on", False: "off"}.get(spec["trim_silence"], str(spec["trim_silence"]))]
            if "sets" in spec:
                cmd += ["--sets", *spec["sets"]]
            if "workers" in spec:
                cmd += ["--workers", str(spec["workers"])]
            if "normalize" in spec:
                cmd += ["--normalize", str(spec["normalize"])]
            if "n_boot" in spec:
                cmd += ["--n-boot", str(spec["n_boot"])]
        elif kind == "script":
            cmd = [PY, *[str(a) for a in spec]]
        else:
            log(f"{prefix}: unknown step type {kind!r}", logfile)
            failures += 1
            continue
        log(prefix, logfile)
        if run(cmd, logfile) != 0:
            failures += 1
    log(f"queue finished with {failures} failed step(s)", logfile)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
