#!/usr/bin/env python
"""GPU packing factor: throughput vs number of concurrent training processes.

Runs N identical single-fold training jobs as N concurrent subprocesses sharing
one device, and reports aggregate throughput (fold-fits/minute). Repeats for
each N in --levels. The point is to find where throughput stops scaling — which
sets the configs-per-GPU-task chunk size for Phase 3.

Also records per-level CPU utilisation, so CPU starvation (the expected
bottleneck on a 4-vCPU g4dn.xlarge, where featurization and dataloading are
CPU-bound) is distinguishable from GPU saturation.

Usage:
    python gpu_pack.py --levels 1 2 4 8 --reps 2 --out gpu_packing.jsonl \
        --model CNN --dataset siRNA1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import psutil
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# One representative config per model class. Kept small enough that a single
# fold finishes in seconds-to-minutes, large enough that the GPU does real work.
CONFIGS = {
    "CNN": {"featurizer": "OneHotEncoder", "featurizer_args": {},
            "model": "CNN",
            "model_args": {"hidden_dim": 64, "depth": 2, "kernel_size": 3}},
    "MLP": {"featurizer": "OneHotEncoder", "featurizer_args": {},
            "model": "MLP", "model_args": {"hidden_dims": [128, 64]}},
    "GRU": {"featurizer": "OneHotEncoder", "featurizer_args": {},
            "model": "GRU", "model_args": {"hidden_dim": 64, "num_layers": 2}},
    "Transformer": {"featurizer": "OneHotEncoder", "featurizer_args": {},
                    "model": "Transformer",
                    "model_args": {"d_model": 128, "nhead": 4,
                                   "dim_feedforward": 128, "num_layers": 2,
                                   "dropout": 0.1}},
    "GNN": {"featurizer": "HELMGraph", "featurizer_args": {},
            "model": "GNN",
            "model_args": {"hidden_dim": 64, "output_dim": 1, "num_layers": 3,
                           "pooling_operation": "max"}},
    "RNAFM_Transformer": {"featurizer": "RNAFMEmbeddings",
                          "featurizer_args": {"flatten": False},
                          "model": "Transformer",
                          "model_args": {"d_model": 128, "nhead": 4,
                                         "dim_feedforward": 128,
                                         "num_layers": 2, "dropout": 0.1}},
}


class SysSampler(threading.Thread):
    """Sample system-wide CPU%, and GPU utilisation/VRAM via nvidia-smi."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.cpu = []
        self.gpu_util = []
        self.gpu_mem = []
        self._stop = threading.Event()

    def run(self):
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            self.cpu.append(psutil.cpu_percent(None))
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    u, mem = out.stdout.strip().splitlines()[0].split(",")
                    self.gpu_util.append(float(u))
                    self.gpu_mem.append(float(mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

        def stats(v):
            if not v:
                return None, None
            return sum(v) / len(v), max(v)

        cm, cx = stats(self.cpu)
        gm, gx = stats(self.gpu_util)
        mm, mx = stats(self.gpu_mem)
        return {"cpu_pct_mean": cm, "cpu_pct_max": cx,
                "gpu_util_mean": gm, "gpu_util_max": gx,
                "gpu_mem_mb_mean": mm, "gpu_mem_mb_max": mx}


def write_config(dirpath, dataset, spec, cv="random"):
    cfg = {"dataset": dataset, "cross_validation": cv, **spec}
    p = os.path.join(dirpath, "config.yaml")
    with open(p, "w") as fh:
        yaml.safe_dump(cfg, fh)
    return p


def run_level(n, dataset, spec, folds, seed, workdir, env):
    """Launch n concurrent calibrate.py processes, one fold each."""
    procs, outs = [], []
    sampler = SysSampler()
    sampler.start()
    t0 = time.perf_counter()
    for i in range(n):
        d = os.path.join(workdir, f"proc{i}")
        os.makedirs(d, exist_ok=True)
        cfgp = write_config(d, dataset, spec)
        out = os.path.join(d, "rec.jsonl")
        outs.append(out)
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "calibrate.py"),
             "--config", cfgp, "--out", out, "--seed", str(seed + i),
             "--folds", str(folds), "--tag", f"pack_n{n}_p{i}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env))
    errs = []
    for p in procs:
        _, e = p.communicate()
        if p.returncode != 0:
            errs.append((p.returncode, (e or b"").decode()[-400:]))
    wall = time.perf_counter() - t0
    sysstats = sampler.stop()

    recs = []
    for o in outs:
        if os.path.exists(o):
            with open(o) as fh:
                recs += [json.loads(ln) for ln in fh if ln.strip()]
    ok = [r for r in recs if r.get("status") == "ok"]
    return {
        "concurrency": n, "wall_s": wall,
        "n_procs": n, "folds_per_proc": folds,
        "fold_fits_completed": len(ok),
        "fold_fits_expected": n * folds,
        "throughput_fits_per_min": len(ok) / wall * 60 if wall > 0 else None,
        "median_fit_wall_s": (sorted(r["wall_s"] for r in ok)[len(ok) // 2]
                              if ok else None),
        "mean_fit_wall_s": (sum(r["wall_s"] for r in ok) / len(ok)
                            if ok else None),
        "max_peak_vram_mb": (max((r.get("peak_vram_mb") or 0) for r in ok)
                             if ok else None),
        "sum_peak_vram_mb": (sum((r.get("peak_vram_mb") or 0) for r in ok)
                             if ok else None),
        "max_peak_rss_mb": (max((r.get("peak_rss_mb") or 0) for r in ok)
                            if ok else None),
        "errors": errs[:3], "n_errors": len(errs),
        **sysstats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--folds", type=int, default=1,
                    help="folds per process per rep")
    ap.add_argument("--dataset", default="siRNA1")
    ap.add_argument("--model", default="CNN", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads-per-proc", type=int, default=1,
                    help="OMP/MKL threads per process")
    a = ap.parse_args()

    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(a.threads_per_proc)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")

    spec = CONFIGS[a.model]
    n_cpu = psutil.cpu_count()
    with open(a.out, "a") as fh:
        for rep in range(a.reps):
            for n in a.levels:
                with tempfile.TemporaryDirectory() as wd:
                    r = run_level(n, a.dataset, spec, a.folds,
                                  a.seed + 1000 * rep, wd, env)
                r.update({"rep": rep, "model": a.model, "dataset": a.dataset,
                          "host_cpus": n_cpu,
                          "threads_per_proc": a.threads_per_proc})
                fh.write(json.dumps(r, default=str) + "\n")
                fh.flush()
                print(f"n={n} rep={rep} wall={r['wall_s']:.1f}s "
                      f"tput={r['throughput_fits_per_min']:.2f}/min "
                      f"gpu_util={r.get('gpu_util_mean')} "
                      f"cpu={r.get('cpu_pct_mean')}")


if __name__ == "__main__":
    main()
