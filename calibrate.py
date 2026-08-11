#!/usr/bin/env python
"""Instrumented single-config runner for OligoGym benchmark calibration.

Wraps the Phase-1 patched harness (`train_model_patched.py`) without modifying
it: the harness's own functions are imported and re-driven fold by fold so that
each stage can be timed separately.

Emitted per fold-fit (one JSON line per fold to --out):
    wall_s              total fold wall seconds
    split_s             fold/split construction
    featurize_s         featurizer.fit_transform + transform
    fit_s               model.fit
    predict_s           model.predict (train + test)
    peak_rss_mb         max RSS of this process, sampled in a thread
    peak_vram_mb        torch.cuda.max_memory_allocated (None on CPU)
    peak_vram_reserved_mb
    epochs              epochs actually run before early stop (Lightning only)
    device              'cuda:<name>' or 'cpu'
    n_params            model parameter count (torch models)
    X_shape             featurized training matrix shape

Determinism: --seed is passed through to the harness's seeding scheme so
timings are repeatable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import train_model_patched as H  # noqa: E402
from oligogym_patch import seed_everything  # noqa: E402


# ---------------------------------------------------------------- resource probes
class RSSSampler(threading.Thread):
    """Sample this process's RSS (plus children) until stopped.

    NOTE: the stop flag must NOT be called `_stop` -- threading.Thread has a
    private `_stop()` method and shadowing it with an Event breaks Thread's own
    teardown with "TypeError: 'Event' object is not callable".
    """

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._halt = threading.Event()
        self._proc = psutil.Process()
        # children() needs elevated privileges on some platforms (macOS);
        # probe once so the sample loop does not pay for a failing call.
        try:
            self._proc.children(recursive=True)
            self._can_children = True
        except Exception:
            self._can_children = False

    def run(self):
        while not self._halt.is_set():
            try:
                rss = self._proc.memory_info().rss
                if self._can_children:
                    try:
                        for c in self._proc.children(recursive=True):
                            try:
                                rss += c.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except Exception:
                        self._can_children = False
                self.peak = max(self.peak, rss)
            except psutil.NoSuchProcess:
                break
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        return self.peak / 1e6


def cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def device_label():
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cpu"


def reset_vram():
    if cuda_available():
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def read_vram():
    if not cuda_available():
        return None, None
    import torch

    return (
        torch.cuda.max_memory_allocated() / 1e6,
        torch.cuda.max_memory_reserved() / 1e6,
    )


def n_params(model):
    for attr in ("model", "net", "module"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "parameters"):
            return int(sum(p.numel() for p in m.parameters()))
    if hasattr(model, "parameters"):
        try:
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:
            return None
    return None


def epochs_run(model):
    tr = getattr(model, "trainer", None)
    if tr is None:
        return None
    # current_epoch is 0-based and already incremented past the last epoch
    return int(getattr(tr, "current_epoch", 0))


def shape_of(X):
    if isinstance(X, np.ndarray):
        return list(X.shape)
    if isinstance(X, pd.DataFrame):
        return list(X.shape)
    if isinstance(X, list):
        return [len(X), "graph"]
    return None


# ---------------------------------------------------------------- one fold
def run_one_fold(config, data, k, seed, rng, feature_cache=None):
    """Re-drive H.run_fold's body with per-stage timing.

    feature_cache: optional dict used to test the featurization cache. Key is
    (dataset, featurizer, featurizer_args, cross_validation, fold).
    """
    rec = {"fold": k}
    reset_vram()
    sampler = RSSSampler()
    sampler.start()
    t_fold = time.perf_counter()

    # ---- split
    t0 = time.perf_counter()
    if config["cross_validation"] == "random":
        Xtr, Xte, ytr, yte, itr, ite = H.prepare_data_fold(data, k, rng=rng, seed=seed)
    else:
        rs = None if rng is None else int(rng.integers(0, 2**31 - 1))
        Xtr, Xte, ytr, yte, itr, ite = H.prepare_data(
            data, split_strategy=config["cross_validation"], random_state=rs
        )
    rec["split_s"] = time.perf_counter() - t0
    rec["n_train"], rec["n_test"] = len(Xtr), len(Xte)

    # ---- featurize (cacheable stage)
    ck = None
    if feature_cache is not None:
        ck = (
            config["dataset"],
            config["featurizer"],
            json.dumps(config.get("featurizer_args") or {}, sort_keys=True),
            config["cross_validation"],
            k,
        )
    t0 = time.perf_counter()
    if ck is not None and ck in feature_cache:
        Xtr_f, Xte_f, extra = feature_cache[ck]
        config["model_args"].update(extra)
        rec["cache_hit"] = True
    else:
        featurizer = H.prepare_featurizer(config)
        before = dict(config["model_args"])
        Xtr_f, Xte_f = H.featurize(Xtr, Xte, featurizer, config)
        extra = {
            kk: vv
            for kk, vv in config["model_args"].items()
            if kk in ("input_dim", "seq_len") and before.get(kk) != vv
        }
        rec["cache_hit"] = False
        if ck is not None:
            feature_cache[ck] = (Xtr_f, Xte_f, extra)
    rec["featurize_s"] = time.perf_counter() - t0
    rec["X_shape"] = shape_of(Xtr_f)

    # RNA-FM guard: upstream silently falls back to 6-dim simple features when
    # the checkpoint fails to load. Assert real 640-dim embeddings.
    if config["featurizer"] == "RNAFMEmbeddings" and not config.get(
        "featurizer_args", {}
    ).get("flatten", False):
        sh = rec["X_shape"]
        if not (sh and len(sh) == 3 and sh[2] == 640):
            raise RuntimeError(
                f"RNA-FM produced shape {sh}, expected (n, L, 640). The "
                "pretrained checkpoint did not load -- upstream silently "
                "substitutes 6-dim _get_simple_features."
            )

    # ---- fit
    model = H.prepare_model(config)
    t0 = time.perf_counter()
    H.predict  # noqa: B018  (documenting that we inline it below)
    model.fit(Xtr_f, ytr)
    rec["fit_s"] = time.perf_counter() - t0

    # ---- predict
    t0 = time.perf_counter()
    ptr = model.predict(Xtr_f)
    pte = model.predict(Xte_f)
    rec["predict_s"] = time.perf_counter() - t0

    rec["wall_s"] = time.perf_counter() - t_fold
    rec["peak_rss_mb"] = sampler.stop()
    v, vr = read_vram()
    rec["peak_vram_mb"], rec["peak_vram_reserved_mb"] = v, vr
    rec["epochs"] = epochs_run(model)
    rec["n_params"] = n_params(model)
    rec["device"] = device_label()

    m = H.regression_metrics(np.asarray(yte).squeeze(), np.asarray(pte).squeeze())
    rec["test_pearson_correlation"] = float(
        m.get("test_pearson_correlation", m.get("pearson_correlation", np.nan))
        if isinstance(m, dict)
        else np.nan
    )
    del model, ptr, pte
    reset_vram()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="run config.yaml")
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="", help="free-form label carried into output")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="enable the per-(dataset,featurizer,args,split,fold) feature cache",
    )
    args = ap.parse_args()

    config = H.load_yaml(args.config)
    H.check_config(config)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    data = H.download_data_with_retries(config)
    load_s = time.perf_counter() - t0

    cache = {} if args.cache else None
    base = {
        "tag": args.tag,
        "dataset": config["dataset"],
        "featurizer": config["featurizer"],
        "featurizer_args": json.dumps(config.get("featurizer_args") or {},
                                      sort_keys=True),
        "model": config["model"],
        "model_args": json.dumps(config.get("model_args") or {}, sort_keys=True),
        "cross_validation": config["cross_validation"],
        "dataset_rows": int(len(data.x)),
        "dataset_load_s": load_s,
        "seed": args.seed,
        "cache_enabled": bool(args.cache),
    }

    with open(args.out, "a") as fh:
        for k in range(args.folds):
            # model_args is mutated by featurize(); restore per fold
            cfg = json.loads(json.dumps(config))
            try:
                rec = run_one_fold(cfg, data, k, args.seed, rng, feature_cache=cache)
                rec["status"] = "ok"
            except Exception as e:  # record the failure, keep going
                rec = {
                    "fold": k,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:],
                }
            rec.update(base)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
