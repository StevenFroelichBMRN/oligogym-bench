#!/usr/bin/env python
"""Validate the OligoGym benchmark image.

--build-time : checks that need no GPU (run in the Docker build; a failure
               fails the build). Imports, sm_75 in the compiled arch list,
               RNA-FM checkpoint loads into a real 99.5M-param model, the 12
               shipped datasets are present, PyG works without compiled
               companions.
--gpu        : additionally require a live CUDA device and run one real
               training fold on it. Run as the first task on a g4dn instance.

Exit code 0 = all checks passed. Prints one line per check.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

RESULTS = []


def check(name, fn, required=True):
    try:
        detail = fn()
        RESULTS.append({"check": name, "ok": True, "detail": detail,
                        "required": required})
        print(f"PASS  {name}: {detail}")
        return True
    except Exception as e:
        RESULTS.append({
            "check": name, "ok": False, "required": required,
            "detail": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-800:],
        })
        print(f"{'FAIL' if required else 'WARN'}  {name}: {type(e).__name__}: {e}")
        return not required


# ---------------------------------------------------------------- checks
def c_imports_models():
    import oligogym.models as M

    exec("from oligogym.models import *", {})
    need = ["LinearModel", "NearestNeighborsModel", "RandomForestModel",
            "XGBoostModel", "CatBoostModel", "MLP", "CNN", "GRU",
            "Transformer", "GNN"]
    missing = [n for n in need if not hasattr(M, n)]
    if missing:
        raise AssertionError(f"missing model classes: {missing}")
    return f"{len(need)} model classes importable"


def c_imports_features():
    import oligogym.features as F

    exec("from oligogym.features import *", {})
    need = ["OneHotEncoder", "KMersCounts", "HELMGraph", "RNAFMEmbeddings",
            "Thermodynamics"]
    missing = [n for n in need if not hasattr(F, n)]
    if missing:
        raise AssertionError(f"missing featurizers: {missing}")
    if not getattr(F, "RNA_FM_AVAILABLE", False):
        raise AssertionError("RNA_FM_AVAILABLE is False -- `import fm` failed")
    return "featurizers importable, RNA_FM_AVAILABLE=True"


def c_torch_arch():
    """The load-bearing GPU check: sm_75 must be compiled into the wheel.

    Newer torch releases have been dropping older architectures. Without sm_75 a
    T4 either fails outright or silently JIT-recompiles via PTX at high cost.
    """
    import torch

    archs = torch.cuda.get_arch_list()
    if not any(a in ("sm_75", "compute_75") for a in archs):
        raise AssertionError(
            f"sm_75 (Tesla T4) NOT in compiled arch list {archs}; "
            "this wheel cannot run natively on g4dn"
        )
    return f"torch {torch.__version__}, cuda {torch.version.cuda}, arch_list includes sm_75 ({archs})"


def c_pyg_no_companions():
    """torch_geometric must work with NO compiled companions installed."""
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_max_pool

    present = []
    for mod in ("torch_scatter", "torch_sparse", "torch_cluster", "pyg_lib"):
        try:
            __import__(mod)
            present.append(mod)
        except ImportError:
            pass

    g = Data(x=torch.randn(6, 9), edge_index=torch.tensor([[0, 1, 2, 3, 4],
                                                           [1, 2, 3, 4, 5]]),
             y=torch.tensor([1.0]))
    dl = DataLoader([g, g], batch_size=2)
    batch = next(iter(dl))
    out = global_max_pool(GCNConv(9, 4)(batch.x, batch.edge_index), batch.batch)
    assert out.shape == (2, 4), out.shape
    return (f"PyG {torch_geometric.__version__} GCNConv+global_max_pool OK; "
            f"compiled companions installed: {present or 'none (intended)'}")


def c_rnafm_checkpoint():
    """RNA-FM must load real pretrained weights, not the silent 6-dim fallback."""
    import fm
    import torch

    model, alphabet = fm.pretrained.rna_fm_t12()
    n = sum(p.numel() for p in model.parameters())
    if n < 90_000_000:
        raise AssertionError(f"RNA-FM has only {n} params; expected ~99.5M")
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([("s", "GGGUCAGGCCGGCGAAAGUCGCCACAGUUUGGGGAAAGCUGUGCAGCCUGUAACCCCCCCACGAAAGUGGG"[:40])])
    model.eval()
    with torch.no_grad():
        rep = model(toks, repr_layers=[12])["representations"][12]
    if rep.shape[-1] != 640:
        raise AssertionError(f"embedding dim {rep.shape[-1]} != 640")
    return f"RNA-FM {n/1e6:.1f}M params, layer-12 repr {tuple(rep.shape)}"


def c_datasets():
    from oligogym.data import DatasetDownloader

    dd = DatasetDownloader()
    keys = ["tlr7", "tlr8", "cytotox lna", "neurotox lna", "neurotox moe",
            "sirnamod", "openaso", "asoptimizer", "sherwood", "huesken",
            "ichihara", "shmushkovich"]
    got = {}
    for k in keys:
        d = dd.download(k)
        got[k] = len(d.x)
    expect = {"tlr7": 192, "tlr8": 192, "cytotox lna": 768, "sherwood": 291551,
              "asoptimizer": 32602, "huesken": 2431}
    bad = {k: (got[k], v) for k, v in expect.items() if got.get(k) != v}
    if bad:
        raise AssertionError(f"row-count mismatch (got, expected): {bad}")
    return f"12/12 datasets load from package resources; rows={got}"


def c_harness_importable():
    import train_model_patched as H
    from oligogym_patch import resolve_dataset_key

    assert H.N_FOLDS == 5
    assert resolve_dataset_key("immune_modulation_TLR7") == "tlr7"
    return "patched harness + oligogym_patch importable"


def c_cpu_fold():
    """One real end-to-end fold on the smallest dataset, CPU path."""
    import numpy as np
    import train_model_patched as H

    cfg = {"dataset": "immune_modulation_TLR7", "featurizer": "OneHotEncoder",
           "featurizer_args": {}, "model": "RandomForestModel",
           "model_args": {"n_estimators": 10}, "cross_validation": "random"}
    H.check_config(cfg)
    data = H.download_data_with_retries(cfg)
    rng = np.random.default_rng(0)
    Xtr, Xte, ytr, yte, _, _ = H.prepare_data_fold(data, 0, rng=rng)
    f = H.prepare_featurizer(cfg)
    Xtr, Xte = H.featurize(Xtr, Xte, f, cfg)
    m = H.prepare_model(cfg)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    assert len(p) == len(yte)
    return f"RF fold on TLR7 OK, X_train {Xtr.shape}"


# ---------------------------------------------------------------- GPU-only
def c_cuda_live():
    import torch

    if not torch.cuda.is_available():
        raise AssertionError("torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    x = torch.randn(512, 512, device="cuda")
    y = (x @ x).sum().item()
    assert y == y  # not NaN
    return (f"{name} sm_{cap[0]}{cap[1]}, {total:.1f} GB, "
            f"matmul OK, n_devices={torch.cuda.device_count()}")


def c_gpu_fold():
    """One real Lightning training fold on the GPU, with VRAM measured."""
    import numpy as np
    import torch
    import train_model_patched as H

    cfg = {"dataset": "immune_modulation_TLR7", "featurizer": "OneHotEncoder",
           "featurizer_args": {}, "model": "CNN",
           "model_args": {"hidden_dim": 32, "depth": 1, "kernel_size": 3},
           "cross_validation": "random"}
    H.check_config(cfg)
    data = H.download_data_with_retries(cfg)
    rng = np.random.default_rng(0)
    Xtr, Xte, ytr, yte, _, _ = H.prepare_data_fold(data, 0, rng=rng)
    f = H.prepare_featurizer(cfg)
    Xtr, Xte = H.featurize(Xtr, Xte, f, cfg)
    torch.cuda.reset_peak_memory_stats()
    m = H.prepare_model(cfg)
    m.fit(Xtr, ytr, max_epochs=3)
    p = m.predict(Xte)
    vram = torch.cuda.max_memory_allocated() / 1e6
    dev = next(m.parameters()).device if hasattr(m, "parameters") else "?"
    assert len(p) == len(yte)
    if vram <= 0:
        raise AssertionError("peak VRAM is 0 -- training did not run on the GPU")
    return f"CNN fold on GPU OK, peak VRAM {vram:.1f} MB, param device {dev}"


def c_rnafm_gpu():
    import torch
    from oligogym.features import RNA_FM_AVAILABLE
    import sys as _s
    _s.path.insert(0, "/opt/oligogym-bench")
    from oligogym_patch import RNAFMEmbeddingsFixed

    assert RNA_FM_AVAILABLE
    torch.cuda.reset_peak_memory_stats()
    fz = RNAFMEmbeddingsFixed()
    X = ["AUGCAUGCAUGCAUGCAUGC", "GGGCCCAAAUUUGGGCCCAA"]
    E = fz.fit_transform(X)
    vram = torch.cuda.max_memory_allocated() / 1e6
    if E.shape[-1] != 640:
        raise AssertionError(f"RNA-FM embeddings dim {E.shape} -- fallback path!")
    return f"RNA-FM on GPU: embeddings {E.shape}, peak VRAM {vram:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-time", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    ok = True
    ok &= check("imports:oligogym.models", c_imports_models)
    ok &= check("imports:oligogym.features", c_imports_features)
    ok &= check("harness importable", c_harness_importable)
    ok &= check("torch arch_list has sm_75", c_torch_arch)
    ok &= check("PyG works without compiled companions", c_pyg_no_companions)
    ok &= check("RNA-FM checkpoint loads (99.5M params)", c_rnafm_checkpoint)
    ok &= check("12 shipped datasets load", c_datasets)
    ok &= check("CPU training fold", c_cpu_fold)

    if a.gpu:
        ok &= check("live CUDA device", c_cuda_live)
        ok &= check("GPU training fold", c_gpu_fold)
        ok &= check("RNA-FM on GPU", c_rnafm_gpu)
    else:
        import torch

        print(f"INFO  cuda.is_available()={torch.cuda.is_available()} "
              "(GPU execution checks skipped; run with --gpu on a g4dn task)")

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(RESULTS, fh, indent=2)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'VALIDATION FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
