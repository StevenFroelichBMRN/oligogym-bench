# OligoGym benchmark container image — build spec and decision record

One image serves both AWS Batch compute classes. Built and validated on a real
x86_64 Docker builder (GitHub Actions), pushed to GHCR.

**Build evidence** (so this claim is checkable rather than asserted): Actions run
[`31526338355`](https://github.com/StevenFroelichBMRN/oligogym-bench/actions/runs/31526338355)
on commit `1c4edddb` completed with conclusion `success`; the GHCR package version
listing returns manifest digest **`sha256:c1bb023c3c5f317e1b071b97af0e0d5608d571bbaf541172b689bab7333ab8a8`**
tagged `latest` and `1c4edddb869f2992d03fd47e446d1cecca327e51`. Because
`validate_image.py --build-time` runs as a `RUN` layer, that successful build *is*
the evidence for every check in §5. Three earlier builds failed and are worth
knowing about: the RNA-FM hub-path bug (§4), a root-owned bind-mount directory,
and GHCR rejecting a mixed-case repository name.

- **Image**: `ghcr.io/stevenfroelichbmrn/oligogym-bench`
  (digest-pinned: `@sha256:c1bb023c3c5f317e1b071b97af0e0d5608d571bbaf541172b689bab7333ab8a8`)
- **Source**: <https://github.com/StevenFroelichBMRN/oligogym-bench> (Apache-2.0,
  derived from [Roche/oligogym](https://github.com/Roche/oligogym) @ `97f5b9f`)
- **Platform**: `linux/amd64` only — g4dn, c6id, r6id and m6id are all amd64.

## 1. Decision: pre-built image, not a Wave-built conda spec

Both compute environments have Wave and Fusion v2 enabled, so a Wave-built image
from a conda/pip spec was the alternative. **A pre-built image is the right choice
here, for three reasons that are specific to this workload:**

1. **The 1.2 GB RNA-FM checkpoint has to be baked in.** Wave's conda/pip path
   builds a layer from a *package specification*; it has no mechanism to `COPY` an
   arbitrary large binary blob into the image. The checkpoint is not a pip package
   — it is a single `.pth` file that must land at a specific path in the torch hub
   cache. The only alternatives are a runtime download per container start (1.2 GB
   × every task, from a host that is unreliable — see §4) or a shared filesystem
   mount (adds an FSx/EFS dependency the compute envs do not currently have).
   Baking it in is the only option that is both cheap and robust.
2. **The `sm_75` guarantee needs a build-time assertion.** The torch wheel must
   come from the `cu126` index *and* must have Turing compiled in. A Wave conda
   spec would resolve `pytorch` from conda-forge, whose CUDA builds and
   architecture lists are a moving target between rebuilds. The Dockerfile pins
   `torch==2.13.0+cu126` explicitly and **fails the build** if `sm_75` is absent
   from `torch.cuda.get_arch_list()`. That assertion is the difference between
   "should work on a T4" and "verified to work on a T4".
3. **A 45,950-fold-fit sweep should not re-resolve its environment.** A digest-
   pinned image means every one of the ~11.5k Phase-3 tasks runs bit-identical
   software. Wave rebuilds are cached but the cache is not a guarantee across
   weeks, and a silent dependency drift mid-sweep would be very expensive to
   detect after the fact.

**Wave stays enabled anyway.** Wave *augmenting* an existing image (which is what
enables Fusion v2's direct-S3 task filesystem) is orthogonal to Wave *building*
one from a conda spec. The pipeline config keeps `wave.enabled = true` and
`fusion.enabled = true` — the image is pre-built, and Wave augments it on the way
to the task. This is the combination that makes the Phase-3 sweep's S3 I/O cheap
without giving up build reproducibility.

Cost of the choice: the image is ~8 GB (torch+CUDA ~5 GB, RNA-FM 1.2 GB). ECR
would give lower pull latency from us-west-2 than GHCR; see §6.

## 2. What is in the image

| Component | Pin | Why it is here |
|---|---|---|
| python | 3.11.15 | matches the Phase-1 validated env |
| torch | **2.13.0+cu126** | CUDA 12.6; `sm_75` compiled in (asserted at build) |
| pytorch-lightning | 2.2.1 | upstream `pyproject.toml` pin |
| scikit-learn | 1.4.0 | upstream `pyproject.toml` pin |
| numpy / pandas / scipy | 1.26.4 / 2.2.3 / 1.17.1 | Phase-1 validated |
| torch_geometric | 2.8.0.post1 | **no compiled companions** — see §3 |
| xgboost / catboost | 2.0.3 / 1.2.10 | catboost undeclared upstream |
| viennarna | 2.7.2 | provides `import RNA`; cp311 manylinux x86_64 wheel |
| rna-fm | 0.2.2 | provides `import fm`; gates `RNA_FM_AVAILABLE` |
| torchinfo, networkx, tqdm, biopython, Levenshtein, tabpfn | see `requirements-image.txt` | 7 of these are undeclared by upstream `pyproject.toml` |
| RNA-FM checkpoint | sha256 `5b5d7d87…3d5e99` | 1194.4 MB, baked at `$TORCH_HOME/hub/checkpoints/` |
| oligogym | `97f5b9f` `--no-deps` | includes the 18 MB of processed datasets |
| patched harness | Phase 1 | `train_model_patched.py`, `oligogym_patch.py` |

**The CPU-vs-CUDA divergence is confined to the torch wheel.** Nothing else in
the list differs between the two compute classes, which is why one image serves
both: on a c6id/r6id/m6id task `torch.cuda.is_available()` is simply `False` and
every model class still runs on CPU.

## 3. torch_geometric without compiled companions — deliberate

`torch_scatter`, `torch_sparse`, `torch_cluster` and `pyg_lib` are **not
installed**, and the build asserts they are absent while verifying that
`GCNConv` + `global_max_pool` on a batched `Data` list works. Phase 1 established
this on the validated environment; the image re-verifies it. torch_geometric ≥ 2.3
is pure Python and needs the compiled companions only for optional accelerated
kernels, none of which `oligogym` touches. Adding them is the usual source of
`torch_scatter`-vs-CUDA-version build failures and buys nothing.

## 4. The RNA-FM trap, and how the build closes it

Upstream `RNAFMEmbeddings` silently falls back to a 6-dimensional
`_get_simple_features` representation when the pretrained model fails to load —
**no exception, no warning**. A run that hits this produces plausible-looking
numbers from the wrong features.

Three independent guards:

1. **Baked checkpoint + sha256 verification** at build time. The canonical CUHK
   URL (`proj.cse.cuhk.edu.hk/rnafm/api/download?...`) returns nginx 403 for
   every path, so the file comes from the RNA-FM authors' own HF org
   (`huggingface.co/cuhkaih/rnafm`).
2. **A path trap, found empirically and worth recording**: `torch.hub.get_dir()`
   is `$TORCH_HOME/`**`hub`**, not `$TORCH_HOME`. The first build of this image
   placed the checkpoint one directory too high; the download and sha256 check
   both passed, and the *load* then tried the network and 403'd. The build-time
   validator caught it. The checkpoint must be at
   `$TORCH_HOME/hub/checkpoints/RNA-FM_pretrained.pth`.
3. **`validate_image.py` monkey-patches `urllib.request.urlopen` and
   `torch.hub.download_url_to_file` to raise** during the RNA-FM load, so any
   attempt to reach the network fails the build rather than silently succeeding
   against a wrong path. It then asserts 99.5M parameters and a 640-dim layer-12
   representation.

The calibration runner carries the same assertion at runtime: a
`RNAFMEmbeddings` config whose featurized shape is not `(n, L, 640)` raises
instead of being measured.

## 5. Validation status

`validate_image.py` runs as a `RUN` layer, so **a build that completes has passed
every non-GPU check.** Measured in the built image (Actions run, x86_64):

| check | result |
|---|---|
| `from oligogym.models import *` (10 classes) | PASS |
| `from oligogym.features import *`, `RNA_FM_AVAILABLE=True` | PASS |
| patched harness + `oligogym_patch` importable | PASS |
| **`sm_75` in `torch.cuda.get_arch_list()`** | **PASS** — torch 2.13.0+cu126, cuda 12.6, arch list `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` |
| PyG works with no compiled companions | PASS — companions confirmed absent |
| RNA-FM loads from cache: 99.5M params, repr `(1, 42, 640)` | PASS |
| 12 shipped datasets load, row counts exact | PASS — incl. sherwood 291,551 |
| CPU training fold (RF on TLR7) | PASS |

GPU-execution checks (`--gpu`: live CUDA device, a real Lightning fold with VRAM
measured, RNA-FM on GPU) are **skipped on the build runner, which has no GPU**,
and are run as the `validate` mode of the Nextflow pipeline on a g4dn task.

## 6. ECR vs GHCR

The image is in GHCR because this sandbox has no Docker daemon and no AWS
credentials, so an ECR push was not possible from here. GHCR is publicly
readable, so AWS Batch can pull it without registry credentials.

**Recommended before Phase 3**: mirror the image into ECR in `us-west-2`. Same
region as the compute envs means a faster, cheaper pull (no NAT egress for an
~8 GB image × every instance scale-out), and pull-through cache makes it a
one-liner:

```bash
aws ecr create-repository --repository-name oligogym-bench --region us-west-2
docker pull ghcr.io/stevenfroelichbmrn/oligogym-bench:latest
docker tag  ghcr.io/stevenfroelichbmrn/oligogym-bench:latest \
  <acct>.dkr.ecr.us-west-2.amazonaws.com/oligogym-bench:latest
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-west-2.amazonaws.com
docker push <acct>.dkr.ecr.us-west-2.amazonaws.com/oligogym-bench:latest
```

Then set `params.image` in `nextflow.config` to the ECR ref. **Pin by digest**
(`@sha256:…`) rather than `:latest` for the sweep, so every task is provably the
same software.

## 7. Reproducing the build

```bash
git clone https://github.com/StevenFroelichBMRN/oligogym-bench
cd oligogym-bench
docker buildx build --platform linux/amd64 -t oligogym-bench:local .
# GPU checks, on a machine with an NVIDIA GPU:
docker run --rm --gpus all oligogym-bench:local \
  python /opt/oligogym-bench/validate_image.py --gpu
```

The build is driven by `.github/workflows/build-image.yml` on every push to
`main`; it builds, validates, records image facts (versions, arch list, layer
sizes), pushes to GHCR, and uploads the validation report as a workflow artifact.
