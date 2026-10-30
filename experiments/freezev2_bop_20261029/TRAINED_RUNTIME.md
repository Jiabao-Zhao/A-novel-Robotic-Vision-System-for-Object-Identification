# Trained feature runtime evidence

The original frozen DINOv2 giant and original trained GeDi now execute together
on the RTX 3060 Ti (8 GB). The real feature gate passed on 2026-10-30 UTC. This
establishes dependency and feature validity; pose predictions and official BOP
scores require their own retained inference and evaluation evidence.

## Environment and entry points

Use this **experiment-specific WSL Python**:

```text
/home/jiabao/.venvs/freezev2-bop-20261029/bin/python
```

Core packages are Torch 2.2.2+cu121, torchvision 0.17.2+cu121, NumPy 1.26.4,
SciPy 1.13.1, Open3D 0.19.0, torchgeometry 0.1.2 and pointnet2-ops 3.0.0. Existing
compatible dependencies are reused through a read-only site-packages path;
new/incompatible packages are installed only in the new experiment environment.
The original environment's torch 2.11.0, torchvision 0.26.0, NumPy 2.2.6 and
Open3D 0.19.0 distribution versions remain unchanged. Its packages were not
uninstalled or upgraded. No global environment, NIST source or driver was changed.

The separate numerical policy is full FP32, CUDA matmul TF32 false and cuDNN
TF32 false. These choices are assumptions where the paper is silent. The giant
is the no-register `dinov2_vitg14`, zero-based block 18, 224 RGB input with
ImageNet normalization; the block/input details remain explicit assumptions.

The output-root readiness manifest is:

```text
D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\outputs\freezev2_bop_20261029\runtime_ready.json
```

It contains exact Windows and WSL asset paths, measured checkpoint hashes,
source pins, compiled-operation fingerprints and links to all runtime audits.
The environment/pip snapshots are in the sibling `runtime/` directory:

- `trained_runtime_environment_20261030.json`
- `trained_runtime_pip_freeze_20261030.txt`
- `trained_feature_smoke_20261030_fp32_v2.json`
- `trained_feature_smoke_20261030_fp32_v2.npz`

The old 2.11 environment could not import Open3D's ML torch operations because
that wheel requires PyTorch 2.2. The new environment resolves that exact ABI
constraint rather than changing the production environment.

## Official assets and provenance

Existing cached DINOv2 source is reused. Its archive has no `.git`, so the
revision `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` is a revision hint, with
`hubconf.py` and `dinov2/models/vision_transformer.py` independently verified
against official bytes. This does not prove the entire source tree clean.

The missing giant weights were downloaded once from the official Meta URL:

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
```

They are retained under `external/dinov2/dinov2_vitg14_pretrain.pth`,
4,546,108,579 bytes, SHA256:

```text
baf8467e50af277596bbbafa06887c177ee899ab46033649c383577d7e9309d3
```

Original GeDi source is an ignored external checkout at
`external/gedi`, revision `b3dd86776750d8221f89d39975118da9839b39f7`.
Its `gedi.py` SHA256 is:

```text
50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249
```

No top-level GeDi license grant was found. Its implementation remains an external
dependency; no GeDi algorithm code is copied into the tracked reproduction.
The original author-linked 3DMatch checkpoint was acquired from Google Drive ID
`1Lpep5QigALjk60h8bNJAUH3DnxtnGcZX`. Only the checkpoint archive was fetched;
the combined demo-data downloader was not run. The ZIP passed complete CRC
validation. The extracted `external/gedi/data/chkpts/3dmatch/chkpt.tar` has
12,512,035 bytes, SHA256:

```text
15723f382ecf60f7efbf9161f70bffda2d1c817025cb0702350baaa04060f5f4
```

`external/gedi_checkpoint_acquisition.json` retains source URL, archive and
checkpoint hashes. Existing caches were searched before obtaining either model;
the cached ViT-L and ViT-S weights were not duplicated or substituted for giant.

## Local CUDA operations

There was no usable precompiled PointNet2 extension or nvcc in existing local
environments. Only the official NVIDIA CUDA 12.1.1 nvcc, cudart and CCCL
redistributables were installed into:

```text
/home/jiabao/.local/share/freezev2_bop_20261029/cuda-12.1
```

Archives, NVIDIA license terms and their official SHA256 values are preserved.
No system CUDA toolkit or driver install was performed. The independent
`build_original_pointnet2.py` entry point builds the original unmodified GeDi
PointNet2 CPP/CUDA sources for the actual device architecture 8.6. It replaces
only the incompatible build-target list in the original setup invocation; it
does not edit original source kernels or add fast-math flags.

The installed extension SHA256 is:

```text
7c56646e58561993a64f1aeaa6c3dc9b3e12ac0bd1cb79b4de165b4c7fd3a6fe
```

The exact wheel and source-build manifest are retained at
`external/wheels/pointnet2_ops-3.0.0-cp312-cp312-linux_x86_64.whl` and
`external/wheels/pointnet2_build_sources.json`.

## Actual trained checks

The smoke loads strict original states, freezes both networks, and uses no GT
poses. GeDi processes 64 CAD points against the 20,000-point surface cloud for
T-LESS object 1, at radii 0.3 and 0.4 native diameter. It returns finite
64-dimensional FP32 descriptors. Two real rendered CAD views produce finite
16×16×1536 giant maps. All giant floating parameters are FP32 and frozen, and
both networks are in evaluation mode. Peak allocated/reserved CUDA memory with
both loaded was 4,635,732,480 / 4,687,134,720 bytes.

Original GeDi sampling boundaries are preserved at 500 queries per batch. Its
LRF preparation uses 4000 neighbors and 512 network samples, seed 0. These
demo-derived settings are assumptions, not values claimed from FreeZeV2.
Changing sampling batch would change the random patch recipe, so memory
reduction is restricted to forward-only microbatching after patches are ready.

| Actual trained comparison | Maximum absolute descriptor difference |
|---|---:|
| Original 64-query forward vs microbatch 32 | 5.7183e-7 |
| Original 64-query forward vs microbatch 16 | 6.0815e-7 |
| Repeated microbatch 32 | 0 |

Both comparisons pass rtol 1e-5 / atol 1e-6. The normal resource setting is 32;
16 is a validated forward-only fallback. It is separate from scientific cache
identity. The adapter exposes `set_forward_microbatch_size(n)` and reports the
actual setting at `provenance['resources']['network_forward_microbatch_size']`.

Earlier failed smoke audits are preserved. The first default-TF32 comparison
failed the strict numerical tolerance (~1.04e-4); a second run passed GeDi but
found an unsupported meta-device giant constructor operation. The final passing
run uses the explicit full-FP32 policy and memory-mapped strict checkpoint
loading into the original CPU architecture, then transfers the actual weights
to GPU. No test GT was consulted to choose these resource/numerical fixes.

The final focused command ran from the reproduction staging root:

```text
wsl --cd "/mnt/c/Users/Jiabao Zhao/OneDrive - The Pennsylvania State University/Documents 1/ChatGPT/RA-L/freezev2_staging" --exec /home/jiabao/.venvs/freezev2-bop-20261029/bin/python -m pytest -q -p no:cacheprovider tests/test_features.py
```

Result: **14 passed in 1.88 seconds**. The real trained smoke is independent of
these adapter/mechanics tests; its duration was 56.74 seconds including model
loading. Individual view extraction times are not full inference speed claims.
