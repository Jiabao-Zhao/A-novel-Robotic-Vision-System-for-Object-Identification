# Frozen feature source mapping

This module is an independent implementation of feature plumbing. It does not
contain GeDi source, a pretrained model, a substitute descriptor, or an accuracy
claim. Real extraction fails if exact local dependencies are absent.

Primary specification: [FreeZeV2, arXiv2506.09784v1](https://arxiv.org/pdf/2506.09784),
11June2025. The earlier [FreeZe paper](https://arxiv.org/pdf/2312.00947) is contextual;
its different registration and symmetry refinement are not V2 base defaults.

| Component | Source | Implementation | Status and validation |
|---|---|---|---|
| CAD visual lifting | III-D, p5 | `sample_feature_map` | Bilinear interpolation explicit. Half-pixel, border and crop conventions are assumptions; checked against torch align_corners=False interpolation. Visibility must be supplied by renderer/depth checks. |
| Multi-view CAD aggregation | III-D, p5 | `aggregate_view_features` | Visible-view weighted average explicit; uniform weights adopted because weights omitted. Counts returned for caller's V=18 retention. Unseen features remain NaN. |
| Backbone | IV-A, p6 | `FrozenDINOv2` | ViT-giant explicit; register variant, block index, normalization and resize omitted. Config must explicitly choose block, never substitute ViT-L. Offline strict state loading; freeze/eval/inference_mode tests use a labeled tiny test double only. |
| 224input, ImageNet normalization | V1 IV-A; DINOv2 official convention | `FrozenDINOv2.extract` |224 is consistent with16×16 patch14 grid but is an explicit V2 assumption. Bilinear antialiased resize, normalized block tokens and unchanged crop background are assumptions. Root must record chosen block/variant. |
| PCA64 | III-D Eq1/2 and IV-A | `PCATransform` | Dimension64 follows two32D GeDi scales and128D fused output. Fitting scope and whitening omitted; adopted CAD-only, no whitening, same transform for observations. Test proves observations do not refit PCA and serialization preserves it. |
| Two GeDi radii | IV-A p6 | `FrozenGeDi.compute` |32D at.30/.40 native object diameter explicit, concatenated into64D. Caller must use the same native units for all points and diameter. |
| GeDi source/checkpoint | [author GeDi repository](https://github.com/fabiopoiesi/gedi) | external adapter | External user-provided `gedi.py` must match configured SHA256; optional git revision and checkpoint SHA verified. No root license found, so source is not vendored. No random, geometric-only or learned replacement. |
| GeDi local patch handling | [official demo](https://github.com/fabiopoiesi/gedi/blob/main/demo.py) | adapter arguments |4000 LRF neighbors,512 network samples,batch500 are demo-derived configurable assumptions, not FreeZeV2 values. Original implementation samples/pads neighborhoods. Seed0 adopted and NumPy caller RNG restored. Original CUDA/Open3D/Pointnet2 dependencies remain required. |
| CAD GeDi support density | V2 III-D does not specify neighborhood support sampling | runner calls `FrozenGeDi.compute` | Queries are the retained5000 CAD points; neighborhood support is the raw20,000-point CAD surface cloud. Using the denser raw support is an explicit reproduction assumption, not a published V2 value. |
| Fusion | III-D Eq1/2 | `fuse_features` | Per-modality L2 then concatenation. No added fusion weight, no final renormalization; correspondence cosine must account for fused norms. Tests verify both unit halves and sqrt2 fused norm. |

The author-linked [FreeZe repository](https://github.com/andreacaraffa/freeze),
revision `c59c02f6c5c5447219d2ff2d5f4ef94759d1bbfc`, is a project website, not a
verified author inference release. Its CC BY-SA4.0 website license does not
license a nonexistent inference implementation. The [DINOv2 source license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
is Apache2.0. This adapter loads an external source tree; it copies no DINOv2 code.

The inspected GeDi source revision is
`b3dd86776750d8221f89d39975118da9839b39f7`. Its `gedi.py` SHA256 is
`50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249`
(8643bytes). Both original `GeDi.r_lrf` and `GeDi.lrf.r_lrf` exist and must
change together for the two physical radii.

The cached DINOv2 hub archive has no `.git`. Its revision string
`e1277af2ba9496fbadf7aec6eba56e8d882d1e35` is recorded as a
`source_revision_hint`, separate from a verified Git revision. Two local files
were compared byte-for-byte against official raw files at that revision:

| Relative source file | Verified SHA256 |
|---|---|
| `hubconf.py` | `e82275b3adcc3631865d1edd4b620b89d42459169acdeb652804cd0eb0fe9615` |
| `dinov2/models/vision_transformer.py` | `8dc1e2ededd4893a335aa2f524c15554bcdb14d824e310ca3df825e16081a268` |

`expected_source_hashes` verifies configured relative Python files before model
loading. Provenance retains actual fingerprints and verified hash pins. This
proves the selected file bytes, not the entire archive or a clean source tree.
Missing/mismatching pins fail before a model is loaded. No official source was
copied during the comparison.

The GeDi checkpoint archive advertised by its author download script has Google
Drive ID `1Lpep5QigALjk60h8bNJAUH3DnxtnGcZX`; demo path is
`data/chkpts/3dmatch/chkpt.tar`. Do not execute the combined data/assets downloader
or download duplicates automatically. Verify existing local files first.

Actual trained feature extraction passed a separate runtime gate on 2026-10-30
UTC, using the original official giant checkpoint and the original GeDi 3DMatch
checkpoint. Unit tests alone do not establish this. The retained audit is
`outputs/freezev2_bop_20261029/runtime/trained_feature_smoke_20261030_fp32_v2.json`;
the output-root `runtime_ready.json` joins its evidence with source, checkpoint,
environment and compiled-operation fingerprints. See `TRAINED_RUNTIME.md` for
exact paths and measured results. This demonstrates feature extraction, not pose
accuracy or reproduction of the paper's score.

The new experiment environment uses full FP32 with CUDA matmul and cuDNN TF32
disabled. Numerical policy is an explicit reproduction assumption because the
paper does not specify it. Official DINOv2 source uses its standard non-xFormers
fallback when xFormers is absent; no backbone or trained weights are substituted.

Original GeDi prepares neighborhoods with sampling batch 500. That boundary is
part of the frozen sampling recipe: changing it changes random neighborhood
selection and must not be presented as a memory-only adjustment. The adapter
instead splits only the already-prepared frozen network forward pass into
microbatches, default 32. `set_forward_microbatch_size(n)` updates this resource
setting at `provenance['resources']['network_forward_microbatch_size']` while
preserving sampling batch 500, seed, neighborhoods and LRF preparation. On 64
actual trained queries, forward microbatches 32 and 16 differed from a single
64-query forward by at most 5.72e-7 and 6.09e-7 respectively; both passed
rtol 1e-5 / atol 1e-6. Repeating microbatch 32 yielded identical outputs.

Giant checkpoint loading uses memory-mapped, weights-only CPU state loading and
strict state assignment into the official architecture before moving full FP32
weights to CUDA. This avoids a second eager checkpoint copy. No random weights
survive strict state loading, and all floating model parameters were verified
FP32, frozen and in evaluation mode during the real runtime gate.
