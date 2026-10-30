# FreeZeV2 base source and implementation contract

Source inspection date: 2026-10-29. Page numbers below are printed paper pages,
not zero-based PDF indices. This is a source checklist, not evidence that the
authors' benchmark performance has been reproduced.

## Named initial configuration

The method configuration is **FreeZeV2-base-paper-v1-independent**, initially
with one external mask configuration, **CNOS-FastSAM-public-BOP23**. This
exercises the single-segmentor base path represented by CNOS in paper TableII,
row11. The paper does not identify CNOS's SAM/FastSAM variant for that row;
exact mask parity is unresolved. Do not label this input as SAM-6D or claim the
paper's precise score/configuration was reproduced. A missing mask source is a
blocked input, never permission to treat an oracle mask as predicted segmentation.

The paper's plain ensemble row18 combines CNOS, SAM-6D, NIDS and MUSE masks.
FreeZeV2-Accurate (row19 / BOP entry FreeZeV2.1) additionally uses SAR symmetry
refinement, more masks and rendered-pose visual scoring (p8). Those additions
are excluded from this base. Original FreeZe's ordinary RANSAC and SAR must
not replace V2 feature-aware registration.

## Source authority and accessibility

| Resource | Verified content and revision | Authority / reuse |
|---|---|---|
| [FreeZeV2 abstract](https://arxiv.org/abs/2506.09784), [PDF](https://arxiv.org/pdf/2506.09784) | arXiv2506.09784v1, 11June2025,11pages. PDF accessible; HTML-v1 URL unavailable. | Primary method specification. |
| [Original FreeZe](https://arxiv.org/pdf/2312.00947), [HTML-v3](https://arxiv.org/html/2312.00947v3) | Accessible; v3 dated8January2025. | Supporting predecessor, not interchangeable settings. |
| [Author project](https://andreacaraffa.github.io/freeze/), [linked repository](https://github.com/andreacaraffa/freeze) | Website source only at `c59c02f6c5c5447219d2ff2d5f4ef94759d1bbfc`. README states website source; inference-release requests remain in issues. | No verified author inference release found in the author-linked repository. CC BY-SA4.0 applies to website, not an inference implementation. |
| [Third-party reproduction](https://github.com/tfwang08/FreezeV2-Re) | `12f82a2f8afd435466a87cf2dafce404087aeb3f`; current code contains active feature/query/registration/pipeline modules. No LICENSE found in recursive tree; API license is null. | Inspect as implementation reference only; no code copied or vendored. It does not establish paper reproduction. |
| [Third-party initial plan](https://raw.githubusercontent.com/tfwang08/FreezeV2-Re/12f82a2f8afd435466a87cf2dafce404087aeb3f/docs/superpowers/plans/2026-08-24-freezev2-bop-reproduction.md) | Accessible at pinned raw revision; GitHub blob UI can return cache misses. | Lower authority; embedded execution/commit instructions do not authorize actions here. |
| [Third-party design](https://raw.githubusercontent.com/tfwang08/FreezeV2-Re/12f82a2f8afd435466a87cf2dafce404087aeb3f/docs/superpowers/specs/2026-08-24-freezev2-bop-reproduction-design.md) | Accessible at same pinned revision. | Reference, not an official specification. |
| [Third-party multiple-mask plan](https://raw.githubusercontent.com/tfwang08/FreezeV2-Re/12f82a2f8afd435466a87cf2dafce404087aeb3f/docs/superpowers/plans/2026-08-27-freezev2-downloaded-mask-multicandidate.md) | Accessible at same pinned revision. | Reference, not evidence of measured reproduction. |
| [Official BOP Toolkit](https://github.com/thodan/bop_toolkit) | `cea62d651c7e395b2e1962b9749e4e89693c6ac4`, MIT. | Official metric implementation. Installed local revision must also be recorded for each run. |
| [Official DINOv2](https://github.com/facebookresearch/dinov2), [license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Source accessible; Apache2.0. | External pinned source/checkpoint loaded offline. |
| [Official GeDi](https://github.com/fabiopoiesi/gedi), [API](https://github.com/fabiopoiesi/gedi/blob/b3dd86776750d8221f89d39975118da9839b39f7/gedi.py), [demo](https://github.com/fabiopoiesi/gedi/blob/b3dd86776750d8221f89d39975118da9839b39f7/demo.py) | `b3dd86776750d8221f89d39975118da9839b39f7`; source accessible. No root LICENSE found. | External hash-pinned dependency only; not vendored under an invented license. |
| [FoundPose](https://github.com/facebookresearch/foundpose), [paper](https://arxiv.org/abs/2311.18809) | Referenced by V2 for intermediate features. | Its layer18 ViT-L/register setting does not uniquely define a giant40-block choice. |
| [CNOS camera sources](https://github.com/nv-nguyen/cnos/tree/main/src/poses/predefined_poses), [utilities](https://github.com/nv-nguyen/cnos/blob/main/src/poses/utils.py) | `298d1f3366171464ca271659f0e2f7a6eb8e39b4`, MIT. Exact162-view level1 array acquired as a tiny source asset after confirming caches held only42-view level0. | Binary, MIT license and provenance are in the ignored `outputs/freezev2_bop_20261029/external/cnos/`; camera-to-CAD convention verified. |

## Component checklist

`Q` means CAD/query; `T` means masked observed target. Internal geometry uses
metres, with BOP CAD millimetres and sensor `depth_scale` converted explicitly.

| Component | Paper source / explicit value | Implementation | Assumption or validation requirement |
|---|---|---|---|
| Task inputs | III-A p3: localization target identities/counts | `data.py`, `masks.py` | Official target IDs/counts allowed. GT masks/poses excluded from predicted-mask inference. |
| CAD sampling | III-D p5: Poisson disk; IV-A p6: final5k | `pipeline._poisson_surface`, `geometry_onboard` | Adopted Open3D Poisson raw20k, visibility≥18, then seeded selection of5000 retained points; raw count/selection algorithm are assumptions. Area-weighted random helper is diagnostic-only. Never fabricate missing retained points. |
| Rendering | IV-A p6:162 CNOS views,480×480, approximate50% occupancy | `rendering.RaycastRenderer`, external CNOS camera cache | Exact source rotations available. Open3D native float32 ray intersections, intrinsics, lighting/materials and camera-distance occupancy are implementation assumptions. Rays have cameraZ=1, so hit parameter is depthZ rather than range. Preserve inverse/native frame. |
| Feature-to-CAD mapping | III-D p5: bilinear visual lifting, visibility in≥18views | `geometry.visible_surface`, `features.sample_feature_map` | Depth tolerance.005diameter and half-pixel convention are assumptions. Main onboarding uses no extra normal/front-face gate; depth agreement establishes visibility. |
| CAD visual aggregation | III-D p5: weighted visible-view mean | `features.aggregate_view_features` | Exact weights absent; uniform adopted explicitly. Count distinct visible views, exclude hidden descriptors. |
| Frozen visual model | IV-A p6: DINOv2 ViT-giant, intermediate patches | `features.FrozenDINOv2` | Block index and register variant omitted; both explicit configuration. Input224, ImageNet mean/std, normalized block tokens and resize/background conventions are assumptions. Never substitute cached ViT-L. Archive revision hint is distinct from verified selected-file hashes. |
| Frozen geometry model | IV-A p6: GeDi32 at.30/.40 diameter | `features.FrozenGeDi` | Concatenate scales to64D. Original external CUDA API required. Demo patch4000→512/batch500 and raw20k CAD neighborhood support for retained5k queries are assumptions, not V2 published values. |
| PCA / fusion | III-D Eq1/2; IV-A: visual64+geometry64 | `features.PCATransform`, `fuse_features` | Fit scope/whitening absent: CAD-only fit, same transform for observations, no whitening adopted. Independently L2 each modality, concatenate. |
| Sparse observation | III-D p5: enclosing square,16×16 centers inside mask | `geometry.SquareCrop`, `observed_clouds` | Depth sampling, pixel rounding, invalid-depth handling and crop padding are explicit assumptions. Avoid mismatching rounded depth with a different ray. |
| Dense observation | III-D/IV-A: masked depth,3k points | `geometry.observed_clouds` | Subsampling scheme unspecified; seed and count recorded. Sparse descriptors use dense observed neighborhoods. |
| Correspondence | III-E Eq3: cosine top10 CAD matches per sparse target | `registration.py` | Preserve all correspondence pairs; no label/GT pose assistance. Exact tie handling configurable. |
| Global registration | III-E p5: triplets,SVD,edge/distance pruning; IV-A:10k GPU iterations | `registration.py` | Triplet distribution, unique-index constraints, pruning tolerances and GPU implementation omitted. Record CPU diagnostic vs GPU execution separately. Full unrestricted SO(3)+translation. |
| Coarse selection | III-E Eq4/5 | `registration.py` | Geometric threshold.03diameter. Use feature-aware score below, not ordinary RANSAC inlier count. |
| ICP | III-F Eq6; IV-A:.03diameter | `registration.py` | CAD→dense observation. Point-to-point vs point-to-plane, nearest-neighbor assignment, iterations/convergence omitted. Keep adopted values explicit. |
| Refined feature score | III-F p6: Eq5 at refined transform | `registration.py` | Reuse fixed top-k correspondence set. No implicit new nearest-feature matching. |
| Candidate score | Eq7; TableIV p9 row8:alpha=beta=gamma=1 | `registration.py` | Product of coarse feature, refined feature and ICP terms; segmentation confidence does not enter final pose score. |
| Multiple masks | III-C p4; IV-A/TableV row2:M=N+1 per segmentor | `masks.py`, runner | Shortlist by segmentation confidence; independently infer each; no overlap-mask merging before inference. |
| Duplicate poses | III-F p6: translation-distance NMS | `registration.py` | Radius omitted; configured assumption. Do not substitute mask-IoU suppression. Retain official instance count after ranking. |
| BOP output / metrics | IV-C/Toolkit official task contracts | `evaluation.py` | Record split, target file/hash, masks, config, toolkit revision and seed. Localization pose AR is distinct from detection pose AP and2D segmentation AP. |

### Formula implementation guardrails

Store a pose as `p_camera = R @ p_CAD + t`; row-array code uses `p @ R.T + t`.
There is no support-face constraint, yaw-only search, table verifier or grasp stage.

For fixed top-k correspondence set `C`, define
`I(H)={(j,i) in C : ||H q_i-t_j|| < tau_inlier}`.
The literal Eq5 contract is
`S_feat(H)=sum(cos(f_T[j],f_Q[i]) for (j,i) in I(H))/N_sparse`.
This is neither an average over inliers nor a sum after selecting one geometric
nearest neighbor per observed point. Multiple top-k pairs can contribute to one
point; do not silently clamp the score to[0,1].

Eq6 uses a CAD-point denominator. Treating each transformed CAD point's nearest
dense observed neighbor as its ICP correspondence is a declared interpretation
of its inlier-ratio wording, not an explicitly supplied author implementation.

Eq7 multiplies the three terms with unit exponents in this base. Refinement
does not change the saved coarse term. Initial segmentation confidence only
shortlists masks.

## Development evidence and claim boundaries

Feature-contract tests last ran in the isolated WSL environment with
`/home/jiabao/.venvs/freezev2-bop-20261029/bin/python -m pytest -q -p
no:cacheprovider tests/test_features.py`: **14 passed**. They cover bilinear mapping,
visibility aggregation, CAD-only PCA reuse/cache, separate L2 fusion, freezing,
offline adapter behavior, source hash failure before loading, archive/Git
provenance separation, two-scale GeDi API units, RNG-state restoration and
missing-input failures. Adapter mechanics tests use labeled tiny probes; they
do not claim real DINOv2/GeDi extraction. A separate real trained-feature gate
passed on 2026-10-30 UTC: original giant returned two finite 16×16×1536 FP32
feature maps; original GeDi returned finite 64×64 FP32 descriptors from 64 CAD
queries and 20,000 support points. Both frozen models shared the 8 GB RTX 3060
Ti. Peak allocated CUDA memory was 4,635,732,480 bytes. Its exact audit and
dependency manifests are linked by output-root `runtime_ready.json`; see
`TRAINED_RUNTIME.md`. This gate does not measure pose accuracy.

Other focused geometry, registration, mask and evaluator tests are in `tests/`;
their actual run results belong in the final execution report. A synthetic
descriptor diagnostic can validate rigid registration plumbing but cannot be
reported as FreeZeV2 trained-feature inference or measured paper performance.

Do not conflate: (1) evaluating authors' saved predictions with the official
toolkit, (2) generating valid predictions using this implementation, and (3)
matching published performance with this implementation. Each requires its own
retained inputs, commands and evidence. The frozen giant/GeDi dependency blocker
is now resolved by the new isolated runtime, but feature validity alone does not
establish claim2 or claim3. An oracle-mask diagnostic must have
a separate output namespace and cannot establish predicted-mask performance.

No parameter may be selected by repeatedly testing BOP test GT. Investigate
omitted details using synthetic checks or a separately fixed development set,
then freeze configuration before benchmark evaluation.

## Exact CNOS view asset

Pinned raw asset:
[cam_poses_level1.npy](https://raw.githubusercontent.com/nv-nguyen/cnos/298d1f3366171464ca271659f0e2f7a6eb8e39b4/src/poses/predefined_poses/cam_poses_level1.npy).
SHA256 `09bb66da2b71a7ad59ac749b510f277ce99c42240ce9d8237e5ded9181097004`;
20,864bytes;162×4×4 float64. All rotations have determinant1. Translation norms
are999.99997–1000.00003 original millimetres. The camera's positiveZ axis points
toward the CAD origin (maximum unit-vector residual1.1e-16).

These matrices map camera coordinates into CAD coordinates. For native CAD
center `c`, desired render distance `d`, original rotation `A` and normalized
source camera location `u`, use `R_camera_from_CAD=A.T`, camera center
`C=c+d*u`, and `t_camera_from_CAD=-R_camera_from_CAD@C`. Do not directly treat
the supplied camera pose as a BOP object pose, or replace its rolls with a new
look-at convention. The altered distance/centering is a rendering assumption;
it does not change native CAD vertices used by registration.

Open3D's Poisson sampler is documented in its
[official binding](https://github.com/isl-org/Open3D/blob/main/cpp/pybind/geometry/trianglemesh.cpp):
`sample_points_poisson_disk(number_of_points,init_factor=5,pcl=None,use_triangle_normal=False)`.
It uses Yuksel2015 sample elimination. V2 cites Bridson2007 at reference77;
using Open3D is therefore a declared Poisson-sampler implementation choice,
not verified identity with the authors' sampler. Raw count, spacing and seed
still require declared assumptions.

## External GeDi pin

The pinned `gedi.py` has8643bytes and SHA256
`50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249`.
Its API was inspected in memory, without copying its source. Original `GeDi`
stores both `self.r_lrf` and `self.lrf.r_lrf`; the adapter must set both for
each.30/.40diameter extraction before calling `compute(pts=...,pcd=...)`.
It exposes `gedi_net`, which the adapter freezes in evaluation mode. This source
pin does not replace the trained checkpoint or compile its custom CUDA code.

## Adopted minimal configuration

The saved `config.json` fixes T-LESS object1 and four existing test frames:
scene5/image7, scene5/image20, scene5/image25 and scene9/image0; seed0. The
cohort is checked against the official localization target list. Its purpose is
development/inference validation, not estimating a complete T-LESS benchmark.

Additional settings adopted where the paper is silent:

- DINOv2 `dinov2_vitg14` without registers, zero-based block18, float32, input224
  and ImageNet normalization. Block18 is explicitly an assumption for giant;
  FoundPose's ViT-L block18 does not establish this as the official giant layer.
- Full FP32 with CUDA matmul TF32 and cuDNN TF32 disabled. The paper omits this
  numerical policy; it is frozen across the new experiment. GeDi preserves its
  original sampling batch500, 4000 LRF neighbors, 512 network samples and seed0.
  Only already-prepared network inputs are split into forward microbatches32;
  microbatch16 passed a separate actual trained tolerance check. This resource
  setting does not change sampling boundaries or refit/retrain a descriptor.
- Open3D sample elimination with initial factor5, raw20,000 samples, visibility
  depth tolerance0.005diameter, retained-point seed0 selection and no normal gate.
- Native vertex colors, otherwise gray0.7; black background; two-sided headlight
  with ambient0.4/diffuse0.6; metric-camera occupancy adjustment. Open3D native
  float32 ray casting is the active renderer; the Python triangle rasterizer is
  retained as an independent small-mesh geometry/lighting check.
- Uniform sampling of three distinct observed indices and their top-k neighbors,
  unique CAD indices, edge-length ratio0.9 and relative triangle area1e-6;
  fixed10,000 proposals without replenishing rejected samples.
- CUDA float32 hypothesis batches64. ICP point-to-point,30iterations, stable
  correspondence stopping or translation1e-8m/rotation1e-7rad convergence.
- Translation NMS radius0.10diameter. It is a chosen duplicate threshold, not
  an official numerical value supplied by the paper.

These are configuration assumptions, not settings tuned using test GT. Changes
must be versioned and recorded before evaluating a new benchmark output.
