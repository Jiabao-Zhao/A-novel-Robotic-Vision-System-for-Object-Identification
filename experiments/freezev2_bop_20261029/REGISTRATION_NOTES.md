# FreeZeV2 registration implementation

`freezev2_repro/registration.py` is an independent implementation of the base
FreeZeV2 registration and candidate scoring path. It does not copy code from
the unlicensed `tfwang08/FreezeV2-Re` implementation. That repository was
inspected at revision `12f82a2f8afd435466a87cf2dafce404087aeb3f` as a secondary
reference; neither its existence nor its plans establish reproduction of the
paper's published performance.

Primary method source: [FreeZeV2, arXiv:2506.09784](https://arxiv.org/abs/2506.09784).
The primary-paper audit should be read alongside this component record. The
module takes external frozen descriptors and masks/point clouds. It does not
extract learned features, access benchmark ground-truth poses, or constrain
poses using table/support geometry.

| Component | Method source / status | Implementation | Validation |
|---|---|---|---|
| CAD-to-camera rigid transform | Full rigid registration, method section | `rigid_fit`, proper 3D SVD, `camera = R @ cad + t` | Three-axis large rotation, translation, proper determinant, degenerate rejection |
| Correspondence set C | Top-k descriptor correspondences; base k=10 | `topk_correspondences`, every sparse observed row has k CAD candidates | Cosine invariance to row magnitude, stable ties, all pairs retained |
| Feature-aware RANSAC | 10,000 hypotheses; geometric tolerance 0.03 diameter | `feature_ransac`, batched 3-point SVD and all-C score | Partial overlap, distractor correspondences, observed outliers, large rotation, fixed-seed repeatability |
| Coarse feature score | Eq. 5: sum inlier-pair cosine / sparse observed point count | `score_pose`, same implementation in hypothesis scorer | Multiple inlier CAD candidates per observed row produce score greater than 1; no clipping or k denominator |
| ICP | CAD-to-dense-observed refinement; threshold 0.03 diameter | `refine_icp`, point-to-point nearest-neighbour fitting, no scale | Residual refinement after a large rotation; no-neighbour failure retains last pose |
| Refined feature score | Eq. 5 evaluated at refined pose | `score_pose` accepts the unchanged saved C | Saved indices unchanged; pose-dependent inlier scores |
| Geometric term | One-sided transformed CAD nearest-neighbour inlier fraction | `score_pose`, `refine_icp` | Hidden/unexplained CAD point remains in denominator |
| Joint candidate score | Base three unit exponents | `joint_score(coarse, refined, geometry)` | Exact product, no probability clipping |
| Multiple candidates | Base ranks resulting poses by joint score, then NMS/instance count | `rank_candidates` | High mask confidence cannot override joint score, same-CAD duplicate suppression, stable ties |
| Random sampling | **Assumption:** uniform three distinct observed rows, independent uniform top-k CAD candidate; rejected triplets consume budget | `RegistrationConfig.seed`; recorded sampler string and hypothesis counts | Deterministic CPU repeatability and explicit failure if all hypotheses invalid |
| Triplet pruning | **Assumption:** repeated CAD IDs removed; noncollinear triangles; edge-length min/max ratio >=0.9 | Configurable `edge_length_ratio` and `min_triangle_area_ratio` | Degenerate test and recorded mutually exclusive pruning counts |
| ICP optimiser / stopping | **Assumption:** point-to-point SVD; maximum 30 updates; stable correspondences or transform deltas <=1e-8 m /1e-7 rad | All settings exposed in `RegistrationConfig` | Synthetic refinement convergence; saved per-iteration count/RMSE/update deltas |
| Translation NMS threshold | **Unspecified in paper:** caller must supply a positive metric threshold | Required `translation_nms_threshold_m` keyword; no default | Reject missing/nonpositive assumption; preserves different CAD IDs |
| CPU/GPU execution | Paper's GPU-parallel RANSAC is available via explicit optional CUDA engine; NumPy default is a **CPU reference variant** | `hypothesis_backend=numpy` or `torch_cuda`; precision/batch size recorded | CPU and actual RTX 3060 Ti CUDA synthetic verification passed; the portable test raises an explicit error when CUDA is unavailable |

All points, diameter, translation and NMS thresholds use metres. CAD model
dimensions are retained. The rigid fit rejects collinear point sets because
their orientation is not fully determined. Coplanar noncollinear triples are
valid. Sparse observation rows and descriptor rows must correspond exactly.

The input array shape for CAD descriptors is `(N_cad,D)` and observed
descriptors `(N_sparse,D)`. `topk_correspondences` returns a `Correspondences`
dataclass with `(N_sparse,k)` CAD indices and cosines. `feature_ransac` returns
a `PoseResult`; failed registration has `R=t=score=None` and explicit status.
No identity pose is fabricated after a failed hypothesis search.

For every candidate: retain `PoseResult.correspondences`, score the coarse pose,
call `refine_icp`, then score the refined pose **using the same correspondences**.
Compute the joint product, write the coarse/refined poses and component scores,
and rank the candidates. Segmentation confidence is used only by the upstream
base candidate selection policy, not by this final pose-ranking function.

The optional CUDA backend uses the same NumPy-generated sampled index bank,
then CUDA batched SVD and pair scoring, with explicitly selected float32 or
float64. It does not force a runtime into GPU mode, change a model's precision,
or fall back to CPU. Floating-point ties and boundary cases can differ between
backends; bitwise equivalence is not claimed. The backend choice belongs in
each run configuration and result provenance.

Offline component tests can be run from this staging directory with:

```powershell
& D:\Python\python.exe -m pytest tests/test_registration.py -q
```

These tests establish numerical implementation behaviour. They do not measure
BOP pose performance, reproduce authors' scores, or verify the frozen feature
networks. BOP evaluation belongs to the separate official-toolkit adapter.

## Executed component verification

All 13 focused tests passed in both native Windows Python and the existing WSL
environment `/home/jiabao/.venvs/lerobot-libero/bin/python`. The WSL run used
actual CUDA, PyTorch 2.11.0+cu130 and an RTX 3060 Ti. Its test exercises the
explicit CUDA backend instead of skipping it.

`runtime/registration_runtime_audit.json` additionally records a 10,000-triplet,
k=10 synthetic test with 200 CAD points, 100 sparse observations, 40% CAD overlap,
20% observed outliers and a [117,-73,158]-degree three-axis rotation. Of the fixed
10,000 proposed triplets, 29 passed the declared pruning checks. CUDA float32
rotation error was 9.906e-7 rad and translation error was 1.694e-8 m; its Eq. 5
score differed from the float64 CPU reference by 5.528e-8. CUDA float64 matched
the CPU score exactly for this case. An Open3D 0.19 point-to-point ICP reference
also agreed with this module to 3.27e-16 rad and 6.66e-16 m on a synthetic
residual-refinement case. These are numerical tests with known synthetic inputs,
not dataset accuracy results. Recorded timings mix cold/warm CUDA execution and
are not a controlled runtime comparison.
