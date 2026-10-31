# Exact public-source execution: notes and unresolved choices

Source audit: **2026-10-30, America/New_York**. Selected external implementation: [tfwang08/FreezeV2-Re](https://github.com/tfwang08/FreezeV2-Re/tree/12f82a2f8afd435466a87cf2dafce404087aeb3f), revision `12f82a2f8afd435466a87cf2dafce404087aeb3f`. This is a third-party code replication with declared inputs, not an author-released implementation. SOURCE_AVAILABILITY.md records the source search and absent verified code license. The original independent baseline and GeDi/CAD-resolution ablation remain historical evidence and must not be mixed into this run.

No benchmark completion or accuracy is asserted by this source checklist. Actual execution settings and receipts belong to the run configuration and outputs. A source-code default, documentation example, paper value and locally adopted choice are different kinds of evidence.

## Required choices: the repository is not a complete run preset

| Choice | What the pinned public source actually supplies | Required treatment |
|---|---|---|
| DINO giant block | `extract-query-visual --layer` is required. README offers layer 30 as one candidate, explicitly not the final FreeZeV2 layer. The plan mentions 29 / 30 / 31 / 39 as development candidates. | Record the selected zero-based block before inference. Neither 30 nor our previous 18 may be described as a published author default. Do not select it by BOP test-GT score. |
| CAD depth visibility tolerance | `--depth-tolerance` is required; README leaves a placeholder. | Record the numeric value/rule and native units. Do not silently reuse our independent tolerance and call it a public constant. |
| Translation NMS | `--nms-translation-threshold-mm` is required for multi-mask and dataset localization. No numeric default is published. It is one global millimetre argument per invocation. | Record it explicitly. Our prior 0.1 × diameter rule is not this CLI's fixed default. |
| Raw CAD point count | Public `sample_query_points` defaults to 5,000. README explicitly calls its 5,000 raw-point example a geometry smoke configuration because the paper does not state N_raw. Visibility filtering may retain fewer points. | Record raw and retained counts independently. Do not claim that 5,000 raw points guarantee the paper's 5,000 final points, or silently insert our 20k sampling/support policy. |
| Other exposed CLI controls | grid 16, dense 3,000, top-k 10, 10,000 RANSAC attempts, seed 0, edge similarity 0.9, ICP 30; visual `min_views=18`, inverse_bilinear depth sampling; no-register dinov2_vitg14 default. | Use the original implementations and record explicit execution values. Edge similarity 0.9 and ICP stop/cap remain the public reconstruction's choices, not fully author-published settings. |

## Active source-to-stage mapping

All links below are to the exact selected revision.

| Stage | Public source / behavior | Boundary for this reproduction |
|---|---|---|
| CAD geometry and cameras | [onboard.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/onboard.py): seeded surface candidates and Poisson rejection, CNOS 162-view camera geometry, per-view 50% camera occupancy fitting. | Import the original functions; no our Open3D sampling/camera replacement. CAD coordinates stay native BOP millimetres. |
| Rendering | `render_templates`: pinned BOP VisPy, flat shading, ambient 0.02, black background, RGB+depth, 480 × 480 pixels. | Original model dimensions/frames. No our headlight/ambient 0.4 photometry or old render caches. |
| CAD DINO input | [features.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/features.py): round dimensions down to multiples of 14 pixels, top-left crop from 480 × 480 to 476 × 476 pixels, 34 × 34 feature grid. | Use the full public wrapper unchanged; our earlier CAD 476 ablation changed only part of the representation and is not a cache substitute. |
| Intermediate DINO tokens | `encode` hooks `model.blocks[layer]`, runs the model, then explicitly calls `model.norm(tokens)` at lines 216–217 before removing class/register tokens. | These are normalized block tokens. Do not describe the public output as raw/unnormalized. The previous snippet-based raw-token claim was corrected before execution. |
| CAD image feature sampling | Public `sample_feature_map` uses FoundPose coordinates `2*point/(width,height)-1` with grid_sample align_corners=False. | Do not replace it with our half-pixel-coordinate sampler. |
| CAD aggregation | Actual `extract-query-visual` branch calls `aggregate_query_visual_features_streaming`, projection/depth visibility, `min_views=18` and view_weights=None. | Do not substitute the alternate query_crop/query_features pixel-lifting path merely because those modules exist. Cache the actual retained points/counts. |
| GeDi | [gedi_bridge.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/gedi_bridge.py): original trained GeDi revision b3dd86776750d8221f89d39975118da9839b39f7; radius-search shim in gedi_radius.py uses CPU torch.cdist. Radii 0.3 / 0.4 × diameter; 500 sampling-batch points / 4,000 LRF points / 512 network points; both NumPy and Torch seeds reset. A fresh GeDi model is constructed/deleted per scale. | Use public failures and sampling. CAD query and support are the same retained points; do not use our 20k support cloud or finite-row repair. |
| PCA and fusion | [fusion.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/fusion.py): CAD-only SVD PCA 64, no our SVD-sign canonicalization; branch L2 normalization then 128D concatenation. CLI saves PCA state as float32. | Use its fit, transform and cache writers, not old features/PCA. |
| Observed feature path | `run_bop.py` early-return extract-target dispatch at lines 1164–1171 calls [target_cli.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/target_cli.py). It uses target_features.py square bbox, PIL EXTENT+BICUBIC RGB crop at 224 × 224 pixels and 16 × 16 direct patch tokens. Sparse 3D rays use fractional patch coordinates, with depth sampled at selected sensor pixels; dense points are seeded masked-depth samples. | This active path is **224-pixel cropped observation processing**, not full-frame target DINO. The later legacy extract-target branch is unreachable after that early return. Do not copy our OpenCV crop/integer-ray mapping. |
| Matching and registration | [matching.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/matching.py), pipeline.py and [pose.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/pose.py): original top-k cosine matches, NumPy float64/RNG triplets, noncollinearity/relative-edge pruning, Kabsch, feature-aware score. | No our CUDA hypothesis backend, alternate tie handling, table/yaw restriction or pose verifier. Public coarse CLI rejects nonfinite best score when no valid hypothesis exists. |
| Refinement and score | Original point-to-point ICP with brute-force/chunked NN, strict 3% diameter gate, 30-step cap and stable entire-NN/inlier-set stop. [refinement.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/freezev2/refinement.py) multiplies coarse feature, fine feature and ICP terms with exponents 1. | No our KDTree/transform-delta stop or altered geometric denominator. |
| Multiple masks | [run_bop.py](https://github.com/tfwang08/FreezeV2-Re/blob/12f82a2f8afd435466a87cf2dafce404087aeb3f/run_bop.py): N+1 confidence shortlist per input source, per-mask original target/coarse/refine commands, pose-product ranking, translation NMS, retain N. | Segmentation confidence is not multiplied into the pose score. Invalid mask cases remain recorded and must remain in official target denominators. |

## Public command/cache interfaces

The public source has no implemented generic onboard CLI even though the plan mentions one. An external orchestration helper may call its original onboarding functions to create geometry/templates. Such orchestration must not replace the algorithms.

The point-aligned public CLI sequence is:

1. Generate the geometry/templates cache using public load_mesh, sample_query_points, make_template_cameras, render_templates and save_onboarding_cache.
2. Run extract-query-visual with explicit layer/tolerance and the original geometry/RGB inputs.
3. Run extract-query-gedi **with the visual NPZ as --onboarding-cache**, so GeDi describes the exact visibility-retained query_points. Its implementation reads query_points from that argument. Passing the raw geometry points can make the original fusion point-alignment check fail.
4. Run fuse-query-features using the corresponding visual/geometric NPZs.
5. Run run-localization with unchanged official target rows/counts and explicit mask paths/NMS.
6. Run the official evaluator on finalized own predictions.

The dataset runner expects `<query-cache-dir>/<dataset>_obj_XXXXXX_query.npz`. It raises for a missing CAD cache; it catches individual candidate exceptions inside the multi-mask path. Any external resumable ledger must distinguish these behaviors and retain all requested targets, not silently count missing caches as complete inference.

## Asset and runtime compatibility

Existing datasets, masks and weights are reused. Public target_cli.py hardcodes rgb/XXXXXX.png and depth/XXXXXX.png. A GT-free compatibility view can expose original JPEG/TIFF/PNG bytes under those names: Pillow detects actual formats from file content. If a transcode is ever necessary, decoded pixel equality must be verified. Original files remain untouched. T-LESS/HB native test_primesense layout and appropriate models_cad/models families need explicit view aliases. Do not bring scene_gt or GT masks into the inference view.

Public DinoExtractor uses a local source checkout with torch.hub.load(pretrained=True), not a direct checkpoint CLI flag. An isolated TORCH_HOME/hub/checkpoints alias can point to the existing exact giant checkpoint to prevent a duplicate download. GeDi source revision and checkpoint are likewise reused. Environment/global installations and old method caches must not be silently changed.

The unchanged public code constructs DINO per target-mask invocation and GeDi per scale. This may make original-source execution slow. A future model-lifetime or resource adapter requires separately documented output-equivalence validation and must not be confused with an already unchanged original execution.

## Evaluation and claim boundaries

The saved downloadable public reference CSVs are the authors' FreeZeV2.1/Accurate predictions. Evaluating them verifies the official evaluator/reference protocol only. It does not show that this repository's base code generated them, nor that our own inference reproduces their 82.1 all-core AR. The base configuration and Accurate's SAR, extra masks and rendered visual rescoring remain distinct.

Own-inference evaluation must retain the exact official BOP19 localization targets and instance counts, including invalid candidates, missing masks and unavailable CAD descriptors. Ground-truth poses are evaluation-only. No test-GT parameter selection is authorized. Public-GT metrics are available locally for five datasets; HB/ITODD inference and output validity can be checked, but their private-GT scores cannot be claimed locally and no seven-dataset aggregate is justified without them.

Source/API inspection establishes the recipe above. It does not establish numerical equivalence of our former implementation, execution success, inference accuracy or published-performance parity. Those claims require their own recorded receipts.