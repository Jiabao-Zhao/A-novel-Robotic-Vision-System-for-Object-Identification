# Proposed twelve-product Qwen/CLIP comparison

Completed results and verification are in [analysis/README.md](analysis/README.md).
DINO alone scores 65/96; CLIP top-five retrieval followed by unchanged DINO
selection scores 66/96, fixing one error and introducing none.

Status: user-authorized on 2026-10-23. Use the middle preview: balanced lighting
with original colors. CLIP retrieves five crops, then DINO makes the final
selection. DINO alone is a separate method. The earlier eight-product draft and
its prepared DINO shortlists are not inputs to this experiment.

## Scene inventory and source

Reuse the eight existing twelve-product N00-N07 layouts from
`experiments/nist12_direct_cad_vlm_20261019`, preserving their object poses,
camera, calibration, and native metric meshes in new output copies.

- Round pins: 8, 12, and 16 mm diameters.
- Rectangular pins: nominal widths 8, 12, and 16 mm. Native cross sections are
  8x7, 12x8, and 16x10 mm; they are not all square. All six pins are 50 mm long.
- Large gear, M16 nut, waterproof connector, USB, D-sub, and BNC.

There are 12 products per scene, 96 retrieved-CAD target tasks, 1,152 candidate
comparisons before shortlisting, and 480 comparisons after a top-five shortlist.
These are not the later eight-product J00-J07 layouts. The prior J-scene 58/64
DINO result is not the baseline for this expanded, relit condition.

## Lighting preview and proposed image control

Preview N00 and selected existing CAD directions in three clearly distinct
conditions: original; balanced lighting preserving original colors; and the same
balanced lighting with all parts uniformly gray in both the scene and CAD.
The third condition is an optional appearance control, not lighting alone.

Balanced lighting uses a camera headlight with ambient 0.60, diffuse 0.25 and
specular 0.02, plus three diagonal directional fills with diffuse 0.12 and
specular 0.02 each. Cast shadows remain disabled. No pixel brightness editing
is used. Camera/part transforms must be unchanged. After review, apply one
common approved condition to all eight scene copies and every CAD reference;
do not tune lighting for individual products or successful outcomes.

The user selected original catalog colors consistently in CAD and scenes. Do not
recolor any part. All six pins already share the same source color, so color alone
cannot distinguish their sizes. The preview is not an illumination-robustness test.

## Fixed matching pipeline

1. Recompute pretrained DINOv2 global and foreground-patch descriptors on the
   final common inputs. Retain the existing depth-based foreground preprocessing.
   Use the existing twenty CAD directions per product, with explicit unavailable
   templates where the established foreground-patch rule fails.
2. Keep DINO as an independent all-twelve-crop baseline. It must not generate the
   proposed method's shortlist. Save its unchanged per-candidate score and best-CLS
   corresponding CAD view for the final matcher and the independent baseline.
3. Use local Qwen3.5-9B to generate keywords and descriptive phrases in two
   separate requests for each crop and each referenced CAD image. Use the same
   paper-derived prompts on both image types. Adapt only the named four-view
   opening to a single unnamed object, supply no true names or dimensions, and
   preserve every raw output without manual corrections or content-based retries.
4. Use cached pretrained CLIP ViT-B/32 image/text encoders with unchanged weights.
   Encode text fields without silent truncation. DINO and CLIP embeddings are
   separate representations; do not compare their vectors directly.
5. For each of the twelve crops and each of the target CAD's twenty views, compute
   the equal mean of four CLIP cosine similarities: image-image, text-text,
   crop-image/CAD-text, and crop-text/CAD-image. A crop's retrieval score is the
   maximum across the twenty CAD views. Keep five distinct crops by this CLIP
   score, with no DINO prefilter. Preserve the best-CLIP view as a diagnostic.
6. Select the highest DINO-scoring crop within the CLIP top five, using exactly
   the independent baseline's per-candidate score and corresponding DINO CAD view.
   The best-CLIP and best-DINO views may differ and are both recorded. CLIP scores
   are not mixed into the final DINO score. No fitted weights, global assignment,
   per-case fallback, or outcome-dependent changes.

The source prompts retain COLOR, SHAPE, and MATERIAL. Score a full-attribute arm
and a separate SHAPE-text arm from those identical outputs. The latter still
uses RGB image features and must not be called a geometry-only image method.
The protected active CAD-description prompt remains unchanged.

## Comparisons and report

Report DINO alone and the primary CLIP-full-top-five then DINO method. Include a
separate SHAPE-text retrieval control using identical authored outputs, and
CLIP retrieval-only top-one diagnostics clearly separated from the main methods.
Freeze inputs, prompts, settings and score equations before new model inference.
Save label-free scores and rankings before consulting simulator identity labels.

Report all 96 tasks, top-five recall, per-product correct/wrong/tied/unavailable
counts, within-family wrong-size matches, DINO errors fixed, and previously
correct DINO matches lost. Show failed cases with CAD views, crops, Qwen outputs,
and score components. Missing evidence or ties remain explicit, not guessed.

This is a controlled study using known CADs and reused simulator layouts. It does
not evaluate upstream CAD retrieval, held-out generalization, real images,
duplicate instances, robot execution, or full 6D pose.
