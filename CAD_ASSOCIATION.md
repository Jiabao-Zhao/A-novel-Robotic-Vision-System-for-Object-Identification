# Training-free CAD-to-observation association

`main.py` now retrieves one CAD per semantic target from `CAD/cad_library.json`
before selecting any workspace object. It preserves that CAD record through
association and final registration. The existing text retrieval, RGB-D localization,
tabletop pose estimator, planner, and robot controller remain in use.

```
semantic target -> text CAD retrieval -> fixed CAD ID
                                         |
complete CAD mesh -> 14 RGB views -> frozen DINOv2-small descriptors
                 -> sampled surface -> local Open3D FPFH descriptors
                                         |
each localized object -> original RGB crop -> same frozen DINOv2 encoder
                      -> raw partial cloud -> local Open3D FPFH
                                         |
all CAD/candidate pairs -> max view cosine + FPFH/RANSAC/ICP
                       -> normalize -> fixed fusion -> rank -> resolve
                       -> same CAD + selected raw cloud -> existing registration
```

Install the association dependencies in your Python environment with
`python -m pip install -r requirements-association.txt`. The existing capture and
planner dependencies are still needed to import/run `main.py`. The first visual
inference downloads the pinned official DINOv2 ViT-S/14 code and pretrained weights
to `outputs/cache/torch_hub`; later runs work from that cache. No training or
object-specific checkpoint is required. `DINO_DEVICE = "auto"` chooses CUDA when
available; set it to `"cpu"` for CPU inference. Rendering uses Open3D CPU ray casting
and requires neither a display nor an OpenGL context.

For the full existing application, set `TARGET_DESCRIPTIONS` and the capture/output
constants in `main.py`, then run `python main.py`. `USE_SAVED_RAW_CAPTURE` reuses
saved RGB-D frames. The existing `OUTPUT_ROBOT_BASE_POSE` setting controls the robot
connection; `RUN_LLM_PLANNER` controls the planner call.

For association alone on an already-localized scene, use:

```python
import json
from pathlib import Path
from main import associate_targets_with_cad, register_resolved_associations

payload = json.loads(Path(
    "outputs/physical/point_cloud_localization/point_cloud_localization.json"
).read_text(encoding="utf-8"))
results, retrieved_cads, summary_path = associate_targets_with_cad(
    ["white gear", "red block"], payload, payload["rgb_path"], payload.get("plane_model"),
)
# Optional downstream handoff, after every requested association resolves:
if all(result["resolution"] == "automatic_match" for result in results):
    records = register_resolved_associations(
        results, payload, payload.get("plane_model"), retrieved_cads=retrieved_cads,
    )
```

The lower-level `associate_cad_to_candidates(cad_model, localization_payload,
rgb_path, plane_model=None, *, target_description, scene_cache=None, cache_dir=...)`
returns a JSON-compatible result. Pass the same dictionary as `scene_cache` for
all target CADs within one scene, and discard it when that scene is finished.
Candidate IDs, inclusive pixel boxes (`roi` or `bbox_2d_xyxy`), and raw-cloud paths
come directly from localization. No candidates are screened by shape, size, color,
or visual score. Invalid candidate evidence remains in the ranking and causes
deferral; missing files fail with their path rather than fabricating features.

## Scores and decisions

- Visual raw metric: maximum cosine across all canonical CAD views. Both renders
  and original RGB crops receive the same 224-pixel letterboxing and ImageNet
  channel normalization, followed by the frozen encoder's CLS descriptor.
- Visual score: `(cosine + 1) / 2`.
- Geometry: metric voxel downsampling, local 33-dimensional Open3D FPFH, local
  feature correspondence RANSAC, and point-to-point ICP. FPFH is never averaged
  into a global descriptor. The source is the partial observation and the target
  is the complete sampled CAD surface.
- Geometry score: observed registration fitness, i.e. correspondence count divided
  by the number of downsampled observation points. The correspondence radius is
  `1.5 * CADRegistrationConfig.voxel_size_m` (7.5 mm by default). RANSAC fitness,
  correspondence counts, inlier RMSE, full observed-to-CAD RMSE, and transform
  direction are retained. Unseen CAD surfaces do not reduce observed fitness.
- Fusion: `0.5 * visual_score + 0.5 * geometry_score`. Set `FUSION_METHOD` to
  `"geometric_mean"` to use `sqrt(visual_score * geometry_score)` instead.

All fusion weights and decision thresholds are explicit module constants in
`cad_object_association.py` and are recorded with each result. They are provisional,
not calibrated identity probabilities, and were not fitted to a test dataset.
Keep them fixed for final evaluation. The initial gate requires fused score >=
0.65, visual score >= 0.60, geometry score >= 0.35, and fused margin >= 0.04.
Strong modality disagreement also defers. It reports:

- `automatic_match`: sufficient evidence and separation; both `selected_object_id`
  and the legacy `final_object_id` contain the winner.
- `ambiguous`: close ranking, conflicting modalities, weak/invalid evidence, or
  all geometric registrations failing. Both selected IDs are null.
- `not_present`: no localized candidates, or every candidate has visual score
  below 0.60 and geometric score below 0.15 with valid evidence. Registration
  failure alone is insufficient to conclude absence.

`proposed_object_id` and `scores` describe the top-ranked hypothesis even when the
selected IDs are null. `main.py` stops registration and downstream calls if any
requested target is unresolved or absent. Pairwise association uses rigid
RANSAC/ICP; its transform is diagnostic. The existing final registration still
uses the supplied table plane and tabletop constraints, without retrieving CAD
again. CAD ID, semantic target, and object ID are included in registration output.

## Caching, timing, and output

CAD descriptors are stored under `outputs/cache/cad_association`. The keys include
mesh content, preprocessing version, encoder revision, rendering settings, Open3D
version, and relevant geometry configuration. Both sampled/downsampled CAD points
and their local FPFH arrays are persisted. Scene cache keys include original image
content, object ID, box, raw cloud content, and feature settings. Changing the
image, cloud, CAD, or feature settings invalidates the corresponding cache entry.

Each target produces `outputs/physical/cad_association/target_NNN.json`, containing
all candidates, all raw and normalized metrics, fusion scores, resolution reason,
and runtime measurements. The small `semantic_associations.json` summary adapts
the planner's existing `associations` / `final_object_id` interface. The VLM
baseline's outputs remain separate.

Runtime records distinguish CAD preprocessing (zero on a descriptor-cache hit),
CAD cache loading, workspace DINOv2, workspace FPFH, visual comparisons, each
pairwise registration, fusion/resolution, scene association, and total association.
The first encoder load/download is included in the first visual extraction that
needs it. Scene-cache hits report zero new feature-extraction time; pairwise
registration still runs for every CAD/candidate pair. Totals include file reads
and cache bookkeeping but exclude localization, text CAD retrieval, result JSON
writing, and downstream final registration. All cache and run artifacts are
ignored through the repository's existing `/outputs/` rule.

## Evaluation and limits

`vlm_module.py` is unchanged and remains the GPT/Gemini semantic-association
baseline. Its token likelihood and provisional threshold are not used by the
CAD association path. For later ablations, rank the same saved candidate rows by
`visual_score` (DINOv2 only), `geometry_score` (FPFH/registration only), or
`fused_score` (multimodal). Evaluate those alongside the VLM baseline with the same
localized scenes and target descriptions; report accuracy, deferrals, and runtime.
The production decision always evaluates both modalities exhaustively.

Canonical CAD views use neutral shading because STL files carry no surface color.
The existing text retrieval must resolve the intended CAD: association cannot
repair a wrong CAD choice or distinguish identical CAD geometry by semantic color
alone. Partial, planar, or symmetric observations can support several CAD fits;
high observed fitness is evidence of surface compatibility, not proof of identity.
This is a baseline, not an accuracy claim. Seeded Open3D RANSAC can still vary with
parallel scheduling; use `OMP_NUM_THREADS=1` before starting Python when comparing
repeatability, and record the environment with evaluation results.

Run focused checks with:

```
python -m pytest tests/test_cad_object_association.py tests/test_vlm_module.py tests/test_point_cloud_localization.py tests/test_llm_planner.py -q
```

The tests use real Open3D rendering, local FPFH, and partial-cloud RANSAC/ICP, plus
controlled encoder outputs for cache/decision/contract checks without network
downloads. Encoder tests verify frozen parameters, evaluation mode, CPU support,
and disabled gradients.

Primary implementation references:
[DINOv2 pretrained backbones](https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub),
[Open3D global registration](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html).
