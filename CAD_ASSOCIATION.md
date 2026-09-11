# Training-free CAD-to-observation association

Supply known CAD IDs from `CAD/cad_library.json` to evaluate association separately
from text retrieval. Each target's CAD remains fixed. The pipeline evaluates every
localized candidate with both modalities and saves visual-only, geometry-only,
and fused rankings without confidence, margin, or presence thresholds.

```
semantic target + known CAD ID -> exact library lookup -> fixed CAD
                                         |
complete CAD mesh -> 14 RGB views -> frozen DINOv2-small descriptors
                 -> sampled surface -> local Open3D FPFH descriptors
                                         |
all localized objects -> batch of original RGB crops -> frozen DINOv2 encoder
                      -> raw partial cloud -> local Open3D FPFH
                                         |
all CAD/candidate pairs -> max view cosine + FPFH/RANSAC/ICP
                       -> normalize -> experimental fixed fusion -> rankings
                       -> detect duplicate target selections -> save evaluation JSON
```

Install the association dependencies in your Python environment with
`python -m pip install -r requirements-association.txt`. The existing capture and
planner dependencies are still needed to import/run `main.py`. The first visual
inference downloads the pinned official DINOv2 ViT-S/14 code and pretrained weights
to `outputs/cache/torch_hub`; later runs work from that cache. No training or
object-specific checkpoint is required. `DINO_DEVICE = "auto"` chooses CUDA when
available; set it to `"cpu"` for CPU inference. Rendering uses Open3D CPU ray casting
and requires neither a display nor an OpenGL context.

Set `TARGET_DESCRIPTIONS` and `TARGET_CAD_IDS` in `main.py`, then run
`python main.py`. Use a complete mapping from semantic target descriptions to
their known CAD IDs. `TARGET_CAD_IDS = None` retains the text-retrieval baseline;
results label the selection method so this baseline can be distinguished from
known-CAD experiments. A supplied mapping with missing, null, or unknown IDs fails
instead of silently falling back to text retrieval. `USE_SAVED_RAW_CAPTURE` reuses
saved RGB-D frames. `RUN_CAD_REGISTRATION = False` makes the default run save the
association experiment and stop there.

For association alone on an already-localized scene, use:

```python
import json
from pathlib import Path
from main import associate_targets_with_cad

payload = json.loads(Path(
    "outputs/physical/point_cloud_localization/point_cloud_localization.json"
).read_text(encoding="utf-8"))
# Example library IDs; supply the CAD identities known for your experiment.
known_cad_ids = {"medium gear": "3", "square block": "5"}
results, retrieved_cads, summary_path = associate_targets_with_cad(
    list(known_cad_ids), payload, payload["rgb_path"], payload.get("plane_model"),
    known_cad_ids=known_cad_ids,
)
```

The lower-level `associate_cad_to_candidates(cad_model, localization_payload,
rgb_path, plane_model=None, *, target_description, scene_cache=None, cache_dir=...)`
returns a JSON-compatible result. Pass the same dictionary as `scene_cache` for
all target CADs within one scene, and discard it when that scene is finished.
Candidate IDs, inclusive pixel boxes (`roi` or `bbox_2d_xyxy`), and raw-cloud paths
come directly from localization. No candidates are screened by shape, size, color,
or visual score. Invalid evidence remains null in the candidate row; missing files
fail with their path rather than fabricating features. Every valid crop is queued
into one workspace encoder call, which uses batches of up to eight images. Every
candidate still receives geometric processing, including candidates with invalid
crops. Cached candidates reuse both descriptors across target CADs.

## Scores and rankings

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

The normalized scores share a numeric range; they are **not calibrated probabilities
or directly equivalent evidence**. The 0.5/0.5 sum is an experimental baseline.
No fusion weights are optimized. Raw cosine, RANSAC fitness, final registration
fitness, and observed-to-CAD RMSE remain available for alternative offline
normalization/fusion analysis.

`candidate_ranking` retains every candidate and all its metrics, sorted by fused
score. `rankings.visual`, `rankings.geometry`, and `rankings.fused` contain ordered
object IDs; `predictions` contains the independent argmax for each modality.
Missing scores sort last and remain null. Exact ties use ascending object ID.
Failures with numeric zero fitness remain part of the ranking. Weak scores, close
scores, and disagreement do not trigger abstention. The target status is:

- `ranking_only`: at least one computable fused score; `selected_object_id` is its
  argmax, without a claim of confidence or target presence.
- `no_candidates`: localization returned no objects; no presence claim is made.
- `no_valid_candidates`: none has a computable fused score. Visual-only or
  geometry-only predictions can still be available.

If independent targets select the same object ID, the summary's `resolution` and
each target's `association_set_resolution` become `conflicting`. `conflicts` records
the object ID, target descriptions, and CAD IDs. Independent `selected_object_id`
values, rankings, and scores remain intact for evaluation; all `final_object_id`
handoffs in that set become null. This includes repeated queries for the same CAD.
All targets are evaluated and saved before `main.py` reports the conflict and
stops. There is no reassignment or global matching optimization.

For an optional nonconflicting downstream run, enable `RUN_CAD_REGISTRATION` or
call `register_resolved_associations(..., retrieved_cads=retrieved_cads)` explicitly.
That adapter also rejects conflicts before any registration work. It preserves
the original CAD identity and accepts the `ranking_only` interface. The final
tabletop registrar, planner, and robot controller are unchanged. Existing
`OUTPUT_ROBOT_BASE_POSE` and `RUN_LLM_PLANNER` settings apply only after the optional
registration stage.

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
pairwise registration, fusion/ranking, scene association, and total association.
The first encoder load/download is included in the first visual extraction that
needs it. Scene-cache hits report zero new feature-extraction time; pairwise
registration still runs for every CAD/candidate pair. Totals include file reads
and cache bookkeeping but exclude localization, text CAD retrieval, result JSON
writing, and downstream final registration. All cache and run artifacts are
ignored through the repository's existing `/outputs/` rule.

Every candidate row has `runtime`: crop preparation, `dino_batch_share_time_s`,
FPFH extraction, visual comparison, geometric matching, fusion, and their sum
`accounted_time_s`. The DINOv2 field is an equal share of the measured workspace
batch duration, **not an independently measured per-crop inference latency**.
Cache hits have zero new feature-extraction cost and retain a cache-hit flag.
The candidate sum excludes shared CAD preparation, image reading/hashing, and
ranking overhead. The original measurements for earlier targets are not overwritten
when later targets reuse scene features.

## Evaluation and limits

`vlm_module.py` is unchanged and remains the GPT/Gemini semantic-association
baseline. Its token likelihood and provisional threshold are not used by the
CAD association path. Compare each modality's saved prediction against the known
object ID for the same scene and CAD. Use all candidate rows to inspect score
distributions, failure counts, and runtime. Keep conflicting independent predictions
in accuracy evaluation and report the set conflict separately. The code does not
claim measured accuracy without ground-truth object IDs. Both modalities are always
evaluated exhaustively, including in visual-only or geometry-only analyses.

Canonical CAD views use neutral shading because STL files carry no surface color.
Known CAD IDs bypass text-retrieval errors, but association cannot distinguish
identical CAD geometry by semantic color alone. Partial, planar, or symmetric
observations can support several CAD fits;
high observed fitness is evidence of surface compatibility, not proof of identity.
This is a baseline, not an accuracy claim. Seeded Open3D RANSAC can still vary with
parallel scheduling; use `OMP_NUM_THREADS=1` before starting Python when comparing
repeatability, and record the environment with evaluation results.

Run focused checks with:

```
python -m pytest tests/test_cad_object_association.py tests/test_vlm_module.py tests/test_point_cloud_localization.py tests/test_llm_planner.py -q
```

The tests use real Open3D rendering, local FPFH, and partial-cloud RANSAC/ICP, plus
controlled encoder outputs for cache/ranking/conflict/contract checks without network
downloads. Encoder tests verify frozen parameters, evaluation mode, CPU support,
and disabled gradients.

Primary implementation references:
[DINOv2 pretrained backbones](https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub),
[Open3D global registration](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html).
