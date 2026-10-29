# CAD-Guided NIST Object Identification

Research on identifying manufacturing parts from a requested CAD model and a
single RGB-D observation, then generating grasp candidates from the aligned CAD.
Current scene and CAD generation use NIST Task Board 1.

```mermaid
flowchart LR
    A[Target request] --> B[Cosine-similarity CAD retrieval]
    C[Observed RGB-D] --> D[Depth localization]
    D --> E[RGB crops and partial point clouds]
    B --> F[DINO appearance and CAD/depth verification]
    E --> F
    F --> G[Object match and aligned CAD pose]
    G --> H[Explicit CAD contact regions]
    M[Known assembly pose and CAD task labels] --> H
    H --> I[Bounded region-guided ray candidates]
    I --> J[Mandatory geometry checks]
    J --> K[Geometric priority selection]
    K --> L[One selected grasp or no valid grasp]
```

The pipeline selects the final grasp itself using geometric criteria and
deterministic priority rules. VLM ranking or pose selection is not part of this
pipeline. The result is the best valid sampled grasp under those rules, rather
than a globally optimal grasp or a VLM shortlist.

The component modules and controlled studies exist. The complete
retrieval-to-grasp workflow has not been validated as one autonomous system.
Grasp candidates do not establish physical pickup, holding or assembly success.

## Start here

- [Published results and visual examples](results/README.md): summaries,
  per-object errors, detailed grasp decisions and selected figures.
- [Cloud analysis guide](docs/cloud_analysis.md): reading order, useful analysis
  questions and the limits of a GitHub checkout.
- [Experiment index](experiments/README.md): current and historical study scopes.
- [Pipeline contract](simulation/pipeline_protocol.md): units, frames and evidence rules.
- [Official NIST board reference](docs/nist_task_board_1_reference.md): fabrication
  dimensions, native assembly transforms and corrections to our receiver layout.

## Files used by each stage

| Stage | Main files |
|---|---|
| CAD library and cosine retrieval | [CAD/cad_library.json](CAD/cad_library.json), [helper_function.py](helper_function.py) — `CADRetrieval` accepts an embedding function and ranks CAD text metadata |
| Calibrated RGB-D input | [camera_capturing.py](camera_capturing.py), [simulation/libero_sensor.py](simulation/libero_sensor.py), [simulation/libero_io.py](simulation/libero_io.py) |
| Depth localization and RGB crops | [point_cloud_localization.py](point_cloud_localization.py) — table/cluster localization; [visualize_localization_projection.py](visualize_localization_projection.py) — projection, masks and crop bounds |
| DINO global and foreground-patch features | [scripts/cad_descriptors.py](scripts/cad_descriptors.py), [scripts/sam6d_patch_matching.py](scripts/sam6d_patch_matching.py), [run_dino.py](experiments/nist12_qwen_clip_reranking_20261023/run_dino.py) |
| CAD/depth fitting and verification | [scripts/surface_verification.py](scripts/surface_verification.py), [geometry.py](experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/geometry.py) — native-scale support hypotheses, table-plane refinement, silhouette/depth agreement |
| Tested DINO/depth fusion | [select_results.py](experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/select_results.py) — historical experiment selector |
| Reusable table-constrained alignment | [CADPointCloudRegistration.py](CADPointCloudRegistration.py) |
| Corrected scene and task references | [Implemented NIST corrections](experiments/nist_scene_corrections_20261029/README.md): full receiver transforms, finished holes, 20 mm pin goals and mating/protected regions |
| Grasp inputs and finite-hand geometry | [inputs.py](experiments/cad_grasp_geometry_20261026/inputs.py), [geometry_kernel.py](experiments/cad_grasp_geometry_20261026/geometry_kernel.py), [hand_model.py](experiments/cad_grasp_geometry_20261026/hand_model.py) |
| Current grasp generation and selection | [Modeled finger-end contacts](experiments/cad_fingertip_contacts_20261029/README.md), [pipeline.py](experiments/cad_fingertip_contacts_20261029/pipeline.py): CAD region rays, static first-contact fitting on original inward finger hull faces, complete-hand checks and deterministic priority selection |
| Preserved surface-contact baseline | [Surface-contact refinement](experiments/cad_grasp_refinement_20261029/README.md): shared one-degree contact rule, verified round-pin side normals and finite-pad witness correction; the preserved 35/96, 495-valid-pose comparison |
| Preserved region-guided comparison | [Assembly-guided fingertip selector](experiments/cad_grasp_selection_20261029/README.md): the preceding allowed-patch sampler and fingertip/contact calibration; preserved for the controlled comparison |
| NIST scenes and reference views | [simulation/nist_task_board_1.py](simulation/nist_task_board_1.py), [simulation/nist_capture.py](simulation/nist_capture.py), [simulation/nist_cad_views.py](simulation/nist_cad_views.py) |

The historically named `nist12_qwen_clip_reranking_20261023` folder still houses
tested DINO/depth code. Its frozen CLIP-top-five study is a comparison with its
own assumptions; the folder name does not make Qwen or CLIP mandatory in the
current framework. Cleanup did not convert that runner into a new benchmark.

## Repository structure

```text
CAD/             Retrieval catalog, source models and asset provenance
simulation/      NIST scene/capture helpers and shared simulator backend
scripts/         Descriptors, geometry checks and capture utilities
experiments/     Our study code, configurations, protocols and documentation
results/         Selected exact copies of saved evidence for GitHub/cloud review
tests/           Shared regression tests
docs/            Cloud handoff, current protocol context and cleanup record
outputs/         Mostly local generated data, caches, captures and galleries
```

The twenty catalog meshes are included at their existing
`outputs/simulation/nist_task_board_1/assets/meshes/` paths, with conversion
[provenance](CAD/NIST_ASSET_NOTICE.md). Other `outputs/` content is local except
the dataset README, small cleanup receipts, and the reviewed
[October 28 grasp run](outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/README.md) explicitly allowed by
[.gitignore](.gitignore).

Local data locations:

- `outputs/datasets/nist_task_board/`: shared captures, camera metadata,
  crops and CAD reference views.
- `outputs/datasets/nist_task_board/legacy_workspace_inputs/`: required legacy
  NIST scene, camera/state metadata and USB CAD assets relocated from the retired
  C: experiment workspace.
- `outputs/simulation/`: NIST board and five/seven-part figure variants.
- `outputs/cad_grasp_*/`: complete grasp-study outputs and galleries.
- `outputs/cache/`: installed models, baseline dependencies and tools.
- `outputs/physical/` and `outputs/recovered/`: physical captures/calibration.

LIBERO/robosuite remain simulator and Panda dependencies. Their installed robot
assets and textures do not enable additional object-layout generators.

## Results available on GitHub

The [October 28 grasp-selection run](outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/README.md)
is published at its original output paths, with all 96 case decisions, candidate
diagnostics, derived CAD validation meshes, audits and eight gallery sheets.
Its publication manifest records the saved file hashes. This is a run-specific
exception in `.gitignore`; other generated output trees remain local by default.

The [evidence bundle](results/README.md) contains 83 exact-copied files,
about 22.25 MiB: classification decisions, per-object and error tables,
depth pin-case illustrations, current grasp decisions and eight contact-atlas
images. [bundle_manifest.json](results/bundle_manifest.json) records each
original path, byte count and SHA256. No new inference, fitting or accuracy
evaluation was performed to publish these copies.

[.gitattributes](.gitattributes) preserves result and pinned experimental bytes
across platforms. Raw captures, feature arrays, exhaustive fit/pose banks,
downloaded packages, model weights and temporary files are excluded from Git,
apart from the explicitly reviewed run evidence described above.
Copied historical reports keep their original links; the evidence index points
to the figures actually included.

Keep these claims separate:

- Recognition studies usually supply a known CAD, so their accuracy does not
  measure upstream text-to-CAD retrieval.
- Matching-module comparisons share crops; native full-scene methods generate
  their own proposals.
- Grasp studies assume correct CAD/crop correspondence and saved observed-depth
  alignment. The [October 28 selector](experiments/cad_grasp_selection_20261028/README.md)
  returns one grasp using explicit priorities after geometry checks. Its CAD-derived
  eight-scene trial selected grasps for 4 of 96 placements; the report records
  the pad-center/table and mesh-normal limitations. The pipeline's geometric
  selector makes the final decision; there is no subsequent VLM selection step.
  Pickup and assembly execution remain paused. The older
  [sampler recommendation](experiments/cad_grasp_method_comparison_20261026/recommendation.md)
  describes its historical downward-only, reduced-pair study.
- [Scene and goal corrections](experiments/nist_scene_corrections_20261029/README.md)
  implement vertical waterproof mounting, corrected USB/DSUB locations, M4/M12
  hole assignments, bolt underside mounting and finished round-hole geometry.
  Native/STL frame mappings and explicit CAD datums transfer task references;
  they are not physical seating validation. The current user-selected pin-tip depth is
  **20 mm below the board top**, recorded in the
  [task requirement](experiments/nist_official_reference_20261029/task_requirements.json).
  Earlier 12 mm review outputs are historical and have not been regenerated.
  The current selector uses corrected task inputs and a calibrated fingertip
  frame. Explicit positive CAD labels intersect the complement of protected
  insertion/seating regions before area-weighted sampling; both contacts must
  remain allowed before the orientation bank and 100-pose cap. Rays and collision
  checks retain the complete CAD. These training-free annotations are our local
  adaptation, not physical grip validation. The
  [preserved surface-contact refinement](experiments/cad_grasp_refinement_20261029/README.md) uses
  the same one-degree normal-alignment tolerance when proposing and checking
  contacts. A verified cylinder-side model applies to the three round pins;
  BNC does not pass that model's verification and retains triangle normals.
  Rays start on the CAD surface, avoiding the old fixed 0.5 mm inward jump.
  All original part triangles and complete palm, finger and pad geometry remain
  in collision checks. Independent failure categories record every available
  check outcome; one pose may fail several categories.
  The completed eight-scene comparison has **33/96 placements and 368 valid
  poses** for the controlled previous construction, versus **35/96 placements
  and 495 valid poses** in the final refined set, from 9,600 proposals per set.
  Four round-pin placements gained a valid grasp and two waterproof placements
  lost one. The final set combines 72 unchanged cases with 24 round-pin cases
  rechecked after correcting how finite pad contact is represented; the
  regenerated round-pin candidate banks were identical. Original trials remain
  preserved. This derived set has no separately measured full-run wall time.
  See the [final report](outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md)
  and [selected grasps with pickup side views](outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html).
  These are modeled grasp availability counts. VLM selection is excluded;
  pickup and assembly execution remain paused.
  Historical results and selector sources remain unchanged; the old scene
  source is preserved in a hash-checked snapshot before updating the active file.
- The current [finger-contact study](experiments/cad_fingertip_contacts_20261029/README.md)
  fits first contact on the saved finger ends rather than extending pad boxes.
  All original hand solids remain present. Its full eight-scene trial selected
  grasps in **37/96 placements**, with **75 valid poses from 9,600 proposals**,
  in **230.30 s**. Against the preserved 35/96 baseline, 13 placements gained a
  selected grasp and 11 lost one: a modest net gain of two, with fewer valid
  candidates because the contact model changed. The round 8 mm pin still has
  no selected grasp in its eight placements. The coarse collision hull bridges
  a groove in the detailed visual finger; physical contact accuracy is unverified.
  See the [report](outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/report.md)
  and [selected contacts with pickup side views](outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/gallery/index.html).
  These are geometric results; VLM selection is excluded and pickup/assembly
  execution remain paused. The derived 35/96 baseline has no standalone wall
  time, so these records do not establish a measured speedup against it.
- The retained robustness scope is 220 NIST inputs and 2,720 queries.
  Classification accuracy evaluation remains incomplete.

Preserve native metric scale, camera calibration, unavailable fits and ties.
State the searched pose degrees of freedom. Single-view partial clouds and
symmetric parts can leave orientation unresolved. Simulator identities and
poses are audit-only, never recognition/alignment inputs. Scores are uncalibrated.

## Local execution

The commands below target the existing local WSL environment **with its cached
assets and dependencies**. This is not a complete fresh-cloud installation recipe.

```bash
PY=/home/jiabao/.venvs/lerobot-libero/bin/python
$PY -B -m pytest -q -p no:cacheprovider tests
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY -m scripts.capture_nist_task_board
```

Capture writes simulated output. Tests/runners requiring omitted captures, fit
banks or installed models cannot be reproduced from source and summaries alone.
`requirements-association.txt` lists only part of the dependency stack:
geometry also uses SciPy, simulation uses MuJoCo/robosuite/LIBERO/LeRobot,
and optional physical capture uses RealSense. Historical description studies
also require their declared local model backend.

Cloud analysis can inspect source, CAD meshes and published evidence without
claiming a rerun. Do not download weights, install environments, start paid APIs
or command a physical robot as a side effect of analysis or default checks.

## Cleanup and historical evidence

The October 27 cleanup removed retired VLM likelihood experiments and results,
plus non-NIST layouts. Shared NIST inputs were extracted before deletion.
Current grasp work, native CAD assets, installed dependencies and physical
calibration/captures were preserved.

Path migrations did not recompute saved numerical outcomes. Original source
inventories describe the original checkout; they do not certify reorganized
sources as the original run. See [cleanup_20261027.json](docs/cleanup_20261027.json).

The separate C: experiment workspace was also removed after its required inputs
and editable presentation assets were copied to D: and hash-verified. Saved
results keep their historical path strings; maintained readers resolve that
specific retired workspace prefix to the D: dataset location. Scene XML changes
are restricted to asset-path attributes and recorded in a verified migration
receipt. See [the C: cleanup record](docs/c_workspace_cleanup_20261027.json).
