# Retained NIST experiments

Start with the [current pipeline](../README.md). Dated directories preserve the
assumptions and results of each study; an older study is not automatically the
current runtime. Cleanup has not rerun recognition or changed saved scores.

## Classification and CAD alignment

| Study | Purpose |
|---|---|
| [NIST12 appearance study](nist12_qwen_clip_reranking_20261023/analysis/README.md) | Earlier Qwen descriptions, frozen CLIP retrieval and DINO comparison on eight twelve-part scenes. Retained as a comparison, not the current selector specification. |
| [DINO/depth verification](nist12_qwen_clip_reranking_20261023/depth_verification_20261023/README.md) | Native-scale, table-constrained CAD fitting to partial observed depth; geometry-only and DINO fusion results. |
| [Matching-module comparisons](nist12_matching_benchmarks_20261023/README.md) | Shared-crop adaptations of CNOS, SAM-6D and NIDS-Net. These are distinct from complete pipelines. |
| [Full-scene comparisons](nist12_fullscene_benchmarks_20261023/README.md) | Native proposal/matching pipelines, with execution failures and compatibility controls disclosed. |
| [NIST robustness study](nist12_robustness_20261024/README.md) | Retained NIST scope: 220 inputs, 2,720 queries. Accuracy evaluation remains incomplete. Noise and non-NIST cohorts are retired. |

Shared N00–N07 captures and CAD views are in
[`outputs/datasets/nist_task_board/`](../outputs/datasets/nist_task_board/README.md).
The old experiment-directory name does not make CLIP or Qwen a required stage
of the current DINO/depth framework.

## Grasp research after alignment

The current pipeline selects its final grasp by geometric validation and
deterministic priority rules. VLM ranking/selection is excluded. Older VLM
studies below describe historical experiments, not a pending pipeline stage.

| Study | Purpose |
|---|---|
| [Official NIST reference inspection](nist_official_reference_20261029/README.md) | Current source audit: instructions, finished holes, saved native assembly transforms and corrections to vertical waterproof mounting, USB/DSUB locations and M4/M12 hole roles. No grasp or execution trials. |
| [Implemented scene and goal corrections](nist_scene_corrections_20261029/README.md) | Current scene/task references: official mounts/holes, explicit frames, 20 mm pin depth and refreshed mating/protected regions. Static side views; no grasp or robot trial. |
| [Current surface-contact refinement](cad_grasp_refinement_20261029/README.md) | Completed eight-scene study: controlled previous 33/96 placements and 368 valid poses; final refined 35/96 and 495, from 9,600 proposals each. Final set preserves 72 cases and uses 24 round-pin cases after finite-pad witness correction with identical candidate banks. Current callable is `support_pipeline.select_grasp`; no new standalone final wall time. [Final report](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md), [selected contacts and pickup side views](../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html). VLM excluded; pickup/assembly paused. |
| [Preserved assembly-guided fingertip selector](cad_grasp_selection_20261029/README.md) | Previous region-guided sampler and shared corrected task/fingertip helpers. Exact allowed-region origins and endpoint restrictions; fixed inward ray offset and triangle-normal contact policy retained for comparison with the current refinement. |
| [NIST mating-goal CAD review](nist_mating_goals_20261028/README.md) | Earlier side-section diagnostics and user-selected 12 mm depth. Mounting/destination assumptions are superseded by the official source audit; numerical measurements are preserved. |
| [CAD grasp priority selector](cad_grasp_selection_20261028/README.md) | Preserved nominal-goal revision: at most 100 complete poses and priority selection; four of 96 placements selected. Its numerical results remain unchanged and are not the controlled baseline for the corrected-goal refinement. |
| [Sampler comparison and recommendation](cad_grasp_method_comparison_20261026/README.md) | Historical comparison of released CAD contact samplers and modeled Panda checks over 96 saved placements. |
| [CAD geometry](cad_grasp_geometry_20261026/README.md) | Aligned CAD/observed-cloud inputs and geometric validation. |
| [Surface-derived grasps](cad_surface_grasps_20261026/README.md) | Surface contact construction and its limitations. |
| [Grasp regions](cad_grasp_regions_20261026/README.md) | Earlier human/VLM-reviewed grasp-region study. |
| [Runtime antipodal study](runtime_antipodal_assembly_20261023/README.md) | Earlier simulated candidate search and assembly investigation. |
| [GR-ConvNet study](grconvnet_vlm_pickup_20261024/README.md) | Earlier depth-based grasp proposals and CAD refinement. |
| [Observed-held assembly](observed_held_assembly_20261025/README.md) | Earlier simulated held-part sensing and transport. |
| [RoboGround grasp review](roboground_grasp_review_20261024/README.md) | Related-method review and grasp-study context. |

These studies supply or assume a CAD association as documented. Modeled candidate
availability, successful pickup and assembly accuracy are separate outcomes.
No physical robot action is part of cleanup or the default tests.

Likelihood classifiers, their test cases and results, and non-NIST layout
generators were removed. See the [cleanup record](../docs/cleanup_20261027.json).
