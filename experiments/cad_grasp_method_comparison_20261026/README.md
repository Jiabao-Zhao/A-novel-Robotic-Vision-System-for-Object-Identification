# Why the nine placements lost every grasp, and comparison with released samplers

**Current choice: `one_budget512` + the tested straight-down Panda conversion, finger/part/table checker, and one-degree local contact alignment.** The user's 27 October clarification retains well-aligned round-side grips through local tangent contact. The true one-degree revalidation retains 804 families and modeled grasps in **89/96 placements**. The former 95/96 result used fifteen degrees and is preserved below as reference evidence. Holding quality and assembly precision have not been compared. See the [plain-language decision](recommendation.md).

The [current contact atlas](../../outputs/cad_grasp_method_comparison_20261026/contact_atlas/20261027T152602Z/index.html) matches the user's reference: eight white 3-by-4 top-down CAD sheets with colored retained-pair lines and endpoints. It shows all twelve parts per scene, including zero-result tiles. Height is not shown. The lines join original sampled CAD points and follow the final gripper closing axis in XY; they do not connect independently selected first-contact witnesses. No new rank or VLM selection is created. The prior [3D finger viewer](../../outputs/cad_grasp_method_comparison_20261026/retained_pose_gallery/20261027T150152Z/index.html) and diagnostic galleries remain preserved.

## One-degree contact-alignment revalidation, 27 October

[Run 20261027T152238Z](../../outputs/cad_grasp_method_comparison_20261026/aligned_contacts/20261027T152238Z/summary.json) reruns the unchanged native-CAD/Panda checker on all **7,826** frozen `one_budget512` primary hand poses, changing only the contact-normal limit from fifteen degrees to one degree in isolated worker memory. It reevaluates all simultaneous first-contact surfaces and selects an aligned witness for each pad. It retains **1,608 hand alternatives, 804 families and candidates in 89/96 placements**. There is no new pose sampling, pose adjustment, centering, classification, polygon overlap, learned score, VLM call or execution.

The five focused alignment tests pass, and the independent audit passes **169,111 checks with zero failures**. The largest reconstructed selected-contact normal error is **0.9833°**. All source poses, pairs, openings and frozen evidence are preserved. Seven zero cases retain their exact rejection counts in the summary. This inherited pool still uses the previous 5 mm reduction before hand validation; it does not establish that no other aligned grip exists.

Simply screening the old saved witnesses would give 73/96 placements. The correct contact revalidation gives 89/96 because a pad can touch several native faces simultaneously and another face may satisfy the new angle condition. These are different tests. The 1°, 3°, 5° and 15° saved-witness sensitivity results are isolated in the run's `saved_witness_diagnostics` directory and are not labeled full reruns.

For a genuinely flat contacted surface, near-parallel pad contact is the intended condition. On a round side, the same normal test describes local tangent contact, consistent with the user's clarification. This one-degree setting is a controlled numerical tolerance, not a validated physical displacement limit. A selected normal does not prove full-pad contact area, balanced force or zero movement during closing. See the [contact-alignment explanation and primary references](contact_alignment_note.md).

This trial starts after successful CAD alignment. It uses the same saved observed-depth CAD fits for N00–N07: **12 parts per scene, 96 placements**, not the earlier single-scene fourteen-part pickup trial. Correct CAD/crop correspondence is supplied. No simulator object pose, learned ranking, VLM request, robot move, pickup, assembly, model-weight download or physical robot is used.

The zero results in the [previous surface-derived experiment](../cad_surface_grasps_20261026/README.md) were **candidate rejections by added mathematical rules**, not nine failed pickup attempts. All 66 existing hand proposals in those nine placements pass the unchanged modeled Panda, target and table checks when the extra contact screens are omitted.

## The exact shared-contact-region rule was ours

Our added rule clipped the native CAD triangles to a 0.025 mm band around each finger's first-contact plane, restricted them to the finite inner pad, projected them into the two directions across the pad, and required the two projected regions to overlap. The later fitted-cylinder checks were evaluated at a pair of points constructed in that overlap. This is **our proposed rule**, not a requirement established by Wan's paper or its reviewed `one` implementation. We have no physical evidence that it is necessary for holding these parts.

There is related prior art, but it is different. [GPG's released antipodal check](https://github.com/atenpas/gpg/blob/master/src/gpg/antipodal.cpp) uses surface samples near each finger, a five-degree normal cone, overlapping coordinate bounds, and at least twelve supported samples per side. [Its caller](https://github.com/atenpas/gpg/blob/master/src/gpg/grasp_set.cpp) supplies a **3 mm** extremal band. Our band was 120 times narrower, and exact polygon overlap is stronger than overlapping bounding intervals. GPG's 3 mm is a support-selection band, not permitted finger penetration or pad compression. Increasing our band to 3 mm does not reproduce GPG.

For flat parts our revision also added a one-degree contacted-normal screen. For round regions it required fitted-cylinder membership and agreement between the constructed contact points, the fitted radial directions and native mesh sectors. These are further local additions. The generic unchanged checker already verifies actual contacted normals within fifteen degrees of the finger closing direction, finite inner-pad contact, near-simultaneous two-sided contact, finger travel, target interference and table clearance.

## Fixed-pose ablation

Run: [20261026T202322Z](../../outputs/cad_grasp_method_comparison_20261026/ablation/20261026T202322Z/report.md). All 518 previous hand proposals are frozen; the CAD, hand solids, poses, table and core checks are unchanged.

| Added contact checks | Passing hand poses | Placements with at least one |
|---|---:|---:|
| Original strict rule, exact replay | 359 | 87/96 |
| Same construction, band widened to 0.05 mm | 351 | 87/96 |
| Same construction, band widened to 0.10 mm | 353 | 87/96 |
| Same construction, band widened to 0.25 mm | 353 | 87/96 |
| Actual first-contact witnesses, shared region required | 363 | 88/96 |
| Same actual witnesses, shared region requirement removed | 389 | 92/96 |
| Added surface-normal screen alone | 415 | 96/96 |
| Unchanged finite-hand/target/table checker alone | 481 | 96/96 |

The clean test of the shared-region requirement is **363 versus 389**, recovering four placements: N00 BNC and N01/N03/N07 large gears. Both columns use identical actual first-contact points for subsequent cylinder checks. Comparing 359 directly with 389 would also change the evaluation points and would overstate the effect of removing overlap alone.

The wider-band results are not monotonic: changing the overlap can change the chosen representative points and therefore the later radial result. This exposes fragility in our construction; a wider tolerance is not a reliable fix. A separate [3 mm sensitivity test](../../outputs/cad_grasp_method_comparison_20261026/ablation/20261026T202730Z_band3000/summary.json) rechecks only the nine missing placements and their 66 frozen proposals. It restores two jaw assignments of one N00 BNC **end grip**, while the other eight placements remain empty. It is not a full 96-placement test or a GPG reproduction.

| Original missing placement | Existing proposals | First strict rejection: shared-region construction | Added normal screen | Added radial check |
|---|---:|---:|---:|---:|
| N00 BNC | 8 | 6 | 2 | 0 |
| N01 BNC | 8 | 4 | 4 | 0 |
| N03 BNC | 8 | 4 | 4 | 0 |
| N04 BNC | 6 | 6 | 0 | 0 |
| N06 BNC | 8 | 4 | 4 | 0 |
| N01 large gear | 8 | 8 | 0 | 0 |
| N03 large gear | 8 | 6 | 2 | 0 |
| N05 large gear | 4 | 0 | 0 | 4 |
| N07 large gear | 8 | 4 | 4 | 0 |

These are first-rejection counts, not mutually independent causes. N05 gear passes when the radial check uses actual first-touch witnesses instead of our constructed overlap points. Four BNC cases remain rejected under that change because of the fitted-cylinder/native-sector or flat-normal conditions. Every proposal in this table passes the unchanged core checker, so these zero-option results do not establish that the hand cannot reach the part.

## What released software actually runs here

| Component | Actual upstream code executed | Declared local adaptation |
|---|---|---|
| [`one`](https://github.com/wanweiwei07/one), commit `ec1bdae83ec33ca6af9de42904d9851647fe012c` | Original NumPy surface-area sampling, normal ray casts and antipodal contact kernel; original normal and physical-gap masks | Default 20 mm spacing versus an exposed-density setting requesting 512 surface samples; bounded ray batches; common downward Panda conversion |
| [EmbodiChain GraspKit](https://github.com/DexForce/EmbodiChain), pinned commit `bb2b65e0de1ccc486e022e57a6a56d8ddd78e969` | Original `AntipodalSampler.sample`, 512 Fibonacci rays and its own antipodal filtering | Existing utility-import adapter; same common Panda conversion |
| [Fabrica](https://github.com/yunshengtian/Fabrica/blob/main/planning/robot/util_grasp.py), commit `215a30fe51e59299588a5b5e417a9cb934fb393e` | Original `filter_sparse_pairs` function | Canonical finger labels and **5 mm** endpoint spacing applied to either sampler; this radius is our comparison setting |

These are **original contact samplers plus a common adapter**, not complete upstream robot-system reproductions. No upstream learned score, arbitrary roll expansion, IK, robot motion, complete Fabrica sampler, assembly optimizer or controller runs. Pinned source and licenses are retained in `vendor/`; GraspKit's pre-existing pinned source is retained in the earlier experiment.

The intrinsic [36-task cache](../../outputs/cad_grasp_method_comparison_20261026/intrinsics/20261026T203042Z/records.json) stores raw contact pairs in native CAD coordinates: 12 CADs × three configurations. No table, hand pose, fitted primitive or overlap rule is used in those caches. [Independent surface verification](../../outputs/cad_grasp_method_comparison_20261026/intrinsics/20261026T203042Z/audit_verified.json) passes all 36 records. The maximum GraspKit endpoint discrepancy from the native surface is 0.283 micrometers; `one` is at floating-point roundoff. Earlier Open3D nearest-triangle diagnostic failures are preserved: independent double-precision checks resolved incorrect adjacent-triangle queries without altering samples.

The common adapter transforms cached pairs using the same saved observed CAD fit. It keeps the approach straight down, allows a declared small adjustment of the closing line to horizontal, and checks the actual finite Panda fingers against target triangles and table. Two controls separate conversion effects:

- **Midpoint:** place the finger-pad center at the sampled pair midpoint.
- **Panda/table:** additionally raise the hand only as required for table clearance and center the hand using its modeled first finger contacts.

Each has an at-most-one-degree and an at-most-fifteen-degree horizontal-projection condition. This one-degree control concerns the **sampled line's tilt**, not the removed one-degree flat-surface-normal filter. Both finger assignments are tested; no early stop at the first passing pose occurs.

The Fabrica function suppresses a nearby pair when both corresponding endpoints are within 5 mm. It keeps the first representative and does not assess quality. In this comparison it runs before full-hand validation, so a rejected representative may suppress a feasible neighbor. A missing result after this reduction is not proof that the raw sampler found no useful pair. No mandatory twelve-grasp cap or invented weighted quality score is added.

## Completed comparison across all eight scenes

Run: [20261026T203838Z detailed results](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/report.md), [machine-readable summary](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/summary.json), and [independent audit](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/comparison_audit.json).

| Contact sampler | Default midpoint conversion, 15° projection | Panda/table conversion, 1° projection | Panda/table conversion, 15° projection | Original nine recovered with Panda/table + 15° |
|---|---:|---:|---:|---:|
| `one`, original 20 mm density | 29/96 | 48/96 | 58/96 | 1/9 |
| `one`, density requesting 512 samples | 40/96 | 88/96 | **95/96** | **9/9** |
| GraspKit, 512 rays | 40/96 | 79/96 | **83/96** | **9/9** |

Counts are placements with at least one modeled passing pose. The table-aware hand conversion matters substantially; a contact pair is not already a complete calibrated Panda grasp. The 1° and 15° conditions matter too: all-nine recovery occurs under the latter. This is not evidence that all nine have a nearly exact horizontal source pair, or that projected grips leave the object unmoved. Final hand approaches remain exactly downward in every condition.

Under Panda/table + 15°, `one` budget512 retains 2,108 passing hand alternatives in 1,054 pair families, versus GraspKit's 2,239 passing alternatives in 1,120 families and default `one`'s 280 alternatives in 140 families. One GraspKit family has only one valid finger assignment. Families range from zero to 39, 42 and 11 per placement respectively. This general sampling baseline is less compact than our analytic-region trial; its larger count is not presented as an improvement in quality. No cap or score hides the remaining redundancy.

The budget512 `one` case without candidates is N06 8 mm round pin. GraspKit misses all eight USB placements and five 8 mm round-pin placements. The default density supplies only three raw pairs for the 8 mm pin and twenty for the gear; it yields no gear grasp in any scene under these conversion settings. These are sampler-plus-adapter results, not proofs of impossibility. The separate sparsity sensitivity examines selected remaining cases without altering this frozen comparison.

| Sampler | Raw CAD sampling, once for all twelve CADs | Median shared preparation + Panda/table checks per placement |
|---|---:|---:|
| Default `one` | 1.608 s | 0.149 s |
| Budget512 `one` | 32.116 s | 1.001 s |
| GraspKit512 | 0.336 s | 0.733 s |

The second timing column is a sum of twelve intrinsic task times, not elapsed wall time. The last column is the instrumented preparation and selected adapter component inside a two-adapter, four-worker CPU experiment; it excludes the other conversion control but is not isolated deployment latency. It excludes pose estimation, VLM, pickup and rendering. The previous strict trial's 0.702 s median includes a different generator and cannot establish a fair speed advantage here. Cache reuse is valid because these are native-CAD samples transformed by the current observed pose; this does not mean the old hand poses are reused without current table/target checking.

The independent full audit passes **558,735 checks across all 288 sampler/placement records**, including native-to-world transformation, original pair suppression, all four protocol counts, both finger assignments, centering arithmetic, native CAD contact witnesses, finite pads and table clearance. All 36 raw caches, original source and protected evidence hashes pass. This audit does not repeat the full swept-triangle kernel or establish frictional holding or assembly success.

The [alignment diagnostics](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/quality_diagnostics.md) distinguish sampled-line tilt from actual contacted-surface alignment. Both dense samplers offer an option within 3° at both selected first-contact normals in every one of the nine recovered placements. Budget512 `one` offers a nearly zero-error gear option in each of the four recovered gear cases; its best BNC options range from 0.753° to 1.838°. Other accepted gear alternatives reach approximately 10°. These are descriptive model measurements, not a new acceptance threshold or proof that the entire pad lies flat, the part will not turn while closing, or assembly will be precise.

The [interactive gallery](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/gallery/index.html) shows the original nine zero-option placements under all three sampler settings: all 200 retained primary families and 400 valid finger assignments. Pink dashed marks show the original sampled pair; separate green dots show actual first contacts; dark-green arrows show native surface normals; the black line shows the actual closing axis. The static [BNC sheet](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/gallery/BNC_Male_examples.png) and [gear sheet](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/gallery/Gear_Large_examples.png) show one unranked example per cell. These are grasp views, with no final assembly view.

## Pair-reduction sensitivity: three selected cases

The [follow-up](../../outputs/cad_grasp_method_comparison_20261026/sparsity_followup/20261026T204553Z/summary.json) changes only the original Fabrica function's spacing parameter. It reuses the same raw samples, table-aware Panda conversion, downward approach and checker. Each 5 mm control exactly reproduces the frozen baseline evidence and groups.

| Placement and sampler | Passing poses at 5 mm spacing | At 2.5 mm | With exact-duplicate removal only |
|---|---:|---:|---:|
| N06 8 mm round pin, budget512 `one` | 0 | 2 (one family) | 14 (seven families) |
| N01 8 mm round pin, GraspKit | 0 | 2 (one family) | 8 (two families) |
| N00 USB, GraspKit | 0 | 0 | 0 |

Every recovered pin pose also meets the one-degree sampled-line projection condition. This proves that these two pin exclusions were caused by reduction **before** full-finger validation; usable pairs already existed in the original samples. GraspKit's USB case still fails the original finite-pad-height test for all fourteen eligible unreduced pairs. The follow-up passes 21,209 independent checks with no failures and preserves old evidence. Its timings include both conversion controls and exclude reused sampling. It is a **three-case sensitivity study**, not a new full-96 result at 2.5 mm or a new 96/96 benchmark.

The supported change in processing order is to validate candidates before reducing the passing set. Use a published/released reduction such as endpoint spacing or documented clustering; do not treat its first retained neighbor as a quality-ranked winner. The comparison has not yet measured the cost of validating the entire unreduced pool across all 96 placements.

## Which existing work is relevant to NIST-shaped parts

The [primary-source review](literature_review.md) covers ten methods; [the recommendation memo](recommendation.md) explains implementation choices and limitations.

- **[Kleeberger et al., Automatic Grasp Pose Generation for Parallel Jaw Grippers](https://arxiv.org/html/2104.11660v1):** the closest shape match in this review: gear, gear shaft, ring screw and U-bolt. It filters surface pairs and uses two-stage k-medoids clustering to retain spatially and directionally varied grasps. This supports reduction without a universal fixed count. Its process is offline and no generator release was verified here.
- **[F-GPG, Rigid Grasp Candidate Generation for Assembly Tasks](https://github.com/psh117/fgpg):** released geometric generation for assembly using mesh edges, opposing faces, finite-finger tests and duplicate removal. Its examples and default sizes are furniture-scale; ROS/PCL/Eigen/FCL dependencies are absent locally. It was reviewed, not executed.
- **[Fabrica's geometric grasp stage](https://github.com/yunshengtian/Fabrica/blob/main/planning/robot/util_grasp.py):** released mesh-ray contact generation and Panda support. Its complete assembly system uses reinforcement learning, while its geometric grasp stage does not require training. Only its pair-reduction function executes here; full trimesh/FCL dependencies are absent.
- **[NIST-MOADv2](https://www.robot-manipulation.org/nist-moad)** supplies exact task-board object geometry and observations, rather than a grasp planner. **[AssembleBench](https://github.com/hud-evals/assemble-bench)** models fourteen NIST-style tasks but uses a Robotiq 2F-85 and upright pegs in presentation bores. It is not evidence for our lying-pin/Panda geometry and does not include our BNC case.

No released training-free generator verified in this review demonstrates our entire exact fourteen-part set. This is a limit of this review, not proof that no such implementation exists.

## Interpretation and reproduction

Candidate availability, actual contacted-normal errors, interference rejection and processing time can be measured here. Holding quality, insertion-end accessibility, precision and assembly success remain unmeasured. More passing candidates alone does not establish better grasp quality. The VLM remains the intended later selector using task context.

The evidence does not support retaining our exact shared-polygon and fitted-cylinder tests as universal hard rejections. The baseline should be a released general mesh contact sampler with explicitly declared downward Panda conversion and actual finite-hand/target/table validation. The comparative output determines which sampler is the better local baseline; none is presented as a proven precision-assembly solution.

Run with the existing WSL environment from the repository root:

```text
python -m experiments.cad_grasp_method_comparison_20261026.ablate
python -m experiments.cad_grasp_method_comparison_20261026.run_compare outputs/cad_grasp_method_comparison_20261026/intrinsics/20261026T203042Z
python -m experiments.cad_grasp_method_comparison_20261026.analyze OUTPUT_COMPARISON_DIRECTORY
python -m experiments.cad_grasp_method_comparison_20261026.verify_comparison OUTPUT_COMPARISON_DIRECTORY
```

All old prompts, saved results and shared geometry/pose-scoring source remain unchanged; new manifests record protected hashes. Source is separate from generated output. Runtime notes distinguish CAD-intrinsic generation once per part from per-placement conversion and checking, and distinguish component timings from an isolated deployment benchmark.

Validation completed: thirteen focused regression tests pass, the full comparison audit has zero failures in 558,735 checks, and the spacing follow-up has zero failures in 21,209 checks. The gallery's selector/hand/normal controls pass its Node verifier; static sheets were visually inspected. This is not a claim that the HTML was inspected in a live browser or that grasp execution was verified.
