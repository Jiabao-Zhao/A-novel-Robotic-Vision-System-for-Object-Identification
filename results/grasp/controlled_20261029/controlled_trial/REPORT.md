# Controlled NIST grasp geometry trial

Eight saved scenes, 12 products, three paired seeds, using the saved Panda hand geometry. Correct CAD association and saved CAD alignment are assumed. These counts measure availability of a geometrically valid modeled grasp, not pickup or assembly success.

## Candidate budget and comparison

Each policy receives at most 100 proposals. Combined mode declares 50 pad and 50 finger contacts before validation when 100 poses are retained; shorter sets keep the two allocations balanced within one. Both samplers use 200 surface-origin draws and at most ten orientations per pair. The task-informed table prefilter can leave fewer than 100 eligible poses; no additional surface draws refill that set. The benchmark checks pad and finger alternatives on the shared raw bank (up to 200 underlying candidates); combined replays its declared subset. It does not select from a union of two 100-candidate searches.

| Sampler | Contact policy | Seed 0 /96 | Seed 1 /96 | Seed 2 /96 | Total /288 | Valid / evaluated candidates |
|---|---|---:|---:|---:|---:|---:|
| random | pad | 35 | 38 | 41 | 114 | 1661 / 28800 |
| random | finger | 37 | 35 | 32 | 104 | 236 / 28800 |
| random | combined | 46 | 44 | 41 | 131 | 924 / 28800 |
| task_informed | pad | 61 | 52 | 61 | 174 | 4430 / 24546 |
| task_informed | finger | 57 | 56 | 61 | 174 | 1024 / 24546 |
| task_informed | combined | 71 | 66 | 66 | 203 | 2751 / 24546 |

The 288 evaluations per arm repeat the same 96 placements under three seeds. They are not 288 independent objects or scenes. Descriptive paired changes follow; three seeds do not establish a universal best sampler. The task-informed arm changes contact-position allocation, explicit orientations and the safe table prefilter together; this comparison cannot assign a gain to one of those components alone. The 50/50 contact allocation is fixed, not an optimized mixture.

| Task-informed versus random | Both valid | Random only | Task-informed only | Neither | Net change |
|---|---:|---:|---:|---:|---:|
| pad | 110 | 4 | 64 | 110 | +60 |
| finger | 95 | 9 | 79 | 105 | +70 |
| combined | 125 | 6 | 78 | 79 | +72 |

## Separate geometry stages

Contact-valid requires permitted contact, opening limits, normal alignment and complete-hand/part clearance. Pickup-valid adds table clearance to contact-valid. Final-goal-valid separately adds final receiver clearance and protected mating surfaces to contact-valid, without the pickup-table requirement. Joint-valid requires all of these for one unchanged grasp. Counts below are placements with at least one candidate passing each stage; a different candidate may supply each stage. Unavailable or unevaluated checks are never passes.

| Sampler | Policy | Contact /288 | Pickup /288 | Final goal /288 | Joint /288 |
|---|---|---:|---:|---:|---:|
| random | pad | 240 | 157 | 232 | 114 |
| random | finger | 248 | 133 | 216 | 104 |
| random | combined | 272 | 172 | 264 | 131 |
| task_informed | pad | 236 | 192 | 220 | 174 |
| task_informed | finger | 259 | 237 | 193 | 174 |
| task_informed | combined | 281 | 260 | 243 | 203 |

## Per-part joint availability

Each cell is seed 0 / seed 1 / seed 2, each out of eight placements.

| Part | Random pad | Random finger | Random combined | Task pad | Task finger | Task combined |
|---|---:|---:|---:|---:|---:|---:|
| BNC Male | 0 / 0 / 0 | 2 / 1 / 2 | 2 / 1 / 2 | 0 / 0 / 0 | 2 / 3 / 5 | 1 / 2 / 3 |
| DSUB Male | 8 / 7 / 6 | 4 / 3 / 2 | 8 / 7 / 5 | 8 / 8 / 8 | 7 / 7 / 4 | 8 / 8 / 7 |
| Gear Large | 0 / 4 / 4 | 4 / 4 / 0 | 4 / 4 / 4 | 4 / 0 / 4 | 4 / 0 / 4 | 4 / 0 / 4 |
| KET12 Square 12mm | 5 / 8 / 8 | 7 / 4 / 5 | 7 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 |
| KET16 Square 16mm | 6 / 6 / 8 | 4 / 8 / 8 | 6 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 |
| KET8 Square 8mm | 5 / 2 / 2 | 8 / 8 / 2 | 8 / 5 / 2 | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 |
| M16 Hex Nut | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 | 5 / 6 / 7 | 4 / 6 / 4 |
| RGOCG12-50 12mm | 2 / 2 / 2 | 0 / 0 / 0 | 2 / 2 / 2 | 7 / 2 / 7 | 2 / 0 / 2 | 7 / 2 / 7 |
| RGOCG16-50 16mm | 1 / 1 / 1 | 1 / 0 / 0 | 1 / 1 / 1 | 8 / 8 / 8 | 0 / 1 / 1 | 8 / 8 / 1 |
| RGOCG8-50 8mm | 1 / 1 / 1 | 0 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 2 / 2 |
| USB Male | 0 / 0 / 0 | 0 / 0 / 8 | 0 / 0 / 0 | 0 / 0 / 0 | 8 / 8 / 8 | 8 / 8 / 8 |
| Waterproof Male | 4 / 4 / 6 | 4 / 3 / 1 | 4 / 4 / 5 | 6 / 6 / 6 | 4 / 6 / 5 | 6 / 6 / 6 |

## Frozen-pose contact-angle sensitivity

Only the final normal-acceptance tolerance changes. Hand surface labels and calibration, random seed-0 proposals, fitted poses and all other tolerances remain fixed. This does not test changing the proposal-angle tolerance, modifying the hand, frictional holding or physical compliance.

| Acceptance angle | Placements /96 | Valid / evaluated candidates | Newly valid candidates: other checks already passed | Newly valid candidates: some other checks previously unevaluated |
|---|---:|---:|---:|---:|
| 0.5° | 35 | 69 / 9600 | 0 | 0 |
| 1° | 37 | 75 / 9600 | 0 | 0 |
| 2° | 39 | 88 / 9600 | 7 | 6 |
| 3° | 42 | 112 / 9600 | 12 | 25 |
| 5° | 42 | 128 / 9600 | 12 | 41 |

Newly valid counts are relative to the fixed 1° baseline. The final column includes checks that become evaluable only after a valid contact witness exists; it must not be described as candidates known to pass every nonangular check at 1°.

## Gallery, contact surfaces and time

[Primary gallery](primary/gallery/index.html) uses task-informed combined mode, seed 0 fixed before results, not the best seed. Selected contact modes: {'finger': 28, 'pad': 43}. It shows saved selected contacts on CAD; failed placements have no replacement grasp.

Pad and finger-end contacts are explicit alternatives in the existing simulation hand model. The convex finger collision shape bridges a groove in the detailed display mesh. Neither is a verified physical hand surface. The source-surface audit reports that discrepancy separately; it does not silently turn display geometry into hardware truth. UR5e/RG2 geometry and hardware contact calibration are not supplied by this trial.

Shared benchmark wall time: 2249.874 s with 4 worker processes, including all policies/seeds and the separately recorded frozen-angle sweep; excluding rendering/publication. The benchmark caches CAD-local and fixed-final-goal checks, recomputes complete pickup table checks and approach preference in every scene, and shares work between policies. Independent policy runtimes were not measured, so this is not a policy speed comparison.

Underlying mode-candidate evaluations: 106,692; logical evaluations across the three policies: 160,038. Distinct retained raw-bank hashes: {'random': 36, 'task_informed': 288}. Raw proposals reused across scenes are not new independent draws.

## Scope and evidence

The geometric priority selector is unchanged: minimum normalized assembly-axis moment arm, then surface-centroid distance, then downward approach angle, then candidate ID, with explicit numerical ties. This is a geometric proxy; insertion force, actual torque, frictional holding, compliance and execution accuracy are not measured. Recognition policies were not changed. VLM selection, robot motion, pickup, assembly, insertion-path and release checks were not run.

[Launch manifest](launch_manifest.json), [run verification](run_verification.json), [full results](results.json), [summary](summary.json), [paired comparisons](comparison.json), [per-part CSV](per_part_comparison.csv), [frozen-angle records](frozen_angle_sensitivity.json).

This report is postprocessing added after launch. It reads saved outcomes without modifying numerical results or selecting another seed.
