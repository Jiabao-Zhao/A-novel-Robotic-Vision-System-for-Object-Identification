# Preserved assembly-guided fingertip grasp selection

This revision samples contact points from explicitly labeled CAD regions before
forming the bounded grasp bank, and implements the agreed fingertip reference.
October 28 sources and results remain unchanged. This directory is the preserved
region-guided implementation used for comparison and the shared corrected-task,
region and fingertip helpers. Its `pipeline.select_grasp` and JSON entry point
remain available with their recorded assumptions.

The preserved surface-contact refinement is
[`experiments/cad_grasp_refinement_20261029`](../cad_grasp_refinement_20261029/README.md).
It shares a one-degree proposal/check normal rule, verifies side models for the
three round pins (BNC retains triangle normals), begins rays on the CAD surface,
and records independent failure categories while retaining complete collision
geometry. The completed eight-scene comparison selected grasps for 33/96
placements with 368 valid poses under the controlled previous construction,
versus 35/96 with 495 valid poses in the final refined set, from 9,600 proposals
per set. Four round-pin placements gained a grasp and two waterproof placements
lost one. The final set combines 72 unchanged cases with 24 round-pin cases
rechecked after a finite-pad witness correction; their regenerated banks were
identical. Original trials are preserved and no standalone final full-run wall
time is reported. Its preserved callable is
`experiments.cad_grasp_refinement_20261029.support_pipeline.select_grasp`.
See the [final report](../../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md)
and [selected-contact/pickup-side gallery](../../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html).

The current callable is
`experiments.cad_fingertip_contacts_20261029.pipeline.select_grasp`, documented
in the [finger-contact study](../cad_fingertip_contacts_20261029/README.md).
It fits first contact on existing inward faces near the ends of the original
finger collision hulls, retaining the complete hand and original pads. Its full
eight-scene trial selected grasps in 37/96 placements, with 75 valid poses from
9,600 proposals in 230.30 s. Against the preserved 35/96, 495-pose baseline,
13 placements gained a selected grasp and 11 lost one. This modest net gain
does not establish physical contact reliability; the collision hull bridges a
groove in the visual finger. The round 8 mm pin still has no selected grasp in
the eight-scene trial. See the [new report](../../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/report.md)
and [pickup gallery](../../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/gallery/index.html).
The detailed equations below describe this preserved sampler, including its
0.95 proposal threshold and fixed inward ray offset; those two choices do not
describe the refined sampler. Geometric availability is separate from physical
holding or assembly. VLM selection is excluded; pickup and assembly execution
remain paused.

## Contact regions before candidate generation

Let $S=\bigcup_i T_i$ be the complete original CAD surface, $H$ the explicitly
labeled positive contact region and $P$ the protected mating/insertion region.
The surface from which contacts may be proposed is

$$
A=(S\cap H)\setminus P.
$$

`sampling_regions.compile_sampling_region` clips triangles at the declared
halfspaces and carries their original face IDs. It also supports face-ID labels
and Boolean unions/intersections. Overlapping clauses are partitioned into
disjoint patches, so their overlap does not receive extra sampling weight.
This handles a long pin triangle crossing the insertion boundary: its exposed
portion remains available instead of keeping or discarding the whole triangle.
Only sampling patches are clipped; the original CAD arrays remain unchanged.
Protected boundaries include the existing 10 nm numerical guard; positive
region boundaries remain closed. This is a numerical convention, not a physical
clearance allowance or an estimate of pose uncertainty.

For the resulting patches $C_j$, with area $a_j$, the origin sampler uses

$$
\Pr(C_j)=\frac{a_j}{\sum_k a_k},\qquad
p_1=(1-\sqrt u)v_0+\sqrt u(1-v)v_1+\sqrt u\,v\,v_2,
\quad u,v\sim U(0,1).
$$

Here $v_0,v_1,v_2$ are the chosen patch's vertices. The first equation selects
patches by area; the second samples uniformly within the selected patch.
One fixed `surface_budget` limits these draws, with no retry loop seeking more
valid grasps. The original face supplies the outward normal $n_1$.

The numerical ray starts slightly inside the CAD and travels inward:

$$
r(t)=p_1-\epsilon n_1-t n_1,\qquad t>0.
$$

Ray intersections use **all complete original CAD triangles**, including bores,
protected features, cable geometry and regions omitted from sampling. All
forward hits are retained by the existing ray query. Each possible second
contact $p_2$ must satisfy

$$
p_1\in A,\qquad p_2\in A,
\qquad
n_1^\top(-d)\ge0.95,\quad n_2^\top d\ge0.95,
\qquad d=\frac{p_2-p_1}{\|p_2-p_1\|}.
$$

Both endpoint-region checks happen **before** creating the pair/orientation bank
and drawing at most 100 complete poses. If $Q$ pairs pass and each has
$N_\theta$ sampled hand angles, the bank has $Q N_\theta$ entries and

$$
N_{\mathrm{proposed}}\le\min(100,Q N_\theta).
$$

The 0.95 normal threshold and full-circle angle construction are retained from
the preceding ray proposal code. Near-tip placement, full-pad contact checks,
complete finger/palm checks, table and receiver checks still run afterwards.
A contact point inside an allowed region does not guarantee that its entire pad
fits there. Criteria and surface-centrality references still use the complete CAD,
not its clipped sampling patches.

This upstream region restriction is **our task-specific adaptation of the
Fabrica-derived sampler**. The NIST labels below are initial CAD annotations,
not trained affordances, labels supplied by Fabrica, or certified grip surfaces.

Guo et al.'s [Stable grasp generation enabled by part segmentation for real-world
robotic applications](https://www.sciencedirect.com/science/article/pii/S0736584525002248)
(*Robotics and Computer-Integrated Manufacturing* 98, 103170, 2026) motivates
using structural part information in grasp generation. Our functional assembly
labels and upstream ray-origin restriction are our adaptation; the full
Algorithm 1 and drill example have not been independently verified here. This
attribution is limited to the available publisher abstract/introduction; the
[authors' repository](https://github.com/nubot-nudt/Aff-Grasp) still lists core
code as “ETA: Soon” when checked on October 29, 2026. Their Instant-Sem perception
stage uses rapid neural training, so it is not our training-free implementation.

## NIST labels shared with the current refinement

`sampling_annotations.py` applies the explicit per-part declarations in
`config.json`, bound to source CAD hashes, triangle counts and component IDs.
Outward side labels use the original face normal and radial direction about the
declared part axis. The declared coordinate limits then clip triangles exactly.
Protected mating/seating geometry is excluded separately.

| Part | Positive proposal region |
|---|---|
| Six 50 mm pins | Outward shaft sides on the **30 mm remaining exposed** after the chosen 20 mm insertion. Axial end caps are excluded. For the centered +Z-tip pin CADs, the insertion boundary is at CAD Z = +5 mm; the protected side and its existing 10 nm numerical guard are excluded |
| BNC | Main barrel side band between measured CAD Z shoulders of approximately −2.495 and +3.085 mm; excludes the narrow rear stem and mating end |
| DSUB | Main housing side walls within X ±20 mm, Y ±8 mm and Z approximately −21.8 to +15.200 mm; excludes fastening wings and contact insert |
| Waterproof | Main housing side walls within X ±21.6 mm, Y ±16.25 mm and Z approximately −17.700 to +1.800 mm; excludes outboard retaining features |
| USB | Connector housing component sides; the long cable and plug/contact components are excluded from sampling |
| Large gear | Main gear's external flanks and hub sides; separate set screw, bore and seating surfaces are excluded |
| M16 nut | External flats and outward bevels; inward thread/bore walls, axial caps and seating surfaces are excluded |

Housing bounds are measured CAD labels for this study. They do not establish
material stiffness, gripping force or whether a full pad fits. Gear tooth flanks
and nut bevels can still fail the unchanged finite-pad checks. The excluded CAD
geometry remains present for ray intersection and collision queries.

The local [region atlas](../../outputs/cad_grasp_selection_20261029/region_guidance/index.html)
and [audit report](../../outputs/cad_grasp_selection_20261029/region_guidance/README.md)
show these proposal surfaces for all twelve current parts. All twelve annotations
are nonempty, and request composition was checked for all 96 placements across
eight saved scenes. These checks compiled regions and inputs only: the atlas
shows allowed surface regions, not generated grasps or successful pickups.

The default `sampling_mode='assembly_region_guided'` requires a compiled
`sampling_contact_region`, or explicit `protected_contact_region` and optional
positive `sampling_region` declarations that the pipeline compiles. The NIST
`task_inputs.build_request` requires a reviewed positive annotation as well;
`allow_protected_only_sampling=True` is an explicit opt-in for a protected-only
comparison. It does not certify the rest of an object as safe housing.
Missing labels return unavailable inputs. Generic requests can explicitly choose
`sampling_mode='legacy_unrestricted'`; it is never a silent fallback and cannot
silently ignore a supplied compiled region. An empty allowed surface returns
zero candidates without casting rays.

Sampler metadata records eligible area and its fraction of full CAD area,
clipped-patch source face IDs, surface points, offset ray starts, second-endpoint
region rejections, pair counts and the bounded pose count. Hashes bind the
compiled region to the complete normalized vertex/face arrays and face ordering.

## Fingertip and contact placement

The commanded pose `cad_T_hand` now names the **midpoint of the physical finger
distal ends**, with the saved hand's X closing direction and +Z approach direction.
`cad_T_grip_site` is also returned explicitly for a robot interface that commands
the original grip site. The transform between those frames is calibrated from
the saved complete hand geometry, not guessed from a paper's gripper offset.

The intended contacts are **near the pad tips**, one declared pad-edge margin
behind the edges (default 0.25 mm). They are no longer constrained to the pad
centers. Full pad surfaces, both complete finger collision components and the
palm remain present in contact, closing-sweep, table and receiver checks.
This is a contact-placement change as well as a pose-frame change.

For the saved simulator hand, usable pads span Z = −11.6 to +4.4 mm in grip-site
coordinates. The **physical fingertip** reference is at approximately +9.249 mm.
The separate **usable pad-tip** frame is at +4.4 mm and the default contact
anchors are at +4.15 mm. The old pad centers were at −3.6 mm: contact
placement moves approximately **7.75 mm toward the pad tips**. Full finger
geometry extends approximately **4.849 mm beyond those pad edges**. It remains
checked; this implementation does not guarantee that a small resting pin clears
the table. These are simulator-model dimensions, not measured physical calibration.

`hand_frame.fingertip_calibration` exposes both physical-tip and usable-pad-tip
frames explicitly. They are not interchangeable. `hand_in_fingertip_frame`
exports a complete transformed copy of the
hand when needed. The active validator instead converts the public fingertip
pose back into the saved grip-site geometry internally. Both representations
put every full-hand vertex at the same CAD/world position.

The bounded ray and angle construction is retained, with the allowed origin
domain and endpoint filtering described above. Complete collision checks,
protected-contact checks and deterministic priorities (moment arm, surface
centrality, downward preference) are unchanged. There is no added scoring term,
training or VLM selection. Assembly goal and receiver corrections are supplied
by the current NIST input builder; the selector never reads simulator object
ground truth.

`task_inputs.build_request(part, world_T_cad_pickup, table_plane_world, ...)`
constructs a request from the corrected native goal registry. Its return contains
`status`, `request` and source records. The caller supplies the observed pickup
alignment and table plane; an optional explicit `world_T_board` composes every
corrected final pose and receiver transform. Missing data, source-hash mismatch,
frame mismatch or a stale pin-depth/protected-region combination returns
`unavailable_inputs` with `request=None`. It does not call the sampler or selector.
Protected unions classify complete contact polygons, not just their centers.
The builder supplies compiled sampling patches separately from complete part and
receiver meshes; no clipped object is passed to ray or collision checking.
The declared 50 nm Gear Large seam weld is retained without changing source CAD
or triangle ordering; no old nominal task-goal derivation is called.

Offline checks:

```powershell
D:\Python\python.exe -B -m pytest -q -p no:cacheprovider tests/test_nist_scene_corrections.py experiments/nist_scene_corrections_20261029 experiments/cad_grasp_selection_20261029
```

The recorded baseline targeted aggregate passed **68 offline tests**. The tests verify
saved-geometry frame conversion, actual contact placement,
complete-hand invariance, table/receiver/collision rejection and current selector
wiring using explicit synthetic candidates, plus corrected task input composition
and unavailable-input behavior. Synthetic sampling checks cover partial-triangle
origins, both endpoint restrictions, a complete tube with protected bore walls,
empty regions, area weights, original face IDs, immutable CAD arrays, seeded
budgets and JSON region compilation. They do not run the NIST trial.
