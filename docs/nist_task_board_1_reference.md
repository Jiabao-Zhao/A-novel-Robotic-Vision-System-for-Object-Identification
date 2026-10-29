# NIST Task Board 1: official design reference

Reviewed on October 29, 2026. This is a source and static CAD inspection, with
grasp generation, pickup, assembly and physics execution paused. The pipeline
selects grasps geometrically; VLM selection is excluded.

The findings below are implemented in the
[current scene/task correction](../experiments/nist_scene_corrections_20261029/README.md).
See its corrected side views and verification receipt. The source inspection
and saved native placements remain separate evidence from physical assembly.

The official design resolves several errors in our scene and inferences in the
October 28 mating review. **The waterproof connector mounts vertically on a
horizontal board. USB and DSUB receiver locations were swapped in our scene;
M4 and M12 fixed bolts were assigned to round-pin holes.** These are setup
corrections, not evidence that the physical NIST parts cannot be assembled.

## Official resources

The [NIST assembly page](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly)
publishes the original Task Board 1 replication package. It includes dimensions,
a bill of materials, fabrication instructions, native assembly CAD and individual
component meshes. It is sufficient to reproduce the board; we should use these
references before making new goal assumptions.

| Resource | Purpose |
|---|---|
| [Replication instructions v4](https://www.nist.gov/document/assemblyinstructionsv4docx) | Bill of materials, finished drill/tap requirements, mounting photographs and hardware |
| [Laser-cut design](https://www.nist.gov/document/gmclaserplatezip) | SVG, Illustrator and SolidWorks drawing; pilot cuts require subsequent fabrication |
| [Native CAD](https://www.nist.gov/document/task-board-1-cad-zip) | SolidWorks parts and full assembly configurations with saved component transforms |
| [Component STL files](https://www.nist.gov/document/taskboard1stlzip) | Triangle meshes; separate exports do not by themselves establish assembled coordinates |
| [Connector housings](https://www.nist.gov/document/connectorhousingszip) | Printed DSUB and RJ45 housings |
| [2020 benchmark template](https://www.nist.gov/document/benchmarknistassemblytaskboard1v10) | Task/completion reporting; distinct from a numerical mating-pose table |
| [2017 competition rules](https://www.nist.gov/system/files/documents/2017/09/16/task1_assemblydisassembly_rev3b.pdf) | Historical task definitions, designated destinations and seating criteria |
| [Official task demonstration](https://www.nist.gov/video/robotic-grasping-and-manipulation-competition-task-1-task-board-assembly) | Human demonstration; not used here to measure numerical seating depth |

The fabrication DOCX downloaded during this review is byte-identical to the
original cached document: SHA256
`9f918be2cbb03eb20017a3ba91e577c528c1a59e127ad2cd3a4ed7ab3a8cf0b1`.
The public packages were already available locally. Lack of a source file was
not the reason for the earlier setup mistakes.

## Dimensions and fabrication

These values come from the replication instructions, Figure 2 and native CAD.
Nominal purchasing dimensions and CAD dimensions remain separate.

| Item | Official reference | Consequence for our setup |
|---|---|---|
| Board outline | Native CAD: 384 × 384 mm. Purchasing text: 15.1 inches square, or 383.54 mm nominal | Preserve native CAD scale; do not rescale it to the rounded purchasing dimension |
| Board thickness | 0.354 inches = 8.9916 mm; agrees with native CAD | Use explicit top and underside planes |
| Space below board | 20 mm from underside to supporting surface | Board top is about 28.9916 mm above that surface |
| Round-pin holes | Drill to 4.1, 8.1, 12.2 and 16.5 mm, from underside toward top | Raw cut/STL dimensions are not an automatic finished-hole specification |
| Fixed nut-receiver bolts | Tap M4, M8, M12 and M16; install bolts from underneath | Bolt head shoulders belong against the underside |
| Prismatic pins | Cut stock to 50 mm; KET8 is 8 × 7 mm, KET12 is 12 × 8 mm, KET16 is 16 × 10 mm | The historical “Square” filenames do not mean all cross-sections are square |
| Rectangular holes | Fit from top; remove excess material only from the lower half if required | Preserve the upper guiding portion of the hole |
| Pin chamfers | Optional 45-degree chamfer, length 0.1 times the largest side/diameter | Chamfer-first is our chosen configuration; the official task also permits tests without chamfer assistance |

The measured original STL loops differ from some specified drill sizes. Those
measurements remain valid descriptions of that mesh, but are insufficient to
claim finished-part assembly clearance. The laser SVG also uses a different
view: its X coordinates are reflected relative to the native board coordinates.
Accounting for that reflection reconciles the labeled pilot holes; an initial
unreflected comparison was discarded.

The current user-selected pin target is **20 mm below the board top**, giving
11.0084 mm protrusion below the underside. This is a deliberate insertion target,
not the fully bottomed-out pin position described by the 2017 rules. Those rules
award full peg seating when the bottom is constrained against the underlying
surface. The current requirement is recorded in
[task_requirements.json](../experiments/nist_official_reference_20261029/task_requirements.json).
It is applied to the current goal transforms and protected regions. Earlier
review outputs use the previous 12 mm target and remain unchanged as historical
measurements.

## What the official design says about the parts

Directions below assume a horizontal board. “Down” means toward the board along
its normal, independent of a standalone part's native CAD axis labels.

| Part | Official arrangement | Correction to the earlier interpretation |
|---|---|---|
| Waterproof | Female MCON-E10-SS faces up; male MCON-E10-SP mates from above. Four M3 × 35 mm bolts attach the female | The horizontal insertion in the earlier review came from leaving a standalone receiver in the wrong mounting orientation. Figure 8 and the native assembly establish the vertical arrangement |
| DSUB | Female SETDSUB-B-ML-PS-25 faces up in its printed housing; matching male is SETDSUB-B-ML-PP-25 | Use the complete housing/connector assembly and its saved mounting transform. Electrical-contact solid overlap alone does not establish that mating is impossible |
| BNC | Female BNC-R faces away from board; male BNCP-1.5A-K mates along that axis | The user's vertical BNC interpretation is consistent with the official design. Source solid overlap is a model/contact issue to interpret, not proof the intended pair is wrong |
| USB | Female U09-AF-AF-B faces up; male cable is U02-AM-BM-0.5 | Restore its official receiver location. Preserve the independently recovered mapping between our cable-containing model and catalog connector |
| M16 nut | Nut threads onto a fixed M16 coarse, 25 mm bolt installed from underneath | It is a threading task, not just pushing the nut down. Mounting, thread phase and the chosen tightening endpoint are separate from the grasp generator |
| Gears | Gear plate and shafts are mounted normal to the board; gears go onto shafts and mesh | Saved subassembly transforms provide a reference; static alignment is not evidence of successful gear meshing |
| Pins | Each has a designated matching hole | Restore the intended hole assignments before evaluating fit or grasp compatibility |

Official mounting photographs are extracted unchanged in the
[reference gallery](../outputs/nist_official_reference_20261029/gallery/index.html).

## Location errors found in our scene

This table uses **original board STL XY in millimetres**, not robot coordinates.
It lists hole/mounting-group locations, rounded for readability. Connector
component bounding-box centers are not necessarily the mounting-group center.

| Feature | Earlier scene assignment | Official role/location |
|---|---|---|
| M4 fixed bolt | (191.45, 347.176) | (191.45, 122.176) |
| Round 4 mm pin | Its destination was occupied by the earlier M4 assignment | (191.45, 347.176) |
| M12 fixed bolt | (41.45, 122.176) | (266.45, 347.176) |
| Round 12 mm pin | Review selected the alternate free hole | (41.45, 122.176); the alternate hole belongs to the M12 bolt |
| USB receiver | (266.45, 197.176) | Mounting group around (41.45, 272.176) |
| DSUB receiver and housing | (41.45, 272.176) | Mounting group around (266.45, 197.176) |

M4/M12 roles are independently established by the official drilling/tapping
Figure 2, laser design and native assembly. USB/DSUB identities are established
by the mounting photographs and native assembly. Full measured boundary records
and source hashes are in
[hole_roles.json](../outputs/nist_official_reference_20261029/hole_roles.json).

The earlier statement that the correct round-12 hole had to be excluded because
it was occupied by M12 was therefore wrong: the fixture was in the wrong hole.
Its measured mesh gaps are preserved as historical measurements of that earlier
assignment, not adopted as the corrected task reference.

## Native assembly placements recovered

The public CAD ZIP contains two full SolidWorks assemblies. We read their saved
component-instance XML and 4 × 4 transforms without installing SolidWorks or
changing the source archive. Named compressed streams are checked by decoded
size and CRC32. The XML matrix convention is checked independently by applying
the recovered transforms to component bounds and reconstructing assembly bounds.

Both configurations agree on all 64 common top-level component placements. The
newer nested assembly adds five male connector/cable components. Pin, nut, gear
and receiver placements are available in the original assembly metadata; the
newer assembly supplies additional male-component reference placements.

The exact matrices and hierarchy are retained in
[native_cad_audit.json](../outputs/nist_official_reference_20261029/native_cad_audit.json)
and its linked decoded instance trees. This is stronger task-reference evidence
than inferring mounting from unrelated standalone STL origins.

Saved configuration placements require careful use:

- They are CAD source references, not measured physical poses or a certificate
  that every modeled contact is interference-free.
- The native assembly's working face points toward native board −Z. Our existing
  simulation uses +Z. A proper, explicit frame conversion is required; changing
  the sign of one axis alone is a reflection, not a rigid rotation.
- A native component frame or its bounding-box center need not equal our
  converted STL frame. Match component geometry/datums before transferring a
  matrix. A cable's bounding-box center is particularly unsuitable as a connector
  mating datum.
- Our chosen 20 mm pin target is a task override, not a claim that the saved
  native pin placement already has that depth.
- Saved waterproof/DSUB references have approximately 0.25 mm Y differences
  against the plate mounting-group centers. Do not equate a matching bounding
  box or saved placement with exact mounting-hole fit. Small pin offsets are
  also retained in the source summary for later datum-based correction.
- Fine seating, contact compliance and threaded engagement are not established
  by decoding a saved transform. We did not decode native B-rep or mate constraints.

## Current status

The official source inspection supersedes the mounting and destination
conclusions in the [earlier mating review](../experiments/nist_mating_goals_20261028/README.md).
Its side-section numbers and all October 28 grasp outcomes are preserved.

The [implemented correction](../experiments/nist_scene_corrections_20261029/README.md)
updates active scene construction, explicit board/STL task transforms, nominal
finished-hole geometry, 20 mm pin targets and protected insertion regions.
The [current selector](../experiments/cad_grasp_selection_20261029/README.md) uses
the agreed fingertip reference. Exact old scene-source bytes are preserved in
a recorded snapshot; the active file is no longer claimed hash-identical to
the old run. Native CAD/STL sources and old numerical outcomes remain unchanged.
No VLM, grasp trial, pickup, insertion, release or robot-planning execution ran.

The inspection scripts and reproduction instructions are in
[the source audit README](../experiments/nist_official_reference_20261029/README.md).
