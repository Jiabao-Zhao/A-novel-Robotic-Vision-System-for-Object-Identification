# October 28 CAD grasp-selection run

This completed run is published at its original paths under a run-specific
`.gitignore` exception. Publishing did not rerun generation or change the saved
numerical outcomes. VLM selection, pickup and assembly execution remain paused.

The trial checked **96 placements: twelve parts in eight scenes**. Each placement
received 100 complete grasp proposals. Eight proposals passed the modeled checks,
supporting selected grasps for four placements: KET16 in N02/N05 and DSUB in
N04/N07. The remaining 92 searches found no valid grasp within their budget.
Candidate availability does not measure pickup or assembly success.

## Read the evidence

- [Detailed report](report.md) and [summary](summary.json).
- [All placement outcomes and transforms](results.json) and [CSV counts](placement_counts.csv).
- [All 96 case folders](cases/), including the 9,600 proposed poses and their check results.
- [CAD-derived mating axes, protected regions and validation meshes](derived_tasks/).
- [Independent selection verification](verification.json), [table-clearance audit](table_clearance_audit.json)
  and [gallery verification](gallery_verification.json).
- [Launch settings and source/input hashes](launch_manifest.json).
- [Publication manifest and file hashes](publication_manifest.json).

The twelve `*_validation_mesh.npz` files contain CAD vertices, triangle indices
and source-face mappings. They are derived geometry used by the checks, not
model weights or captured observations. The transient `progress.json` stays local.
Other raw captures, saved alignment/calibration inputs and installed dependencies
outside this run are not included by this exception. A fresh clone can review
the evidence and gallery; these files alone do not establish a reproducible run.
Historical absolute paths in receipts are provenance, not portable file links.

## Selected-grasp gallery

Open [gallery/index.html](gallery/index.html) locally after cloning for the scene
selector and complete pickup-transform details. GitHub displays the PNG sheets
directly through the links below. Each sheet contains twelve gray CAD models;
blue lines show selected contact pairs and empty tiles report failed searches.
There are no final assembly views.

| Scene | CAD contact sheet |
|---|---|
| N00 | [View N00](gallery/N00.png) |
| N01 | [View N01](gallery/N01.png) |
| N02 | [View N02](gallery/N02.png) |
| N03 | [View N03](gallery/N03.png) |
| N04 | [View N04](gallery/N04.png) |
| N05 | [View N05](gallery/N05.png) |
| N06 | [View N06](gallery/N06.png) |
| N07 | [View N07](gallery/N07.png) |

The report records the nominal assembly-goal limitations, pad/table conflicts,
strict round-surface normal checks and the recorded Gear Large seam weld.
These limits were preserved when publishing the results.
