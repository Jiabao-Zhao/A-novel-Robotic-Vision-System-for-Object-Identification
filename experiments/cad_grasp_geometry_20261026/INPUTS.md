# Inputs for the eight-scene geometry test

This isolated study assumes that the correct part identity and corresponding
depth crop have already been supplied. The saved audit identity table is used
only to instantiate that correspondence. It is not reported as successful
classification. No simulator object position or orientation is read.

The eight N00--N07 captures contain twelve products each: large gear, M16 nut,
round pins of 8/12/16 mm, rectangular pins of 8/12/16 mm, waterproof male, USB
with cable, DSUB male and BNC male. Medium gear, small gear and M12 nut from the
older fourteen-part scene are absent. These are existing static supported-pose
synthetic captures, not new dynamically settled scenes.

For each supplied CAD/crop pair, the input pose is the saved observation-only
CAD/depth fit from the `contact_0mm` arm of the October 23 depth-verification
study. We use its pose with the best projected-depth/silhouette agreement.
The fitting method uses a finite set of CAD support orientations plus continuous
table-plane position and rotation fitting. It is not an unrestricted full-6D
pose guarantee, and symmetric orientation may remain unobservable.

The old top-five retrieval did not include N06's 12 mm round-pin crop (candidate
8), so that correct-crop fit was unavailable in the old study. We ran the same
fitting code for this one crop, retaining only its original zero-clearance arm.
The new fit took 9.504 seconds and is saved under this study's `extra_fits`.
No old fit, shortlist or result was changed. All 96 input poses are available.

The world pose is computed as `world_T_camera @ T_camera_from_cad`, using the
capture's saved camera calibration. The observed depth table plane is transformed
to world coordinates and oriented upward. Its full normalized coefficients are
saved as `table_plane_world`; `table_z_m` is only its intercept at world x=y=0.
The observed plane is slightly tilted, so use the full coefficients for clearance
checks. Existing simulation camera calibration is not a physical calibration.

`inputs.py` exposes `load_inputs()` and `local(path)`. The input manifest is
`outputs/cad_grasp_geometry_20261026/inputs.json`. Each record includes scene,
part, contact-crop ID, native CAD mesh/hash, pose source/hash, world transform,
observed plane and explicit assumption/status fields. Native CAD scale is kept
in metres. USB uses its original external cable/connector mesh; the resolver
supports both repository paths and the saved Windows/WSL paths.

Generation and validation receive these supplied poses. They do not read the
scene XML, capture object audit poses, simulator product state or live robot
state. Other objects and robot-arm checks are outside the requested geometry
study. Geometry validity does not establish pickup success or assembly accuracy.
