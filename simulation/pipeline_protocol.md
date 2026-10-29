# Current NIST CAD/Depth Pipeline

1. The target request retrieves a CAD model from `CAD/cad_library.json` by cosine
   similarity between supplied text embeddings.
2. Depth localization separates observed objects. Calibrated projection provides
   the RGB crops and corresponding camera-frame partial point clouds.
3. DINO global/foreground-patch appearance and native-scale CAD/depth agreement
   identify the matching candidate and estimate the aligned CAD pose.
4. Grasp research generates and checks candidates using that aligned CAD and the
   partial observation. The pipeline then selects one final valid grasp using
   geometric criteria and deterministic priorities. If none passes, it reports
   no valid grasp. VLM ranking/selection is not part of this pipeline. Physical
   holding and assembly quality remain separate questions under investigation.

The October 28 grasp selector proposes at most 100 complete, calibrated
pad-center poses and rejects invalid geometry before selecting one by explicit
priority: assembly moment arm, area-weighted CAD surface centrality, then pickup
downward-angle preference. It has no mandatory downward gate, grasp grouping,
independent finger-depth sampling or weighted score. Required mating-axis,
protected-surface, receiver and calibration inputs must be explicit; missing
inputs never produce an invalid fallback. The CAD-derived eight-scene geometry
trial selected grasps in 4 of 96 placements under the current checks. This is
candidate availability, not pickup or assembly performance. Geometric priority
selection is the final decision, not a preliminary ranking for a VLM. Pickup
and assembly execution remain paused.

Grasp generation is paused during the official source audit documented in
`docs/nist_task_board_1_reference.md`. NIST native assemblies and fabrication
instructions establish vertical waterproof mating, correct swapped USB/DSUB
locations and correct M4/M12 bolts previously assigned to pin holes. They
supersede those assumptions in `experiments/nist_mating_goals_20261028`.
Use documented finished-hole sizes rather than treating pilot/export geometry
as a fabrication tolerance specification. Saved native assembly transforms need
explicit conversion to our board and component STL frames; native working-face
-Z is not our simulation +Z. Component bounding-box centers are not mating datums.
The saved CAD placements are references, not physical seating validation.
The simulation layout has not been changed or run. Preserve the user-selected
12 mm pin-tip depth below the board top and the chosen chamfer-first convention.
Do not reuse historical protected-end masks after changing pin direction, or
silently substitute nominal goals for unresolved inputs. Historical numerical
results and source locks remain preserved. The earlier registry is not connected
to a resumed grasp or execution runner.

NIST Task Board 1 is the only enabled scene/CAD-generation family. Individual
studies declare their chosen NIST products, sizes, scene variants and cameras.
Keep the shared simulator/camera/Panda backend and required installed assets.

The older likelihood classifier and CAD-description/SAM3 scene-selection protocol
are not the active pipeline. Optional saved baseline comparisons do not replace
the depth-localization stage or establish an end-to-end deployed system.

Use metres and explicit camera/CAD/world/robot transforms. Preserve native scale,
unavailable fits, tied candidates and symmetry/visibility limits. Ground-truth
identities, masks and simulator poses are evaluation-only. Report the searched
pose degrees of freedom, not an unsupported full-6D claim.

The cleanup changes source organization and asset paths, not saved numerical
results. The cleanup record documents removed artifacts and preserved shared
inputs. Historical manifests/source locks remain descriptions of their original
runs, not certification of the reorganized checkout.
