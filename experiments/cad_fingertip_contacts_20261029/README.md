# Contacts on the modeled finger ends

This revision implements actual contact placement on the existing inward-facing
surfaces near the finger ends. The preceding method reported a fingertip pose
but placed its contacts 5.099 mm behind the farthest finger vertex. Its saved
results and sources remain unchanged.

The completed [eight-scene trial](../../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/report.md)
selected a grasp in **37/96 placements**, versus **35/96** previously. It retained
75 geometrically valid poses from the same 9,600 raw proposals, versus 495.
Thirteen placements gained a grasp and eleven lost one. BNC gained two scenes,
the large gear gained four, and square 8 mm pins increased from five to eight.
Round 8 mm pins have no passing selected grasp in the new bank. This is a
contact-model change with mixed results, not a general performance improvement.
The complete new trial took 230.30 seconds; the preserved merged baseline has
no standalone complete-run time for a direct speed comparison.

[Selected CAD contacts and pickup side views](../../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/gallery/index.html)
show the final choices. Selected contacts lie 0.215–3.476 mm behind the farthest
finger vertex; the first-contact calculation determines their actual positions.

The new callable is `pipeline.select_grasp`. It uses the same aligned CAD,
allowed/protected task regions, known assembly goals, full hand and fixed-budget
ray proposals as the preceding refinement. It changes where the modeled hand
contacts the part. There is no VLM, robot movement, physics step, pickup or
assembly execution. Insertion-path and release checks remain paused.

## Contact construction

`contact_audit.make_contact_calibration` derives permitted inward-facing
surfaces from the original convex finger solids. They must face along the
closing direction within the existing 1° limit and extend beyond the old pad
edge. The complete palm, finger bodies and old pads remain collision solids.
No finger is shortened and no pad box is extended.

The common left/right starting contact region is reduced by the existing
0.25 mm edge margin in its side-plane projection. At shared Y = 0, its highest
point has grip-site Z = 7.490822 mm. The farthest finger vertex is at 9.249034 mm,
so the starting contact is 1.758213 mm behind that vertex. This is an actual
modeled contact location, rather than a coordinate-frame rename. The margin
controls the initial contact anchor; subsequent physical contact witnesses may
touch the original permitted face boundaries.

For each original CAD pair and hand angle, `sampler.fit_candidate` holds that
orientation and starting finger depth fixed. It calculates the first contact
of each complete jaw with the complete CAD. It bisects the growing swept volume
from maximum opening, which cannot skip an earlier contact on a concave part.
The two independent jaw positions determine a common opening and a translation
along the closing axis. No extra CAD draws, angles or depth search are added.
These are static solid-geometry queries, not a simulated closing execution.

For the saved symmetric jaw motion (left moves by −q/2 and right by +q/2),
independent first-contact commands q_L and q_R give

$$
q=\frac{q_L+q_R}{2},\qquad
\delta_x=\frac{q_R-q_L}{4}.
$$

The hand moves by delta_x along its local closing axis. This reproduces both
independent jaw positions at one shared command; it does not assume that closing
forces leave a physical part motionless.

The true contact points are taken from the resulting CAD/finger intersection
witnesses. Their original grip-site coordinates are stored explicitly in
`actual_contacts_hand_m`; they are transformed into CAD coordinates for scoring.
The original ray points and normals remain separately labeled as proposals.
An old pad touching first is rejected in this finger-end contact mode; no
fallback pretends that it was contact at the finger ends.

`geometry.GeometryValidator` requires both contacts on permitted original
finger surfaces, reconstructs their coordinates, and checks the target surface
direction against each actual finger face. It also checks complete contact
regions, closing-volume overlap, table clearance, final receiver overlap and
protected assembly surfaces. The 1° rule and existing numerical collision
tolerance are unchanged. A numerical boundary contact is distinguished from a
finite-area contact on verified round sides. All valid samples participate in
the preceding moment-arm, surface-centroid, downward-preference and ID priorities.

Identical CAD proposal banks reuse their static contact fits across saved scenes.
The cache key includes all CAD arrays, hand/contact calibration, proposal fields
and opening-clearance settings; observed poses are applied only downstream.

## Evidence and limits

[Contact calibration and source audit](../../outputs/cad_fingertip_contacts_20261029/contact_audit.json)
and [side comparison](../../outputs/cad_fingertip_contacts_20261029/finger_contact_side_audit.png)
record source surfaces, dimensions and file hashes. The original collision
model has a coarse convex face spanning a central groove visible in the detailed
display mesh. This revision uses the saved collision geometry consistently; it
does not establish that a physical finger has contact material throughout that
face. Physical finger/contact calibration remains necessary for hardware use.

The separate [declared straight-down 8 mm pin check](../../outputs/cad_fingertip_contacts_20261029/declared_side_grip_audit.json)
changes complete-hand table clearance from −1.102949 mm to +2.237873 mm. Its
actual contacts are about 1.670 mm behind the farthest finger point. That pose
still fails the unchanged 1° rule (about 1.107°) and final board clearance. This
isolates the height correction; it is not a passing grasp or a pickup result.
The [saved side comparison](../../outputs/cad_fingertip_contacts_20261029/declared_side_grip_comparison.png)
shows both placements with the same complete modeled hand.

## Local commands

```powershell
D:\Python\python.exe -B -m pytest -q -p no:cacheprovider experiments/cad_fingertip_contacts_20261029
D:\Python\python.exe -B -m experiments.cad_fingertip_contacts_20261029.run_nist --workers 4
D:\Python\python.exe -B -m experiments.cad_fingertip_contacts_20261029.report RUN_DIRECTORY
D:\Python\python.exe -B -m experiments.cad_fingertip_contacts_20261029.verify RUN_DIRECTORY
D:\Python\python.exe -B -m experiments.cad_fingertip_contacts_20261029.gallery RUN_DIRECTORY
```

The trial covers twelve CAD products in all eight saved scenes, using the same
100 retained raw proposals per placement and seeds as the saved 35/96 reference.
The complete hand geometry, observed alignment, corrected receiver goals and
20 mm pin-depth requirement remain fixed. The runner verifies every original
ray pair and approach against the reference and checks launch hashes. Results
describe bounded geometric candidate availability, not physical success.

The independent post-launch audit checks every fitted pose's contact frames and
complete-hand table distances, all valid samples' criteria and selection
priorities, and selected contacts against the actual permitted original finger
planes. It does not independently rerun the collision engine. Source hashes,
protected inputs and all 96 raw proposal banks match their recorded references.
The original 148-test targeted suite passed before launch; later postprocessing
tests increased the completed targeted suite to **152 passing tests**. Neither diagnostic rendering nor audits generate
additional grasp candidates.

The [per-part failed checks](../../outputs/cad_fingertip_contacts_20261029/runs/20261029T180942Z/failure_analysis.md)
include unavailable later checks after a failed first-contact prerequisite;
these unperformed checks are never counted as passes.
