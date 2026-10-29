# CAD contact construction and independent grasp checks

This study implements the four agreed follow-ups: round-side contact
construction, complete-finger height verification, consistent proposal/contact
checks with safe ray starts, and independent failure explanations with pickup
side views. It generates and selects CAD-relative grasps. It performs no robot
motion, physics stepping, VLM selection, pickup or assembly execution.

The completed eight-scene comparison selected a grasp for **35 of 96 part
placements**, versus **33 of 96** with the preceding sampler. There were
**495 valid sampled poses**, versus **368**, from 9,600 retained proposals in
each method. These numbers measure geometric candidate availability.
[Final results and limits](../../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/report.md)
and the [selected-grasp gallery](../../outputs/cad_grasp_refinement_20261029/runs/20261029T160919Z/support_corrected/gallery/index.html)
contain the per-part outcomes. Pickup and assembly remain paused.

The October 28 published results and the October 29 region-guided selector
sources are preserved. The controlled comparison uses the same corrected final
poses, protected surfaces, positive contact regions, complete hand geometry,
observed pickup alignments and table planes in both variants. Pin-tip depth
remains 20 mm below the board top. Insertion-path and release checks remain
outside this study.

## Controlled comparison

- `previous`: the October 29 region-guided sampler, including its fixed 0.5 mm
  inward ray shift and broad proposal-normal test; contact validation uses the
  original flat-face directions.
- `refined`: verified cylindrical side regions can supply radial contact
  directions and opposed contacts; other regions retain original face
  directions. Proposal and validation use the same contact-angle requirement.
  Rays retain the complete CAD and avoid the fixed inward jump that could skip
  thin opposing surfaces.
- `support_corrected`: the final result, using `support_pipeline.select_grasp`.
  A real flat CAD face in contact with a pad is checked using that face's
  direction; intended points and line contacts use the verified cylinder's
  radial direction. Numerical clipping strips at a true line contact are
  reduced to their exact intersection with the inner pad plane. This fixes a
  representation mismatch without changing the one-degree contact requirement
  or the complete collision model. The strip guard is 0.8 nm, a computational
  tolerance rather than a physical contact-accuracy claim.

Both variants use the same exhaustive geometry checker so their diagnostic
counts and measured wall times include comparable work. This checker reports
independent categories rather than stopping at the first failure. Their counts
can overlap. A low score cannot compensate for any failed mandatory check.

Only geometry-supported cylinder regions are used. A textured BNC barrel is
not automatically labeled as a smooth cylinder. The complete target surface,
full finger bodies, pads, palm, table and receiving geometry remain in checks.
The saved hand and its usable-contact labels are not shortened or enlarged.
Changing the commanded reference frame cannot change physical clearance.

The fixed search budget remains 200 surface draws, ten hand angles per accepted
pair and at most 100 complete proposals per placement. There is no retry loop,
clustering, independent finger-depth shift or learned model. All valid sampled
poses participate in the unchanged priority selection:

$$
\min r_\tau\;\rightarrow\;\min r_S\;\rightarrow\;\min\theta_{\rm down}.
$$

Here the first quantity measures distance from the mating axis under an assumed
axial-load model; it is not a measured assembly torque. The second uses the
complete CAD surface centroid, not center of mass. Downward approach is a final
preference, not a mandatory constraint.

## Local commands

Use the existing local Python environment and cached CAD/alignment inputs:

```powershell
D:\Python\python.exe -B -m pytest -q -p no:cacheprovider experiments/cad_grasp_refinement_20261029 experiments/cad_grasp_selection_20261029 experiments/nist_scene_corrections_20261029 tests/test_nist_scene_corrections.py
D:\Python\python.exe -B -m experiments.cad_grasp_refinement_20261029.run_nist --workers 4
```

The runner creates a new directory under
`outputs/cad_grasp_refinement_20261029/runs/`, containing separate `previous`
and `refined` results and candidate diagnostics. Its launch record hashes the
source, hand model, observed input and corrected task registry before launch.
Sources and protected inputs are checked again afterwards. Each variant covers
the current twelve CAD products in eight saved scenes: 96 placements each.

The original full trials are preserved. All 24 round-pin cases were rerun after
the pad-contact correction, with identical generated poses; the remaining 72
case files were copied exactly. Per-case hashes and source provenance are in
`support_corrected/derivation.json`. The final set gains four round-pin
placements and loses two waterproof-connector placements because the changed
proposal rule changes the finite random sample. It does not improve every part.
The final set has no separately measured complete-run wall time.

After the original full comparison, these commands apply the round-contact
correction and report/render the saved final result. `support_correction` creates
a new subfolder and must not be repeated inside an existing completed run:

```powershell
D:\Python\python.exe -B -m experiments.cad_grasp_refinement_20261029.support_correction RUN_DIRECTORY
D:\Python\python.exe -B -m experiments.cad_grasp_refinement_20261029.final_report RUN_DIRECTORY
D:\Python\python.exe -B -m experiments.cad_grasp_refinement_20261029.render_gallery RUN_DIRECTORY/support_corrected
```

`render_gallery.render(run_directory)` reads saved decisions only. The main gallery
retains white 3×4 CAD sheets with one contact pair per selected part. Optional
pickup side sheets show the complete modeled hand and table. Hand vertices use
the original grip-site transform; the physical fingertip frame is a separate
marker. No final assembly view is rendered.

`report_streaming.report(comparison_directory)` compares the original two saved variants and
independently recomputes selected priorities, frame reconstruction and table
clearance. It does not introduce another collision engine or establish physical
holding stability. The October 28 result is historical context only because its
goals, contact convention and sampling domain differ from this controlled study.

The final independent audit recomputes all retained pose frames, pad anchors,
selection criteria and complete-hand table distances. It counts the saved
target/receiver/protected-surface decisions; it does not independently rerun
their contact or collision engine. The original post-processing helper is
preserved alongside a receipt correcting its component-versus-whole-hand
height comparison. No grasp outcome changed in that reporting repair.

The complete fingers extend 5.099 mm beyond the near-tip contact anchors. A
separate straight-down side-grip construction for the 8 mm pin reaches about
1.1 mm into the table in the seven lying scenes. Its upright N05 placement has
a selected grasp. This is a limit of the checked construction and saved hand,
not proof that every possible grasp of a lying pin fails. Recorded fitted-
cylinder chord-sag estimates are 3.915–7.830 micrometres; vertex-fit residuals
are recorded separately. Physical contact accuracy remains unverified.

The targeted offline suite passed **124 tests** after the reporting correction.
