# Eight-scene CAD-derived grasp selection

The geometric trial evaluated **96 placements: twelve parts in N00–N07**.
Each placement received 100 complete pose proposals. Eight poses passed all
modeled checks, supporting **four selected placements**. The other 92 searches
found no valid grasp within their 100-pose budget. These are geometric selection
outcomes, not pickup or assembly success measurements.

The selected-only [gallery](gallery/index.html) preserves the white 3×4 CAD
contact-sheet format. Each blue line joins the selected pad-center contacts.
An empty tile has an explicit failed-search label. All eight scenes are present.
Height and hand orientation are not visible in these views from above; the
collapsed gallery details contain the complete pickup transform and criteria.
No final assembly view is included.

## What ran

- The same frozen CAD-to-observed-depth alignments and table planes were read
  from `outputs/cad_grasp_geometry_20261026/inputs.json`.
- CAD features supplied the previously missing axis and protected-surface inputs.
- The revised sampler generated at most 100 complete poses per placement, with
  full-circle approach orientations and calibrated pad-center placement.
- Mandatory target, closing, table, protected-surface and nominal final receiver
  checks preceded selection. The 1-degree contact-normal requirement was retained.
- Valid candidates were selected by moment arm, surface centrality, downward
  preference, then numerical tie/candidate-ID rules. No grouping or weighted
  score was introduced.

No VLM, new pose estimation, robot motion, physics stepping, pickup, assembly,
insertion-path or release experiment ran. Simulator object poses were not used.
The current eight-scene inventory contains twelve CAD products, not the earlier
fourteen-product experiment inventory.

## CAD-derived task features

The adapter uses explicit rules for these CAD families. It is not a generic
semantic CAD recognizer or a learned model. Each complete feature receipt is in
`derived_tasks/<part>.json`.

| Part family | Axis derivation | Protected region |
|---|---|---|
| Six square/round pins | Longitudinal CAD extent and transverse center; native CAD −Z for the unchanged nominal goal | Portion at/below the board top in the existing nominal seated pose; the remaining side stays available |
| Large gear | Centers of circular inner bore-mouth loops on both terminal planes; radius about 5.300001 mm | Actual bore-visible CAD faces and bottom seating faces |
| M16 nut | Two terminal inner bore-mouth circles; radius about 8 mm | Bore/thread-facing faces and bottom seating faces |
| BNC | Largest internal planar shoulder and open-end cap coverage; −Z | Shoulder at CAD z≈−2.495003 mm and the entire exposed mating feature toward −Z |
| DSUB | Largest internal flange/shoulder and open front/pin end; +Z | Shoulder at CAD z≈15.200002 mm and the front feature toward +Z |
| Waterproof | Internal flange plane and open contact end; +Z | Shoulder at CAD z≈1.799999 mm and the front feature toward +Z |
| Frozen USB cable model | Coincident interface planes between separate plug and body shells; −X | Plug side of CAD x≈−12.785003 mm |

Connector regions are **conservative feature envelopes**, not recovered exact
engagement depths. Full contact polygons, including their interiors, are
classified. Pin half-spaces are checked by polygon extrema; gear/nut regions
use source triangle labels.

The nut's threaded middle cross-section was not used as its axis datum: it can
produce an offset axis. The final adapter uses both circular bore mouths.

## Nominal-goal limitations preserved in this run

The existing final transforms were not silently changed to improve results.
They remain the user-authorized nominal simulation assumptions.

The connector goals specify zero engagement and bounding-plane contact. Their
newly derived mating features do not validate those final poses. DSUB and
Waterproof front directions also differ from the old generic lower-plane
placement convention. The frozen USB input includes a cable and has a −X plug
axis; the older nominal goal came from a different USB CAD orientation. That
goal/model relationship remains unverified.

The native pins' chamfer is at the positive CAD end, while the old nominal goals
insert the negative end. The declared nominal inserted segment is preserved;
this trial does not verify a physical lead-in or insertion outcome.

Task goals and receivers are placed consistently at **nominal board-center
workspace XY=(0,0)** with the original world Z. The observed pickup poses are
left untouched. This nominal board placement is not a new observation of the
static board in a saved simulation scene. Receiver-relative endpoint checks
use the same declared transform for the board and goal.

## Selected grasps

| Scene | Part | Candidate ID | Valid poses / proposed | Moment arm | Surface-center distance | Angle from downward | Minimum full-hand table clearance |
|---|---|---:|---:|---:|---:|---:|---:|
| N02 | KET16 Square 16 mm | 71 | 3 / 100 | 5.3226 mm | 8.3552 mm | 62.12° | 0.8767 mm |
| N04 | DSUB Male | 72 | 1 / 100 | 1.7746 mm | 2.3211 mm | 75.08° | 15.1837 mm |
| N05 | KET16 Square 16 mm | 71 | 3 / 100 | 5.3226 mm | 8.3552 mm | 62.12° | 0.8767 mm |
| N07 | DSUB Male | 56 | 1 / 100 | 4.4131 mm | 11.1986 mm | 21.48° | 14.4224 mm |

KET16's selected candidate is the same CAD-relative pose in N02 and N05 because
the CAD seed/bank is reused and the corresponding posed constraints admit it.
The algorithm still generated and checked 100 poses separately for every case.
A selected grasp is the best within the valid sampled set under the declared
priority; for example, a 5.32 mm moment arm is not evidence of an optimum or
measured fine-assembly accuracy.

| Part | Placements with a selected grasp |
|---|---:|
| BNC Male | 0 / 8 |
| DSUB Male | 2 / 8 |
| Gear Large | 0 / 8 |
| KET12 Square 12 mm | 0 / 8 |
| KET16 Square 16 mm | 2 / 8 |
| KET8 Square 8 mm | 0 / 8 |
| M16 Hex Nut | 0 / 8 |
| RGOCG12-50 12 mm | 0 / 8 |
| RGOCG16-50 16 mm | 0 / 8 |
| RGOCG8-50 8 mm | 0 / 8 |
| USB Male | 0 / 8 |
| Waterproof Male | 0 / 8 |

## Why the current rules reject so many candidates

The validator stops at its first failed check, so its rejection histogram is
**not** a count of all constraints each candidate violates:

| First rejection recorded | Candidate count |
|---|---:|
| Intended pad-center normal outside 1-degree tolerance | 3,848 |
| Hand intersects nominal final board | 3,200 |
| Hand intersects final gear plate or M16 screw | 16 |
| Palm intersects target during closing | 1,408 |
| Other finger body intersects target during closing | 464 |
| Full hand violates pickup table clearance | 574 |
| Contact outside usable inner pad face | 48 |
| Incompatible surrounding pad contact | 24 |
| Contact occupies a protected mating surface | 10 |

These sum to 9,592 rejected poses; eight poses passed. A board-first rejection
may also violate table clearance. It would be incorrect to interpret the 574
table-first entries as the total table problem.

### Independent full-hand table audit

The post-run [table audit](table_clearance_audit.json) directly transformed all
9,600 saved hand poses and component vertices at holding/pre-open configurations.
For this linear fixed-orientation closing model, endpoint plane minima give
the full closing-sweep minimum.

- 8,788 proposals put some hand geometry below the table.
- Five more have nonnegative clearance smaller than the required 0.1 mm.
- 807 proposals pass table clearance: **8,793 / 9,600 fail it**.
- All 9,600 holding/pre-opening command pairs are within the calibrated limits.
- 5,752 proposals pass the sampled-point normal condition, but only 546 pass
  both that condition and the independent table check.
- Recomputed table values match all 592 values recorded by the production
  validator within 4.2×10⁻¹⁷ m.

The saved pads are 16×16 mm. When a small pin lies on the table, forcing its
contacts to the full pad centers can place the lower pad portions through the
table. In N00 the 8 mm round pin lies flat, and all 100 proposals fail table
clearance. The 12 mm round pin is upright in N00: 34 proposals clear the table,
but none pass both table and the normal condition. Do not generalize the
lying-pin height argument to every scene or to upright pins.

The earlier fingertip-depth experiments allowed the object to contact another
location on the pad. Those are excluded by the current pad-center convention.
No such depth shift was restored or substituted in this run.

### Round-pin surface-normal sampling limitation

The round-pin triangle meshes have neighboring opposing barrel facets whose
normal mismatch is about 1.2675°–2.5353°. A ray cast exactly along one facet's
inward normal can therefore pass the broad 0.95 proposal test but fail the
unchanged 1-degree second-pad test. For N00, 96/100 proposals fail that first
normal test for each 12/16 mm round pin; 99/100 do for the 8 mm pin.

This is a conflict between the sampled faceted contacts and the strict normal
rule. It is not proof that the underlying round part has no feasible side grip.
Combined normal/table coverage is zero for each round-pin CAD across all eight
scenes. Neither the normal tolerance nor the candidate budget was relaxed.

## Timing, geometry preparation and checks

Selection finished in **116.04 seconds wall time using four worker processes**.
The summed case times are 289.73 seconds; they overlap and must not be presented
as elapsed time. Wall time includes task-feature extraction and per-case geometry
selection, but excludes gallery rendering, tests and post-run auditing.

Gear Large required the already documented numerical seam weld: a 50 nm
neighbor threshold merges 26 vertex pairs, moving vertices by at most 10.703 nm.
All 22,884 triangles retain their order, and the result has closed outward
components. The original CAD file is unchanged. The validation-copy mapping,
counts and displacement are recorded in `derived_tasks/Gear_Large*`.

**69 offline tests passed** before launch. Post-run
[verification](verification.json) independently recomputed the selected priority,
raw/normalized criteria, pickup/final transforms, calibrated contact centers and
table clearances. All eight selected contact points are within 1 micrometre of
the original CAD. The largest pad-center reconstruction error is 2.76×10⁻¹⁷ m.
Collision decisions use the production geometric validator; no second collision
engine or physical experiment was used.

Protected observed inputs, hand calibration, prior results, VLM code/prompts and
all launched Python source hashes remain unchanged. The gallery contains eight
CAD sheets and exactly four selected contact-pair lines. Gallery JavaScript has
syntax checks; browser interaction is not claimed as tested.

## Artifacts

- [Selected-grasp gallery](gallery/index.html)
- [All 96 placement counts and selected metrics](placement_counts.csv)
- [Machine-readable summary](summary.json)
- [Input/source hashes and sampling settings](launch_manifest.json)
- [Independent saved-result audit](verification.json)
- [Independent table-clearance audit](table_clearance_audit.json)
- `derived_tasks/`: twelve CAD feature receipts and validation meshes
- `cases/<scene>/<part>/`: one selection outcome and all 100 proposal diagnostics

The result identifies a limitation of the current sampled-contact, full-pad-center
and geometry-check combination. It does not establish physical pickup failure,
assembly failure, or superiority over a different grasp method.
