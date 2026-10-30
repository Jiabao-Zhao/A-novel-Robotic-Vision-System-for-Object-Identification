# Final CAD grasp refinement results

The current twelve CAD products were checked in all eight saved scenes: 96 placements. The region-guided baseline selected grasps for **33/96** placements; the corrected refinement selected **35/96**. Valid sampled poses increased from **368** to **495**, out of 9,600 retained proposals per method. These are geometric candidate counts, not pickup or assembly success.

## What was checked

Both methods use the same corrected assembly goals, protected regions, positive sampling labels, full hand, observed CAD alignments, table planes, seeds and 100-pose cap. Priorities remain mating-axis moment arm, surface-centroid distance and downward preference. There is no VLM selection, robot movement, physics, insertion-path or release execution. All poses and task input frames remain explicit.

The first full refinement exposed a contact-representation mismatch. Intended contacts used verified cylinder directions, while a finite flat CAD facet was incorrectly checked using curved-surface directions at all its corners. The correction distinguishes actual coplanar facet support from numerical edge ribbons. It retains the 1° requirement and complete collision geometry. Numerical ribbons within 0.8 nm of the actual inner-pad plane are reduced to their exact line/point intersection; genuine penetration and other pad surfaces are not removed.

All 24 round-pin cases were rerun, and every generated pose field was verified identical to the first refined bank. The other 72 case files are exact copies. The original two full trials remain unchanged. The final 96-case result is an incrementally corrected result, with explicit per-case provenance, rather than a newly timed full run.

| Part | Baseline selected /8 | Corrected selected /8 | Corrected valid poses /800 |
|---|---:|---:|---:|
| BNC Male | 0 | 0 | 0 |
| DSUB Male | 8 | 8 | 66 |
| Gear Large | 0 | 0 | 0 |
| KET12 Square 12mm | 5 | 5 | 70 |
| KET16 Square 16mm | 6 | 6 | 136 |
| KET8 Square 8mm | 5 | 5 | 55 |
| M16 Hex Nut | 3 | 3 | 12 |
| RGOCG12-50 12mm | 0 | 2 | 62 |
| RGOCG16-50 16mm | 0 | 1 | 31 |
| RGOCG8-50 8mm | 0 | 1 | 29 |
| USB Male | 0 | 0 | 0 |
| Waterproof Male | 6 | 4 | 34 |

## Remaining failed checks

Counts below overlap: one pose may fail several checks. They describe the sampled banks and do not establish that no feasible grasp exists.

| Part | Contact alignment | Pad contact | Closing collision | Table | Final receiver | Protected surfaces |
|---|---:|---:|---:|---:|---:|---:|
| BNC Male | 0 | 800 | 800 | 754 | 448 | 328 |
| DSUB Male | 0 | 112 | 264 | 593 | 360 | 200 |
| Gear Large | 0 | 792 | 736 | 552 | 392 | 16 |
| KET12 Square 12mm | 0 | 0 | 32 | 668 | 544 | 288 |
| KET16 Square 16mm | 0 | 0 | 80 | 568 | 544 | 280 |
| KET8 Square 8mm | 0 | 0 | 72 | 701 | 592 | 256 |
| M16 Hex Nut | 0 | 464 | 456 | 758 | 688 | 288 |
| RGOCG12-50 12mm | 0 | 224 | 88 | 692 | 536 | 176 |
| RGOCG16-50 16mm | 0 | 272 | 64 | 740 | 520 | 200 |
| RGOCG8-50 8mm | 0 | 336 | 56 | 748 | 528 | 216 |
| USB Male | 0 | 776 | 544 | 800 | 408 | 8 |
| Waterproof Male | 0 | 352 | 464 | 586 | 440 | 408 |

Newly available placements: N00 RGOCG12-50 12mm, N04 RGOCG16-50 16mm, N05 RGOCG8-50 8mm, N06 RGOCG12-50 12mm.

Baseline placements absent from the final fixed-budget bank: N01 Waterproof Male, N07 Waterproof Male.

The changed proposal rule changes which pairs and angles enter the seeded 100-pose sample. It can lose a previously sampled valid pose; this revision does not improve every placement. No retry search was used to repair the reported outcomes.

## Finger height and timing

The saved complete fingers extend 5.099 mm beyond the declared near-tip contact anchors. The separate straight-down 8 mm pin audit finds about 1.1 mm table penetration in seven lying scenes, while N05 is upright. That audit concerns its declared side-grip construction; it does not rule out every other orientation. The final search retains only the upright N05 8 mm round-pin placement. The hand geometry was not shortened or enlarged.

Original complete trial wall times: baseline **843.84 s**; initial refinement **751.07 s**. Additional 24-case correction: **137.62 s**. No separate full-run wall time was measured for the final merged result. Trial times include exhaustive checking and case output; rendering and independent reporting are separate. This evidence does not establish an overall runtime improvement.

## Verification

The audit independently recomputes all retained hand transforms, near-tip contact anchors, normalized criteria, deterministic selection priorities and complete-hand table distances. Each component is compared with its own endpoint minimum, and the whole hand with the overall minimum. It does not rerun the target/receiver collision engine or measure physical holding/assembly accuracy.

Only the three round-pin sides verify the declared cylindrical model. BNC retains its original surface directions because its textured barrel is not one straight cylinder. Nominal fitted-cylinder chord-sag estimates are recorded in selected validity metadata, with vertex-fit residuals separately recorded in the contact model; physical precision remains unverified.

The first post-processing audit incorrectly compared per-component table heights against the global minimum. Its exact pre-repair source is preserved and hash-checked. The reporter was corrected without changing generated grasps, geometry decisions or original trial results. The source resolution is explicit in verification.json.

[Selected grasps and pickup side views](gallery/index.html) · [Independent verification](verification.json) · [All placement counts](placement_counts.csv) · [Exact case provenance](derivation.json) · [Identical regenerated banks](../support_correction/bank_identity.json) · [Finger-height audit](../fingertip_height_audit.json) · [Original full-trial report](../report.md)
