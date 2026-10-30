# Finger-surface contact trial

The saved finger-contact trial selected a geometrically valid grasp in 37/96 placements, with 75 valid poses from 9600 proposals. The preceding baseline selected 35/96, with 495 valid poses from 9600 proposals.

13 placements gained a selected grasp and 11 lost one. These counts measure modeled geometric availability, not pickup or assembly success.

## What moved and what stayed fixed

The contact anchors are on existing inward-facing triangles of the saved finger collision hulls. The full original palm, fingers and pad solids remain present. No pad box was lengthened and no hand vertices were removed. The physical fingertip pose reference remains distinct from the actual contact positions.

For selected grasps, contact Z in the original grip-site frame ranges from 5.773 to 9.034 mm; baseline contacts range from 4.150 to 4.150 mm. Z follows the finger axis and is not height above the table.

The saved hand's furthest vertex is at grip-site Z = 9.249 mm. Remaining full-hand extent beyond each selected contact along +Z ranges from 0.215 to 3.476 mm. Actual contacts were read from each selected candidate's actual_contacts_hand_m and checked against its CAD-to-grip-site transform.

**Collision-model limitation:** detailed finger_vis.stl has a central V-groove bridged by the coarse convex collision hull atY=0; contact here is a collision-model approximation The study does not establish that the physical finger has contact material at these locations. Physical contact calibration, holding and precise assembly remain unverified.

## Per-part comparison

| Part | Baseline selected / 8 | Baseline valid | New selected / 8 | New valid | New proposed |
|---|---:|---:|---:|---:|---:|
| BNC Male | 0/8 | 0 | 2/8 | 7 | 800 |
| DSUB Male | 8/8 | 66 | 4/8 | 6 | 800 |
| Gear Large | 0/8 | 0 | 4/8 | 4 | 800 |
| KET12 Square 12mm | 5/8 | 70 | 7/8 | 13 | 800 |
| KET16 Square 16mm | 6/8 | 136 | 4/8 | 12 | 800 |
| KET8 Square 8mm | 5/8 | 55 | 8/8 | 16 | 800 |
| M16 Hex Nut | 3/8 | 12 | 3/8 | 9 | 800 |
| RGOCG12-50 12mm | 2/8 | 62 | 0/8 | 0 | 800 |
| RGOCG16-50 16mm | 1/8 | 31 | 1/8 | 1 | 800 |
| RGOCG8-50 8mm | 1/8 | 29 | 0/8 | 0 | 800 |
| USB Male | 0/8 | 0 | 0/8 | 0 | 800 |
| Waterproof Male | 4/8 | 34 | 4/8 | 7 | 800 |

## Timing and evidence

New trial elapsed wall time: 230.30 s. The 35/96 baseline was assembled from 72 original and 24 corrected cases and has no standalone complete-run wall time. Its original trial and supplemental correction times must remain separate; this report does not invent a baseline speedup.

Saved alignments, table planes and CAD mesh paths match the baseline. The contact calibration is bound to the original hand file hash and a digest of every complete hand component and opening motion. This report verifies selected contact coordinates and counts; it does not rerun candidate generation or collision checking.

VLM selection is excluded. Pickup and assembly execution remain paused. The gallery contains selected CAD contacts and pickup side views, with no final assembly views.

[Selected contacts and pickup side views](gallery/index.html) · [Contact measurements](selected_contacts.csv) · [Machine-readable report](contact_report.json)
