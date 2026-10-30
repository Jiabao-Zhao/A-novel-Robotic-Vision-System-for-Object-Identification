# Saved geometry stages

Each cell is passing candidate-placement evaluations / placements with at least one passing candidate. These are modeled endpoint and contact checks, not successful pickups, insertions or assemblies. The same CAD-relative candidate bank may be evaluated in several scenes.

## pad_reference_35

| Part | Contact geometry | Pickup endpoint | Final endpoint and protected surfaces | Combined checks |
|---|---:|---:|---:|---:|
| BNC_Male | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| DSUB_Male | 472 / 8 | 138 / 8 | 288 / 8 | 66 / 8 |
| Gear_Large | 8 / 8 | 4 / 4 | 0 / 0 | 0 / 0 |
| KET12_Square_12mm | 768 / 8 | 132 / 7 | 256 / 8 | 70 / 5 |
| KET16_Square_16mm | 720 / 8 | 232 / 8 | 256 / 8 | 136 / 6 |
| KET8_Square_8mm | 728 / 8 | 99 / 8 | 208 / 8 | 55 / 5 |
| M16_Hex_Nut | 328 / 8 | 12 / 3 | 32 / 8 | 12 / 3 |
| RGOCG12-50_12mm | 512 / 8 | 94 / 2 | 248 / 8 | 62 / 2 |
| RGOCG16-50_16mm | 488 / 8 | 51 / 8 | 248 / 8 | 31 / 1 |
| RGOCG8-50_8mm | 448 / 8 | 44 / 1 | 232 / 8 | 29 / 1 |
| USB_Male | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Waterproof_Male | 336 / 8 | 87 / 6 | 184 / 8 | 34 / 4 |
| TOTAL | 4808 / 80 | 893 / 55 | 1952 / 72 | 495 / 35 |

## finger_end_37

| Part | Contact geometry | Pickup endpoint | Final endpoint and protected surfaces | Combined checks |
|---|---:|---:|---:|---:|
| BNC_Male | 96 / 8 | 9 / 3 | 48 / 8 | 7 / 2 |
| DSUB_Male | 32 / 8 | 6 / 4 | 32 / 8 | 6 / 4 |
| Gear_Large | 16 / 8 | 8 / 8 | 8 / 8 | 4 / 4 |
| KET12_Square_12mm | 40 / 8 | 16 / 7 | 32 / 8 | 13 / 7 |
| KET16_Square_16mm | 24 / 8 | 12 / 4 | 24 / 8 | 12 / 4 |
| KET8_Square_8mm | 64 / 8 | 24 / 8 | 40 / 8 | 16 / 8 |
| M16_Hex_Nut | 104 / 8 | 12 / 6 | 24 / 8 | 9 / 3 |
| RGOCG12-50_12mm | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| RGOCG16-50_16mm | 8 / 8 | 1 / 1 | 8 / 8 | 1 / 1 |
| RGOCG8-50_8mm | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| USB_Male | 8 / 8 | 0 / 0 | 0 / 0 | 0 / 0 |
| Waterproof_Male | 24 / 8 | 7 / 4 | 24 / 8 | 7 / 4 |
| TOTAL | 416 / 80 | 95 / 45 | 240 / 72 | 75 / 37 |

Contact geometry combines the recorded input, opening, reconstructed contacts, normal alignment, finite contact support and target closing checks. Pickup endpoint adds table clearance. Final endpoint adds receiver clearance and protected-surface compatibility. Combined checks require all of them. An unavailable or unevaluated check never counts as a pass. Insertion paths, release, physical stability and execution accuracy were not evaluated.
