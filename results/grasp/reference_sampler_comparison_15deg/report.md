# Original contact samplers with common Panda adapters

This executes pinned upstream contact-generation functions, not their complete robot frameworks. All 96 saved CAD placements and the target/table/hand checker are held fixed. The shared-region, fitted-cylinder and 1-degree flat-face hard filters are absent. No learned ranking, pickup, assembly or physical holding test is run.

| Sampler | Conversion | Poses checked | Poses passing | Placements with candidates | Original nine recovered |
|---|---|---:|---:|---:|---:|
| graspkit_512 | midpoint_proj1 | 4904 | 1191 | 40/96 | 4/9 |
| graspkit_512 | midpoint_proj15 | 5530 | 1191 | 40/96 | 4/9 |
| graspkit_512 | panda_table_proj1 | 4904 | 1957 | 79/96 | 5/9 |
| graspkit_512 | panda_table_proj15 | 5530 | 2239 | 83/96 | 9/9 |
| one_budget512 | midpoint_proj1 | 6624 | 1032 | 40/96 | 4/9 |
| one_budget512 | midpoint_proj15 | 7826 | 1044 | 40/96 | 4/9 |
| one_budget512 | panda_table_proj1 | 6624 | 1806 | 88/96 | 4/9 |
| one_budget512 | panda_table_proj15 | 7826 | 2108 | 95/96 | 9/9 |
| one_default | midpoint_proj1 | 862 | 146 | 29/96 | 0/9 |
| one_default | midpoint_proj15 | 970 | 146 | 29/96 | 0/9 |
| one_default | panda_table_proj1 | 862 | 234 | 48/96 | 0/9 |
| one_default | panda_table_proj15 | 970 | 280 | 58/96 | 1/9 |

`midpoint` places the finite finger-pad center at the sampled pair midpoint. `panda_table` also raises the hand to clear the table and centers it using modeled first finger contacts. `proj1` permits at most 1 degree adjustment of the native pair to horizontal; `proj15` permits 15 degrees. Every resulting hand approaches straight down and is independently rechecked against native target triangles and the table.

| Sampler | Cached CAD generation, sum for 12 parts (s) | Median runtime preparation + Panda checking (s) | Retained primary families per placement |
|---|---:|---:|---:|
| graspkit_512 | 0.336 | 0.733 | [0, 42] |
| one_budget512 | 32.116 | 1.001 | [0, 39] |
| one_default | 1.608 | 0.149 | [0, 11] |

CPU times under four concurrent workers. Primary component adds shared preparation and Panda-adapter checking; excludes other control checks but is instrumented inside a two-control run, not an isolated deployment timing. CAD-intrinsic sampling once per part is reported separately. No pose estimation, VLM, physics or pickup time.

## Original nine missing placements

| Placement | one default | one budget 512 | GraspKit 512 |
|---|---:|---:|---:|
| N00/BNC_Male | 2 | 3 | 6 |
| N01/BNC_Male | 0 | 5 | 5 |
| N01/Gear_Large | 0 | 20 | 18 |
| N03/BNC_Male | 0 | 5 | 5 |
| N03/Gear_Large | 0 | 19 | 18 |
| N04/BNC_Male | 0 | 4 | 6 |
| N05/Gear_Large | 0 | 19 | 18 |
| N06/BNC_Male | 0 | 5 | 5 |
| N07/Gear_Large | 0 | 19 | 18 |

Counts in the last table are distinct pair families under `panda_table_proj15`, with both passing jaw assignments preserved in the detailed records. Availability does not establish that the grasp leaves the insertion end free.

## Per-part primary coverage

| Part | one default | one budget 512 | GraspKit 512 |
|---|---:|---:|---:|
| BNC_Male | 4/8 | 8/8 | 8/8 |
| DSUB_Male | 8/8 | 8/8 | 8/8 |
| Gear_Large | 0/8 | 8/8 | 8/8 |
| KET12_Square_12mm | 5/8 | 8/8 | 8/8 |
| KET16_Square_16mm | 8/8 | 8/8 | 8/8 |
| KET8_Square_8mm | 3/8 | 8/8 | 8/8 |
| M16_Hex_Nut | 6/8 | 8/8 | 8/8 |
| RGOCG12-50_12mm | 7/8 | 8/8 | 8/8 |
| RGOCG16-50_16mm | 8/8 | 8/8 | 8/8 |
| RGOCG8-50_8mm | 1/8 | 7/8 | 3/8 |
| USB_Male | 0/8 | 8/8 | 0/8 |
| Waterproof_Male | 8/8 | 8/8 | 8/8 |
