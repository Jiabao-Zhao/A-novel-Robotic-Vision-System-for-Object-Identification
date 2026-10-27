# Alignment diagnostics for the nine recovered placements

Read-only analysis of comparison/20261026T203838Z. No frozen result or code was changed. The error below is the larger angle between either selected first-contact CAD triangle normal and the gripper closing direction. It measures modeled local alignment, not assembly precision.

| Method | Placement | Passing pair families | Best / worst actual normal error (deg) | Families within1 /3 /5deg | Native tilt of best-aligned candidate (deg) |
|---|---|---:|---:|---:|---:|
| one_budget512 | N00/BNC_Male | 3 | 1.805 / 3.089 | 0 / 1 / 3 | 9.995 |
| one_budget512 | N01/BNC_Male | 5 | 1.838 / 4.071 | 0 / 1 / 5 | 10.716 |
| one_budget512 | N01/Gear_Large | 20 | 0.001 / 10.096 | 1 / 5 / 8 | 0.001 |
| one_budget512 | N03/BNC_Male | 5 | 0.753 / 4.183 | 1 / 2 / 5 | 12.007 |
| one_budget512 | N03/Gear_Large | 19 | 0.001 / 10.096 | 1 / 5 / 7 | 0.001 |
| one_budget512 | N04/BNC_Male | 4 | 1.677 / 4.143 | 0 / 3 / 4 | 13.113 |
| one_budget512 | N05/Gear_Large | 19 | 0.001 / 10.096 | 2 / 5 / 7 | 0.001 |
| one_budget512 | N06/BNC_Male | 5 | 0.754 / 4.190 | 1 / 3 / 5 | 12.006 |
| one_budget512 | N07/Gear_Large | 19 | 0.000 / 10.096 | 1 / 5 / 7 | 0.000 |
| graspkit_512 | N00/BNC_Male | 6 | 0.002 / 4.303 | 2 / 2 / 6 | 0.002 |
| graspkit_512 | N01/BNC_Male | 5 | 0.771 / 4.158 | 1 / 3 / 5 | 1.123 |
| graspkit_512 | N01/Gear_Large | 18 | 1.147 / 9.746 | 0 / 3 / 4 | 0.001 |
| graspkit_512 | N03/BNC_Male | 5 | 1.681 / 4.179 | 0 / 2 / 5 | 13.584 |
| graspkit_512 | N03/Gear_Large | 18 | 1.661 / 9.746 | 0 / 1 / 1 | 0.000 |
| graspkit_512 | N04/BNC_Male | 6 | 0.788 / 4.133 | 1 / 2 / 6 | 8.405 |
| graspkit_512 | N05/Gear_Large | 18 | 1.661 / 9.514 | 0 / 1 / 2 | 0.000 |
| graspkit_512 | N06/BNC_Male | 5 | 1.681 / 4.178 | 0 / 2 / 5 | 13.583 |
| graspkit_512 | N07/Gear_Large | 18 | 1.661 / 9.746 | 0 / 1 / 1 | 0.001 |

Both dense samplers provide at least one pose within 3 degrees at both selected first-contact normals in every recovered placement. At 1 degree, one_budget512 covers 6/9 and GraspKit covers 3/9. These differ from the 4/9 and 5/9 recovered by the strict native-pair projection control: the sampled contact line and the actual first-contact surfaces are different quantities. Some BNC candidates start with 10–14 degrees native pair tilt but reach actual contacts within about 1–2 degrees after the common finite-pad conversion.

This does not make every retained option equally aligned. Accepted large-gear alternatives reach 10.096 degrees in one_budget512 and 9.746 degrees in GraspKit. The one sampler also supplies a nearly zero-error gear option in every originally missing gear placement; GraspKit best gear options are approximately 1.147–1.661 degrees. The BNC best error ranges are 0.753–1.838 degrees for one and 0.002–1.681 degrees for GraspKit. A VLM view or metadata should expose these differences; the availability count alone hides them.

The modeled finger-stop mismatch among the recovered passing poses is at most 4.290 micrometers for one and 1.732 micrometers for GraspKit. These are CAD/model numerical outputs, not observed physical accuracy. Triangle normals describe selected witnesses, not flat contact across the full pad. Opposing contact forces, friction, grasp moments, slipping and assembly clearance were not evaluated.

The remaining one_budget512 miss is N06/RGOCG8-50_8mm: 503 native pairs, 9 pairs after the common 5 mm sparsity, 18 primary hand assignments, all rejected because the proposed contact lies outside the inner finger pad. This does not establish that the raw 503 pairs contain no feasible grip; early sparsity and the chosen height conversion remain possible bottlenecks.

The 95/96 result is candidate availability across 12 distinct CAD parts in 8 saved placements. It is not pickup success, physical quality, assembly precision, or a claim that all 14 parts were evaluated. Normal thresholds 1/3/5 degrees here are descriptive slices and have not been selected as a new acceptance policy.
