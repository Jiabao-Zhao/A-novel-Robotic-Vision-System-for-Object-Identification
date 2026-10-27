# Smaller grasp search on all 14 parts

The completed comparison picked up **14/14 parts with each search**. Total
search time fell from **17.49 to 4.28 seconds** across all parts, a **75.5%**
reduction. Median search time per part fell from **1.250 to 0.266 seconds**;
including scene setup, the medians were **2.883 and 1.963 seconds**.
The smaller search generated **92.4% fewer intermediate gripper poses** and
saved **156 shortlisted gripper poses across 85 distinct contact pairs**:
12 poses for most parts, 10 for the 12 mm round pin, and 2 for the 8 mm round
pin. The last two exchange the jaws at the same contact pair. The 8 mm pin
used +8 mm depth and lifted 179.9 mm.

- [Full per-part time and pickup comparison](../../../outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z/report.md)
- [Exact generation, score, shortlist and table-check mathematics](MATH.md)
- [Proposed downward-grasp and modeled-contact refinement](REFINED_METHOD.md)
- [Every shortlisted grasp on all 14 parts](../../../outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z/shortlist_gallery/index.html)
- [All kept contact points drawn on every CAD, from two sides](../../../outputs/runtime_antipodal_assembly_20261023/contact_point_views/20261026T143511Z/index.html)
- [Time comparison chart](../../../outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z/search_times.png)

The contact-point overview redraws the saved results without generating new
grasps or running physics. The 156 kept poses share 85 contact pairs (170 points).
Matching letters identify a pair; the G numbers map it to the original poses.
All contacts remain visible through the CAD, including those on the hidden side.

This local simulation compares a small runtime grasp search with a fresh larger
search on the same 14 starting part poses. A separate comparison retains the
original saved run. Part identity and initial CAD alignment are assumed correct.

The fixed small search uses 512 surface rays and expands at most 24 pairs of
opposing finger touches. It keeps up to 12 table-clear gripper poses, favouring
different contact regions before adding a second option at one region. The
larger control uses 2,048 rays, 320 pairs and up to 48 table-clear poses in the
original score order. Both check robot movement until the first passing pose.
Neither retries a failed physical pickup with another pose.

Both searches keep the same six finger-depth settings: -8, -4, 0, +4, +6 and
+8 mm. A larger positive setting puts the contact nearer the finger ends;
for a downward grasp this raises the gripper. The last two settings were added
in the successful earlier 8 mm round-pin follow-up. No part-specific grasp
location is supplied.

Each arm starts from the prepared `objects_centered` scene and commands an
18 cm lift followed by a one-second hold. A pickup passes if the final part
height increased by at least 12 cm, both fingers still touch it, and recorded
robot/self/table/board overlap stays below 1 mm. The same contact-driven
controller and contact settings are used. Parts are never attached to the hand.
Only one target moves; the other 13 loose parts stay visible, with their
collisions disabled. Their saved positions are restored every physics substep;
the recorded one-step gravity shift is 0.03924 mm and does not accumulate.
The board and fixed hardware remain physical.
The planner checks the robot, table and open-hand approach to the target;
contacts with the board and its hardware are audited during physical movement.
This is pickup testing; no assembly motion is attempted.

The two searches run serially in one process, alternating which goes first for
each part. Stage times separate setup, finding contact pairs, forming gripper
poses, checking the table, forming the shortlist and checking robot movement.
Search timing excludes imports, scene setup, bulky saved-evidence writes,
physics execution and pictures. Setup-inclusive planning time is also recorded.
There is one timing observation and one possible physical attempt per part per
condition; this is not a speed or reliability study across different scenes.

The original 14-part run used four depths and searched assembly routes before
trying pickup-only routes. It checked pickup at the end of the lift without
the new hold. Its time and success count are historical context, not the direct
comparison of search budgets. The new larger control removes these differences.

Sources:

- [Search and shortlist](search.py)
- [Run the comparison](run_comparison.py)
- [Independent record audit](audit.py)
- [Show every shortlisted pose](gallery.py)
- [QuickGrasp and the banana example](QUICKGRASP.md)

Run in the existing WSL environment from the repository root:

```bash
/home/jiabao/.venvs/lerobot-libero/bin/python -m experiments.runtime_antipodal_assembly_20261023.smaller_search_20261023.run_comparison
```

Every run creates a new timestamped output folder. Frozen earlier records and
shared pose-estimation/classification sources are not rewritten.
