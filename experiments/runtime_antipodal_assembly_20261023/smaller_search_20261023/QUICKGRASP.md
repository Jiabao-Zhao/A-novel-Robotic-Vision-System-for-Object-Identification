# QuickGrasp and the banana example

**2026-10-24 correction:** the user identified **RoboGround** as the intended
paper. The identification guess below was incorrect. This note is retained as
a review of a different method. See the [RoboGround paper/code review and VLM
grasp-ranking proposal](../../roboground_grasp_review_20261024/README.md).

The likely paper is Ravie et al., **QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds**, ICRA 2025 ([author manuscript](https://arxiv.org/abs/2504.19716)). This is a strong identification, rather than a confirmed recollection of the earlier conversation.

## What the paper shows and does

[Figure 2, PDF page 3](https://arxiv.org/pdf/2504.19716#page=3) shows banana/flask point clouds and fitted surface patches. [Figure 3, page 5](https://arxiv.org/pdf/2504.19716#page=5) shows grippers and finger contacts.

The published workflow groups surface points into approximately flat patches, finds opposing overlapping patches, models contact forces, optimizes a stability score, and selects the best candidate. Algorithm 1 does not specify a fixed shortlist size. Section II-D derives gripper poses from contacts and gripper dimensions; Section III-C separately uses MoveIt for robot movement.

The analytical grasp stage uses no trained grasp network. The complete demonstration additionally uses pretrained SAM and MCC3D. Section III-B reports approximately 2 seconds on a six-core laptop with 8 GB RAM and 3.3 seconds on Raspberry Pi 4. These are the authors' measurements, not our timing comparison.

No author-linked runnable release was verified in the manuscript/arXiv record or targeted code searches on 2026-10-23. This does not establish that no code exists.

## Difference from our experiment

These statements describe our checked local source, rather than QuickGrasp:

| Step | Current GraspKit integration | Smaller-search experiment |
| --- | --- | --- |
| Find possible finger contacts | Cast rays at aligned CAD, then qualify both surface normals and remove duplicate pairs | Same geometric contact sampler, 512 rays instead of 2,048 |
| Limit contact pairs used for gripper positions | Up to 320 | Up to 24 |
| Construct gripper positions | Try generic approach rotations, exchange the jaws, and vary finger depth | Same construction, including the +6 and +8 mm depths from the successful low-pin follow-up |
| Keep candidates for robot movement checks | First 48 after table filtering | At most 12, preferring different contact regions before a second depth |
| Execute pickup | First candidate whose planned pickup passes the checks | Same rule; one simulated physical attempt per part and condition |

Source: [search.py](search.py), [run_comparison.py](run_comparison.py), and [original contact/frame construction](../grasp_planning.py). The full finger-depth set is +4, 0, -4, -8, +6 and +8 mm along the gripper's local approach axis. These are grip-site/contact offsets, not distances measured from the physical fingertip.

The smaller search remains **GraspKit plus our own bounded search**. It is not a QuickGrasp implementation or reproduction.

## Practical CAD implementation, if tested separately

The following is our proposed engineering work, not a claim of published or completed implementation:

1. Sample the native-scale CAD into a point cloud with outward surface normals; transform it into the world frame using the supplied aligned pose. This replaces the need to reconstruct unseen surfaces from a camera.
2. Group neighboring surface points by normal and distance, using tolerances expressed in metres and degrees. Preserve curved surfaces as several local patches. Check minimum patch area against the real finger contact size.
3. Compare opposite-facing patch regions within the gripper's opening range. Project them onto a shared two-dimensional plane and intersect their footprints. Retain actual surface points and original plane offsets.
4. Choose paired contact locations in that overlap. Use the CAD volume centre only when the mesh is closed and a uniform-density assumption is declared; actual mass distribution may differ.
5. Implement and verify the paper's force/stability optimization with explicit friction and finger-force limits. A geometric score alone must not be relabelled a physical stability guarantee.
6. Derive a small set of gripper orientations/depths from each contact pair, check the full finger and palm geometry against the target/table, then check robot movement. Keep the successful low-pin depths.
7. Benchmark the separate implementation against the same fixed scenes, hardware and pickup test. Report time by stage and failed searches explicitly.

Two contact points define the jaw-closing line and separation. They do **not** uniquely determine rotation around that line, where the contacts lie along the fingers, or a reachable collision-free robot movement. Sparse contact illustrations therefore do not remove those checks.

A mathematical detail needs resolution before copying the published equations: Section II-B writes an ordinary inverse for an orthogonal projection matrix. For a unit normal, `P = I - n n^T` has rank two and no ordinary inverse. In implementation, retain the plane offset and lift two-dimensional overlap coordinates back onto each original surface plane; do not blindly invert that matrix.

No QuickGrasp code, reconstructed cloud, optimizer, or different grasp method is introduced by the current smaller-search trial.

