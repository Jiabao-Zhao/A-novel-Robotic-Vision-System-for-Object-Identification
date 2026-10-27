# Recommendation for the fixed-down NIST candidate generator

## Current decision, in plain language

**Use the released `one` contact sampler with about 512 surface samples, the tested downward Panda conversion and finger/part/table checks, and the new one-degree local contact-alignment requirement.** `one_budget512` is our experiment label for the sampler setting, not a learned model. The user's 27 October clarification keeps well-aligned side grips on round parts: these use local tangent alignment, rather than requiring two genuinely planar CAD faces.

The implemented trial rechecks **all 7,826 frozen primary hand poses** with the existing native-CAD/Panda checker and a one-degree normal limit. It retains **1,608 passing hand alternatives, 804 families and options in 89/96 placements**. Both pads must have a selected first-contact local surface normal within one degree of the closing direction. This is a controlled numerical setting for near-square contact, not a calibrated physical precision limit. The hand poses, openings, table, target solids and original source are unchanged. See the [audited results](../../outputs/cad_grasp_method_comparison_20261026/aligned_contacts/20261027T152238Z/summary.json).

The full stage order is: aligned CAD → original `one` opposing-contact sampler → saved 5 mm nearby-pair reduction → straight-down Panda placement with calculated finger height and centering → modeled finger/part/table validation with one-degree contact selection → all passing poses. No fitted-cylinder classification or exact shared-patch-overlap rejection is applied. The new angle rule applies uniformly to flat and curved surfaces; it is distinct from the older classifier-dependent one-degree flat-face screen. Moving pair reduction after full validation remains a supported next change, not a completed full-96 benchmark. VLM selection has not run.

“Parallel” and “lined up” describe different things. Parallel describes surface angle; lined-up contact areas refers to overlap in their positions when viewed along finger closure. The old thin-band polygon-overlap requirement remains removed. The new trial instead tightens actual local contact direction from fifteen to one degree. On a round side, a rigid flat pad can touch along a tangent line. Neither a selected CAD normal nor opposition alone guarantees a finite flat patch, force balance or zero movement while closing. The [contact-alignment note](contact_alignment_note.md) explains that distinction.

The [current gallery](../../outputs/cad_grasp_method_comparison_20261026/contact_atlas/20261027T152602Z/index.html) follows the user's reference: one white 3-by-4 sheet per scene, gray CAD silhouettes, colored sampled-pair lines and endpoint dots. It shows all retained families and keeps zero-result tiles. Lines follow the final gripper closing axis in the top-down projection; they join original sampled surface points, not independently selected pad-contact witnesses. Height and full contact patches are not shown. The previous 3D finger viewer remains historical.

The reference comparison at the former fifteen-degree contact limit found candidates in 95/96 placements for `one_budget512` and 83/96 for GraspKit. That selected the generator for the current trial; **95/96 is not the current one-degree result**. These remain contact-kernel adaptations, not complete upstream framework reproductions or demonstrated physical holding/assembly precision. No stricter full-scene GraspKit comparison is claimed.

The `one_budget512` surface density is set through an exposed parameter to obtain a comparable budget on each small part. GraspKit is the independent Open3D/Torch comparison. These results cover twelve CAD parts in eight placements, not all fourteen original parts. Both recover all nine previously missing BNC/gear placements with at least one passing candidate whose two selected first-contact normals are within three degrees of the closing direction. Some other accepted gear options deviate by approximately ten degrees. More accepted grasps therefore does not establish better grasp quality; see [alignment diagnostics](../../outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/quality_diagnostics.md). No pickup or assembly success has been measured in this comparison.

## What should replace the shape-specific hard filters

Use the sampler's opposing-contact condition, then check the **actual first surfaces reached by the finite fingers** under the common downward Panda conversion. Keep the modeled checks for finger travel, inner-pad contact, two-sided contact agreement, target interference, and table clearance. These apply to a cylinder, a gear, and a nut without requiring a successful cylinder fit or arbitrarily meaningful CAD X/Y axes. The reference source defaults to fifteen degrees; the new isolated trial tightens actual contact selection to one degree without modifying that frozen source file.

Neither the one-degree angle setting nor the fifty-micrometer modeled contact-agreement tolerance is physically calibrated. Record those settings and rejection reasons. An opposing point pair does not prove the whole pad lies flat on a gear tooth or threaded surface. A first-contact witness is reselected from all simultaneous contacts under the new limit; merely filtering the old saved witness would incorrectly lose additional options. The finite-hand check remains necessary even when an upstream sampler accepts the points.

GPG is a useful later independent contact-support comparison: its released code uses a 3 mm extremal band, shared transverse bounds and sample support. It should not be described as validating our exact 25 micrometer polygon-intersection rule. F-GPG is an assembly-specific alternative, but its furniture-scale edge sampling and ROS/PCL/FCL requirements make it a less immediate baseline for small round pins. [GPG source](https://github.com/atenpas/gpg/blob/master/src/gpg/antipodal.cpp), [F-GPG source](https://github.com/psh117/fgpg).

## Reduction without a mandatory count

[Kleeberger et al.](https://arxiv.org/html/2104.11660v1) first group candidate grasp midpoints by translation, then group the normalized closing directions inside each translation group. PAM k-medoids returns existing candidates as representatives, rather than averaging valid poses into a potentially invalid pose. The generic objective is

\[
M^*=\arg\min_{M\subseteq G,\;|M|=k}\sum_{g\in G}\min_{m\in M}d(g,m).
\]

Their two stages use position and closing-direction features separately. Our fixed-down approach removes the need to cluster a freely varying approach direction. A local implementation would still need an explicitly declared distance, cluster count or separation stopping rule, and treatment of swapped fingers. It would be a documented adaptation, not the authors' original software: no released generator was verified. Their paper targets offline processing and reports clustering times from minutes to hours. Do not promise its original implementation meets our runtime budget.

The present Fabrica sparsity function is a lighter alternative: keep a pair and suppress another only when **both corresponding contact endpoints** are within a radius. It does not require twelve outputs and does not rank grasp quality.

## Exact scope of the current implementation

| Component | What actually executes now | What is our adaptation |
| --- | --- | --- |
| `one` | Pinned upstream contact-generation/helper function bodies; ray batches bound memory. | Density settings, physical-gap interface, fixed-down pose, heights, first-contact centering, both finger assignments and finite-hand/table checks. Upstream roll expansion, weighted ranking and stop-after-fifty planner are absent. |
| GraspKit | Pinned `AntipodalSampler.sample` contact kernel using Open3D/Torch. | Same common Panda conversion/validation; added closest-triangle normal metadata does not change its sampler acceptance. The complete EmbodiChain framework is absent. |
| Fabrica | Original `filter_sparse_pairs` body using SciPy KD-trees. | Endpoint canonicalization, **5 mm radius**, use on either sampler's pairs and its position in our processing order. Its complete ray sampler, clearance spheres, gripper states, ranking and assembly optimizer are absent. |

Fabrica's original function accepts a caller-supplied distance; **5 mm is our comparison setting**, not a published universal optimum or a physical contact tolerance. Do not label the resulting column “Fabrica grasp planner.” Suitable wording is “upstream contact kernel + Fabrica pair sparsity at 5 mm + common downward Panda adapter.” [Original function](https://github.com/yunshengtian/Fabrica/blob/main/planning/robot/util_grasp.py).

The full Fabrica sampler cannot execute under the stated environment without trimesh/FCL. GPG requires its C++ PCL/Eigen build; F-GPG additionally needs ROS/FCL. The extracted Fabrica sparsity function needs only available NumPy/SciPy and does execute separately from that unavailable stack.

The source attribution in `upstream.py` and `run_compare.py` is explicit. Two interpretation limits remain: sparsity currently runs **before** finite-hand validation, so a failing representative can suppress a nearby passing pair; its first-kept representative is not the best-scored one. Also, the one-degree versus fifteen-degree projection control measures the sampled contact line's tilt before horizontal projection; it is different from the removed one-degree flat-surface-normal rule.

Finally, report intrinsic CAD sampling time separately from per-placement conversion/validation. The current runtime includes both conversion controls and both finger assignments, as its manifest states. It is not the deployment latency of one chosen protocol, the full upstream framework time, or the VLM/pose-estimation/pickup cycle time.
