# Runtime grasp method selection

This experiment assumes successful CAD identification and an aligned metric CAD
pose. It evaluates grasp generation and simulated manipulation separately from
classification and pose-estimation accuracy. The existing perception prompts,
scorers and frozen experiments are not changed.

## Selected implementation

Use the existing **EmbodiChain GraspKit AntipodalSampler**, pinned to commit
`bb2b65e0de1ccc486e022e57a6a56d8ddd78e969`:
[exact upstream source](https://github.com/DexForce/EmbodiChain/blob/bb2b65e0de1ccc486e022e57a6a56d8ddd78e969/embodichain/toolkits/graspkit/pg_grasp/antipodal_sampler.py).
The source is Apache-2.0. The narrow vendored component requires NumPy, Open3D,
PyTorch and Python standard-library support. It uses PyTorch for tensor math;
there is no learned or pretrained grasp stage, model checkpoint, grasp dataset,
or manually annotated grasp pose.

The executable source changes only the upstream configuration/logger imports.
The byte-exact upstream file, license, local modification notice, SHA-256 hashes
and upstream Git blob identities are retained under `vendor/`. Import and output
checks passed in the existing local Python runtime without installing the full
EmbodiChain framework. See `vendor/provenance.json` for the synthetic-box check.

The pinned algorithm casts Fibonacci-sphere rays toward the mesh, uses surface
normals to seek opposite contact points and filters pair width and hit-normal
angle. It returns contact pairs, rather than robot trajectories or validated
grasps. Its maximum-angle threshold must be positive; the random ray-rotation
helper is not called by the pinned sampling path.

## Primary-source comparison

| Candidate | Training-free component and implementation | Integration assessment |
| --- | --- | --- |
| [EmbodiChain GraspKit](https://dexforce.github.io/EmbodiChain/v0.2.4/api_reference/embodichain/embodichain.toolkits.html) | Apache-2.0 geometric mesh sampling; existing Open3D/PyTorch source | Best fit for this experiment when only the contact sampler is vendored. The full framework adds an unrelated simulator and many dependencies. Versioned documentation describes an older API; the pinned source above is authoritative. |
| [WRS](https://github.com/wanweiwei07/wrs) | MIT geometric antipodal planning with robot/gripper models | Broad planning system with Panda3D and its own collision/robot abstractions. Its repository identifies itself as archived. Integrating that framework would duplicate existing MuJoCo infrastructure. |
| [BURG toolkit](https://github.com/mrudorfer/burg-toolkit) | MIT antipodal mesh sampler with optional simulation | Useful established reference, but [packaging](https://github.com/mrudorfer/burg-toolkit/blob/master/setup.py) pins Python below 3.9 and Open3D 0.12. Its sampler also warrants repairs before use: the width predicate uses OR where bounds normally require AND, and the sample loop has no failure budget. |
| [QuickGrasp](https://arxiv.org/abs/2504.19716) | Paper proposes analytical contact-region selection and optimization | Relevant alternative method. An official, runnable source release suitable for integration was not established in this review; this is not a claim that none exists. Its reported experiments do not establish superiority for this task board. |

This selection is **best fit for the available runtime and integration scope**,
not a claim of universally best grasp success or a benchmark comparison between
these methods. The other systems were inspected, not installed or tested here.

## Runtime use and limits

The host transforms the CAD vertices with the supplied current pose and invokes
the sampler anew. No part-specific grasp library is loaded. The sampler sees
the aligned target geometry; it does not observe the table, robot or other
objects. Host-side geometry and kinematic checks supply those constraints.
Precomputing contact candidates would be an optional optimization, not a method
requirement, and is not the protocol used here.

The host constructs jaw frames from contacts, considers generic approach rolls
and pad depths, and evaluates the complete operation. Noncontact gripper/target
penetration, gripper opening and the approach/closure path still matter even
when other loose parts are deliberately excluded. Contact-angle heuristics do
not establish force closure for an arbitrary real finger/contact model.

The requested simplified planning scope is robot self-collision and the table,
with other loose products excluded. Receiving task-board contacts must be
distinguished from loose-object obstacles and audited during physical insertion.
Endpoint reachability or a gripper pose alone is not evidence of successful
grasping, transport, release or insertion. Runtime limits, unavailable candidates,
slips, collisions and final assembly errors must remain explicit in results.
