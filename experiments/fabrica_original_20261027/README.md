# Original Fabrica grasp-generation stage on NIST CAD

This study runs the released Fabrica contact sampler, gripper pose construction,
and target/gripper collision checks. The previous `fabrica_candidates_20261027`
study used a NumPy port and our finite-pad checker. This study executes the
unchanged upstream function and class-method bodies instead.

The scope remains **candidate generation only**. VLM selection and its prompt,
pickup, assembly, training and physical robot execution are untouched. The input
is the saved observed CAD alignment for twelve parts in eight scenes, not the
simulator's object-state estimate. No fixture is supplied or generated.

## Original source and dependencies

The thirteen upstream files, including Panda hand/finger collision meshes and
the MIT license, are saved under `vendor/fabrica/`. Their exact byte lengths and
Git blob hashes are recorded in `vendor/source_manifest.json` and tested against
[commit 215a30f](https://github.com/yunshengtian/Fabrica/tree/215a30fe51e59299588a5b5e417a9cb934fb393e).

`trimesh` performs surface sampling and triangle-ray queries. `python-fcl`
provides collision checks for clearance spheres and the gripper. `rtree`
accelerates ray queries. They are needed to execute these original code paths;
installing them alone does not reproduce the paper's complete assembly results.
Those also depend on its task assets, settings, planning, fixtures and learned
assembly policies, which are outside this candidate-only trial.

The dependencies are isolated under
`outputs/cache/fabrica_native_20261027/python/`; the shared environment is not
modified. Installed versions:

| Package | Version | Reason |
|---|---|---|
| trimesh | 4.4.1 | Fabrica's pinned version |
| python-fcl | 0.7.0.6 | Fabrica's pinned version |
| rtree | 1.3.0 | Fabrica's pinned version |
| NumPy | 1.26.4 | Compatible with this FCL wheel and existing Python 3.12 |
| SciPy | 1.12.0 | Compatible with the isolated NumPy and Python 3.12 |

Fabrica's older NumPy/SciPy pins do not support this existing Python version.
Consequently, this is original algorithm/source execution on our NIST inputs,
not a claim of an identical historical software environment or paper benchmark.

## Object-only context

The user requested removal of our physical constraints for this trial. We do not
apply our downward-only rule, 1 degree contact-normal rule, pad-depth variants,
contact-centering adjustment, table checker, duplicate grouping or custom score.

The original assembly constructor expects assembly paths and a fixture workcell.
`native.py` supplies a one-part identity-path context and executes the original
`GraspGenerator.generate_grasps` and `check_grasp_feasible` method bodies. An AST
loader excludes unused top-level imports and drivers; it does not rewrite the
function or method bodies. Trace wrappers save decisions without changing them.

Two workspace settings are explicitly different from the native default:

- `compute_antipodal_pairs(..., ground_z=-inf)` disables its ground-height gate
  using the function's existing parameter.
- The ground collision manager is empty. There is no table, fixture, robot arm
  or other loose part in this object-only context.

Fabrica's own target/gripper rules remain active:

- Opposite sampled normals satisfy `n2 dot (-n1) > 0.95`.
- Native surface budget 200, clearance-sphere radius 1 cm, pair spacing 0.05 cm.
- Ten angles are sampled around each contact line. The original code rejects
  upward-pointing hand-base directions, normally leaving five orientations. It
  allows tilted approaches; it does not require our former straight-down pose.
- Panda opening is at most 95% of its 80 mm range: 76 mm.
- Original target/gripper FCL checks test the grasp and wider retract opening.
  Their numerical opening margin of 0.05 ratio is preserved. This is not our
  finite-pad first-contact or finger-closing simulation.
- `max_n_grasp=None`: no random final cap, VLM selection or custom shortlist.

The 200-point budget is not a promise of exactly 200 returned pairs. The original
sampler accumulates complete batches until enough pairs pass, then removes close
pairs and orders their midpoints by distance to the mesh surface center. A final
batch can take the count above 200; branching ray hits can also add pairs.

## Native ground limitation

The native sampler requires both contacts to satisfy `z > ground_z + 1 cm` and
uses an unbounded loop until its pair budget is reached. A resting 8 mm pin
cannot satisfy the default gate. The runner records a maximum-vertex-height
preflight against the literal default `ground_z=0`; it does not run that known
nonterminating case or call it a successful native-default trial.

The actual object-only run uses the explicitly declared `ground_z=-inf`. Each
case additionally has an external 90 second timeout. A timeout is reported as
unavailable, without claiming that there is no grasp or saving partial output
as a completed result.

## Units and frames

CAD files, frozen input transforms and exported results use **metres**. Fabrica's
native geometry uses **centimetres**. Aligned CAD vertices are multiplied by 100
before native execution; contacts and hand translations are divided by 100 on
export. CAD rotation and native mesh geometry are not changed to fit a desired
outcome.

Poses are exported as `world_T_fabrica_panda_hand`, not as our old grip-site
frame. Native Panda local Y is the closing axis; quaternions use `wxyz`. Its hand
base is 11.24 cm from the contact-pair center. No unverified hand-to-grip-site or
camera calibration transform is invented.

## Gallery preference

Use the user's requested format for grasp research: white background, top-down
CAD silhouettes, one tile per part, and colored lines/dots joining contact pairs.
Do not replace this with motion-failure charts, gripper close-ups or final
assembly views unless requested.

One contact pair can support multiple hand orientations. Show pair and full-pose
counts separately; a line cannot show height or orientation. The gallery saves
all-pair images and permits paging through twelve pairs per part in saved order.
Twelve is only a display page size. No candidate is removed, scored or selected
by pagination; every pair remains reachable, with an all-pairs view available.

## Outputs and verification

Every case saves all sampled pairs, all generated hand poses with native
pass/reject decisions, every retained 6D pose, and the pairs supporting retained
poses. Root results, summary, protected-file hashes and a count/frame audit are
saved in a fresh timestamped output directory.

The offline tests check exact upstream files, isolated dependency versions and
real FCL queries, analytic box contacts, Panda pose axes/offsets, and the original
target collision checker on a clear pose and a deliberately overlapping palm.
Candidate availability and an FCL pass do not establish holding stability,
unchanged part position during closing, or precise insertion.

Run from the repository root using the existing WSL Python. Put the absolute
dependency-cache path on `PYTHONPATH` before starting pytest, and disable pytest
plugin autoload so a plugin cannot preload the shared NumPy version.

```bash
PYTHONPATH="$PWD/outputs/cache/fabrica_native_20261027/python" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/jiabao/.venvs/lerobot-libero/bin/python -B -m pytest -q -p no:cacheprovider experiments/fabrica_original_20261027/test_native.py

PYTHONPATH="$PWD/outputs/cache/fabrica_native_20261027/python" \
/home/jiabao/.venvs/lerobot-libero/bin/python -B -m experiments.fabrica_original_20261027.run_trial

/home/jiabao/.venvs/lerobot-libero/bin/python -B -m experiments.fabrica_original_20261027.gallery RUN_DIRECTORY
```
