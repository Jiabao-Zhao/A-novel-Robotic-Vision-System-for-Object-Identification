# Pinned upstream `one` contact-generation functions

Repository: https://github.com/wanweiwei07/one

Revision: `ec1bdae83ec33ca6af9de42904d9851647fe012c`.

The four Python files are unchanged official source files. `LICENSE` is the
repository's MIT license. No model weights, binary runtime or robot assets are
downloaded. The complete `one` framework is not installed or executed.

`../../upstream.py` executes these original function bodies from their source
AST, with NumPy and the original math constants:

- `antipodal._antipodal_candidates`
- `geometry_ops.triangle_areas`
- `geometry_ops.sample_count_from_area`
- `geometry_ops.sample_surface`
- `geometry_ops.ray_triangles_batch_far`

The last helper is invoked in batches of sixteen rays to bound memory on the
large native industrial meshes. Each ray is still checked against every native
triangle and retains the original farthest-hit rule. A deterministic regression
compares all returned distances and triangle indices with one unchunked call;
they are identical. The saved original file itself is not edited.

The normal and width masks reproduce the next stage in `antipodal_iter` for
the host's logical physical-gap interface: normals within the original twenty
degree tolerance, zero contact-depth correction, and physical gap0--79mm.
Panda's actuator command is1mm larger than its inner-pad gap. That conversion,
fixed-down frame construction, table and full hand checks belong to the common
host adapter and are not attributed to upstream.

Recorded variants preserve upstream20mm point spacing, change it to2mm, or
derive point spacing from native area to request a declared512-sample budget.
The latter two change the exposed density parameter, not the sampling algorithm.
Every variant keeps the original2mm clearance per side and NumPy float32 epsilon
unless a separately declared numerical override is requested. The original
iterator's roll expansion, heuristic sorting, collision framework and top-fifty
stop are not executed by this contact-only adapter. Therefore results must be
labeled **original upstream contact kernel plus common Panda adapter**, not an
unmodified end-to-end `one` experiment.
