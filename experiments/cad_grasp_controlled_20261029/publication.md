# Publishing the saved grasp evidence

`publish.py` prepares reviewable copies without running generation, selection,
fitting, a model, a robot or a git command. Original study JSON and historical
numerical outcomes remain unchanged. It does not stage, commit or push files.

The explicit allowlist includes the saved summary, results, reports, verification,
calibration and launch/derivation receipts; each case's selection and complete
candidate diagnostics; saved task inputs; and the selected-grasp gallery's HTML,
PNG sheets and named receipts. Unknown files, weights, captures, installed
dependencies and native CAD duplicates are excluded. `plan_run(source)` inventories
the whole source tree before enumerating the exact included paths and bytes.
Additional provenance files require explicit source/destination pairs.

`publish_run(source, destination, label=..., supplements=[...])` writes a new
directory and rejects an existing destination. Complete results and per-case
JSON banks are stored as deterministic gzip streams, with a 95 MiB limit per
stored file. Each stream is decompressed again to verify its exact original
SHA256 and byte count. The source JSON is not reformatted or overwritten.
Small reports and images remain exact uncompressed copies. The publication
manifest records the pre-copy inventory and every source/stored hash and size.
`decisions.csv` is clearly labeled as a derived compact table of saved values;
the original full decision and diagnostic banks remain available alongside it.

`publish_inputs([(source, relative_name), ...], destination)` publishes explicitly
enumerated shared hand, observed-alignment, goal and source-contact references.
It preserves their bytes with the same hash checks. Native NIST CAD meshes
already tracked by the repository stay at their existing paths.

Use `verify_bundle(destination)` to independently recheck all stored/decompressed
hashes and the compact table. `write_index(publication_root)` links each saved
run's report, decisions, manifest and self-contained selected-grasp gallery.
Historical report links are preserved rather than silently rewritten; the new
index identifies the files actually included. Compressed JSON can be read with
Python's standard `gzip.open` and `json.load`.

`write_stage_tables(stage_counts_path, destination)` formats the scene-audit
schema's per-part contact, pickup-endpoint, final-endpoint/task and combined
counts. It verifies that per-part totals match the saved totals and records
source/output hashes. It does not reevaluate geometry or interpret an unavailable
check as a pass. The table labels explicitly distinguish modeled endpoint
geometry from executed pickup or assembly success.

The preserved 35/96 support result is a derived set of 72 unchanged cases plus
24 round-pin corrections, with exact case provenance and no standalone 96-case
elapsed time. The 37/96 finger-contact result uses a different contact mode and
retains 75 valid candidates rather than 495. Those historical counts are not a
controlled timing comparison or physical performance evidence. The new trial
documents its own controlled conditions separately. All published grasp counts
are modeled candidate availability. VLM selection is excluded; pickup and
assembly execution remain paused.

The publication namespace is `results/grasp/controlled_20261029/`. Its narrow
`.gitignore` exceptions admit only the named compressed result/selection/diagnostic
files, rather than exposing all local `outputs/` or arbitrary model/pose banks.
Eleven individually listed existing receiver/plate files are also allowed at
their original asset paths: nine required receiver meshes, the original virtual
plate and its finished-hole derivation receipt. Their hashes and roles are in
`receiver_assets_manifest.json`; publication does not modify or duplicate them.

The new `publish_controlled.py` adapter was added after the controlled benchmark
launch. It supports that runner's full 1,728-row policy matrix and 576 shared
bank directories, rather than adapting its results into a 96-row historical
layout. Existing bank gzip bytes are copied exactly and their decoded JSON
hashes are also checked. Original policy-selection/result JSON is compressed
without reformatting. The compact complete decision table identifies strategy,
seed, policy, selected contact mode, stage counts and linked full shared banks.
Combined mode is reconstructed with the saved `combined_assignment`, never an
after-the-fact union of the two diagnostic lists.

`publish_controlled(source, destination, supplements=[...])` inventories before
writing, checks the complete matrix and policy budgets, rejects an existing
destination, and stores its own exact post-launch source alongside the evidence.
`verify_controlled(destination)` checks stored/source/decoded-bank hashes.
The explicit allowlist includes `REPORT.md`, `comparison.json` and
`per_part_comparison.csv`, saved calibration/launch/run verification, task inputs,
part receipts, angle sensitivity and the primary gallery. Additional contact or
independent audit files require explicitly enumerated supplements if their names
are not already allowed. No existing launched Python or historical study output
is changed to prepare this publication.
