# October 29 grasp evidence

These are preserved geometric trials and a controlled follow-up. Availability of a valid modeled grasp does not establish pickup or assembly success. VLM selection is excluded; pickup and assembly execution remain paused. Each run has its own exact-byte publication manifest and complete compressed candidate diagnostics. Historical numerical outcomes are unchanged.

| Saved trial | Placements selected | Valid candidates | Stored evidence | Gallery |
|---|---:|---:|---:|---|
| [Preserved modeled finger contacts: 37/96](finger_contacts/README.md) | 37/96 | 75 | 26.38 MiB | [finger_contacts/gallery](finger_contacts/gallery/index.html) |
| [Preserved support-contact baseline: 35/96](support_baseline/README.md) | 35/96 | 495 | 39.34 MiB | [support_baseline/gallery](support_baseline/gallery/index.html) |

The 35/96 support baseline combines 72 unchanged cases and 24 corrected round-pin cases; its derivation and source manifests are included. It has no standalone 96-case elapsed time. The 37/96 finger-contact trial changes the contact model, so its lower number of valid candidates is not an isolated improvement or regression under an identical checker. The new controlled comparison reports its own shared conditions and timing scope.

Shared task and hand references are included separately when listed in their manifest. Native CAD files already tracked by the repository are not duplicated. The publication process performs no trial, robot action, git stage, commit or push.

Read the [saved baseline stage counts](shared_inputs/review_baseline/saved_stage_counts.md), [all stage counts](shared_inputs/review_baseline/saved_stage_counts.json), [exact shared-input manifest](shared_inputs/publication_manifest.json), and [required receiver asset manifest](receiver_assets_manifest.json). The receiver meshes remain at their original individually allowed repository paths. The exact historical publisher source is preserved in `publisher_sources/` under the SHA256 recorded by each publication manifest.

<!-- controlled-trial-publication -->

## Complete controlled comparison

The [complete controlled evidence](controlled_trial/README.md) preserves 1728 policy-placement records, including every failure, all shared banks, the fixed primary gallery and exact publication hashes. These counts describe modeled geometry availability; VLM selection is excluded and no pickup or assembly was executed.

| Proposal strategy | Contact policy | Selected placements over three seeds /288 | Valid modeled candidates |
|---|---|---:|---:|
| random | pad | 114 | 1661 |
| random | finger | 104 | 236 |
| random | combined | 131 | 924 |
| task_informed | pad | 174 | 4430 |
| task_informed | finger | 174 | 1024 |
| task_informed | combined | 203 | 2751 |

Read the [paired report](controlled_trial/REPORT.md), [all policy decisions](controlled_trial/decisions.csv), [per-part comparison](controlled_trial/per_part_comparison.csv), [primary selected-grasp gallery](controlled_trial/primary/gallery/index.html), and [manifest](controlled_trial/publication_manifest.json). The displayed branch is task-informed combined mode, seed 0 fixed before results, rather than a selected best seed.

The 288 rows per arm repeat the same 96 placements under three seeds. The benchmark evaluates two contact versions of each shared raw bank; its total work differs from each policy's 100-candidate budget. Its wall time covers the complete shared comparison and frozen-angle sweep, not an independently measured runtime or speedup for one policy. Detailed visual/collision surface mismatch remains a separate model limitation.
