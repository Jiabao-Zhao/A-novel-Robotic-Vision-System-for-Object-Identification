# Shared NIST Task Board data

- `shared_inputs/direct_cad_captures/`: twelve-part N00–N07 RGB-D captures,
  camera metadata, depth-localized crops and audit-only labels.
- `shared_inputs/cad_reference_views/`: CAD meshes, reference images and render
  metadata used by retained NIST comparisons.
- `board_captures/`: eighteen earlier NIST board captures.
- `shared_inputs/fixture_assets/`: meshes required to reopen preserved scene
  XMLs, including a fixed background fixture and unused historical mesh
  declarations. These are dependencies, not additional enabled object layouts.
- `shared_inputs/generation_models.json`: local description-model metadata.
  It contains no token-likelihood classifier scores.

The shared inputs were extracted from retired experiment folders before deletion.
`shared_inputs/migration_20261027.json` records 421 byte-identical copied files.
The fixture and source-retarget receipts in that directory, plus the
[path migration receipt](../../cleanup_20261027/path_migration.json), document path
updates. Image pixels, CAD geometry, numerical scores and poses were not changed.
Original hash inventories remain records of the original checkout.

Simulator identities and poses are for auditing only. Recognition and alignment
must use observed RGB-D, camera calibration and the requested CAD.
