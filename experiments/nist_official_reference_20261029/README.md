# Official NIST Task Board 1 reference inspection

Read the [findings and corrections](../../docs/nist_task_board_1_reference.md)
and [official-source gallery](../../outputs/nist_official_reference_20261029/gallery/index.html).
This is document/static CAD inspection. Grasp generation, pose estimation,
VLM selection, pickup, assembly and physics execution did not run.

The current [task requirement](task_requirements.json) is a **20 mm pin-tip depth
below the board top**. It is implemented in the
[scene/task correction](../nist_scene_corrections_20261029/README.md), with the
current fingertip selector and corrected goal/protected-region inputs.
The earlier 12 mm mating-review outputs are preserved as historical evidence.

## Inputs and evidence

- NIST replication instructions v4, all 38 bill-of-material rows and nine original
  embedded figures. A fresh official download matches the saved DOCX byte-for-byte.
- Official laser SVG/AI/SolidWorks drawing and benchmark template.
- Previously saved official `cad.zip`, including both full SolidWorks assemblies.
- Unmodified original board STL and the earlier measured plane boundaries.

The public resource URLs are recorded in
[`web_findings.json`](../../outputs/nist_official_reference_20261029/web_findings.json).
Downloaded originals remain under `outputs/`; no models or global dependencies
were installed. DOCX page rendering was unavailable because LibreOffice was
absent. Original embedded figures were inspected directly instead.

## Local reproduction

Run with the existing local Python environment containing NumPy, SciPy, Shapely
and Matplotlib:

```powershell
D:\Python\python.exe -B -m experiments.nist_official_reference_20261029.extract_instructions
D:\Python\python.exe -B -m experiments.nist_official_reference_20261029.map_hole_roles
D:\Python\python.exe -B -m experiments.nist_official_reference_20261029.native_cad_audit
D:\Python\python.exe -B -m experiments.nist_official_reference_20261029.native_cad_audit --check
D:\Python\python.exe -B -m experiments.nist_official_reference_20261029.build_reference
```

The source files provide small, bounded inspection helpers:

- `extract_instructions.py`: read OOXML paragraphs, tables and original figures;
  paragraph positions are not rendered page numbers.
- `map_hole_roles.py`: associate documented drill/tap roles with measured native
  board boundaries; preserve raw mesh sizes separately from finishing tools.
- `native_cad_audit.py`: recover named metadata streams, saved instance transforms
  and native previews. Verify decompressed size/CRC and assembly bounding boxes.
  Native B-rep and mate constraints are not decoded.
- `task_requirements.json`: current user-chosen depth and paused stage status;
  this is not an executed or corrected goal registry.
- `build_reference.py`: create the static source gallery and a preservation receipt;
  it does not create new mating or grasp poses.

The scripts need the cached official archives and previous boundary audit, which
are not all included in a fresh GitHub checkout. They do not call the simulator
or a learned model. Saved SolidWorks placements must be converted explicitly to
our board and component STL frames before any resumed trial.

`web_findings.json`, `instructions_findings.json` and the compact
`native_cad/placement_summary.json` are authored inspection notes accompanying
the extracted raw evidence; they are not automatically inferred task goals.
Offline checks cover stream decoding/CRC, matrix transpose and parent chaining,
173 proper component transforms and reconstruction of assembly bounds.

All historical numerical measurements remain unchanged. A visible supersession
notice on the earlier mating review identifies the corrected mounting and hole
assignments. Frozen October 28 grasp outputs and selector inputs remain intact.
The active scene source changed during implementation; its exact original bytes
are saved and hash-checked through the recorded source migration.
