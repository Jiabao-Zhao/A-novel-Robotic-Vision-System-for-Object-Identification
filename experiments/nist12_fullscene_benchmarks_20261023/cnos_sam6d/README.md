# Official full-scene identification runs

`acquire_sources.py` downloads exact official source archives into this trial's
`sources/cnos` and `sources/sam6d`, with complete source SHA-256 receipts. CNOS is
revision `298d1f3366171464ca271659f0e2f7a6eb8e39b4`; SAM-6D is
`1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`. Source trees remain unedited.

`render_templates.py` calls the official CNOS Pyrender `render` function with
the released level-0-in-level-2 42-pose mapping, 400 mm camera distance, original
CAD colors and native camera/light defaults. This is the released CAD/Pyrender
reference variant, not public BOP PBR training-image onboarding. Every method
reads those RGBA images through its own official BOP template loader. An extra
canonical view301 supports NOCTIS's different native view mapping. Native metric
CAD geometry is retained for SAM-6D projection; centering occurs only in renders.

`run_native.py cnos propose` and `run_native.py sam6d propose` generate separate
SAM ViT-H proposal banks from complete RGB frames. Released CNOS stability0.97
and SAM-6D stability0.85, width640, native proposal filters and point grid remain.
Decoder point batches are reduced64to8 for memory. Native bilinear upsampling is
performed in chunks then copied to CPU. Stored float16 soft masks must roundtrip
exactly to the original float32 values; no threshold changes enter inference.

`run_native.py cnos match` and `run_native.py sam6d match` call official native
preprocessing, DINOv2 ViT-L/14, class assignment, score thresholds and per-class
NMS. All12 CAD classes compete as in the released full identification pipeline.
SAM-6D RGB is the explicit semantic-plus-appearance ablation of its RGB-D ISM;
both use SAM-6D's own proposals. Its subsequent pose-estimation module is outside
identification. Neither method consumes old ROIs, scene masks, world transforms,
simulator poses, target labels or prescribed proposal counts.

Official full-image preprocessing is microbatched by2 to fit8 GB. SAM and DINO
run in separate processes. SAM-6D uses its original 2,048 sampled native CAD
points, selected-template rotation and mean observed depth translation. CPU
projection chunks keep working memory bounded, while the released full-batch
IoU guard and final coverage-weighted score remain intact. A singleton projection
chunk is duplicated solely to accommodate the upstream `squeeze_` call, and its
duplicate is discarded. No ICP, support orientation search or table constraint
is added to these baselines.

Each scene exports final `detections.json`, native-frame `masks.npz`, native
size-filter counts and raw proposals in a separate directory. Binary mask export
uses the official BOP default `force_binary_mask` rule, soft mask>0. Source,
checkpoint, input, template and output hashes are recorded. Evaluation belongs
to a separate post-inference audit; missing detections remain misses.

All extra Python packages live in this experiment's `deps` directory. Missing
GLU/OpenGL libraries were downloaded as Ubuntu packages and extracted inside
`cnos_sam6d/native_libs`, without system installation. Existing cached BOP toolkit
sources supply its loader imports. No global environment is modified.

Focused CPU checks are in `test_native.py`: exact native/chunked mask and box
resize parity, lossless storage, proposal-ID preservation through native size
filtering, native class matching, and official42 reference mapping/preprocessing.
