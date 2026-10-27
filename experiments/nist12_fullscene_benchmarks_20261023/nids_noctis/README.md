# Native NIDS-Net and NOCTIS full-scene runs

These runs consume the eight raw RGB scenes and twelve native-scale CADs in
[`../inputs.json`](../inputs.json). Each method generates its own boxes and masks.
No supplied scene crops, simulator labels or object poses enter inference. All
weights are frozen; NIDS's learned adapter is disabled. This is the basic NIDS
variant, not a reproduction of the paper's trained weight-adapter result.

## Sources and execution

The official source archives remain unchanged under `../sources/`:

| Source | Pinned commit |
| --- | --- |
| [NIDS-Net](https://github.com/IRVLUTD/NIDS-Net) | `c7685a442157a1f28f2d7771e10dd9c7afdd7154` |
| [NOCTIS](https://github.com/code-iai/noctis) | `ba76bbc5176e2e3604bc596e656f8334188ad486` |
| GroundingDINO | `2b62f419c292ca9c518daae55512fabc3fead4a4` |
| SAM | `6fdee8f2727f4506cfbbe553e23b895e27956588` |
| MobileSAM | `c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed` |

Archive hashes are in [`source_pins.json`](source_pins.json) and
[`dependency_source_pins.json`](dependency_source_pins.json). `run_native.py`
orchestrates the official proposal, descriptor, matching and NMS routines in
separate processes for the two conflicting native `src` namespaces. Completed
outputs are `../nids_basic/N00..N07/` and `../noctis/N00..N07/`.

| Setting | NIDS basic | NOCTIS |
| --- | --- | --- |
| Proposal models | GroundingDINO Swin-T OGC + SAM ViT-H | GroundingDINO Swin-B cogcoor + Ultralytics SAM2.1-L |
| Released variant | Native `use_adapter=False`, appearance enabled | `cnos2AppCT5ConfGBSAM21LPyrenderMode` |
| Detection prompt / thresholds | `objects`; box/text 0.1 | `objects`; box/text 0.1, mask 0.01 |
| Descriptor | Frozen DINOv2 ViT-L/14, native 224 preprocessing | Frozen DINOv2 ViT-L/14, native 224 preprocessing |
| Matching | Max foreground-feature cosine across views; mean with native patch appearance at the winning view | Top-five mean CLS similarity; max cyclic patch appearance (distance threshold 5, factor 2); mean, multiplied by mean box/mask confidence |
| Score threshold | FFA score strictly >0.01 | Semantic similarities below 0.2 become zero; final score strictly >0.2; maximum 100 instances |
| Size filter | Box area/image area >0.05² and mask sum/image area >0.0003 | Same native thresholds |
| NMS | Per object ID, box IoU 0.25 | Per object ID, box IoU 0.25 |

Proposal checkpoints are task-local under `../weights/`:
`groundingdino_swint_ogc.pth`, `groundingdino_swinb_cogcoor.pth`,
`sam_vit_h_4b8939.pth`, and `yolosam/sam2.1_l.pt`. The last is the
**Ultralytics** checkpoint. Local `bert-base-uncased/` supplies the text encoder.
The cached `dinov2_vitl14_pretrain.pth` is loaded strictly from the repository's
`outputs/cache/torch_hub/checkpoints/`; no implicit downloads occur at inference.
Frozen method configurations record source, template and checkpoint hashes.

The shared `../templates42/` bank uses the released CNOS Pyrender renderer,
RGBA 640×480 images, native CAD scale and catalogue colors. Each method uses its
own official BOP template loader and view-index file. NIDS consumes 42 distinct
canonical views. NOCTIS's released 42-entry mapping contains **21 distinct views**;
that mapping is preserved, including its repeated index 301. The renderer supplied
the additional 301 image for every CAD. See the bank's metadata and completion
receipt; this run does not reuse the earlier twenty-view matching study.

## Scoped compatibility and memory use

`compat.py` suppresses only NIDS's import-time
`python setup.py build develop --user` command. It resolves local checkpoints and
uses the stock SAM ViT-H registry for NIDS's wrapper. Official source files are
not patched.

GroundingDINO requires BERT APIs absent from the installed Transformers 5 stack.
Only these runners prepend `hf_compat/`: Transformers 4.38.2, tokenizers 0.15.2
and huggingface_hub 0.23.5. The global environment is unchanged. Installation
hashes are in `hf_compat_install_report.json`; the failed initial installation and
Transformers-5 CPU probe are retained separately.

The compiled GroundingDINO CUDA extension is unavailable. A scoped monkeypatch
calls its **shipped PyTorch deformable-attention implementation** on the selected
GPU. Both GroundingDINO checkpoints
loaded with zero missing keys. Only the extra released checkpoint entries
`label_enc.weight` and `bert.embeddings.position_ids` were unused.

Proposal models are released before DINO matching. Native descriptor batches are
two; NOCTIS patch-scoring query/object chunks are one. NIDS's Pyrender loader
returns masks without a channel dimension; the runner restores that singleton
only for the native reference-patch routine. Scores and thresholds are unchanged.

## Native NIDS failure and secondary control

The unchanged `nids_basic` execution completed but retained zero detections in
every scene. Six **USB_Male** template slots, 16, 19, 20, 21, 24 and 25
(canonical views 304, 316, 320, 324, 336 and 340), have nonempty 224² masks but zero
mask mass after native bilinear sampling to 16². Native FFA pooling divides by
zero, producing NaN reference vectors. Native max over views and then objects
propagates these NaNs to every proposal's selection score. All scene query FFA
vectors are finite. This is an unavailable native reference-feature condition,
not evidence of zero classification accuracy.

[`nids_zero_mask_diagnostic.json`](nids_zero_mask_diagnostic.json) records the
CAD-only measurements. The independent native-loader/score replay is
[`../cnos_sam6d/nids_reference_audit.json`](../cnos_sam6d/nids_reference_audit.json).
Original native configurations, tensors and outputs are preserved.

`run_finite_templates.py` produces the explicitly secondary
`../nids_basic_finite_templates/` control. It excludes only those six invalid
per-view scores with negative infinity before the native maximum. It does not
replace reference NaNs, remove USB's remaining 36 valid views, remove a class,
or change thresholds, filters, appearance scoring or NMS. Saved original query
FFA vectors and own proposals are reused. Appearance vectors are recomputed from
the same frozen DINO/images/masks because the original zero-selection run did
not serialize them; recomputed FFA must equal the saved FFA exactly. Every scene
records exact equality of all valid view scores and finite final scores in
`finite_template_check.json`. This is a numerical compatibility control, not an
untouched official-method result.

## Checks and output contract

All **18 CPU tests** pass: eight per native method in `test_native.py`, plus two
finite-template guard tests. Receipts are `*_CPU_tests.json`. CPU checkpoint
probes verify both GroundingDINO models and both segmentation wrappers without
inference; see `*_cpu_probe.json` and `*_segmentor_cpu_probe.json`.

Every scene exports `raw_proposals.npz/json`, `proposal_completion.json`,
`proposal_filter.json`, `detections.json`, native-resolution `masks.npz`,
`matching_scores.npz` and a hashed `completion.json`. `proposal_id` remains mapped
through filtering and NMS. Exported masks use the native strict >0 conversion.
`all_runs_completion.json` summarizes all 24 completed scene conditions;
`prediction_tensor_health.json` checks serialized score tensors. No ground truth
was used for these runs or checks; correctness evaluation is separate.
