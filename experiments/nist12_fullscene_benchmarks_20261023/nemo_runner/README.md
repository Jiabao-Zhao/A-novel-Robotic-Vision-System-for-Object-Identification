# NeMO full-scene experiment

This runner calls the official NeMO encoder, decoder and multiscale full-scene
segmentation routine at commit `76478815baf2c99002cccbee98426773fa4e59cd`
from <https://github.com/DLR-RM/NeMO>. It uses the released complete checkpoint;
all 511 keys load strictly. No weights are updated. All objects' masks are
predicted from full RGB scenes, without supplied scene masks, crops, poses,
depth localization or a forced object count.

The paper's model-based evaluation uses 32 PBR reference images. Its released
repository does not supply a CAD renderer. Our prospectively declared custom
CAD onboarding uses 32 of the official CNOS Pyrender sphere views, selected
without replacement using NumPy seed 0. `view_selection.json` freezes the
selection before NeMO inference. Isolated CAD alpha determines the smallest
square enclosing the foreground, with black padding on the shorter axis;
this avoids distorting pin length/width with the native square resize.
`primary_references.json` and `primary_crop_records.json` record every source,
output hash and crop box in render-image pixels. No scene regions determine
these CAD crops.
This is official NeMO inference with custom CAD-rendered references, rather
than an exact recreation of the paper's BOP PBR reference condition.

Before any benchmark labels or evaluation were read, an onboarding check found
that the initial raw CAD render references contained excessive black background.
The paper describes object-centric templates and bbox variation in supplement
7.3; its released loader expects ready-made reference images rather than
cropping the object itself. The initial arm is preserved unchanged under
`nemo_raw_reference_diagnostic`, with its exact runner and README copied beside
this file. `pre_label_protocol_correction.json` records the correction before
the primary run. The fixed 32 views, native colors, weights, scene inputs and
detector settings are the same in both arms. No condition is chosen using
evaluation outcomes.

Native preprocessing, encoder and detector behavior are retained:

- Reference images: native `load_images(normalize=False)`, followed by the
  model's own 224-pixel transformations and ImageNet normalization.
- Encoder: 1,500 uniformly sampled points, seed 0, native 32-view encoding.
- Full scenes: sliding windows at minimum image dimension divided by 1, 2 and
  3, with stride equal to window size divided by 3. No scene regions are supplied.
- Native 0.5 pixel threshold, union of masks across the three scales, mean
  confidence across scales, and amodal connected components of at least
  1,024 pixels determine boxes. The corresponding modal mask inside each box
  is exported. A small pin removed by this rule remains a miss.
- Each output retains its conditioned CAD identity. There is no extra cross-CAD
  suppression or externally supplied classification stage.

The decoder processes two windows at a time to bound memory. Window positions,
preprocessing, outputs and folding are unchanged. `check_batching.py` compares
the released aggregation with bounded batches on CPU and verifies exact agreement.
The complete checkpoint includes the DINOv2 backbone, so its cached official
architecture is constructed without a redundant initial weight download and
then loaded strictly from the NeMO checkpoint.

`run.py` exports per-scene `detections.json`, native-resolution `masks.npz`,
per-object raw amodal/modal maps, and completion records. Only the scene RGB
and CAD-rendered reference images reach the network. Evaluation labels are
reserved for the separate benchmark evaluator.

Run `prepare_references.py` after the 42-view rendering bank completes, followed
by `build_objectcentric_references.py`. Run `run.py` only when the benchmark's
sequential GPU scheduler grants the slot. The exact original raw-reference
runner is frozen for provenance, rather than intended to overwrite the primary
output directory.
