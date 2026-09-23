# Local SAM3

The active automatic-mask experiments use the point-prompt tracker from
`facebook/sam3`, with a 32x32 uniform point grid and four prompts per batch.
Object names, Roboflow, and network access are not needed for these runs.
The model is not a VLM or an object-identity classifier.

## This computer

- Checkpoint: `D:\AI\models\sam3` (WSL: `/mnt/d/AI/models/sam3`).
- Python: `/home/jiabao/.venvs/lerobot-libero/bin/python` in Ubuntu-22.04.
- RTX 3060 Ti, BF16 inference, Transformers 5.5.4, CUDA PyTorch 2.11.0.
- The Windows Python installation has CPU-only PyTorch; use WSL for these scripts.
- Model files are loaded with `local_files_only=True`; there is no hosted fallback.
- The loader extracts the pretrained tracker/backbone from the combined checkpoint
  and rejects missing or mismatched weights. The adapter casts NMS scores to FP32
  to handle a Transformers 5.5.4 BF16 postprocessing incompatibility.

Run in WSL from the repository root:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=outputs/cache/bop_eval_deps:outputs/cache/bop_toolkit
PY=/home/jiabao/.venvs/lerobot-libero/bin/python
$PY -m scripts.run_bop_automatic_sam_masks
$PY -m scripts.run_bop_automatic_sam_comparison
```

The second command runs the much longer CAD-matching comparison. It reuses the
existing frozen depth baseline and template assets, but computes fresh SAM3
features and scores. Setting `PYTHONPATH` before startup keeps BOP's existing
NumPy 1.26.4/COCO fork together; mixing its extension with NumPy 2 causes an ABI error.

The separate RGB/inverse-depth workbench experiment runs with:

```bash
$PY -m scripts.run_workbench_depth_sam
$PY -m scripts.compare_workbench_framework
```

New outputs use `bop_automatic_sam3_local_20260921`,
`workbench_depth_sam3_local_20260921`, and `workbench_framework_sam3_local_20260921`
under `outputs/`. Protocols record weight/config/source hashes, versions, precision,
and sampling thresholds. Existing frozen protocols must match before caches are reused.
`predicted_iou` is SAM3's mask-quality prediction. Per-mask stability scores are
not exposed by this pipeline, so they are null and the applied threshold is recorded.

## Validation And History

```bash
$PY -m pytest -q tests/test_local_sam3.py tests/test_bop_automatic_sam.py \
  tests/test_workbench_depth_sam.py tests/test_bop_sam_depth_comparison.py
```

Measured local full-grid smoke results are in `outputs/sam3_local_smoke_20260921`.
They establish runtime feasibility, not superior segmentation accuracy. The RGB
and inverse-depth inputs may produce very different proposal coverage at the
existing thresholds; run the held-out evaluation before drawing quality conclusions.

Historical ViT-H and hosted SAM3 masks, metrics, and diagnostic scripts remain
unchanged as research evidence. `run_bop_sam3_masks`, `check_workbench_sam3`,
`run_surface_sam_masks`, and `check_image_sam3` are older **hosted** entrypoints,
not the local point-grid implementation. No active script imports `segment_anything`
or loads the old ViT-H checkpoint. SAM-6D scoring helpers are not ViT-H dependencies.

Reference: https://huggingface.co/docs/transformers/model_doc/sam3_tracker
