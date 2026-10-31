# Controlled GeDi repair and CAD DINO resolution comparison

This isolated experiment connects the finite prepared-patch GeDi repair to full pose inference. It compares a fixed seven-target T-LESS diagnostic cohort with the saved strict baseline, then compares CAD224 and CAD476 while keeping observed crops at224. It is an independent robustness experiment, not a full reproduction of the public FreeZeV2-Re code or its reported performance.

The cohort has7 original official object-image target rows,7 requested instances,6 images and5 CADs. It includes CAD27/28/30 onboarding failures, scene3/image476/object8 observed failure, and three computationally healthy controls. Selection used recorded numerical failures and sorted official IDs, never GT accuracy. Every target remains in official evaluation even if it produces no pose. Mask inputs are the same11 CNOS-FastSAM public BOP23 records selected by the baseline N+1 policy. Object IDs/counts come from the official localization targets; this does not evaluate novel CAD identity classification or segmentation AP.

`gedi_valid_rows.py` runs both original GeDi radii with the original query order, sampling batches, dense support, checkpoint and RNG consumption. Only finite prepared patches enter the frozen network. The intersection of finite output rows is applied to the3D points and visual descriptors using explicit original indices. It does not invent a frame, zero a descriptor, jitter points, or resample a failure. Empty neighborhoods and zero/nonfinite output descriptors remain explicit failures. This handling is our method change, not a verified author-released policy.

`pose_alignment.py` enforces row alignment. The complete original5000-point CAD cloud remains in ICP and the geometric scoring denominator, and the complete3000-point observed dense cloud remains unchanged. Sparse feature scoring uses retained observations, so its denominator is explicitly recorded. Arm repair224 uses the unchanged CAD-only PCA fit on all5000 original visual rows before GeDi row removal. Healthy existing visual/PCA caches are reused byte-for-byte; missing blocked-CAD aggregates are reconstructed from the original hash-checked224 maps. Arm repair476 fits its own CAD-only PCA from476 features and applies that same basis to observations.

`visual_ablation.py` implements the second controlled change: the same cached480 render is cropped at the top left to476 and encoded into34x34 DINO patch tokens, rather than resized to224/16x16. Frozen giant checkpoint, zero-based block18, ImageNet normalization, FP32, TF32 disabled, renderer, camera/view/surface samples, visibility, half-pixel feature mapper, observation224, fusion, RANSAC, ICP and scoring remain those of the completed baseline. This tests CAD input resolution; it does not test a new renderer or the public repository's other implementation differences. The public reference revision is12f82a2f8afd435466a87cf2dafce404087aeb3f. No public source code was copied.

`run_pose_ablation.py` reads only RGB-D, calibration, CAD, predicted masks and official target IDs/counts. GT access occurs only in the separate `evaluate_cohort.py` invocation. Evaluation uses the pinned official BOP Toolkit cea62d651c7e395b2e1962b9749e4e89693c6ac4, BOP19 model-based localization, VSD/MSSD/MSPD and inst_count most-visible GT with visib_gt_min=-1. The reported metric is pose AR, not AP. Evaluations use the exact selected original target rows and counts; failures do not disappear from the denominator. These selected failure-heavy cohort scores cannot establish full-dataset improvement or paper parity.

Resolved inputs are D:/BOP_SAM3_Pilot/data and existing checkpoints/masks under D:/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029. The completed baseline source and outputs remain untouched. New outputs are outputs/freezev2_bop_20261029/gedi_pose_ablation_v1_20261030. Source/config/cohort identities and cached-array hashes prevent incompatible resume. One GPU and two CPU threads are used; caches retain a15GiB free-disk reserve. A genuine source/config change requires a new output version. No training, download, environment installation, robot action or remote push is performed.

From the repository root in PowerShell, run focused checks:

```powershell
wsl.exe --exec env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/jiabao/.venvs/freezev2-bop-20261029/bin/python -m pytest -q -p no:cacheprovider /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_gedi_valid_rows_20261030/test_gedi_valid_rows.py /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_gedi_valid_rows_20261030/test_pose_alignment.py /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_gedi_valid_rows_20261030/test_visual_ablation.py /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_gedi_valid_rows_20261030/test_evaluate_cohort.py
```

Run/resume inference (repair224 first, then repair476 automatically). Confirm another GPU inference worker is not active before starting:

```powershell
wsl.exe --exec env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiabao/.venvs/freezev2-bop-20261029/bin/python /mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/experiments/freezev2_gedi_valid_rows_20261030/launch_pose_ablation.py --arm all
```

Individual arms can resume with --arm repair224 or --arm repair476. Do not launch both concurrently. Inference checkpoints are not a metric receipt. After inference ends, run evaluations sequentially:

```powershell
$repo='/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification'
$study="$repo/experiments/freezev2_gedi_valid_rows_20261030"
$output="$repo/outputs/freezev2_bop_20261029/gedi_pose_ablation_v1_20261030"
wsl.exe --exec /home/jiabao/.venvs/freezev2-bop-evaluation-20261030/bin/python -B "$study/evaluate_cohort.py" --root $repo --cohort "$study/cohort.json" --csv "$repo/outputs/freezev2_bop_20261029/classic_core_base_v2/datasets/tless/predictions/freezev2base_tless-test.csv" --output "$output/baseline/evaluation"
foreach ($arm in @('repair224','repair476')) {
  wsl.exe --exec /home/jiabao/.venvs/freezev2-bop-evaluation-20261030/bin/python -B "$study/evaluate_cohort.py" --root $repo --cohort "$study/cohort.json" --csv "$output/$arm/freezev2cohort_tless-test.csv" --output "$output/$arm/evaluation"
}
```

Each evaluated candidate retains its unchanged mask, RGB crop, original/retained sparse-point indices, CAD original-index map, correspondence indices/similarities, coarse/refined pose, all three scores, selected/suppressed status, runtime and failure trace. GT comparisons and official metric artifacts stay separate from inference. `REPORT.md` and `results_summary.json` under the new output directory contain measured outcomes only after evaluation; `visual_review.json` records actual visual inspection. The original all-seven benchmark and its published evidence remain separate.
The launcher exposes only the already cached pure-Python png.py module through a run-local symlink for official CSV validation. It does not place the full cached evaluation dependencies ahead of the inference venv's NumPy/PyTorch. The first224 execution saved all seven poses before its format check found the missing png import; the unchanged runner resumes those exact checkpoints. Dependency provenance is in pypng_runtime.json.
