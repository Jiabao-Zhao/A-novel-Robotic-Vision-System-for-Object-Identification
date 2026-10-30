# Local runtime and external-input inventory

This file preserves the initial inspection before dependency provisioning.
Its missing-model/environment observations are historical. The current
validated trained runtime, checkpoints, isolation and actual CUDA evidence
are recorded in [TRAINED_RUNTIME.md](TRAINED_RUNTIME.md); all-seven execution
uses `protocol.json` and the immutable `classic_core_base_v2` source snapshot.

Inspected 2026-10-29 local / 2026-10-30 UTC. This inventory did not change the NIST pipeline, install packages, replace environments, or move original datasets/model weights.

## Existing inputs and runtime

The retained BOP data is `D:\BOP_SAM3_Pilot\data`: the seven BOP19 image cohorts (3,871 RGB/gray images, sensor depth and camera metadata), 125 CADs, selected PBR training references, and public evaluation annotations. The earlier SAM3 predictions and experimental scripts were removed at the user's request; they are not available as external segmentation inputs. The PBR cache's 42 views per CAD are not the paper's 162 rendered reference views and are not silently reused as a faithful onboarding substitute.

The existing WSL interpreter `/home/jiabao/.venvs/lerobot-libero/bin/python` has Python 3.12.14, Torch 2.11.0+cu130, Torchvision 0.26.0, CUDA on an 8GB RTX 3060 Ti, NumPy 2.2.6, SciPy 1.18.1, Open3D 0.19.0, Pillow 12.3.0, OpenCV and pytest 9.1.1. Invoke the default WSL distribution; a distribution named `Ubuntu` is not registered. An actual Open3D ICP smoke test recovered the synthetic transform with fitness 1.0 and maximum matrix error 2.22e-16. No FreeZeV2 or GeDi experiment workers were found during the inventory.

Native `D:\Python\python.exe` has NumPy, SciPy, Torch, OpenCV, Pillow, pycocotools and pytest. Its Torch CUDA is unavailable and Open3D is absent, so the existing WSL interpreter is the inference runtime candidate. Native Python was used for focused mask/CSV tests and data inspection only.

Existing repository caches include `outputs/cache/torch_hub`, `bop_toolkit`, `bop_cocoapi`, `bop_eval_deps`, `blender`, and `blenderproc_env`. Blender 3.3.1 Linux is retained. The official BOP Toolkit cached revision is `cea62d651c7e395b2e1962b9749e4e89693c6ac4` (MIT). `bop_eval_deps/png.py` supplies the official parser's missing pure-Python png dependency without installation; format validation appends this directory after the interpreter's installed libraries to avoid shadowing native NumPy with cached Linux binaries.

## Frozen-feature blockers

The cached DINOv2 source is `outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35`. Checkpoints found are `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes) and `dinov2_vits14_pretrain.pth` (88,283,115 bytes). No DINOv2 ViT-g checkpoint was found in the repository cache, D: model directories, or Windows/WSL user model caches. ViT-L is not substituted for the paper's ViT-g in the named base configuration.

GeDi source/checkpoint and `pointnet2_ops` were not found. The WSL environment has gcc/g++ but no nvcc, CUDA toolkit directory, cmake, or ninja. Consequently a compatible GeDi descriptor backend remains an essential execution blocker; availability of Torch CUDA alone does not supply its compiled PointNet2 operations. FAISS, trimesh, pyrender and moderngl were also absent. NumPy correspondence search and the independently implemented renderer avoid treating those optional packages as required substitutes for the paper's frozen networks.

## Public predicted masks and localization metadata

The official archive is [BOP23 default detections for Task 4](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip), 30,040,248 bytes, with HTTP byte-range support. Only its T-LESS CNOS-FastSAM member was acquired (10,775,250 compressed bytes, 74,524,999 uncompressed bytes). It contains 58,667 predicted detections. Its SHA256 is `db010fbce92149a54ae7a252176d6dee80823353a7e5d704c0f33657c5b1ecec`; its ZIP central-directory CRC32 is `13f07ac2`. Acquisition used bounded range reads and checked the central-directory CRC, including the ZIP data-descriptor convention. The local file is:

`D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\outputs\freezev2_bop_20261029\external\cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json`

Its adjacent `cnos_fastsam_tless_acquisition.json` records URL, ETag, archive size, exact member/offset, CRC and SHA256. These are external predictions, not ground-truth masks. The paper labels its single-source experiment “CNOS” without identifying its SAM/FastSAM backend; therefore this run must be named explicitly as a CNOS-FastSAM input configuration, and equivalence to the paper's exact mask set remains unresolved. Multiple segmentor inputs are supported independently; no second source is invented.

The existing `data/bop22_annotations/tless/test_targets_bop19.json` is an image-only derived list and has no `obj_id` or `inst_count`; it is unsuitable as a localization target file. The original official list was selectively extracted from the tiny pinned `tless_base.zip` (49,597 bytes) at Hugging Face revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. The target-file SHA256 matches the retained input manifest: `5ced8796c421baad79816fe905a36fe6ad440efe78cd640d167fc4b8f26c3cfe`. Its 4,904 rows include only allowed localization object IDs/counts and no pose initialization. It is stored beside the masks as `tless_test_targets_bop19.json`, with `tless_targets_acquisition.json` provenance.

The first three sorted object-1 targets in that official file are scene 5 / images 7, 20 and 25, with one requested instance and one CNOS-FastSAM candidate each. Their segmentation confidences are 0.52421635, 0.59044647 and 0.59772670. Scene 1 / image 1 contains a predicted category-1 detection but is not an official object-1 localization target. Fixed-cohort selection must use the official localization metadata, not a detector's potentially incorrect category.

The first later official object-1 target having at least two public candidate masks is scene 9 / image 0, requesting three instances and providing five detections. It was added only to exercise the multi-candidate input path; the N+1 policy selects its four highest-confidence masks. No ground-truth pose, overlap score, or measured registration outcome was consulted to choose this fourth image. `external/tless_obj1_fixed4_targets.json`, `tless_obj1_fixed4_predictions.json`, and `tless_obj1_fixed4_provenance.json` retain the small fixture, original detection record indices, source hashes, official source URL/revision, and selection rule. All four source RGB/depth images exist. Four predicted-mask previews under `diagnostics/input_masks` were visually inspected: overlays coincide with their binary predicted regions, and the multi-candidate preview distinguishes the selected four from the fifth. This verifies input mapping and does not establish correct CAD identity or pose.

`load_mask_candidates(..., target_keys={(scene_id, image_id, object_id), ...})` skips other records before dense RLE decoding and retains their original file record indices. Loading all 58,667 masks into dense arrays is unnecessary and can exhaust memory. Both compressed COCO and uncompressed integer-run RLEs are accepted and checked; provenance is mandatory. Oracle input requires an explicit separate flag and is rejected by the default benchmark candidate selection.

## Official evaluator limits

`evaluation.py` writes model-to-camera SO(3) poses and converts internal metre translations to BOP millimetres. All predictions from the same image carry the complete measured image wall time. Its official-parser integration used the actual cached `bop_toolkit_lib.inout.check_bop_results`, not a reimplemented benchmark score. A valid CSV only demonstrates output-format validity.

The cached official evaluator uses `models_eval` for T-LESS VSD/MSSD/MSPD, while the retained data currently has only `models_cad`. Those official evaluation meshes are missing. Public pose annotations are retained under `full_eval_gt`, not the standard inference dataset tree, and must be exposed only to an evaluation process/view. The missing official evaluation assets are distinct from the neural-feature blockers and do not justify reacquiring the existing RGB/depth/CAD dataset. No `models_eval` was acquired in this task segment.

The three separate claims remain: evaluating published predictions reproduces their score; our inference produces valid predictions; our inference reproduces published performance. Only the tests and input/parser checks actually executed here support claims. No inference-accuracy or paper-performance score was measured by this inventory.

`evaluate.py` is an explicit evaluation-only entry point. `--dry-run` prints the exact official command and checks the strict CSV, pinned official parser, evaluation model family, target IDs/counts, scene GT/camera paths and sensor depth paths. It does not read GT pose contents. `--execute` is the only mode that launches official metrics, after the input and evaluator-runtime checks pass. Missing `models_eval` causes a clear blocking result; the CLI does not create GT links or alias CAD meshes. The invocation pattern is:

```powershell
& 'D:\Python\python.exe' evaluate.py --config config.json --results-file '<completed BOP CSV>' --dataset-view '<explicit evaluation-only BOP root>' --targets-filename test_targets_bop19.json --dry-run
```

The actual dry-run probe used the already existing synthetic diagnostic CSV and the retained data root. Its recorded result is `runtime/evaluation_cli_dryrun_synthetic_probe.json`: official CSV format passed; all 30 official T-LESS evaluation meshes plus `models_eval/models_info.json` were absent; the required original target filename was not present inside that explicit dataset view. The original target file is available in `external` and should be placed only into a separately prepared evaluation view. The native metric subprocess also lacks an installed png module; the standalone parser check succeeds via the cached-dependency append, while a full metric process still needs a separately compatible evaluator environment. No metric process was launched and no GT pose contents were opened.

## Focused test command

From the isolated implementation directory, using process-scoped environment entries:

```powershell
$env:FREEZEV2_BOP_TOOLKIT = 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\outputs\cache\bop_toolkit'
$env:FREEZEV2_BOP_SUPPORT_PATH = 'D:\GitHub\A-novel-Robotic-Vision-System-for-Object-Identification\outputs\cache\bop_eval_deps'
& 'D:\Python\python.exe' -m pytest -q tests/test_masks_evaluation.py
```

The focused suite passed all 26 tests, including a four-frame targets-to-filtered-masks-to-candidate-pool fixture, official evaluation-mesh-family preflight, and an evaluation CLI dry-run. It tests non-square COCO RLE and actual pycocotools interoperability, invalid input rejection, explicit oracle separation, official localization IDs/counts, per-segmentor N+1 confidence pool selection, unrestricted rotations, metre/mm CSV round-trip, per-image timing, no-overwrite output, and the actual official BOP format parser. It does not run the official pose metrics or load frozen visual/geometric networks. Native Python prints a pyreadline3 console destructor warning after the successful test result; pytest exits with status zero.
