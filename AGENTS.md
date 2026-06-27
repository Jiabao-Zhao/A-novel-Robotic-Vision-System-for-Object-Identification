# AGENTS.md

## Project goal

This repository develops a compact robotic vision system for object identification and tabletop object pose reporting.

Current milestone:
Intel RealSense RGB-D data -> segmented object point cloud -> object center, dimensions, roll, pitch, yaw, yaw confidence, rotation matrix, and saved numeric results.

The code already works as a prototype. The main job for future agents is not to add features. The main job is to remove redundant code, keep the working behavior, and make the repository clean enough for paper/demo use.

Do not expand the project into LLM planning, robot control, grasp planning, GUI, voice control, CAD registration, active scanning, or full VLM classification unless the user explicitly asks for that exact feature.

## Ruthless cleanup priority

When editing this repository, prefer deletion over abstraction.

Remove code that is:
- unused
- duplicated
- generated but committed as source
- defensive without a real failure mode
- configurable only for the sake of configurability
- unrelated to the current RealSense point-cloud pose pipeline
- a wrapper around one line of real work

Do not preserve code just because it works. If it works but is redundant, simplify it while keeping the behavior.

## Current repository issues to watch

The current codebase has specific redundancy problems:

- `argparse` is overused for prototype-only constants.
- `realsense_pointcloud_pose.py` embeds a full visualization script as a giant string.
- `Output/visualization/visualize_pose.py` duplicates visualization functions already present in the main script.
- `main.py` is only a tiny wrapper around `realsense_pointcloud_pose.main()`.
- Generated outputs under `Output/` are tracked as repository source.
- Visualization logic and capture/pose-estimation logic are mixed in one large file.

Future agents should treat these as cleanup targets, not patterns to copy.

## CLI and argparse policy

Do not add `argparse` by default.

Use direct constants, a small config dataclass, or clearly named module-level settings when the user is running a controlled local prototype.

Only keep or add a CLI when at least one of these is true:
- the user explicitly asks for command-line options
- the script is meant to be reused by other people with different hardware/workspaces
- the option cannot reasonably be represented as a fixed config value

If a CLI is kept, keep it tiny. Avoid long lists of rarely changed flags such as every DBSCAN threshold, crop bound, warmup count, and visualization switch. Those belong in a config block unless the user asks otherwise.

## Code style

Write concise, direct Python.

Prefer:
- small functions with one job
- dataclasses only for real structured results
- explicit units in names, such as `center_mm` and `voxel_size_m`
- clear geometry math
- fail-fast errors for missing hardware, missing files, invalid transforms, or empty point clouds
- readable constants over sprawling argument parsers

Avoid:
- unnecessary classes
- abstract base classes
- plugin systems
- broad try/except blocks
- fake fallback logic
- large if/elif chains
- excessive validation wrappers
- logging frameworks unless requested
- unrelated helper modules
- embedded source-code strings
- duplicated visualization utilities

## File organization rules

Keep source code separate from generated artifacts.

Recommended direction:
- `realsense_pointcloud_pose.py`: capture, point-cloud conversion, segmentation, clustering, pose estimation, and saving numeric results
- optional `visualize_pose.py`: visualization only, if the user wants a reusable visualization script
- `main.py`: delete it unless it provides real value beyond calling another `main()`
- `Output/`: generated local run artifacts, not source code

Do not commit generated PLY/JSON output files unless the user explicitly asks for sample outputs for the paper/demo.

If generated outputs must stay for demonstration, label them as sample data and do not let the pipeline rewrite committed source files during normal execution.

## Dependencies

Use only the libraries explicitly needed for the current task.

Allowed for the current milestone:
- pyrealsense2
- open3d
- numpy
- scipy
- json
- pathlib
- dataclasses

Avoid adding new packages unless requested.

`argparse` is not a default dependency. Use it only when the CLI policy above is satisfied.

## Coordinate and unit rules

Work internally in meters unless the task says otherwise.

Output robot- or paper-facing quantities in:
- millimeters for translation and dimensions
- degrees for roll, pitch, and yaw

Never invent calibration values. If a transform is required, require it from the user or load it from a specified file.

## Pose-estimation rules

For tabletop components:
- estimate object center from the object point cloud or oriented bounding box
- estimate yaw from the dominant horizontal PCA axis
- estimate roll and pitch from the support/table plane normal
- report yaw confidence from PCA eigenvalue separation
- do not pretend yaw is meaningful for symmetric objects

Treat the phrase "6D pose" carefully. Some objects have unobservable orientation components due to symmetry.

## Output expectations

When saving perception results, prefer JSON for numeric results and PLY for local point-cloud inspection.

Do not generate extra files unless they are necessary for the requested run.

Do not write a Python script from inside another Python script unless the user specifically asks for code generation. A generated visualization script is usually the wrong design; put reusable visualization code in a normal source file or keep visualization inside the active run.

## Review rule

Before finishing any change, check this list:

1. Did I add a feature outside the requested scope? Remove it.
2. Did I keep a CLI flag that should be a constant? Remove it.
3. Did I duplicate visualization or transformation code? Consolidate it.
4. Did I commit generated `Output/` files? Remove them unless requested.
5. Did I make the working prototype harder to understand? Simplify it.

The final repository should be smaller, clearer, and more defensible than before.