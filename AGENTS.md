# AGENTS.md

## Project Goal

This repository enables an amtomuouse robot framework for adaptive manufacturing assembly.

The full research pipeline is:

1. Capture RGB image and depth information from an Intel RealSense RGB-D camera.
2. Use depth-based processing to localize tabletop objects and produce ROI (region-of-interest) using the boundary box (x1, y2, x2, yx) in the pixel coordinates. 
3. Send the annotated RGB image, one explicit semantic target description, and localized object information to a Vision-Language Model (VLM) for object association.
4. Resolve the VLM association through a confidence-guided human-deferral gate, then retrieve the most relevant CAD model using the semantic target description while selecting the observed point cloud using the resolved object ID.
5. Compare the observed partial 3D point cloud with the CAD model or CAD-derived geometry.
6. Estimate object pose information, including x, y, z, roll, pitch, yaw, rotation matrix.
7. Send the object identity, pose, and task context to an LLM-based task planner.
8. Convert the task plan into robot-executable actions through a low-level robot controller.

The proposed framework uses VLM with depth-localization for semantic object association, CAD/point-cloud comparison for geometry-aware pose estimation, and LLM planning for downstream assembly execution.

## Current Repository Scope

This repository is currently the compact perception front end of the full framework, not the entire robot intelligence stack.

Current milestone:

Intel RealSense RGB-D data -> segmented object point cloud -> object localization summary -> one-target VLM association prompt -> confidence-guided resolution -> saved numeric and visual results.

The code should remain focused on:

- RealSense RGB-D capture
- depth-to-point-cloud conversion
- tabletop plane segmentation
- object clustering
- 2D bounding box extraction
- 3D centroid and size reporting
- approximate roll, pitch, yaw estimation when geometrically meaningful
- ROI or bounding-box visual prompting for VLM input
- compact VLM association using one semantic target description plus localized object metadata
- saving JSON/PNG/PLY outputs needed for testing, paper figures, or demo inspection

Do not expand this repository into the complete downstream system unless the user explicitly asks for that exact feature.

Avoid adding full implementations of:

- LLM task planning
- robot control
- grasp planning
- GUI or voice control
- CAD registration
- active scanning
- multi-view reconstruction
- full assembly execution
- database-backed CAD retrieval

It is acceptable to keep lightweight interfaces or clearly named placeholders only when they make the perception output easier to connect to the full framework later.

## Research Terms Used In This Project

Use consistent terminology when writing code comments, prompts, README text, or paper-facing outputs.

| Input or Signal | Preferred Term | Meaning |
| --- | --- | --- |
| Human instruction such as "put the green block on the base" | task text prompt or language instruction | Provides task intent to the downstream LLM planner |
| Object phrase such as "green block" | semantic target description | Names one object for an independent VLM association query |
| RGB image sent to the VLM | visual input | Provides appearance information |
| Bounding boxes, highlighted regions, masks, or ROI images | visual prompt or region prompt | Guides the VLM toward already-localized objects |
| Object centroid, 2D box, 3D size, height, point count, or spatial relation | geometric prompt or spatial prompt | Gives the VLM structured localization evidence |
| Depth map from RealSense | depth image or depth map | Per-pixel distance measurement from the camera |
| 3D points reconstructed from the depth map and camera intrinsics | point cloud | Geometric representation derived from depth information |
| Depth-based object detection before VLM classification | depth-guided localization | Localizes objects without asking the VLM to solve geometry |
| VLM choice of an object_id from localized candidates | object grounding or object association | Connects semantic classification to a detected object region |
| CAD and observed point-cloud comparison | CAD-to-observation alignment or CAD-guided pose estimation | Uses known geometry to estimate or refine pose |

Depth information is not the same thing as a point cloud. The depth image is the raw per-pixel distance measurement. The point cloud is reconstructed from the depth image using the camera intrinsics.

## VLM Classification Policy

The VLM should associate one semantic target description with an already-localized object, not perform primary localization or task-role reasoning.

Preferred VLM input:

- RGB image
- optional ROI-optimized or bounding-box-annotated image
- one semantic target description
- localized object list with object_id, bbox_2d_xyxy, centroid_3d, 3D size, height, and related geometry

Internally map deterministic candidate labels to object IDs and ask the VLM to return one compact label. The public result should map that label back to an object ID and preserve output-token log probabilities, for example:

```json
{
  "target_description": "green block",
  "vlm_object_id": "object_001",
  "association_score": 0.91,
  "final_object_id": "object_001",
  "resolution": "vlm_accepted"
}
```

The association score must be computed from provider-returned output-token log probabilities. It is a raw response likelihood, not a calibrated probability of correct identity. If usable log probabilities are unavailable, keep the score null and defer to a human. The VLM and human paths must both produce the same `final_object_id` interface.

The VLM prompt should make clear:

- localization has already been performed by the depth pipeline
- object_id must be selected only from the provided localized candidates
- the model should not invent coordinates, bounding boxes, or objects
- a NONE choice is valid when the target is absent
- manipulation, destination, source, and reference roles belong to the downstream LLM planner

Use bounding boxes, masks, ROI highlighting, or geometric metadata to improve classification accuracy when helpful. These are guidance signals for semantic reasoning, not a replacement for the depth localization module.

## Ruthless Cleanup Priority

When editing this repository, prefer deletion over abstraction.

Remove code that is:

- unused
- duplicated
- generated but committed as source
- defensive without a real failure mode
- configurable only for the sake of configurability
- unrelated to the current RealSense point-cloud + VLM perception pipeline
- a wrapper around one line of real work

Do not preserve code just because it works. If it works but is redundant, simplify it while keeping behavior.

## Current Repository Issues To Watch

Future agents should treat these as cleanup targets, not patterns to copy:

- prototype constants should not become long command-line argument lists
- visualization should not be generated as a giant string inside another Python script
- generated outputs under `Output/` or `outputs/` should not be treated as source code
- visualization logic and capture/pose-estimation logic should be separated when the file becomes hard to read
- VLM prompts should stay compact and directly tied to localized object metadata
- downstream framework language should not cause agents to add unrelated planning or control modules

## CLI And Argparse Policy

Do not add `argparse` by default.

Use direct constants, a small config dataclass, or clearly named module-level settings when the user is running a controlled local prototype.

Only keep or add a CLI when at least one of these is true:

- the user explicitly asks for command-line options
- the script is meant to be reused by other people with different hardware or workspaces
- the option cannot reasonably be represented as a fixed config value

If a CLI is kept, keep it tiny. Avoid long lists of rarely changed flags such as every DBSCAN threshold, crop bound, warmup count, model name, and visualization switch. Those belong in a config block unless the user asks otherwise.

## Code Style

Write concise, direct Python.

Prefer:

- small functions with one job
- dataclasses only for real structured results
- explicit units in names, such as `center_mm`, `centroid_3d_m`, and `voxel_size_m`
- clear geometry math
- fail-fast errors for missing hardware, missing files, invalid transforms, empty point clouds, or missing API keys
- readable constants over sprawling argument parsers
- compact JSON-compatible outputs for perception results and VLM responses

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
- speculative downstream modules that are not used by the current run

## File Organization Rules

Keep source code separate from generated artifacts.

Recommended direction:

- `clean_pointcloud_vlm_test.py` or `realsense_pointcloud_pose.py`: capture, point-cloud conversion, segmentation, clustering, pose estimation, VLM prompt construction, VLM call, and saving current test results
- optional `visualize_pose.py`: visualization only, if the user wants a reusable visualization script
- `main.py`: delete it unless it provides real value beyond calling another `main()`
- `outputs/` or `Output/`: generated local run artifacts, not source code

Do not commit generated PLY/JSON/PNG output files unless the user explicitly asks for sample outputs for the paper or demo.

If generated outputs must stay for demonstration, label them as sample data and do not let the pipeline rewrite committed source files during normal execution.

## Dependencies

Use only the libraries explicitly needed for the current task.

Allowed for the current milestone:

- `pyrealsense2`
- `open3d`
- `numpy`
- `scipy`
- `opencv-python` / `cv2`
- `openai`
- `json`
- `pathlib`
- `dataclasses`

Avoid adding new packages unless requested.

`argparse` is not a default dependency. Use it only when the CLI policy above is satisfied.

## Coordinate And Unit Rules

Work internally in meters unless the task says otherwise.

Output robot- or paper-facing quantities in:

- millimeters for translation and dimensions
- degrees for roll, pitch, and yaw

Never invent calibration values. If a camera-to-robot or camera-to-world transform is required, require it from the user or load it from a specified file.

Be explicit about coordinate frames:

- camera frame
- table frame
- robot base frame
- object local frame

Do not silently mix these frames.

## Pose-Estimation Rules

For tabletop components:

- estimate object center from the object point cloud or oriented bounding box
- estimate yaw from the dominant horizontal PCA axis when the object geometry supports it
- estimate roll and pitch from the support/table plane normal when appropriate
- report yaw confidence from PCA eigenvalue separation when available
- do not pretend yaw is meaningful for rotationally symmetric objects
- do not claim full 6D pose if some orientation components are unobservable

Treat the phrase "6D pose" carefully. Some objects have unobservable orientation components due to symmetry, occlusion, or insufficient point-cloud coverage.

For the larger proposed framework, CAD comparison can refine pose, but CAD registration should not be implemented in this repository unless the user asks for it.
When explicitly requested, the CAD alignment baseline is: CAD sampling -> FPFH/RANSAC global registration -> ICP refinement -> augmented object point cloud.

## Output Expectations

When saving perception results, prefer:

- JSON for numeric results, structured prompts, VLM responses, and token usage
- PNG for RGB images, ROI images, masks, and annotated VLM inputs
- PLY for local point-cloud inspection only when needed

Do not generate extra files unless they are necessary for the requested run.

Do not write a Python script from inside another Python script unless the user specifically asks for code generation. A generated visualization script is usually the wrong design; put reusable visualization code in a normal source file or keep visualization inside the active run.

## Review Rule

Before finishing any change, check this list:

1. Did I add a feature outside the requested scope? Remove it.
2. Did I keep a CLI flag that should be a constant? Remove it.
3. Did I duplicate visualization, prompting, or transformation code? Consolidate it.
4. Did I commit generated `Output/` or `outputs/` files? Remove them unless requested.
5. Did I let the full proposed framework turn this perception prototype into an unrelated planning/control project? Narrow it.
6. Did I make the working prototype harder to understand? Simplify it.

The final repository should be smaller, clearer, and more defensible for paper and demo use.
