# AGENTS.md

## Project goal

This repository develops a novel object identification by integrating 3D point-cloud localization and VLM classification for new object recognition.

The first milestone is simple and geometric:
Intel RealSense RGB-D data → segmented object point cloud → object center, dimensions, yaw, roll, pitch, yaw confidence, and saved JSON outputs.

Do not expand the project into LLM planning, VLM classification, grasp planning, robot control, GUI, voice control, CAD registration, or active scanning unless explicitly asked.

## Code style

Write concise, direct Python.

Prefer:
- small functions
- dataclasses for structured outputs
- explicit units
- clear geometry math
- fail-fast errors

Avoid:
- unnecessary classes
- abstract base classes
- plugin systems
- broad try/except blocks
- fake fallback logic
- large if/elif chains
- excessive validation wrappers
- logging frameworks
- unrelated helper modules

## Dependencies

Use only the libraries explicitly requested in the task.

For the first milestone, allowed dependencies are:
- pyrealsense2
- open3d
- numpy
- scipy
- argparse
- json
- pathlib
- dataclasses

Do not add new packages unless requested.

## Coordinate and unit rules

Work internally in meters unless the task says otherwise.

Output robot- or paper-facing quantities in:
- millimeters for translation and dimensions
- degrees for roll, pitch, yaw

Never invent calibration values. If a transform is required, require it from the user or load it from a specified file.

## Pose-estimation rules

For tabletop components:
- estimate object center from the object point cloud or oriented bounding box
- estimate yaw from the dominant horizontal PCA axis
- estimate roll and pitch from the support/table plane normal
- report yaw confidence from PCA eigenvalue separation
- do not pretend yaw is meaningful for symmetric objects

The phrase "6D pose" must be treated carefully. Some objects have unobservable orientation components due to symmetry.

## Output expectations

When creating scripts, include a minimal CLI.

When saving perception results, prefer JSON for numeric results and PLY for point clouds.

Do not create extra files unless the task explicitly asks for them.

## Review rule

Before finishing, check whether the code added features outside the requested scope. Remove them.