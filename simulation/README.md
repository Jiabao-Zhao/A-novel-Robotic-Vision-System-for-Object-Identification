# Current Simulation

The active experiment is the 13-object industrial workbench, not the retired
six-object wrist association or NIST assembly benchmark. Read the root
[`README.md`](../README.md) and [current protocol](human_cad_sam3_protocol.md).

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jiabao/.venvs/lerobot-libero/bin/python -m scripts.capture_industrial_workbench
```

This uses existing local assets and creates a new timestamped capture under
`outputs/simulation/industrial_workbench`. The frozen current capture remains
`20260922T214852Z`. Saved data includes wrist RGB, metric depth, intrinsics,
camera-to-world transform and evaluation-only object poses.

NIST asset preparation and the LIBERO sensor/environment adapters remain shared
dependencies. Older names such as `wrist_cluster.py` describe that implementation
history, not a request to restore the removed classification experiments.

The workbench dimensions and materials are designed simulation values, not physical
calibration. Official BOP products retain native metric geometry. The LM-O glue
bottle rests upright; no bottom view or tip-down placement is used. CAD-view
selection is operator-controlled. Simulator depth does not reproduce real sensor
failures, and this inspection layout is not a held-out benchmark.

No segmentation labels or simulator object poses may be used as scoring inputs.
No object-identification result is an authorization to move the real robot.
