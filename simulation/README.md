# NIST Simulation

The supported generator is NIST Task Board 1. The five-part and seven-part
framework-figure scenes are subsets of the same NIST product family.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jiabao/.venvs/lerobot-libero/bin/python -m scripts.capture_nist_task_board
```

Official task-board meshes are prepared under
`outputs/simulation/nist_task_board_1/assets/`. The converted `_m.stl` files use
metres. Existing source assets and camera calibration are retained.

Shared historical N00-N07 captures and twenty-view CAD references live under
`outputs/datasets/nist_task_board/shared_inputs/`; the migration receipts record
original hashes and path-only XML edits. A background-fixture mesh needed by
some historical NIST captures is retained as a dependency, not a separate layout
or FMB object-generation workflow.

`libero_env.py`, `libero_sensor.py`, `libero_io.py` and `libero_control.py` provide
the shared simulator, RGB-D bridge and Panda controls. The LIBERO name does not
mean household layouts are part of the current experiment. Installed backend
textures/robot assets must remain available.

Read [the pipeline protocol](pipeline_protocol.md) before changing capture,
projection, matching or pose assumptions. No capture command authorizes real
robot execution, and no scene defines validated mechanical assembly success.
