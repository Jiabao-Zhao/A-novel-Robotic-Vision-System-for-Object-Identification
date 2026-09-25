# Local SAM3 Modes

The current framework uses **text-prompted concept segmentation**: each short VLM
CAD-view description is a separate query to `Sam3Model`/`Sam3Processor`, sharing
the scene embedding. Its detection score is not calibrated target correctness.
The active query and ungated scoring runners live in the separate local workspace
listed in the root [README](../README.md).

`local_sam3.py` is a retained point-grid tracker adapter using `Sam3TrackerModel`
and `Sam3TrackerProcessor`. It can generate unnamed masks from sampled image
locations. It is not the text-query path and its predicted mask IoU must not be
mixed with text-detection or combined scores. Its isolated tests remain in
`tests/test_local_sam3.py`; old BOP/hosted comparison drivers are retired.

Existing local checkpoint: `D:/AI/models/sam3`, or `/mnt/d/AI/models/sam3` in WSL.
The GPU environment is `/home/jiabao/.venvs/lerobot-libero/bin/python` in
Ubuntu-22.04. Keep the local checkpoint and model provenance. No hosted fallback
or new login is needed for already-cached inference. A fresh machine still needs
its own approved model access and compatible environment.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jiabao/.venvs/lerobot-libero/bin/python -m pytest -q tests/test_local_sam3.py
```

Do not restore SAM ViT-H, the old hosted Roboflow drivers, or the previous 0.8
detection gate when working on the current per-view text-prompt experiment.
