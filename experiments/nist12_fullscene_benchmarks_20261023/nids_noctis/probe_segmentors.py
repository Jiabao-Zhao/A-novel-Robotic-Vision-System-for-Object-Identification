"""Load local segmentation checkpoints on CPU without inference or downloads."""

import gc
from pathlib import Path
import sys
from unittest import mock

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common
from compat import import_native

common.setup()
method = sys.argv[1]
native = import_native(method, common)
import torch

if method == "nids_basic":
    import robokit.ObjDetection as official
    from segment_anything import sam_model_registry

    def local_sam(checkpoint):
        return sam_model_registry["vit_h"](checkpoint=str(common.WEIGHTS / "sam_vit_h_4b8939.pth"))

    with mock.patch.object(torch.cuda, "is_available", return_value=False), \
         mock.patch.dict(official.sam_model_registry, {"vit_h": local_sam}):
        segmentor = official.SegmentAnythingPredictor(vit_model="vit_h")
    model = segmentor.sam
else:
    from src.model.grounded_sam import YOLOSAMPredictor
    segmentor = YOLOSAMPredictor(str(common.WEIGHTS), vit_model="sam2.1_l.pt", device="cpu")
    segmentor.predictor.setup_model()
    model = segmentor.predictor.model

assert all(parameter.device.type == "cpu" for parameter in model.parameters())
result = dict(method=method, segmentation_checkpoint_load=True, device="cpu", inference_run=False,
              parameters=sum(p.numel() for p in model.parameters()), checkpoint_local=True)
common.save(LOCAL / f"{method}_segmentor_cpu_probe.json", result)
print(result, flush=True)
del segmentor, model
gc.collect()
