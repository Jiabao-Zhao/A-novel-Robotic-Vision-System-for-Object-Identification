"""CAD-only diagnosis of zero-mask native FFA denominators; no scene labels."""

from pathlib import Path
import sys

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common
from compat import import_native

inputs = common.setup()
native = import_native("nids_basic", common)
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

dataset = native.dataloader.BOPTemplate(str(common.HERE / "templates42"),
    obj_ids=[o["object_id"] for o in inputs["objects"]],
    processing_config=OmegaConf.create(dict(image_size=224)), level_templates=0, pose_distribution="all")
cache = common.HERE / "nids_basic/native_reference_features.pt"
reference = torch.load(cache, map_location="cpu", weights_only=True)
invalid = (~torch.isfinite(reference["descriptors"])).any(-1)
rows = []
for obj, view in invalid.nonzero().tolist():
    mask = dataset[obj]["template_masks"][view]
    sampled = F.interpolate(mask[None, None], (16, 16), mode="bilinear")[0, 0]
    pooled = F.avg_pool2d(mask[None, None], 14, 14)[0, 0]
    rows.append(dict(object_id=dataset.obj_ids[obj], target=inputs["objects"][obj]["target"],
        template_slot=view, canonical_view_id=int(dataset.index_templates[view]),
        mask_support_pixels=int((mask > 0).sum()), mask_sum=float(mask.sum()),
        native_FFA_bilinear_mask_sum=float(sampled.sum()),
        appearance_valid_patches=int((pooled > .5).sum()),
        FFA_nonfinite_values=int((~torch.isfinite(reference["descriptors"][obj, view])).sum())))

result = dict(CAD_only=True, labels_read=False, native_reference_sha256=common.digest(cache),
    cause="Native bilinear mask sampling produces zero denominator for six thin USB views; NaNs propagate through max(view), then max(object).",
    invalid_views=rows, finite_scene_query_vectors=True,
    unchanged_native_run="nids_basic: eight completed scene outputs, all zero retained detections",
    no_threshold_changes=True)
common.save(LOCAL / "nids_zero_mask_diagnostic.json", result)
print(result, flush=True)
