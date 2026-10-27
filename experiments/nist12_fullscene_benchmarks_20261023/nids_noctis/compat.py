"""Task-local loading adapters; official inference and scoring stay in sources/."""

import importlib
import os
import sys
from types import SimpleNamespace
from unittest import mock


def import_native(method, common):
    source = common.SOURCES / ("nids" if method == "nids_basic" else "noctis")
    sys.path[:0] = [str(common.SOURCES / name) for name in ("sam1", "mobilesam", "groundingdino")]
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(common.HERE / "nids_noctis/hf_compat"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(common.HERE / "nids_noctis/ultralytics_settings"))
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    import torch
    torch.set_num_threads(4)
    # Upstream NIDS imports an installer through robokit. Suppress that exact
    # import-time command; running it would modify the user's global environment.
    with mock.patch("os.system", return_value=0) as calls:
        detector = importlib.import_module("src.model.detector")
    blocked = [str(call.args[0]) for call in calls.call_args_list]
    assert all(command == "python setup.py build develop --user" for command in blocked), blocked
    return SimpleNamespace(source=source, detector=detector,
        descriptor=importlib.import_module("src.model.dinov2"),
        utils=importlib.import_module("src.model.utils"),
        dataloader=importlib.import_module("src.dataloader.bop"),
        blocked_import_commands=blocked)


def deformable_attention_fallback():
    """Use GroundingDINO's shipped PyTorch operator if its extension is absent."""
    import groundingdino.models.GroundingDINO.ms_deform_attn as module
    if hasattr(module, "_C"):
        return "official_compiled_extension"

    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return module.multi_scale_deformable_attn_pytorch(value, spatial_shapes,
                                                         sampling_locations, attention_weights)
    module.MultiScaleDeformableAttnFunction.apply = staticmethod(apply)
    return "official_pytorch_operator_on_selected_device"


def load_grounding(config, checkpoint, bert_dir, device):
    import torch
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    assert checkpoint.is_file(), checkpoint
    assert (bert_dir / "config.json").is_file(), bert_dir
    args = SLConfig.fromfile(str(config))
    args.device = str(device)
    args.text_encoder_type = str(bert_dir)
    model = build_model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    assert not incompatible.missing_keys, incompatible
    model._benchmark_checkpoint_load = dict(missing_keys=incompatible.missing_keys,
                                            unexpected_keys=incompatible.unexpected_keys)
    model.eval().to(device)
    return model


def nids_predictors(native, common):
    import robokit.ObjDetection as official
    from segment_anything import sam_model_registry

    def local_grounding(self, config_file, repo_id, filename):
        return load_grounding(native.source / config_file, common.WEIGHTS / filename,
                              common.WEIGHTS / "bert-base-uncased", self.device)

    def local_sam(checkpoint):
        return sam_model_registry["vit_h"](checkpoint=str(common.WEIGHTS / "sam_vit_h_4b8939.pth"))

    with mock.patch.object(official.GroundingDINOObjectPredictor, "load_model_hf", local_grounding):
        detector = official.GroundingDINOObjectPredictor(use_vitb=False, threshold=0.10)
    with mock.patch.dict(official.sam_model_registry, {"vit_h": local_sam}):
        segmentor = official.SegmentAnythingPredictor(vit_model="vit_h")
    return detector, segmentor


def noctis_predictor(native, common):
    import src.model.grounded_sam as official

    def checkpoint_path(repo_id, filename, **kwargs):
        path = common.WEIGHTS / filename
        assert path.is_file(), path
        return str(path)

    def local_grounding(config_file, checkpoint_file, device):
        return load_grounding(common.path(config_file), common.path(checkpoint_file),
                              common.WEIGHTS / "bert-base-uncased", device)

    with mock.patch.object(official, "hf_hub_download", checkpoint_path), \
         mock.patch.object(official.GroundingDINO, "load_model", staticmethod(local_grounding)):
        result = official.GroundedSAM.load_grounded_sam_model(
            checkpoints_dir=str(common.WEIGHTS), grounded_dino_config_dir=str(native.source / "src/cfg/gdino"),
            grounded_dino_use_vitb=True, box_threshold=0.1, text_threshold=0.1,
            use_yolo_sam=True, sam_vit_model="sam2.1_l.pt", mask_threshold=0.01,
            prompt_text="objects", segmentor_width_size=640, device="cuda")
    result.sam_predictor.chunk_size = 2
    return result
