"""Full-scene native NIDS (no adapter) and NOCTIS with staged GPU residency.

Usage: run_native.py {nids_basic|noctis} {prepare|proposals|score|all|imports}
The runner never consumes an existing scene crop, mask, label, or object pose.
"""

import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest import mock

import numpy as np

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent))
import common
from compat import import_native, deformable_attention_fallback, load_grounding, nids_predictors, noctis_predictor


def write_once(path, value):
    if path.exists():
        assert common.read(path) == value, f"Frozen evidence differs: {path}"
    else:
        common.save(path, value)


def method_directory(method):
    result = common.HERE / method
    result.mkdir(exist_ok=True)
    return result


def configuration(method, inputs):
    source = common.SOURCES / ("nids" if method == "nids_basic" else "noctis")
    paths = [LOCAL / "run_native.py", LOCAL / "compat.py", LOCAL / "source_pins.json",
             LOCAL / "dependency_source_pins.json", LOCAL / "hf_compat_install_report.json",
             LOCAL / "test_native.py", LOCAL / "probe_segmentors.py",
             LOCAL / f"{method}_CPU_tests.json", LOCAL / f"{method}_cpu_probe.json",
             LOCAL / f"{method}_segmentor_cpu_probe.json", common.HERE / "inputs.json", common.HERE / "protocol.json"]
    paths += sorted(source.rglob("*.py")) + sorted(source.rglob("*.yaml"))
    paths += sorted(source.rglob("*.npy"))
    for dependency in ("groundingdino", "sam1", "mobilesam"):
        paths += sorted((common.SOURCES / dependency).rglob("*.py"))
    paths += [common.HERE / "templates42/metadata.json", common.HERE / "templates42/completion.json"]
    paths += sorted((common.HERE / "templates42").rglob("*.png"))
    hub = common.REPO / "outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
    paths += sorted(hub.rglob("*.py")) + [common.REPO / "outputs/cache/torch_hub/checkpoints/dinov2_vitl14_pretrain.pth"]
    weights = ["groundingdino_swint_ogc.pth", "sam_vit_h_4b8939.pth"] if method == "nids_basic" else ["groundingdino_swinb_cogcoor.pth", "yolosam/sam2.1_l.pt"]
    assert all((common.WEIGHTS / name).is_file() for name in weights), weights
    paths += [common.WEIGHTS / name for name in weights]
    paths += sorted((common.WEIGHTS / "bert-base-uncased").glob("*"))
    paths = [path for path in paths if path.is_file()]
    indices = np.load(source / ("src/poses/predefined_poses/idx_all_level0_in_level2.npy" if method == "nids_basic" else "datasets/predefined_poses/ico/indices_object_all_level0_in_level_2.npy"))
    return dict(method=method, task="Full scene -> native retained instance detections for supplied CAD catalogue",
        scenes=[s["scene_id"] for s in inputs["scenes"]], object_ids=[o["object_id"] for o in inputs["objects"]],
        CAD_templates="Released Pyrender variant, native source template loader and canonical view indices",
        native_template_view_indices=indices.tolist(), unique_template_views=int(len(np.unique(indices))),
        supplied_scene_proposals=False, simulator_poses_or_labels_used=False, training=False,
        nids=dict(adapter=False, appearance_enabled=True, semantic_aggregation="max", confidence_threshold=0.01,
                  object_assignment="native independent per-proposal argmax", nms_threshold=0.25),
        noctis=dict(released_variant="cnos2AppCT5ConfGBSAM21LPyrenderMode", semantic_aggregation="avg_5",
                    semantic_minimum=0.2, appearance_aggregation="max", appearance_cycle_threshold=5,
                    appearance_extra_factor=2, detector_confidence=True, final_confidence_threshold=0.2,
                    nms_threshold=0.25, segmentor_width=640),
        common_native_postprocess=dict(min_box_size=0.05, min_mask_size=0.0003),
        compatibility=["Block only NIDS's import-time global installer.",
          "Resolve all pretrained checkpoints and BERT from task-local paths; no implicit model downloads.",
          "Load Transformers 4.38.2/tokenizers 0.15.2/huggingface_hub 0.23.5 only from this runner's isolated hf_compat directory; global packages stay untouched.",
          "Use shipped PyTorch deformable attention if compiled GroundingDINO extension is unavailable.",
          "Stage proposal generation and descriptor matching separately; retain raw proposal outputs.",
          "Native query preprocessing runs in batches of two; scoring formula and thresholds unchanged.",
          "Native NOCTIS patch scoring uses query/object chunks of one to fit 8 GB.",
          "NIDS uses native no-adapter default, including patch appearance; FFA-only values are diagnostic.",
          "NIDS Pyrender loader returns BxHxW masks; restore the singleton channel required by its native reference-patch routine (PBR already returns it).",
          "Empty proposal sets export empty detections rather than failing at an upstream concatenation."],
        source_sha256={common.relative(p): common.digest(p) for p in sorted(set(paths))})


def prepare(method, inputs):
    config = configuration(method, inputs)
    assert len(config["scenes"]) == 8 and len(config["object_ids"]) == 12
    assert (common.HERE / "templates42/metadata.json").is_file()
    write_once(method_directory(method) / "configuration.json", config)
    return config


def verify_configuration(method):
    config = common.read(method_directory(method) / "configuration.json")
    for name, expected in config["source_sha256"].items():
        assert common.digest(common.path(name)) == expected, name
    return config


def native_rgb(scene):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    image = Image.open(common.path(scene["rgb"])).convert("RGB")
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=(.485, .456, .406), std=(.229, .224, .225))])
    inverse = transforms.Normalize(mean=(-.485/.229, -.456/.224, -.406/.225),
                                   std=(1/.229, 1/.224, 1/.225))
    return np.uint8(inverse(normalize(image)).cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255)


def proposals(method, inputs):
    import torch
    verify_configuration(method)
    native = import_native(method, common)
    operator = deformable_attention_fallback()
    if method == "nids_basic":
        detector, segmentor = nids_predictors(native, common)
    else:
        segmentor = noctis_predictor(native, common)
    from PIL import Image
    for scene in inputs["scenes"]:
        out = method_directory(method) / scene["scene_id"]
        out.mkdir(exist_ok=True)
        if (out / "proposal_completion.json").is_file():
            continue
        image = native_rgb(scene)
        started = perf_counter()
        with torch.inference_mode():
            if method == "nids_basic":
                pil = Image.fromarray(image)
                boxes, phrases, confidences = detector.predict(pil, "objects")
                boxes = detector.bbox_to_scaled_xyxy(boxes, pil.width, pil.height)
                if len(boxes):
                    boxes, masks = segmentor.predict(pil, boxes)
                    raw = dict(boxes=boxes.detach().cpu().numpy(),
                               masks=masks[:, 0].detach().cpu().numpy().astype(np.float32),
                               boxes_scores=confidences.detach().cpu().numpy())
                else:
                    raw = dict(boxes=np.empty((0, 4), np.float32), masks=np.empty((0, *image.shape[:2]), np.float32),
                               boxes_scores=np.empty(0, np.float32))
            else:
                detected = segmentor.generate_masks(image)
                raw = {key: value.detach().cpu().numpy() for key, value in detected.items()} if detected is not None else dict(
                    boxes=np.empty((0, 4), np.float32), masks=np.empty((0, *image.shape[:2]), np.float32),
                    boxes_scores=np.empty(0, np.float32), masks_scores=np.empty(0, np.float32))
        assert raw["masks"].shape == (len(raw["boxes"]), *image.shape[:2])
        assert all(np.isfinite(array).all() for array in raw.values())
        np.savez_compressed(out / "raw_proposals.npz", **raw)
        write_once(out / "raw_proposals.json", [dict(proposal_id=i, bbox_xyxy=box.tolist(),
            box_confidence=float(raw["boxes_scores"][i]),
            mask_confidence=float(raw["masks_scores"][i]) if "masks_scores" in raw else None)
            for i, box in enumerate(raw["boxes"])])
        write_once(out / "proposal_completion.json", dict(scene_id=scene["scene_id"], raw_proposals=len(raw["boxes"]),
            shape_hw=list(image.shape[:2]), seconds=perf_counter()-started,
            raw_proposals_sha256=common.digest(out / "raw_proposals.npz"),
            RGB_sha256=common.digest(common.path(scene["rgb"])), operator=operator,
            grounding_checkpoint_load=(detector.model if method == "nids_basic" else segmentor.ground_dino.model)._benchmark_checkpoint_load,
            blocked_import_commands=native.blocked_import_commands, labels_read=False))
        print(f'{method} {scene["scene_id"]}: {len(raw["boxes"])} own proposals', flush=True)
        del raw
        torch.cuda.empty_cache()
    verify_configuration(method)


def descriptor_and_templates(method, native, inputs):
    import torch
    from omegaconf import OmegaConf
    checkpoint = common.REPO / "outputs/cache/torch_hub/checkpoints/dinov2_vitl14_pretrain.pth"
    hub = common.REPO / "outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
    backbone = torch.hub.load(str(hub), "dinov2_vitl14", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    backbone = backbone.eval().requires_grad_(False).to("cuda")
    cfg = OmegaConf.create(dict(image_size=224))
    kwargs = dict(template_dir=str(common.HERE / "templates42"),
                  obj_ids=[o["object_id"] for o in inputs["objects"]], processing_config=cfg,
                  level_templates=0, pose_distribution="all")
    if method == "nids_basic":
        descriptor = native.descriptor.CustomDINOv2(model_name="dinov2_vitl14", model=backbone,
            token_name="x_norm_clstoken", image_size=224, chunk_size=2, descriptor_width_size=640)
    else:
        descriptor = native.descriptor.CustomDINOv2(model_name="dinov2_vitl14", model_or_repo=backbone,
                                                   image_size=224, chunk_size=2, valid_patch_mask_threshold=.5)
        kwargs.update(predefined_poses_dir=str(native.source / "datasets/predefined_poses/ico"), max_level_templates=2)
    dataset = native.dataloader.BOPTemplate(**kwargs)
    template_metadata = common.read(common.HERE / "templates42/metadata.json")
    if method == "nids_basic":
        np.testing.assert_array_equal(dataset.index_templates, template_metadata["index_templates"])
    assert len(dataset.index_templates) == 42
    return descriptor, dataset, checkpoint


def build_matcher(method, native, descriptor, dataset, directory):
    from omegaconf import OmegaConf
    post = OmegaConf.create(dict(mask_post_processing=dict(min_box_size=.05, min_mask_size=3e-4),
                                nms_thresh=.25, nms_threshold=.25))
    onboarding = OmegaConf.create(dict(rendering_type="pyrender", reset_descriptors=True, level_templates=0))
    if method == "nids_basic":
        from src.model.loss import PairwiseSimilarity
        matching = SimpleNamespace(aggregation_function="max", confidence_thresh=.01,
                                   metric=PairwiseSimilarity(metric="cosine", chunk_size=16))
        with mock.patch.object(native.detector, "GroundingDINOObjectPredictor", return_value=None), \
             mock.patch.object(native.detector, "SegmentAnythingPredictor", return_value=None):
            model = native.detector.NIDSNET(None, descriptor, onboarding, matching, post, 5, str(directory))
        assert model.use_adapter is False
    else:
        matching = OmegaConf.create(dict(semantic_score_weight_factor=1, semantic_aggregation_function="avg_5",
            semantic_confidence_threshold=.2, appearance_score_weight_factor=1, appearance_confidence_threshold=0,
            appearance_cycle_threshold=5, appearance_aggregation_function="max", appearance_score_extra_weight_factor=2,
            use_detector_confidence=True, final_confidence_threshold=.2, max_num_instances=100))
        model = native.detector.NOCTIS(None, descriptor, onboarding, matching, post, "native", 5,
                                       str(directory), storage_device="cpu", return_detections=True)
    model.ref_dataset = dataset
    return model


def reference_features(method, native, descriptor, dataset, out):
    import torch
    cache = out / "native_reference_features.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)
    pooled, patches = [], []
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = dataset[i]
            images, masks = item["templates"].to("cuda"), item["template_masks"].to("cuda")
            if method == "nids_basic":
                ffa, _, _ = descriptor.compute_features(images, "x_norm_clstoken", masks)
                patch = descriptor.compute_masked_patch_feature(images, masks[:, None] if masks.ndim == 3 else masks)
                pooled.append(ffa.cpu()); patches.append(patch.cpu())
            else:
                cls, patch = descriptor.compute_cls_and_patch_features(images, masks)
                pooled.append(cls.cpu()); patches.append(patch.cpu())
            print(f"{method} native CAD object {dataset.obj_ids[i]}: {len(images)} views", flush=True)
            del images, masks
    result = {"descriptors": torch.stack(pooled), "appe_descriptors": torch.stack(patches)} if method == "nids_basic" else {
        "cls_descriptors": torch.stack(pooled), "patch_descriptors": torch.stack(patches)}
    torch.save(result, cache)
    write_once(out / "reference_completion.json", dict(views_per_object=42, object_ids=dataset.obj_ids,
        index_templates=np.asarray(dataset.index_templates).tolist(), unique_views=len(set(dataset.index_templates.tolist())),
        cache_sha256=common.digest(cache), native_loader=True, labels_read=False))
    return result


def query_features(method, native, descriptor, image, detections):
    import torch
    all_features = None
    with torch.inference_mode():
        for start in range(0, len(detections), 2):
            subset = SimpleNamespace(masks=detections.masks[start:start+2].to("cuda"),
                                     boxes=detections.boxes[start:start+2].to("cuda"))
            parts = descriptor(image, subset)
            assert all(len(part) == len(subset.boxes) for part in parts), "Native preprocessing skipped a proposal; mapping must stay explicit"
            if all_features is None:
                all_features = [[] for _ in parts]
            for values, value in zip(all_features, parts):
                values.append(value.detach().cpu())
    return [torch.cat(values) for values in all_features]


def export_detections(method, native, inputs, out, detections, raw_count, filtered_count, seconds, score_details):
    import torch
    rows = []
    masks = detections.masks.detach().cpu().numpy() > 0
    object_ids = [o["object_id"] for o in inputs["objects"]]
    for i in range(len(detections)):
        rows.append(dict(proposal_id=int(detections.proposal_ids[i]), object_id=object_ids[int(detections.object_ids[i])],
                         score=float(detections.scores[i]), bbox_xyxy=detections.boxes[i].tolist(), mask_index=i))
    np.savez_compressed(out / "masks.npz", masks=masks)
    write_once(out / "detections.json", rows)
    np.savez_compressed(out / "matching_scores.npz", **score_details)
    write_once(out / "completion.json", dict(method=method, raw_proposals=raw_count,
        proposals_after_native_size_filter=filtered_count, retained_detections=len(rows), seconds=seconds,
        detections_sha256=common.digest(out / "detections.json"), masks_sha256=common.digest(out / "masks.npz"),
        raw_proposals_sha256=common.digest(out / "raw_proposals.npz"), matching_scores_sha256=common.digest(out / "matching_scores.npz"),
        proposal_filter_sha256=common.digest(out / "proposal_filter.json"),
        labels_read=False, native_scores_and_NMS=True, mask_export_threshold="strictly >0, matching official force_binary_mask"))


def cpu_probe(method):
    native = import_native(method, common)
    operator = deformable_attention_fallback()
    config = native.source / ("robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py" if method == "nids_basic" else "src/cfg/gdino/GroundingDINO_SwinB_cfg.py")
    checkpoint = common.WEIGHTS / ("groundingdino_swint_ogc.pth" if method == "nids_basic" else "groundingdino_swinb_cogcoor.pth")
    started = perf_counter()
    model = load_grounding(config, checkpoint, common.WEIGHTS / "bert-base-uncased", "cpu")
    result = dict(method=method, CPU_load_success=True, seconds=perf_counter()-started,
                  checkpoint_load=model._benchmark_checkpoint_load, operator=operator,
                  checkpoint_sha256=common.digest(checkpoint), inference_run=False)
    common.save(LOCAL / f"{method}_cpu_probe.json", result)
    print(json.dumps(result), flush=True)


def score(method, inputs):
    import torch
    verify_configuration(method)
    native = import_native(method, common)
    descriptor, dataset, checkpoint = descriptor_and_templates(method, native, inputs)
    directory = method_directory(method)
    matcher = build_matcher(method, native, descriptor, dataset, directory)
    matcher.ref_data = reference_features(method, native, descriptor, dataset, directory)
    for scene in inputs["scenes"]:
        out = directory / scene["scene_id"]
        if (out / "completion.json").is_file():
            continue
        done = common.read(out / "proposal_completion.json")
        assert common.digest(out / "raw_proposals.npz") == done["raw_proposals_sha256"]
        raw = dict(np.load(out / "raw_proposals.npz"))
        raw_count = len(raw["boxes"])
        raw["proposal_ids"] = np.arange(raw_count)
        detections = native.utils.Detections({key: torch.from_numpy(value) for key, value in raw.items()})
        started = perf_counter()
        detections.remove_very_small_detections(matcher.post_processing_config.mask_post_processing)
        filtered_count = len(detections)
        write_once(out / "proposal_filter.json", dict(
            retained_proposal_ids=detections.proposal_ids.tolist(),
            raw_proposals=raw_count, retained_after_native_size_filter=filtered_count))
        detail = {}
        if filtered_count:
            features = query_features(method, native, descriptor, native_rgb(scene), detections)
            with torch.inference_mode():
                if method == "nids_basic":
                    ffa, appearance, _ = features
                    selected, identities, ffa_scores, best_views = matcher.find_matched_proposals(ffa)
                    detail = dict(proposal_ids=detections.proposal_ids.numpy(), FFA_features=ffa.numpy())
                    detections.filter(selected)
                    app_scores, _ = matcher.compute_appearance_score(best_views, identities, appearance[selected])
                    scores = (ffa_scores + app_scores) / 2
                    detail.update(selected_proposal_ids=detections.proposal_ids.numpy(), ffa_score=ffa_scores.numpy(),
                                  appearance_score=app_scores.numpy(), selected_object_indices=identities.numpy(), best_template_indices=best_views.numpy())
                else:
                    from src.model import scoring
                    cls, appearance = [value.to("cuda") for value in features]
                    semantic = matcher.compute_semantic_score(cls)
                    app_views = scoring.compute_appearance_similarity(appearance, matcher.ref_data["patch_descriptors"],
                        min_threshold=0, cycle_threshold=5, chunk_size=1, obj_chunk_size=1)
                    app_scores = matcher.aggregate_scores(app_views * 2, "max")
                    scores = (semantic + app_scores) / 2
                    confidence = (detections.boxes_scores.to("cuda") + detections.masks_scores.to("cuda")) / 2
                    scores = (scores * confidence[:, None]).clamp(0, 1)
                    detail = dict(proposal_ids=detections.proposal_ids.numpy(), semantic_scores=semantic.cpu().numpy(),
                                  appearance_scores=app_scores.cpu().numpy(), detector_confidence=confidence.cpu().numpy(),
                                  combined_scores=scores.cpu().numpy())
                    selected, identities, scores, distribution = matcher.choose_best_and_filter(scores, .2, 100)
                    detections.filter(selected.cpu())
                    identities, scores = identities.cpu(), scores.cpu()
                detections.add_attribute("object_ids", identities)
                detections.add_attribute("scores", scores)
                if len(detections):
                    if method == "nids_basic":
                        detections.apply_nms_per_object_id(nms_thresh=.25)
                    else:
                        detections.apply_nms_per_object_id(nms_threshold=.25)
        if not hasattr(detections, "object_ids"):
            detections.add_attribute("object_ids", torch.empty(0, dtype=torch.long))
            detections.add_attribute("scores", torch.empty(0))
        export_detections(method, native, inputs, out, detections, raw_count, filtered_count, perf_counter()-started, detail)
        print(f'{method} {scene["scene_id"]}: {raw_count} raw -> {filtered_count} size-filtered -> {len(detections)} retained', flush=True)
        del detections, raw
        torch.cuda.empty_cache()
    verify_configuration(method)


if __name__ == "__main__":
    method, stage = sys.argv[1:3]
    assert method in ("nids_basic", "noctis")
    inputs = common.setup()
    if stage == "imports":
        native = import_native(method, common)
        print(json.dumps(dict(imported=method, blocked=native.blocked_import_commands,
                              deformable_attention=deformable_attention_fallback())))
    elif stage == "prepare":
        prepare(method, inputs)
    elif stage == "probe":
        cpu_probe(method)
    elif stage == "proposals":
        proposals(method, inputs)
    elif stage == "score":
        score(method, inputs)
    elif stage == "all":
        prepare(method, inputs)
        proposals(method, inputs)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        score(method, inputs)
    else:
        raise ValueError(stage)
