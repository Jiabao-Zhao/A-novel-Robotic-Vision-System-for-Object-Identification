"""Official NeMO encoder and own full-scene multiscale segmentation.

Only execution batching and output serialization surround the released model.
CAD references are a prospectively frozen 32-view subset of custom CAD renders.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))
from common import REPO, SOURCES, digest, path, read, save, setup

COMMIT = '76478815baf2c99002cccbee98426773fa4e59cd'
SOURCE = SOURCES / 'nemo'
sys.path.insert(0, str(SOURCE / 'src'))
CHECKPOINT = HERE / 'weights' / 'checkpoint.pth'
OUTPUT = BENCHMARK / 'nemo'
MASK_THRESHOLD = 0.5
MINIMUM_AREA = 32 * 32
SAMPLE_POINT_COUNT = 1500
PATCH_BATCH_SIZE = 2


def load_native_detector():
    filename = SOURCE / 'examples' / 'run_detection_on_video.py'
    spec = importlib.util.spec_from_file_location('official_nemo_detector', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(device):
    import torch
    from omegaconf import OmegaConf
    from nemolib.model import Model

    checkpoint = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    assert all(key.startswith('model.') for key in state)
    state = {key[len('model.'):]: value for key, value in state.items()}
    config = checkpoint['hyper_parameters']['config']
    original_hub_load = torch.hub.load

    def cached_architecture(repo, name, *args, **kwargs):
        # Complete NeMO weights include the backbone, so no second weight file is needed.
        if repo != 'facebookresearch/dinov2':
            raise RuntimeError(f'Unexpected model source {repo}')
        cached_source = (REPO / 'outputs/cache/torch_hub' /
                         'facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35')
        if not cached_source.is_dir():
            raise FileNotFoundError(f'Expected cached official DINOv2 source: {cached_source}')
        return original_hub_load(str(cached_source), name, source='local', pretrained=False)

    torch.hub.load = cached_architecture
    try:
        model = Model(config)
    finally:
        torch.hub.load = original_hub_load
    coverage = model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.eval().requires_grad_(False).to(device)
    save(OUTPUT / 'model_load.json', {
        'source_commit': COMMIT,
        'checkpoint_sha256': digest(CHECKPOINT),
        'missing_keys': coverage.missing_keys, 'unexpected_keys': coverage.unexpected_keys,
        'config': OmegaConf.to_container(config, resolve=True),
        'backbone_initialization': 'Cached official DINOv2 architecture pretrained=False; full checkpoint loaded strictly.',
        'weight_updates': False,
    })
    return model


class BatchedDecoder:
    """Bound activation memory without changing native windows or aggregation."""
    def __init__(self, model):
        self.model = model

    def __call__(self, run, images, features_3d):
        import torch
        assert run == 'xyz' and images.shape[0] == 1
        outputs = {'mask': [], 'mask_full': []}
        for start in range(0, images.shape[1], PATCH_BATCH_SIZE):
            result = self.model(run, images=images[:, start:start + PATCH_BATCH_SIZE],
                                features_3d=features_3d)
            for key in outputs:
                outputs[key].append(result[key])
        return {key: torch.cat(values, dim=1) for key, values in outputs.items()}


def build_memories(model, manifest, references, device):
    import torch
    from nemolib.utils import load_images

    memories = {}
    for item in manifest['objects']:
        object_id = item['object_id']
        destination = OUTPUT / 'memories' / f'obj_{object_id:06d}.pt'
        if destination.exists():
            previous = read(destination.with_suffix('.json'))
            expected = references['objects'][str(object_id)]['sha256']
            assert previous['reference_sha256'] == expected
            assert previous['checkpoint_sha256'] == digest(CHECKPOINT)
            memories[object_id] = destination
            continue
        images = [str(path(value)) for value in references['objects'][str(object_id)]['images']]
        assert len(images) == 32
        torch.manual_seed(0)
        image_tensor = load_images(images, device=device, normalize=False).unsqueeze(0)
        sample_points = torch.rand(1, SAMPLE_POINT_COUNT, 3, device=device) * 2 - 1
        started = time.perf_counter()
        with torch.no_grad():
            nemo = model.encode_images(image_tensor, sample_points)
            nemo['features_3d_updated'] = (
                nemo['features_3d'] + model.point_encoder(nemo['surface_points']))
            nemo['scale_factor'] = None
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in nemo.items()}, destination)
        save(destination.with_suffix('.json'), {
            'object_id': object_id, 'target': item['target'],
            'reference_sha256': {value: digest(path(value)) for value in references['objects'][str(object_id)]['images']},
            'checkpoint_sha256': digest(CHECKPOINT),
            'seed': 0, 'sample_point_count': SAMPLE_POINT_COUNT,
            'seconds': time.perf_counter() - started,
        })
        memories[object_id] = destination
        del image_tensor, sample_points, nemo
        torch.cuda.empty_cache()
        print(f'encoded {object_id} {item["target"]}', flush=True)
    return memories


def segment_one(model, detector, features, rgb, device):
    import torch
    import numpy as np
    from nemolib.utils import masks_to_bounding_boxes

    image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().to(device) / 255
    height, width = image.shape[-2:]
    patch_sizes = [min(height, width) // divisor for divisor in (1, 2, 3)]
    amodal = torch.zeros((1, 1, height, width), device=device)
    amodal_conf = torch.zeros_like(amodal)
    modal = torch.zeros_like(amodal)
    modal_conf = torch.zeros_like(amodal)
    for patch_size in patch_sizes:
        result = detector.segment_objects(
            BatchedDecoder(model), features, image, mask_thr=MASK_THRESHOLD,
            patch_size=patch_size, stride=patch_size // 3, device=device)
        amodal += result[0]
        amodal_conf += result[1]
        modal += result[2]
        modal_conf += result[3]
    # Preserve the official union across scales and mean confidence.
    amodal = amodal / len(patch_sizes) > 0.0
    modal = modal / len(patch_sizes) > 0.0
    amodal_conf /= len(patch_sizes)
    modal_conf /= len(patch_sizes)
    boxes, scores, _ = masks_to_bounding_boxes(amodal, amodal_conf, minimum_area=MINIMUM_AREA)
    masks = []
    for x0, y0, x1, y1 in boxes[0]:
        mask = torch.zeros((height, width), dtype=torch.bool, device=device)
        mask[y0:y1, x0:x1] = modal[0, 0, y0:y1, x0:x1]
        masks.append(mask.cpu().numpy())
    raw = {'amodal_mask': amodal[0, 0].cpu().numpy(), 'modal_mask': modal[0, 0].cpu().numpy(),
           'amodal_confidence': amodal_conf[0, 0].cpu().numpy(),
           'modal_confidence': modal_conf[0, 0].cpu().numpy()}
    return boxes[0], scores[0], masks, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=None)
    parser.add_argument('--encode-only', action='store_true')
    args = parser.parse_args()
    manifest = setup()
    references = read(HERE / 'references.json')
    import torch
    import numpy as np
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the authorized CUDA device.')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device('cuda')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    detector = load_native_detector()
    memories = build_memories(model, manifest, references, device)
    if args.encode_only:
        return
    for scene in manifest['scenes']:
        if args.scene and scene['scene_id'] != args.scene:
            continue
        destination = OUTPUT / scene['scene_id']
        if (destination / 'completion.json').exists():
            previous = read(destination / 'completion.json')
            assert previous['references_sha256'] == digest(HERE / 'references.json')
            assert previous['rgb_sha256'] == digest(path(scene['rgb']))
            print(f'already complete {scene["scene_id"]}', flush=True)
            continue
        rgb = np.asarray(Image.open(path(scene['rgb'])).convert('RGB'))
        detections, all_masks = [], []
        destination.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for item in manifest['objects']:
            object_id = item['object_id']
            features = torch.load(memories[object_id], map_location=device,
                                  weights_only=False)['features_3d_updated']
            with torch.no_grad():
                boxes, scores, masks, raw = segment_one(model, detector, features, rgb, device)
            np.savez_compressed(destination / f'raw_obj_{object_id:06d}.npz', **raw)
            for box, score, mask in zip(boxes, scores, masks):
                index = len(detections)
                detections.append({'proposal_id': index, 'object_id': object_id,
                                   'score': float(score), 'bbox_xyxy': list(map(int, box)),
                                   'mask_index': index})
                all_masks.append(mask)
            del features
            torch.cuda.empty_cache()
            print(f'{scene["scene_id"]} {item["target"]} detections={len(boxes)}', flush=True)
        masks_array = np.stack(all_masks) if all_masks else np.zeros((0, *rgb.shape[:2]), dtype=bool)
        np.savez_compressed(destination / 'masks.npz', masks=masks_array)
        save(destination / 'detections.json', detections)
        save(destination / 'completion.json', {
            'scene_id': scene['scene_id'], 'detections': len(detections),
            'seconds': time.perf_counter() - started,
            'rgb_sha256': digest(path(scene['rgb'])),
            'source_commit': COMMIT, 'runner_sha256': digest(__file__),
            'references_sha256': digest(HERE / 'references.json'),
            'native_detector_sha256': digest(SOURCE / 'examples' / 'run_detection_on_video.py'),
            'minimum_area_pixels': MINIMUM_AREA, 'mask_threshold': MASK_THRESHOLD,
            'patch_batch_size': PATCH_BATCH_SIZE, 'device': str(torch.cuda.get_device_name()),
            'supplied_scene_regions': False, 'depth_used': False,
        })


if __name__ == '__main__':
    main()
