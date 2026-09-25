"""Shared DINOv2 inputs and tokens for independent CAD-view branches."""
import gc
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from visualize_localization_projection import mask_bbox, project_rgb_on_white

ROOT = Path(__file__).resolve().parents[1]
DINO_HUB_DIR = Path('outputs/cache/torch_hub')
DINO_REPO = 'facebookresearch/dinov2:e1277af2ba9496fbadf7aec6eba56e8d882d1e35'
IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID_SIZE = 16
BATCH_SIZE = 8


def letterbox_rgb(image):
    """Return the exact 224x224 RGB canvas used before DINO normalization."""
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError('DINOv2 requires a nonempty RGB image.')
    scale = IMAGE_SIZE / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    y, x = (IMAGE_SIZE - resized.shape[0]) // 2, (IMAGE_SIZE - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def prepare_masked_input(rgb, mask):
    """Use the existing inclusive tight crop and white letterbox."""
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('Expected a binary mask in the original RGB image coordinates.')
    bounds = mask_bbox(mask)
    if bounds is None:
        raise ValueError('An empty mask is unavailable, not a blank DINO candidate.')
    white = project_rgb_on_white(rgb, mask)
    x1, y1, x2, y2 = bounds
    crop = white[y1:y2 + 1, x1:x2 + 1]
    return letterbox_rgb(crop), crop, bounds


def letterbox_mask(mask):
    """Same RGB resize dimensions/offset; binary nearest-neighbor, zero padding."""
    height, width = mask.shape
    scale = IMAGE_SIZE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y, x = (IMAGE_SIZE - size[1]) // 2, (IMAGE_SIZE - size[0]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return canvas


def patch_occupancy(mask):
    """Row-major 16x16 grid: fraction of object pixels within each 14x14 patch."""
    mask = np.asarray(mask)
    if mask.shape != (224, 224) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Patch occupancy requires a binary 224x224 object mask.')
    return mask.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE).mean(axis=(1, 3))


def encode(model_name, dimension, images, patch_indices, output, device='cpu'):
    """Preserve existing preprocessing and FP32 inference; normalize in float64."""
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    stage = perf_counter()
    torch.hub.set_dir(str(ROOT / DINO_HUB_DIR))
    model = torch.hub.load(DINO_REPO, model_name, pretrained=True,
                           trust_repo=True, skip_validation=True).eval().requires_grad_(False).to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    runtime = {'device': device, 'model_load_s': perf_counter() - stage}
    assert model.patch_size == 14 and model.embed_dim == dimension and not model.training
    cls, patches = [], {}
    stage = perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), BATCH_SIZE):
            tensors = []
            for image in images[start:start + BATCH_SIZE]:
                assert image.shape == (224, 224, 3)
                np.testing.assert_array_equal(letterbox_rgb(image), image)
                normalized = (image.astype(np.float32) / 255 - [0.485, .456, .406]) / [.229, .224, .225]
                tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)))
            features = model.forward_features(torch.stack(tensors).to(device))
            raw = features['x_norm_clstoken'].cpu().numpy().astype(np.float64)
            assert raw.shape == (len(tensors), dimension)
            raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
            cls.extend(raw)
            for offset in range(len(tensors)):
                index = start + offset
                if index in patch_indices:
                    p = features['x_norm_patchtokens'][offset].cpu().numpy().astype(np.float64)
                    assert p.shape == (256, dimension)
                    p /= np.linalg.norm(p, axis=-1, keepdims=True)
                    patches[index] = p.reshape(16, 16, dimension)
            if start % 80 == 0:
                print(f'{model_name}: encoded {min(start + BATCH_SIZE, len(images))}/{len(images)}', flush=True)
    runtime['feature_extraction_s'] = perf_counter() - stage
    runtime['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    runtime['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved() if device == 'cuda' else None
    cls = np.stack(cls)
    assert np.isfinite(cls).all() and all(np.isfinite(p).all() for p in patches.values())
    np.testing.assert_allclose(np.linalg.norm(cls, axis=-1), 1, atol=1e-12)
    np.savez_compressed(output / 'features.npz', cls_features=cls,
                        patch_indices=list(patches), patch_tokens=np.stack(list(patches.values())))
    del features, model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return cls, patches, runtime
