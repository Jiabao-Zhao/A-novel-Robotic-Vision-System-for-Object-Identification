"""CPU check that bounded decoder batches preserve native window aggregation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
run.setup()
import torch
import torch.nn.functional as F


class IndependentWindows:
    def __call__(self, mode, images, features_3d):
        assert mode == 'xyz'
        batch, count, channels, height, width = images.shape
        pixels = F.interpolate(images.reshape(batch * count, channels, height, width),
                               size=(224, 224), mode='bilinear', align_corners=False)
        pixels = pixels.reshape(batch, count, channels, 224, 224).mean(dim=2)
        return {'mask': pixels, 'mask_full': (pixels * 0.8 + 0.1)}


def main():
    detector = run.load_native_detector()
    torch.manual_seed(17)
    image = torch.rand(3, 89, 123)
    features = torch.zeros(1, 7, 3)
    model = IndependentWindows()
    results = []
    for batch_model in (model, run.BatchedDecoder(model)):
        results.append(detector.segment_objects(batch_model, features, image,
                                               patch_size=43, stride=14, device='cpu'))
    differences = [float((first - second).abs().max())
                   for first, second in zip(*results)]
    assert differences == [0.0] * 4, differences
    run.save(run.HERE / 'batching_check.json', {
        'native_detector_sha256': run.digest(run.SOURCE / 'examples' / 'run_detection_on_video.py'),
        'test': 'Unchanged native window/fold segmentation with full vs bounded decoder batch.',
        'image_shape_chw': list(image.shape), 'patch_size': 43, 'stride': 14,
        'max_abs_difference_amodal_mask_confidence_modal_mask_confidence': differences,
        'passed': True,
    })
    print('Native window aggregation and bounded decoder batches agree exactly.')


if __name__ == '__main__':
    main()
