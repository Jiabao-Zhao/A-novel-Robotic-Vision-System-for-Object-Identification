"""Verify label-free NeMO outputs against their saved native confidence maps."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run
manifest = run.setup()
import numpy as np
import torch
from PIL import Image
from nemolib.utils import masks_to_bounding_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=None)
    args = parser.parse_args()
    records = []
    for scene in manifest['scenes']:
        if args.scene and args.scene != scene['scene_id']:
            continue
        destination = run.OUTPUT / scene['scene_id']
        completion = run.read(destination / 'completion.json')
        detections = run.read(destination / 'detections.json')
        masks = np.load(destination / 'masks.npz')['masks']
        size = np.asarray(Image.open(run.path(scene['rgb']))).shape[:2]
        assert masks.dtype == bool and masks.shape == (len(detections), *size)
        assert [value['proposal_id'] for value in detections] == list(range(len(detections)))
        assert [value['mask_index'] for value in detections] == list(range(len(detections)))
        assert len(detections) == completion['detections']
        assert run.digest(run.path(scene['rgb'])) == completion['rgb_sha256']
        assert run.digest(run.REFERENCES) == completion['references_sha256']
        replayed = 0
        for item in manifest['objects']:
            object_id = item['object_id']
            raw = np.load(destination / f'raw_obj_{object_id:06d}.npz')
            assert set(raw.files) == {'amodal_mask', 'modal_mask', 'amodal_confidence', 'modal_confidence'}
            amodal = torch.from_numpy(raw['amodal_mask'][None, None])
            confidence = torch.from_numpy(raw['amodal_confidence'][None, None])
            boxes, scores, _ = masks_to_bounding_boxes(amodal, confidence,
                                                       minimum_area=run.MINIMUM_AREA)
            actual = [value for value in detections if value['object_id'] == object_id]
            assert len(actual) == len(boxes[0])
            for detection, box, score in zip(actual, boxes[0], scores[0]):
                assert detection['bbox_xyxy'] == list(map(int, box))
                assert detection['score'] == score and np.isfinite(score)
                expected = np.zeros(size, dtype=bool)
                x0, y0, x1, y1 = box
                expected[y0:y1, x0:x1] = raw['modal_mask'][y0:y1, x0:x1]
                assert np.array_equal(masks[detection['mask_index']], expected)
                replayed += 1
        records.append({'scene_id': scene['scene_id'], 'native_output_replay_exact': True,
                        'detections': replayed, 'mask_shape': list(masks.shape),
                        'detections_sha256': run.digest(destination / 'detections.json'),
                        'masks_sha256': run.digest(destination / 'masks.npz'),
                        'completion_sha256': run.digest(destination / 'completion.json')})
    output = run.OUTPUT / ('verification_' + args.scene + '.json' if args.scene else 'verification.json')
    run.save(output, {'scenes': records, 'passed': True, 'labels_read': False,
                      'evaluation_performed': False})
    print(f'Exact native score/box/mask replay passed for {len(records)} scenes.', flush=True)


if __name__ == '__main__':
    main()
