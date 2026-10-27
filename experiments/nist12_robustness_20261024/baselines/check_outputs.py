"""Check serialized baseline outputs without reading simulator audit data."""
import numpy as np
from PIL import Image
import common

METHODS = ['cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_basic_finite_templates', 'noctis', 'nemo']


def main():
    inputs = common.setup()
    methods = {}
    for method in METHODS:
        scenes = []
        for scene in inputs['scenes']:
            directory = common.HERE / method / scene['scene_id']
            done = common.read(directory / 'completion.json')
            rows = common.read(directory / 'detections.json')
            with np.load(directory / 'masks.npz') as bank:
                masks = bank['masks']
            with Image.open(common.path(scene['rgb'])) as image:
                shape = (image.height, image.width)
            assert masks.shape == (len(rows), *shape)
            assert masks.dtype == np.bool_
            for index, row in enumerate(rows):
                assert row['mask_index'] == index and 1 <= row['object_id'] <= 12
                assert np.isfinite(row['score'])
                assert len(row['bbox_xyxy']) == 4 and np.isfinite(row['bbox_xyxy']).all()
            for filename in ('detections.json', 'masks.npz'):
                key = filename.replace('.', '_') + '_sha256'
                # Native receipts name detections_sha256 / masks_sha256.
                key = filename.split('.')[0] + '_sha256'
                if key in done:
                    assert common.digest(directory / filename) == done[key]
            scenes.append(dict(scene_id=scene['scene_id'], retained_detections=len(rows)))
        methods[method] = dict(scenes=len(scenes), retained_detections=sum(x['retained_detections'] for x in scenes),
                               per_scene=scenes)
    common.save(common.HERE / 'output_checks.json', dict(passed=True, labels_read=False,
                input_sha256=common.digest(common.HERE / 'inputs.json'), methods=methods))
    print('All six methods have valid finite serialized outputs for all twenty scenes.', flush=True)


if __name__ == '__main__':
    main()
