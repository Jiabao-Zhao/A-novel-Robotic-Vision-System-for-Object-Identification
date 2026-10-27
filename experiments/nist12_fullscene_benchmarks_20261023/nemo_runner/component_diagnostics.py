"""Saved CAD-conditioned component boxes before native size pruning; no labels."""
from pathlib import Path
import sys
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, save, digest


def main():
    benchmark = HERE.parent
    manifest = read(benchmark / 'inputs.json')
    records = []
    for scene in manifest['scenes']:
        folder = benchmark / 'nemo' / scene['scene_id']
        detections = read(folder / 'detections.json')
        for item in manifest['objects']:
            filename = folder / f'raw_obj_{item["object_id"]:06d}.npz'
            with np.load(filename) as archive:
                amodal = archive['amodal_mask']
            number, labels, stats, _ = cv2.connectedComponentsWithStats(amodal.astype(np.uint8), connectivity=8)
            candidates = [row for row in detections if row['object_id'] == item['object_id']]
            retained_index = 0
            components = []
            for index in range(1, number):
                x, y, width, height, area = map(int, stats[index])
                box = [x, y, x + width, y + height]
                retained = area >= 1024
                proposal_id = None
                if retained:
                    original = candidates[retained_index]
                    assert original['bbox_xyxy'] == box
                    proposal_id = original['proposal_id']
                    retained_index += 1
                components.append({'component_id': index, 'bbox_xyxy': box,
                                   'amodal_area_pixels': area, 'retained_by_native_size_filter': retained,
                                   'retained_proposal_id': proposal_id})
            assert retained_index == len(candidates)
            records.append({'scene_id': scene['scene_id'], 'object_id': item['object_id'],
                            'target': item['target'], 'raw_prediction_sha256': digest(filename),
                            'components': components})
        per_scene = [record for record in records if record['scene_id'] == scene['scene_id']]
        save(folder / 'conditional_components.json', {
            'method': 'NeMO', 'scene_id': scene['scene_id'],
            'source_sha256': digest(__file__),
            'box_convention': 'Half-open pixel edges: x,y,x+w,y+h.',
            'native_minimum_area_pixels': 1024,
            'CAD_conditioned': True, 'class_agnostic_proposal_bank': False,
            'main_predictions_changed': False, 'ground_truth_read': False,
            'objects': {str(record['object_id']): {
                'target': record['target'],
                'raw_prediction_sha256': record['raw_prediction_sha256'],
                'pre1024_boxes_xyxy': [value['bbox_xyxy'] for value in record['components']],
                'pre1024_areas_pixels': [value['amodal_area_pixels'] for value in record['components']],
                'postfilter_boxes_xyxy': [value['bbox_xyxy'] for value in record['components']
                                          if value['retained_by_native_size_filter']],
                'retained_proposal_ids': [value['retained_proposal_id'] for value in record['components']
                                           if value['retained_by_native_size_filter']],
            } for record in per_scene},
        })
    output = {
        'source': 'Frozen nativeNeMO CAD-conditioned amodal prediction masks; no shared generic proposal bank.',
        'box_convention': 'Half-open pixel edges: x,y,x+w,y+h.',
        'threshold': 'Nativeper-scale0.5 mask union, already frozen in rawmaps.',
        'native_minimum_area_pixels': 1024,
        'main_predictions_changed': False, 'ground_truth_read': False, 'accuracy_computed': False,
        'records': records,
    }
    save(HERE / 'conditioned_component_diagnostics.json', output)
    print(f'Validated native size pruning for {len(records)} scene/CAD branches; no labels.')


if __name__ == '__main__':
    main()
