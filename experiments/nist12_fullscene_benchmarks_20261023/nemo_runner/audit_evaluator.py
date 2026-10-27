"""Independent label-free audit of target selection and NeMO export conventions."""
from pathlib import Path
import ast
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, save, digest, path
import evaluate


def main():
    manifest = read(HERE.parent / 'inputs.json')
    records = []
    for scene in manifest['scenes']:
        filename = HERE.parent / 'nemo' / scene['scene_id'] / 'detections.json'
        rows = read(filename)
        for item in manifest['objects']:
            candidates = [row for row in rows if row['object_id'] == item['object_id']]
            chosen, status = evaluate.select(rows, item['object_id'])
            if not candidates:
                assert chosen is None and status == 'no_retained_detection'
            else:
                values = np.array([row['score'] for row in candidates])
                maximum = float(values.max())
                tied = np.count_nonzero(np.abs(values - maximum) <= 1e-12) > 1
                if tied:
                    assert chosen is None and status == 'tie'
                else:
                    expected = candidates[int(values.argmax())]
                    assert status == 'ok' and chosen == expected
                    assert chosen['object_id'] == item['object_id']
                    assert evaluate.box_edges(chosen['bbox_xyxy'], 'nemo') == chosen['bbox_xyxy']
            records.append({'scene_id': scene['scene_id'], 'target': item['target'],
                            'retained_target_detections': len(candidates), 'status': status,
                            'selected_proposal_id': None if chosen is None else chosen['proposal_id']})
    # Importing evaluate must define functions only, without running its label-opening main.
    tree = ast.parse((HERE.parent / 'evaluate.py').read_text(encoding='utf-8'))
    main_calls = [node for node in tree.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == 'main']
    assert not main_calls
    save(HERE / 'evaluator_label_free_audit.json', {
        'evaluation_source_sha256': digest(HERE.parent / 'evaluate.py'),
        'primary_references_sha256': digest(HERE / 'primary_references.json'),
        'native_output_verification_sha256': digest(HERE.parent / 'nemo/verification.json'),
        'selection_tasks': len(records), 'selection_matches_independent_numpy_argmax': True,
        'NeMO_box_convention': 'OpenCV x,y,w,h -> x,y,x+w,y+h, half-open; no+1 correction.',
        'NeMO_mask_convention': 'Native modal binary mask within own amodal connected-component box, in originalRGBframe.',
        'NeMO_score_convention': 'Mean averaged amodalconfidence over connected component, not modalIoU or objectcorrectness.',
        'minimum_area_pixels': 1024, 'pixel_threshold': 0.5,
        'NeMO_stage_attribution': 'CAD-conditioned dense segmentation; no shared class-agnostic proposal bank. Missing output cannot by itself isolate recognition vs segmentation failure.',
        'target_training': False, 'ground_truth_opened': False, 'accuracy_computed': False,
        'records': records,
    })
    print(f'Independent target-identity maximum selection passed for {len(records)} tasks; no labels opened.')


if __name__ == '__main__':
    main()
