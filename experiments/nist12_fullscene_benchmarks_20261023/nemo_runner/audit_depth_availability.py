"""Post-lock explanation of the frozen all-shortlisted-fits availability rule."""
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, save, digest
from independent_results_audit import edges, intersection_over_union


def main():
    benchmark = HERE.parent
    rows = read(benchmark / 'analysis/per_task.json')
    assert read(benchmark / 'analysis/independent_results_verification.json')['passed']
    records = []
    for row in rows:
        if row['method'] != 'ours' or row['status'] != 'unavailable':
            continue
        scene, target = row['scene_id'], row['target']
        proposals = read(benchmark / 'ours' / scene / 'localization.json')['proposals']
        decision = next(value for value in read(benchmark / 'ours' / scene / 'decisions.json')
                        if value['target'] == target)
        truth = read(benchmark.parents[1] / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures/scenes' /
                     scene / 'capture_audit.json')['audit_only_objects'][target]['bbox']
        candidates = []
        for proposal_id in decision['shortlist_ids']:
            filename = benchmark / 'ours/fits' / f'{scene}__{target}__p{proposal_id:03d}.json'
            fit = read(filename)['geometry']
            arm = fit['arms'][0]
            best = arm.get('best_geometry')
            available = best is not None
            iou = intersection_over_union(edges(proposals[proposal_id]['bbox_xyxy'], 'ours'),
                                          edges(truth, 'audit'))
            candidates.append({'proposal_id': proposal_id, 'target_bbox_IoU': iou,
                               'target_box_match': iou >= .5, 'fit_available': available,
                               'score': None if not available else best['verification']['geometry_score'],
                               'projection_status_counts': dict(Counter(
                                   value['verification']['status'] for value in arm['pose_bank'])),
                               'fit_sha256': digest(filename)})
        records.append({'scene_id': scene, 'target': target,
                        'available_candidate_count': sum(value['fit_available'] for value in candidates),
                        'target_candidate_fit_available': any(value['target_box_match'] and value['fit_available']
                                                             for value in candidates),
                        'candidates': candidates})
    result = {
        'primary_decisions_changed': False,
        'diagnostic_only_after_prediction_lock': True,
        'frozen_rule': 'Emit decision only if every shortlisted CAD/crop fit has an available score.',
        'unavailable_tasks': len(records),
        'tasks_with_at_least_one_available_candidate': sum(value['available_candidate_count'] > 0 for value in records),
        'tasks_with_available_correct_target_fit': sum(value['target_candidate_fit_available'] for value in records),
        'records': records,
    }
    save(benchmark / 'analysis/depth_availability_policy_diagnostic.json', result)
    print({key: value for key, value in result.items() if key != 'records'})


if __name__ == '__main__':
    main()
