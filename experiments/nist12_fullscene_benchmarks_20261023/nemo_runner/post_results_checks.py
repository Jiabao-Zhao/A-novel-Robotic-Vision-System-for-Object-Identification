"""Post-lock artifact integrity and explicitly conditional failure diagnostics."""
from collections import Counter
from pathlib import Path
import sys
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read, save, digest


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def main():
    benchmark = HERE.parent
    analysis = benchmark / 'analysis'
    assert read(analysis / 'independent_results_verification.json')['passed']
    rows = read(analysis / 'per_task.json')
    checked_images = 0
    for row in rows:
        if row['correct']:
            assert row['case_image'] is None
            continue
        assert row['case_image']
        filename = analysis / row['case_image']
        assert filename.is_file() and filename.stat().st_size > 0
        with Image.open(filename) as image:
            image.verify()
        checked_images += 1
    methods = list(dict.fromkeys(row['method'] for row in rows))
    diagnostics = {}
    for method in methods:
        selected = [row for row in rows if row['method'] == method]
        data = {'all_requests': len(selected), 'correct': sum(row['correct'] for row in selected),
                'status_counts': dict(Counter(row['status'] for row in selected))}
        if method == 'nids_basic':
            data['classification_comparison_available'] = False
            diagnostics[method] = data
            continue
        raw = [row for row in selected if row['raw_proposal_bbox_recall'] is True]
        retained = [row for row in selected if row['after_native_size_filter_bbox_recall'] is True]
        if any(row['raw_proposal_bbox_recall'] is not None for row in selected):
            data['raw_target_box_recall_count'] = len(raw)
        if any(row['after_native_size_filter_bbox_recall'] is not None for row in selected):
            correct = sum(row['correct'] for row in retained)
            assert data['correct'] == correct
            data.update(target_box_survives_native_filter_count=len(retained),
                        correct_among_surviving_target_boxes=correct,
                        conditional_correct_given_target_box_survives=ratio(correct, len(retained)))
        if method == 'ours':
            available = [row for row in selected if row['status'] == 'ok']
            data['conditional_correct_given_available_depth_decision'] = ratio(
                sum(row['correct'] for row in available), len(available))
        if method == 'nemo':
            pre = [row for row in selected if row['conditional_mask_bbox_recall_before_area_filter'] is True]
            post = [row for row in selected if row['conditional_mask_bbox_recall_after_area_filter'] is True]
            assert data['correct'] == sum(row['correct'] for row in post)
            data.update(CAD_conditioned_target_box_before_area_filter_count=len(pre),
                        CAD_conditioned_target_box_after_area_filter_count=len(post),
                        conditional_correct_given_requested_CAD_mask_contains_target_box=ratio(data['correct'], len(post)),
                        conditional_area_filter_lost_target_tasks=[{'scene_id': row['scene_id'], 'target': row['target']}
                            for row in selected if row['conditional_mask_bbox_recall_before_area_filter'] is True
                            and row['conditional_mask_bbox_recall_after_area_filter'] is False])
        diagnostics[method] = data
    save(analysis / 'independent_secondary_checks.json', {
        'case_images_verified': checked_images, 'case_image_mapping_complete': True,
        'all_primary_metrics_unchanged': True,
        'interpretation': 'Conditional diagnostics use audit labels after prediction locking; they are not deployment performance or common-proposal experiments.',
        'input_per_task_sha256': digest(analysis / 'per_task.json'),
        'diagnostics': diagnostics,
    })
    print({'case_images_verified': checked_images, 'diagnostics': diagnostics})


if __name__ == '__main__':
    main()
