"""Freeze completed label-free NeMO predictions without evaluating correctness."""
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run
manifest = run.setup()


def main():
    verification = run.read(run.OUTPUT / 'verification.json')
    assert verification['passed'] and len(verification['scenes']) == 8
    source_audit = run.read(HERE / 'source_audit.json')
    for value, expected in source_audit['native_files'].items():
        assert run.digest(run.SOURCE / value) == expected
    raw_archive = run.read(HERE / 'raw_reference_archive.json')
    raw_root = run.OUTPUT.parent / 'nemo_raw_reference_diagnostic'
    for value, expected in raw_archive['archive_sha256'].items():
        assert run.digest(raw_root / value) == expected
    references = run.read(run.REFERENCES)
    for item in references['objects'].values():
        for value, expected in item['sha256'].items():
            assert run.digest(run.path(value)) == expected
    scenes = []
    for scene in manifest['scenes']:
        root = run.OUTPUT / scene['scene_id']
        completion = run.read(root / 'completion.json')
        assert completion['runner_sha256'] == run.digest(HERE / 'run.py')
        assert completion['references_sha256'] == run.digest(run.REFERENCES)
        detections = run.read(root / 'detections.json')
        counts = {item['target']: sum(detection['object_id'] == item['object_id']
                                       for detection in detections)
                  for item in manifest['objects']}
        scenes.append({'scene_id': scene['scene_id'], 'detections': len(detections),
                        'per_object_detection_counts': counts,
                        'scene_seconds': completion['seconds']})
    files = {str(filename.relative_to(run.OUTPUT)).replace('\\', '/'): run.digest(filename)
             for filename in sorted(run.OUTPUT.rglob('*')) if filename.is_file()}
    result = {
        'method': 'NeMO', 'status': 'completed',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'official_source_commit': run.COMMIT,
        'official_source_files_unchanged': True,
        'native_model_and_full_scene_detector_run': True,
        'native_score_box_mask_replay_exact': True,
        'reference_condition': '32object-centricCAD-alpha-squarecrops throughnativeNeMOloader',
        'custom_render_backend': 'OfficialCNOSPyrender sphere, same32viewselection fixedbeforeinference; notBOPPBRreproduction',
        'primary_references_sha256': run.digest(run.REFERENCES),
        'runner_sha256': run.digest(HERE / 'run.py'),
        'checkpoint_sha256': source_audit['checkpoint']['sha256'],
        'backbone_architecture_source': 'CachedofficialDINOv2 e1277af2ba9496fbadf7aec6eba56e8d882d1e35',
        'weight_updates': False, 'supplied_scene_regions': False,
        'ground_truth_read': False, 'evaluation_performed': False,
        'gpu_released': True, 'scenes': scenes,
        'total_detections': sum(value['detections'] for value in scenes),
        'native_scene_inference_seconds_total': sum(value['scene_seconds'] for value in scenes),
        'cad_encoding_seconds_total': sum(run.read(filename)['seconds']
                                           for filename in (run.OUTPUT / 'memories').glob('*.json')),
        'frozen_output_sha256': files,
        'preserved_raw_reference_diagnostic': 'nemo_raw_reference_diagnostic',
        'raw_reference_archive_original_hashes_verified': True,
        'pre_label_protocol_correction_sha256': run.digest(HERE / 'pre_label_protocol_correction.json'),
    }
    run.save(run.OUTPUT / 'completion.json', result)
    print({'scenes': len(scenes), 'total_detections': result['total_detections'],
            'per_scene': [value['detections'] for value in scenes],
            'scene_seconds_total': result['native_scene_inference_seconds_total']}, flush=True)


if __name__ == '__main__':
    main()
