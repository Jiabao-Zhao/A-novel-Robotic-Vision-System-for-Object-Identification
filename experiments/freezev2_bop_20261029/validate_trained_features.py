"""Actual trained-model smoke test; no GT pose access or benchmark claim."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from freezev2_repro.features import FrozenDINOv2, FrozenGeDi, configure_frozen_fp32


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main(experiment_output, dino_source, output):
    root, output = Path(experiment_output), Path(output)
    configure_frozen_fp32(torch)
    if output.exists():
        raise RuntimeError('Preserve the existing smoke audit; choose a versioned filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {'status': 'running', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'python': sys.executable, 'python_version': platform.python_version(),
             'torch': torch.__version__, 'cuda_version': torch.version.cuda,
             'device': torch.cuda.get_device_name(0), 'stages': {},
             'GT_pose_access': False, 'claim': 'Actual trained feature/runtime validity only; no pose or benchmark accuracy claim.',
             'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
             'tf32_cudnn': torch.backends.cudnn.allow_tf32}

    def save():
        output.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')

    save()
    start = time.perf_counter()
    try:
        external = root / 'external'
        geometry_dir = root / 'base_v1/onboarding/tless/obj_000001'
        data = np.load(geometry_dir / 'geometry.npz', allow_pickle=False)
        query, support = data['points_m'][:64], data['support_points_m']
        models_info = Path('/mnt/d/BOP_SAM3_Pilot/data/tless/models_cad/models_info.json')
        diameter = json.loads(models_info.read_text())['1']['diameter'] * 1e-3
        torch.cuda.reset_peak_memory_stats()
        gedi = FrozenGeDi(
            external / 'gedi', external / 'gedi/data/chkpts/3dmatch/chkpt.tar',
            expected_source_sha256='50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249',
            expected_revision='b3dd86776750d8221f89d39975118da9839b39f7',
            checkpoint_sha256='15723f382ecf60f7efbf9161f70bffda2d1c817025cb0702350baaa04060f5f4',
            samples_per_batch=500, forward_microbatch_size=32, seed=0,
        )
        value32 = gedi.compute(query, support, diameter)
        repeated32 = gedi.compute(query, support, diameter)
        gedi.set_forward_microbatch_size(None)
        reference = gedi.compute(query, support, diameter)
        gedi.set_forward_microbatch_size(16)
        value16 = gedi.compute(query, support, diameter)
        difference32 = float(np.max(np.abs(reference - value32)))
        difference16 = float(np.max(np.abs(reference - value16)))
        close32 = bool(np.allclose(reference, value32, rtol=1e-5, atol=1e-6))
        close16 = bool(np.allclose(reference, value16, rtol=1e-5, atol=1e-6))
        import pointnet2_ops._ext as extension
        audit['stages']['gedi'] = {
            'provenance': gedi.provenance, 'query_count': len(query), 'support_count': len(support),
            'diameter_m': diameter, 'shape': list(value32.shape), 'finite': bool(np.isfinite(value32).all()),
            'dtype': str(value32.dtype), 'sampling_batch_unchanged': 500,
            'comparison': {'reference_network_batch': len(query), 'microbatches': [32, 16],
                           'max_abs_difference_32': difference32, 'max_abs_difference_16': difference16,
                           'max_abs_repeat_difference_32': float(np.max(np.abs(repeated32 - value32))),
                           'allclose_32': close32, 'allclose_16': close16, 'rtol': 1e-5, 'atol': 1e-6},
            'extension': str(extension.__file__), 'extension_sha256': sha(extension.__file__),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }
        save()
        print(json.dumps({'stage': 'gedi', 'finite': bool(np.isfinite(value32).all()),
                          'max_diff32': difference32, 'max_diff16': difference16}), flush=True)
        if not close32 or not close16:
            raise RuntimeError('Trained GeDi forward microbatch equality gate failed')
        gedi.set_forward_microbatch_size(32)
        torch.cuda.empty_cache()
        dino = FrozenDINOv2(
            dino_source, external / 'dinov2/dinov2_vitg14_pretrain.pth', layer_index=18,
            device='cuda', checkpoint_sha256='baf8467e50af277596bbbafa06887c177ee899ab46033649c383577d7e9309d3',
            source_revision_hint='e1277af2ba9496fbadf7aec6eba56e8d882d1e35',
            expected_source_hashes={
                'hubconf.py': 'e82275b3adcc3631865d1edd4b620b89d42459169acdeb652804cd0eb0fe9615',
                'dinov2/models/vision_transformer.py': '8dc1e2ededd4893a335aa2f524c15554bcdb14d824e310ca3df825e16081a268',
            },
        )
        torch.cuda.reset_peak_memory_stats()
        dino_maps, dino_records = [], []
        for view in (0, 20):
            image_path = geometry_dir / 'views' / f'{view:03d}.png'
            image = np.asarray(Image.open(image_path).convert('RGB'))
            tick = time.perf_counter()
            features = dino.extract(image)
            torch.cuda.synchronize()
            dino_maps.append(features)
            dino_records.append({'view': view, 'image_sha256': sha(image_path),
                                 'shape': list(features.shape), 'dtype': str(features.dtype),
                                 'finite': bool(np.isfinite(features).all()), 'runtime_s': time.perf_counter() - tick})
        audit['stages']['dinov2'] = {
            'provenance': dino.provenance, 'views': dino_records,
            'all_parameters_frozen': not any(p.requires_grad for p in dino.model.parameters()),
            'training': dino.model.training,
            'floating_parameter_dtypes': sorted({str(p.dtype) for p in dino.model.parameters() if p.is_floating_point()}),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }
        artifact = output.with_suffix('.npz')
        np.savez_compressed(artifact, gedi_micro32=value32, gedi_reference=reference,
                            gedi_micro16=value16, dino_maps=np.array(dino_maps),
                            CAD_query_points_m=query)
        audit.update(status='passed', runtime_s=time.perf_counter() - start,
                     descriptor_artifact=str(artifact), descriptor_artifact_sha256=sha(artifact))
    except Exception as exc:
        audit.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc(), runtime_s=time.perf_counter() - start)
        save()
        print(json.dumps({'status': 'failed', 'error': str(exc), 'audit': str(output)}), flush=True)
        raise
    save()
    print(json.dumps({'status': audit['status'], 'audit': str(output), 'runtime_s': audit['runtime_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-output', required=True)
    parser.add_argument('--dino-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.experiment_output, args.dino_source, args.output)
