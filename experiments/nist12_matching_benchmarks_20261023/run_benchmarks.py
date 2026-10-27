"""Controlled CAD-to-crop matching modules; no observed identities enter inference."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE.parent / 'nist12_qwen_clip_reranking_20261023'
DEPTH = BASE / 'depth_verification_20261023'
HUB = REPO / 'outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35'
WEIGHT = REPO / 'outputs/cache/torch_hub/checkpoints/dinov2_vitl14_pretrain.pth'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
METHODS = ('cnos', 'sam6d_rgb', 'sam6d_rgbd', 'nids_noadapter', 'nids_ffa_only')
TIE_ATOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(value):
    value = str(value).replace('\\', '/')
    if len(value) > 1 and value[1] == ':' and sys.platform != 'win32':
        value = '/mnt/' + value[0].lower() + value[2:]
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path):
    path = local(path)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        value = path.as_posix()
        return value[5].upper() + ':' + value[6:] if value.startswith('/mnt/') else value


def save(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Frozen file differs: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def render_rotation(view):
    """CAD frame -> OpenCV camera frame, including the render's image roll."""
    backward = np.asarray(view['camera_direction_cad'], dtype=float)
    backward /= np.linalg.norm(backward)
    hint = np.array([0., 1., 0.]) if abs(backward[2]) > .99 else np.array([0., 0., 1.])
    right = np.cross(hint, backward)
    right /= np.linalg.norm(right)
    gl_camera_in_cad = np.column_stack([right, np.cross(backward, right), backward])
    result = np.diag([1., -1., -1.]) @ gl_camera_in_cad.T
    np.testing.assert_allclose(result @ result.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(result) - 1) < 1e-12
    return result


def sample_surface(vertices, faces, count=2048):
    """Deterministic area-uniform sampling at native CAD scale in meters."""
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    rng = np.random.default_rng(0)
    faces = rng.choice(len(triangles), size=count, p=area/area.sum())
    selected = triangles[faces]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return selected[:, 0] + uv[:, :1]*(selected[:, 1]-selected[:, 0]) + uv[:, 1:]*(selected[:, 2]-selected[:, 0])


def check_sources():
    protocol = read(HERE / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert digest(local(name)) == expected, name
    return protocol


def prepare():
    import torch
    import open3d as o3d
    from PIL import Image
    old = module('frozen_dino_preparation', BASE / 'run_dino.py')
    records = read(BASE / 'data/input_records.json')
    capture = read(BASE / 'data/capture_manifest.json')
    catalog = read(BASE / 'data/cad_inputs.json')
    images, masks, indices, statuses, consumed = old.prepare_images(records, capture, catalog)
    assert len(records) == len(indices) == 336 and indices == list(range(336))
    previous = np.load(BASE / 'dino/inputs_and_masks.npz')
    np.testing.assert_array_equal(images, previous['images'])
    np.testing.assert_array_equal(masks, previous['masks'])
    np.testing.assert_array_equal(indices, previous['encoded_record_indices'])
    np.savez_compressed(HERE / 'preprocessed.npz', images=images, masks=masks,
                        encoded_record_indices=np.asarray(indices))
    save(HERE / 'input_records.json', records)
    save(HERE / 'preparation_statuses.json', statuses)
    majority = (masks.reshape(336,16,14,16,14).mean(axis=(2,4)) > .5).sum(axis=(1,2))
    soft = torch.nn.functional.interpolate(torch.as_tensor(masks[:,None], dtype=torch.float32),
                                            size=(16,16), mode='bilinear', align_corners=False)
    soft_sums = soft.sum(dim=(1,2,3)).numpy()
    template_pool = {target:[i for i,r in enumerate(records) if r['kind']=='cad' and
                    r['target']==target and majority[i]>0 and soft_sums[i]>0] for target in catalog}
    assert all(len(pool)>=5 for pool in template_pool.values())
    save(HERE/'template_pool.json', dict(eligible_record_indices=template_pool,
        CAD_view_statuses=[dict(record_index=i,target=r['target'],view_index=r['view_index'],
            majority_patch_count=int(majority[i]),bilinear_mask_sum=float(soft_sums[i]),
            status='available' if majority[i]>0 and soft_sums[i]>0 else 'unavailable_empty_foreground_at_patch_resolution')
            for i,r in enumerate(records) if r['kind']=='cad'],
        shared_across_all_new_methods=True, decided_before_encoding=True))
    cads, pose_audit, samples = {}, [], {}
    for target, item in catalog.items():
        path = local(item['mesh_file'])
        assert digest(path) == item['mesh_sha256']
        consumed.add(path)
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        np.testing.assert_allclose(np.ptp(vertices, axis=0), item['native_extent_m'], atol=1e-8)
        points = sample_surface(vertices, np.asarray(mesh.triangles))
        samples[target] = points.astype(np.float32)
        rotations = {}
        for view in item['views']:
            R = render_rotation(view)
            rotations[str(view['view_index'])] = R.tolist()
            camera_position = np.asarray(view['camera_position_cad_m'])
            camera_points = (vertices-camera_position) @ R.T
            assert np.all(camera_points[:, 2] > 0)
            focal = 448 / (2*np.tan(np.deg2rad(30)/2))
            uv = camera_points[:, :2]/camera_points[:, 2:]*focal + 224
            mask = np.asarray(Image.open(local(view['mask_file']))) > 0
            yy, xx = np.nonzero(mask)
            projected = [float(uv[:, 0].min()), float(uv[:, 1].min()),
                         float(uv[:, 0].max()), float(uv[:, 1].max())]
            observed = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
            error = float(np.max(np.abs(np.asarray(projected)-observed)))
            assert error < 2.5, (target, view['view_index'], error)
            pose_audit.append(dict(target=target, view_index=view['view_index'],
                projected_bbox=projected, rendered_mask_bbox=observed,
                max_bbox_difference_px=error, rotation_cad_to_camera=R.tolist()))
        cads[target] = dict(mesh_file=relative(path), mesh_sha256=digest(path),
            native_extent_m=np.ptp(vertices, axis=0).tolist(),
            native_bbox_center_m=((vertices.min(0)+vertices.max(0))/2).tolist(),
            sample_count=2048, sample_seed=0, rotations=rotations)
    np.savez_compressed(HERE / 'cad_surface_samples.npz', **samples)
    save(HERE / 'cad_rotation_audit.json', pose_audit)
    states = {state: {key: row[key] for key in ('depth_file', 'intrinsics_file',
                     'plane_model_camera', 'plane_distance_threshold_m', 'crops')}
              for state, row in capture['states'].items()}
    save(HERE / 'inputs.json', dict(states=states, cads=cads))
    paths = list(consumed) + [BASE / 'data/input_records.json', BASE / 'data/capture_manifest.json',
        BASE / 'data/cad_inputs.json', BASE / 'run_dino.py', BASE / 'dino/inputs_and_masks.npz',
        REPO / 'scripts/cad_descriptors.py', REPO / 'scripts/sam6d_patch_matching.py',
        BASE / 'build_inputs.py', WEIGHT, HUB / 'hubconf.py',
        REPO / 'docs/cad_view_prompt.txt', DEPTH / 'selections.json',
        DEPTH / 'protocol.json', BASE / 'clip/shortlist.json']
    for source in ('sam6d_matching_source/model/detector.py', 'sam6d_matching_source/model/loss.py',
                   'sam6d_matching_source/model/dinov2.py'):
        paths.append(REPO / 'outputs/cache' / source)
    paths += [HERE / n for n in ('run_benchmarks.py', 'nids_matching.py', 'test_nids_matching.py',
        'nids_source_audit.json', 'cnos_source_audit.json', 'sam6d_geometry.py', 'test_sam6d_geometry.py', 'sam6d_source_audit.json',
        'preprocessed.npz', 'input_records.json', 'preparation_statuses.json', 'inputs.json',
        'cad_surface_samples.npz', 'cad_rotation_audit.json', 'template_pool.json')]
    paths += list(HUB.rglob('*.py'))
    protected = {relative(path): digest(path) for path in sorted(set(paths))}
    assert digest(REPO / 'docs/cad_view_prompt.txt') == '180dc9df66ab2cbd0b9b8057a054314bc1252d74b47db9d5dc98a15c391d2019'
    save(HERE / 'protocol.json', dict(task='One supplied CAD -> one of twelve unchanged scene crops',
        states=sorted(states), targets=list(cads), tasks=96, candidates_per_task=12, candidate_pairs=1152,
        input_images=336, intended_CAD_views_per_target=20,
        eligible_CAD_views={t:len(p) for t,p in template_pool.items()},
        reference_availability='Same CAD-only valid reference pool for all four methods: at least one strict-majority patch and positive bilinear mask sum. Empty thin reference views remain explicit unavailable and are excluded before image encoding, without using scene scores or labels.',
        primary_methods=list(METHODS[:4]),
        auxiliary_methods=['nids_ffa_only'], backbone='Frozen cached DINOv2 ViT-L/14, non-register, FP32, batch 2',
        matched_shortlist_controls=['clip5_'+m for m in METHODS],
        preprocessing='Exact saved 224x224 tight foreground crop, white background, aspect-preserving letterbox; native CAD masks and observed >2mm plane-depth masks. Verified byte-equal to previous study.',
        adaptation='Matching modules on common inputs, not full detector/segmentation reproductions. Common20-view bank and white224 input differ from released preprocessing/default42 templates. No SAM proposals, class-assignment filtering, confidence gate, NMS, target fitting, training, or model downloads.',
        cnos='Mean top-five DINOv2 CLS cosine similarities clamped[0,1] over the shared available reference pool.',
        sam6d_rgb='Equal mean semantic top5 CLS score and foreground patch maximum similarity at best CLS template; strict patch occupancy>0.5 and explicit reference availability.',
        sam6d_rgbd='(semantic+appearance+r*bbox_IoU)/(2+r); r from reverse patch similarity, not physical visibility. Fixed best CLS template CAD-to-camera rotation; mean observed-depth XYZ translation; native2048 sampledCADpoints; integer projected UV clamped to image; continuous bbox widths without+1. No ICP, table fitting, or geometry residual.',
        nids='Official BOP no-adapter FFA max-view score plus best-FFA-view appearance as audited in nids_source_audit.json; also report FFA-only auxiliary to distinguish paper variants. Raw patch tokens retained before feature averaging.',
        geometry_units='Native meters; OpenCV camera x right/y down/z forward, observed depth camera Z; pixel offset0.',
        selection='Independently rank all12 for each target, no one-to-one assignment. Unique max within1e-12; unavailable candidates or top ties stay explicit; no fallback.',
        comparator='Exactly saved CLIP top5 -> contact_0mm_geometry (94/96); clearance_1mm_geometry separate simulator sensitivity. Candidate budget12 vs5 explicitly reported.',
        evaluation='Freeze new scores and rankings before separate analysis reads audit identities. Scenes previously inspected development data; not blind/heldout.',
        protected_prompt_unchanged=True, simulator_pose_input=False, source_sha256=protected,
        software=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__)))
    check_sources()
    print(json.dumps(dict(prepared=True, protected_files=len(protected),
        protocol_sha256=digest(HERE/'protocol.json'), rotation_audits=len(pose_audit))), flush=True)


def encode():
    import torch
    check_sources()
    assert not (HERE / 'encoding_completion.json').exists()
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    model = torch.hub.load(str(HUB), 'dinov2_vitl14', source='local', pretrained=False)
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    model = model.eval().requires_grad_(False).to('cuda')
    assert model.embed_dim == 1024 and model.patch_size == 14
    inputs = np.load(HERE / 'preprocessed.npz')
    images = inputs['images']
    cls, patches = [], []
    start_time = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(images), 2):
            array = images[start:start+2].astype(np.float32)/255
            array = (array-np.asarray([.485,.456,.406],dtype=np.float32))/np.asarray([.229,.224,.225],dtype=np.float32)
            batch = torch.from_numpy(array.transpose(0,3,1,2)).to('cuda')
            raw = model.forward_features(batch)
            cls.extend(raw['x_norm_clstoken'].cpu().numpy())
            patches.extend(raw['x_norm_patchtokens'].cpu().numpy())
            if start % 32 == 0 or start+2 >= len(images):
                print(f'ViT-L: {min(start+2,len(images))}/{len(images)}', flush=True)
    cls, patches = np.asarray(cls), np.asarray(patches)
    assert cls.shape == (336,1024) and patches.shape == (336,256,1024)
    assert np.isfinite(cls).all() and np.isfinite(patches).all()
    np.savez_compressed(HERE / 'features.npz', cls=cls, patches=patches,
                        encoded_record_indices=inputs['encoded_record_indices'])
    save(HERE / 'encoding_completion.json', dict(images=336, model='dinov2_vitl14',
        raw_patch_tokens_preserved=True, no_patch_L2_before_FFA=True,
        feature_file_sha256=digest(HERE / 'features.npz'), seconds=perf_counter()-start_time,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        checkpoint_sha256=digest(WEIGHT), protocol_sha256=digest(HERE/'protocol.json'),
        evaluation_labels_read=False))
    check_sources()


def rank(rows, method, state, target):
    available = [r for r in rows if r['status']=='available']
    base = dict(method=method, state=state, target=target,
                candidate_ids=[r['proposal_id'] for r in rows], selected_ids=[],
                status='unavailable_incomplete_candidates', candidate_scores=rows)
    if len(available) != len(rows):
        return base
    ordered = sorted(rows, key=lambda r:(-r['score'],r['proposal_id']))
    best = ordered[0]['score']
    tied = [r['proposal_id'] for r in ordered if abs(r['score']-best)<=TIE_ATOL]
    return dict(base, status='selected' if len(tied)==1 else 'tied', selected_ids=tied,
                ranked_ids=[r['proposal_id'] for r in ordered], top_score=best,
                gap=best-ordered[1]['score'])


def score():
    import torch
    from scripts import sam6d_patch_matching as visual
    from scripts import cad_descriptors
    import nids_matching as nids
    from sam6d_geometry import compute_geometry
    old = module('frozen_foreground_helper', BASE/'run_dino.py')
    protocol = check_sources()
    assert digest(HERE/'features.npz') == read(HERE/'encoding_completion.json')['feature_file_sha256']
    torch.set_num_threads(1)
    features = np.load(HERE/'features.npz')
    inputs = np.load(HERE/'preprocessed.npz')
    records = read(HERE/'input_records.json')
    state_inputs = read(HERE/'inputs.json')
    points = np.load(HERE/'cad_surface_samples.npz')
    np.testing.assert_array_equal(features['encoded_record_indices'],np.arange(336))
    patches = [visual.masked_patch_descriptors(p,m) for p,m in
        zip(features['patches'],[cad_descriptors.patch_occupancy(m) for m in inputs['masks']])]
    embeddings = [nids.nids_embedding(p,m) for p,m in zip(features['patches'],inputs['masks'])]
    save(HERE/'descriptor_statuses.json', [dict(record_index=i,
        sam6d_foreground_patches=int(torch.count_nonzero(p.sum(-1))),
        nids_status=e['status'], nids_metadata=e.get('metadata'))
        for i,(p,e) in enumerate(zip(patches,embeddings))])
    refs = read(HERE/'template_pool.json')['eligible_record_indices']
    observed = {(r['state'],r['proposal_id']):i for i,r in enumerate(records) if r['kind']=='crop'}
    scores, selections = [], []
    shortlists={(r['state'],r['target']):r['candidate_ids'] for r in read(BASE/'clip/shortlist.json')
                if r['method']=='clip_full'}
    started = perf_counter()
    for state in protocol['states']:
        scene = state_inputs['states'][state]
        depth = np.load(local(scene['depth_file']))
        K = np.load(local(scene['intrinsics_file']))
        masks = {}
        for crop in scene['crops']:
            roi=crop['roi']; x1,y1,x2,y2=[roi[k] for k in ('x1','y1','x2','y2')]
            full=np.zeros(depth.shape,dtype=bool)
            full[y1:y2+1,x1:x2+1]=old.foreground(depth,K,roi,scene['plane_model_camera'],scene['plane_distance_threshold_m'])
            masks[crop['proposal_id']]=full
        for target in protocol['targets']:
            reference=refs[target]
            per_method={m:[] for m in METHODS}
            for (s,pid),index in observed.items():
                if s!=state:
                    continue
                sem=visual.semantic_score(features['cls'][index],features['cls'][reference])
                best=reference[sem['best_view_index_1based']-1]
                app=visual.appearance_score(patches[index],patches[best])
                R=state_inputs['cads'][target]['rotations'][str(records[best]['view_index'])]
                geo=compute_geometry(points[target],np.asarray(R),masks[pid],depth,K)
                nm=nids.nids_score(embeddings[index],[embeddings[j] for j in reference])
                # NIDS source-specific best-view appearance is implemented in its isolated module.
                na=None
                if nm['status']=='available':
                    nb=reference[nm['best_view_index']]
                    na=nids.nids_appearance(features['patches'][index],inputs['masks'][index],
                                            features['patches'][nb],inputs['masks'][nb])
                values={'cnos':sem['semantic_score'],
                    'sam6d_rgb':visual.combined_visual_score(sem['semantic_score'],app['appearance_score']),
                    'sam6d_rgbd':(sem['semantic_score']+app['appearance_score']+app['visible_ratio_diagnostic']*geo['geometry_bbox_iou'])/(2+app['visible_ratio_diagnostic']) if geo['status']=='available' else None,
                    'nids_ffa_only':nm['score'] if nm['status']=='available' else None,
                    'nids_noadapter':(nm['score']+na['score'])/2 if na is not None and na['status']=='available' else None}
                row=dict(state=state,target=target,proposal_id=pid,record_index=index,
                    semantic=sem,appearance=app,geometry=geo,nids=nm,nids_appearance=na,
                    sam6d_best_CAD_view=records[best]['view_index'],scores=values)
                scores.append(row)
                for name,value in values.items():
                    per_method[name].append(dict(proposal_id=pid,score=value,
                        status='available' if value is not None and np.isfinite(value) else 'unavailable'))
            selections += [rank(per_method[m],m,state,target) for m in METHODS]
            selected_ids=shortlists[state,target]
            selections += [rank([r for r in per_method[m] if r['proposal_id'] in selected_ids],
                                'clip5_'+m,state,target) for m in METHODS]
        print(f'{state}: scored12targets x12crops x{len(METHODS)}methods',flush=True)
    assert len(scores)==1152 and len(selections)==960
    check_sources()
    save(HERE/'candidate_scores.json',scores)
    save(HERE/'selections.json',selections)
    all_methods=list(METHODS)+['clip5_'+m for m in METHODS]
    save(HERE/'completion.json',dict(candidate_pairs=1152,tasks_per_method=96,methods=all_methods,
        scores_sha256=digest(HERE/'candidate_scores.json'),selections_sha256=digest(HERE/'selections.json'),
        features_sha256=digest(HERE/'features.npz'),protocol_sha256=digest(HERE/'protocol.json'),
        seconds=perf_counter()-started,evaluation_labels_read=False,source_hashes_verified=True,
        ties={m:sum(r['status']=='tied' for r in selections if r['method']==m) for m in all_methods},
        unavailable={m:sum(r['status'].startswith('unavailable') for r in selections if r['method']==m) for m in all_methods}))
    print(json.dumps(read(HERE/'completion.json')),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'encode':encode,'score':score,'check':check_sources}[sys.argv[1]]()
