"""Execute pinned upstream contact samplers without importing robot frameworks.

The original functions are loaded verbatim from the saved source AST. Geometry
helpers are also upstream code; only ray batches are chunked to bound RAM.
The common Panda adapter, fixed-down frame, table and continuous hand checks
are outside these samplers and must not be attributed to their original code.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

VENDOR=Path(__file__).with_name('vendor')/'one'
ONE_COMMIT='ec1bdae83ec33ca6af9de42904d9851647fe012c'
GRASPKIT_COMMIT='bb2b65e0de1ccc486e022e57a6a56d8ddd78e969'
METHODS=('one_default','one_2mm','one_budget512','graspkit_512')
_ONE=None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_functions(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'),filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):
        raise ValueError('Pinned source no longer contains expected functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def _load_one():
    global _ONE
    if _ONE is not None:
        return _ONE
    math_path=VENDOR/'math.py'
    spec=importlib.util.spec_from_file_location('comparison_one_math',math_path)
    math=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    geo=original_functions(VENDOR/'geometry_ops.py',
        ('triangle_areas','sample_count_from_area','sample_surface','ray_triangles_batch_far'),
        {'np':np,'oum':math})
    original_ray=geo['ray_triangles_batch_far']
    def chunked_rays(origins,directions,v0,v1,v2,eps=1e-6):
        hits=[]
        ids=[]
        for start in range(0,len(origins),16):
            t,i=original_ray(origins[start:start+16],directions[start:start+16],v0,v1,v2,eps)
            hits.append(t)
            ids.append(i)
        return np.concatenate(hits),np.concatenate(ids)
    helper=SimpleNamespace(**geo)
    helper.ray_triangles_batch_far=chunked_rays
    module=original_functions(VENDOR/'antipodal.py',('_antipodal_candidates',),
                             {'np':np,'oum':math,'osgop':helper})
    _ONE=SimpleNamespace(candidate=module['_antipodal_candidates'],math=math,geo=geo,
                         helper=helper,original_ray=original_ray)
    return _ONE


def source_manifest():
    return dict(one=dict(repository='https://github.com/wanweiwei07/one',commit=ONE_COMMIT,
        license='MIT',original_functions=['_antipodal_candidates','triangle_areas','sample_count_from_area','sample_surface','ray_triangles_batch_far'],
        source_sha256={p.name:sha(p) for p in VENDOR.glob('*.py')},
        license_sha256=sha(VENDOR/'LICENSE'),
        original_framework_executed=False,original_contact_functions_executed=True,
        adaptation='AST executes unchanged function bodies; ray batches of 16 bound RAM without changing per-ray calculations'),
        graspkit=dict(repository='https://github.com/DexForce/EmbodiChain',commit=GRASPKIT_COMMIT,
        license='Apache-2.0',source_provenance='experiments/runtime_antipodal_assembly_20261023/vendor/provenance.json',
        original_function='AntipodalSampler.sample',original_framework_executed=False,
        adaptation='Existing pinned vendor changes only utility imports, not sampler algorithm'))


def _one_pairs(vertices,faces,seed,density_mm,ray_epsilon=None):
    backend=_load_one()
    vs=np.asarray(vertices,dtype=np.float64)
    fs=np.asarray(faces,dtype=np.int64)
    triangles=vs[fs]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
    spacing=float(density_mm)/1000.
    samples=backend.helper.sample_count_from_area(vs,fs,spacing)
    old_state=np.random.get_state()
    old_epsilon=backend.math.eps
    started=time.perf_counter()
    try:
        np.random.seed(seed)
        if ray_epsilon is not None:
            backend.math.eps=float(ray_epsilon)
        values=backend.candidate(vs,fs,normals,spacing,20.,30.,.002)
    finally:
        np.random.set_state(old_state)
        backend.math.eps=old_epsilon
    metadata=dict(backend='one_original_contact_kernel',density_m=spacing,
        input_and_output_frame='caller_supplied_mesh_frame',units='meters',
        upstream_density_default_m=.02,surface_samples=samples,seed=int(seed),
        normal_tolerance_deg=20.,upstream_clearance_per_side_m=.002,
        upstream_roll_step_deg=30.,roll_not_expanded_in_contact_kernel=True,
        logical_jaw_gap_limits_m=[0.,.079],ray_batch_size=16,
        ray_epsilon=float(old_epsilon if ray_epsilon is None else ray_epsilon),
        numerical_epsilon_override=ray_epsilon is not None,
        sampling_seconds=time.perf_counter()-started,
        full_framework_executed=False,
        local_shared_contact_region_rule_used=False,local_cylinder_rule_used=False,
        local_strict_flat_normal_rule_used=False,
        source_commit=ONE_COMMIT)
    if values is None:
        return dict(pairs_world_m=np.empty((0,2,3)),normals_world=np.empty((0,2,3)),metadata=dict(metadata,ray_hits=0,qualified_pairs=0))
    center,width,n,nq,cosine,threshold,_=values
    # These are the original antipodal_iter normal and physical-gap masks;
    # contact-depth correction is zero because the adapter describes jaw gap,
    # not the Panda actuator command. Its +1mm conversion is done by the host.
    good=(cosine>=threshold)&(width>=0.)&(width<=.079)
    separation=width-2*.002
    p=center+.5*separation[:,None]*n
    q=center-.5*separation[:,None]*n
    pairs=np.stack((p,q),axis=1)[good]
    pair_normals=np.stack((n,nq),axis=1)[good]
    metadata.update(ray_hits=len(center),qualified_pairs=len(pairs),
                    normal_or_width_rejections=int((~good).sum()))
    return dict(pairs_world_m=pairs,normals_world=pair_normals,metadata=metadata)


def _graspkit_pairs(vertices,faces,seed,n_sample):
    import open3d as o3d
    import torch
    from experiments.runtime_antipodal_assembly_20261023.vendor import AntipodalSampler,AntipodalSamplerCfg
    vs=np.asarray(vertices,dtype=np.float64)
    fs=np.asarray(faces,dtype=np.int64)
    started=time.perf_counter()
    # The pinned Fibonacci/raycast path is deterministic and does not use this
    # seed. Record it for a common experiment interface without claiming RNG.
    sampler=AntipodalSampler(AntipodalSamplerCfg(n_sample=int(n_sample),
        min_length=1e-6,max_length=.079,max_angle=np.deg2rad(15.)))
    raw=sampler.sample(torch.as_tensor(vs,dtype=torch.float32),torch.as_tensor(fs,dtype=torch.int64))
    pairs=raw.detach().cpu().numpy().astype(np.float64)
    finite=np.all(np.isfinite(pairs),axis=(1,2))
    pairs=pairs[finite]
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vs.astype(np.float32)),o3d.core.Tensor(fs.astype(np.int32))))
    if len(pairs):
        nearest=scene.compute_closest_points(o3d.core.Tensor(pairs.reshape(-1,3).astype(np.float32)))
        normals=nearest['primitive_normals'].numpy().astype(np.float64).reshape(-1,2,3)
    else:
        normals=np.empty((0,2,3))
    return dict(pairs_world_m=pairs,normals_world=normals,metadata=dict(
        backend='graspkit_original_fibonacci_sampler',source_commit=GRASPKIT_COMMIT,
        input_and_output_frame='caller_supplied_mesh_frame',units='meters',
        surface_samples=int(n_sample),seed=int(seed),deterministic_fibonacci=True,
        normal_tolerance_deg=15.,contact_separation_limits_m=[1e-6,.079],
        qualified_pairs=len(pairs),raw_pairs=len(raw),nonfinite_removed=int((~finite).sum()),
        sampling_seconds=time.perf_counter()-started,full_framework_executed=False,
        normal_metadata_source='native closest-triangle query, not a changed sampler filter',
        local_shared_contact_region_rule_used=False,local_cylinder_rule_used=False,
        local_strict_flat_normal_rule_used=False))


def generate_pairs(vertices,faces,method,seed=0,density_mm=None,n_sample=512,ray_epsilon=None):
    """Return native-mesh pairs plus recorded upstream settings and stage counts.

    `one_default` preserves 20mm upstream spacing; `one_2mm` changes only that
    documented density parameter. `one_budget512` derives density from area to
    request the declared sample budget; it changes no sampling algorithm.
    `graspkit_512` runs the pinned real sampler.
    No width67mm/fixed-down/table/Panda/shared-region filtering is hidden here.
    """
    if method in ('one_default','one_2mm'):
        spacing=(20. if method=='one_default' else 2.) if density_mm is None else density_mm
        return _one_pairs(vertices,faces,seed,spacing,ray_epsilon)
    if method=='one_budget512':
        backend=_load_one()
        area=float(np.sum(backend.helper.triangle_areas(np.asarray(vertices),np.asarray(faces))))
        spacing=1000*np.sqrt(area/float(n_sample)) if density_mm is None else density_mm
        result=_one_pairs(vertices,faces,seed,spacing,ray_epsilon)
        result['metadata'].update(requested_surface_sample_budget=int(n_sample),
            density_rule='sqrt(native_mesh_surface_area / requested_sample_budget)',
            method_variant='documented density parameter adjusted for controlled sample budget')
        return result
    if method=='graspkit_512':
        return _graspkit_pairs(vertices,faces,seed,n_sample)
    raise ValueError('Unknown upstream method: '+str(method))


def verify_ray_chunk_equivalence():
    """Check the exact original helper and memory adapter return identical hits."""
    backend=_load_one()
    rng=np.random.default_rng(71)
    triangle=rng.normal(size=(37,3,3))
    origins=rng.normal(size=(45,3))
    directions=rng.normal(size=(45,3))
    expected=backend.original_ray(origins,directions,*triangle.transpose(1,0,2),eps=float(backend.math.eps))
    actual=backend.helper.ray_triangles_batch_far(origins,directions,*triangle.transpose(1,0,2),eps=float(backend.math.eps))
    if not all(np.array_equal(a,b) for a,b in zip(expected,actual)):
        raise AssertionError('Memory batching changed original per-ray outputs')
    return dict(rays=len(origins),triangles=len(triangle),identical=True)


if __name__=='__main__':
    import json
    from experiments.cad_grasp_geometry_20261026 import run_trial as geometry
    from experiments.cad_grasp_geometry_20261026.inputs import load_inputs
    geometry.initialize()
    print(json.dumps(verify_ray_chunk_equivalence()))
    rows=load_inputs()
    for row in rows[:12]:
        vertices,faces,_=geometry.load_target(row)
        for method in ('one_default','one_budget512','graspkit_512'):
            result=generate_pairs(vertices,faces,method,seed=23)
            print(json.dumps(dict(scene=row['scene'],part=row['part'],method=method,**result['metadata'])),flush=True)
