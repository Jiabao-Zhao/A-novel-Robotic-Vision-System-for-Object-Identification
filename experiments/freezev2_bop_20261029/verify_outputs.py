"""CPU-only validation of this experiment's cached geometry, not benchmark accuracy."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d

from freezev2_repro.pipeline import load_config, local_path, config_hash, source_hashes
from freezev2_repro.data import BOPDataset, sha256, write_json
from freezev2_repro.geometry import visible_surface, validate_pose


def main():
    cfg=load_config(Path(__file__).with_name('config.json'))
    root=local_path(cfg['paths']['output_root'])
    folder=root/'onboarding'/cfg['dataset']/f"obj_{cfg['object_id']:06d}"
    manifest=json.loads((folder/'geometry_manifest.json').read_text())
    assert manifest['signature']['config_sha256']==config_hash(cfg)
    assert manifest['signature']['source_sha256']==source_hashes()
    for relative,checksum in manifest['files'].items():
        assert sha256(folder/relative)==checksum, relative
    cad=BOPDataset(local_path(cfg['paths']['dataset_root']),cfg['dataset']).model(cfg['object_id'])
    assert sha256(cad['path'])==manifest['signature']['model_sha256']
    cache=np.load(folder/'geometry.npz',allow_pickle=False)
    points=cache['points_m']
    assert points.shape==(5000,3)
    assert cache['visibility'].shape==(162,5000)
    minimum=int(cache['visibility'].sum(0).min())
    assert minimum>=18
    ray_scene=o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(cad['vertices'].astype(np.float32)),o3d.core.Tensor(cad['triangles'].astype(np.uint32))))
    distances=ray_scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    assert np.max(distances)<1e-7
    worst=0.
    for view in range(162):
        saved=np.load(folder/'views'/f'{view:03d}.npz',allow_pickle=False)
        depth=saved['depth_m']
        assert depth.shape==(480,480) and np.isfinite(depth).all() and np.min(depth)>=0
        validate_pose(saved['R_m2c'],saved['t_m'])
        assert np.allclose(saved['R_m2c'],cache['R'][view]) and np.allclose(saved['t_m'],cache['t_m'][view])
        uv,visible=visible_surface(points,None,cache['R'][view],cache['t_m'][view],cache['K'],depth,.005*cad['diameter_m'])
        # Original visibility was computed at full raycast float64; cache depth
        # float32 can change exact threshold-boundary membership. Verify every
        # retained lift directly within a small serialization tolerance.
        selected=cache['visibility'][view]
        pixels=np.floor(uv[selected]+.5).astype(int)
        z=(points[selected]@cache['R'][view].T+cache['t_m'][view])[:,2]
        residual=np.abs(z-depth[pixels[:,1],pixels[:,0]])
        if len(residual):
            worst=max(worst,float(residual.max()))
            assert residual.max()<=.005*cad['diameter_m']+1e-7
    inputs=json.loads((root/'input_geometry_diagnostic'/'report.json').read_text())
    candidates=[c for image in inputs['images'] for c in image['candidates']]
    assert len(inputs['images'])==4 and len(candidates)==7
    assert all(c['status']=='valid_observed_geometry' for c in candidates)
    for image in inputs['images']:
        for key in ('rgb','depth','camera'):
            assert sha256(image['source'][key])==image['source'][key+'_sha256']
    report={'status':'passed','claim':'Cache integrity/native CAD geometry/observed mapping only; no trained inference or benchmark accuracy.',
            'config_sha256':config_hash(cfg),'source_sha256':source_hashes(),'hashed_files':len(manifest['files']),
            'rendered_views':162,'retained_CAD_points':5000,'minimum_visible_views':minimum,
            'maximum_CAD_point_to_original_mesh_distance_m':float(distances.max()),
            'maximum_retained_lift_depth_residual_m':worst,'visibility_threshold_m':.005*cad['diameter_m'],
            'input_images':4,'selected_external_mask_candidates':7,
            'sparse_counts':[c['sparse_count'] for c in candidates], 'dense_counts':[c['dense_count'] for c in candidates],
            'max_observed_reprojection_error_px':max(c['reprojection_max_error_px'] for c in candidates),
            'GT_pose_access':False}
    write_json(root/'cache_validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
