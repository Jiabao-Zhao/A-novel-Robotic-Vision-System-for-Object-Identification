"""Resumable native CAD onboarding for the single frozen all-seven recipe."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .data import sha256
from .durable import atomic_json, atomic_npz, digest, ensure_identity, read_json, require_disk
from .geometry import project, transform_points, visible_surface
from .rendering import camera_poses, RaycastRenderer
from .features import PCATransform, sample_feature_map, fuse_features


def _check_npz(path,identity,**shapes):
    with np.load(path,allow_pickle=False) as value:
        if str(value['identity'])!=digest(identity):raise ValueError(f'Checkpoint identity mismatch: {path}')
        output={k:np.array(value[k]) for k in value.files if k!='identity'}
    for key,shape in shapes.items():
        if output[key].shape!=shape or not np.isfinite(output[key]).all():raise ValueError(f'Invalid checkpoint {path}:{key}')
    return output


def onboard_geometry(model,folder,recipe,cnos_path,implementation,*,minimum_free_gb=15,heartbeat=None):
    """Each complete view is durable; an interrupted object resumes at its view."""
    folder=Path(folder)
    identity={'schema':'metric-CAD-onboarding-v2','model_sha256':model['sha256'],
              'texture_sha256':model.get('texture',{}).get('sha256') if model.get('texture') else None,
              'render':recipe['render'],'cad':recipe['cad'],'seed':recipe['seed'],
              'cnos_sha256':sha256(cnos_path),'implementation':implementation}
    ensure_identity(folder,identity)
    complete=folder/'complete.json'
    if complete.exists():
        result=read_json(complete)
        if result['identity']!=identity:raise ValueError('CAD geometry cache recipe differs')
        for name,checksum in result['files'].items():
            if sha256(folder/name)!=checksum:raise ValueError(f'CAD geometry cache corruption: {folder/name}')
        for view,record in result['view_file_hashes'].items():
            if sha256(folder/'views'/f'{int(view):03d}.npz')!=record['sha256']:raise ValueError('CAD render view checksum changed')
        return result
    require_disk(folder,minimum_free_gb)
    start=time.perf_counter()
    progress=read_json(folder/'progress.json') if (folder/'progress.json').exists() else {'active_runtime_s':0.,'views':{}}
    initialization=folder/'sampling_and_cameras.npz'
    count=recipe['cad']['initial_poisson_points']
    if initialization.exists():
        data=_check_npz(initialization,identity,raw_points_m=(count,3),R=(162,3,3),t_m=(162,3),K=(3,3))
    else:
        import open3d as o3d
        o3d.utility.random.seed(recipe['seed'])
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(model['vertices']),o3d.utility.Vector3iVector(model['triangles']))
        cloud=mesh.sample_points_poisson_disk(number_of_points=count,init_factor=5)
        K,poses=camera_poses(model['vertices'],resolution=recipe['render']['resolution'],occupancy=recipe['render']['occupancy'],camera_to_model=np.load(cnos_path,allow_pickle=False))
        data=dict(raw_points_m=np.asarray(cloud.points).copy(),K=K,R=np.array([p[0] for p in poses]),t_m=np.array([p[1] for p in poses]))
        atomic_npz(initialization,identity=np.array(digest(identity)),**data)
    renderer=RaycastRenderer(model['vertices'],model['triangles'],model['colors'],texture=model.get('texture'))
    raw=data['raw_points_m']; visibility=[]
    resolution=recipe['render']['resolution']
    for view in range(162):
        if heartbeat:heartbeat(stage='CAD_rendering',view=view,views=162)
        path=folder/'views'/f'{view:03d}.npz'
        if str(view) in progress['views']:
            if sha256(path)!=progress['views'][str(view)]['sha256']:raise ValueError(f'Render checkpoint corruption: {path}')
            saved=_check_npz(path,identity,rgb=(resolution,resolution,3),depth_m=(resolution,resolution),visible=(count,))
        else:
            rendered=renderer.render(data['R'][view],data['t_m'][view],data['K'],height=resolution,width=resolution)
            uv,visible=visible_surface(raw,None,data['R'][view],data['t_m'][view],data['K'],rendered['depth_m'],recipe['render']['visibility_tolerance_fraction']*model['diameter_m'])
            visible&=(uv[:,0]>=0)&(uv[:,0]<=resolution-1)&(uv[:,1]>=0)&(uv[:,1]<=resolution-1)
            saved=dict(rgb=rendered['rgb'],depth_m=rendered['depth_m'].astype(np.float32),visible=visible)
            atomic_npz(path,identity=np.array(digest(identity)),**saved)
            progress['views'][str(view)]={'sha256':sha256(path)}
            progress['active_runtime_s']+=time.perf_counter()-start;start=time.perf_counter()
            atomic_json(folder/'progress.json',progress)
        visibility.append(saved['visible'])
    visibility=np.asarray(visibility,bool)
    eligible=np.flatnonzero(visibility.sum(0)>=recipe['cad']['min_visible_views'])
    retained=recipe['cad']['points']
    if len(eligible)<retained:raise ValueError(f'CAD has only{len(eligible)} samples visible in required views; no threshold relaxation')
    chosen=np.random.default_rng(recipe['seed']).choice(eligible,retained,replace=False)
    atomic_npz(folder/'geometry.npz',identity=np.array(digest(identity)),points_m=raw[chosen],support_points_m=raw,
               visibility=visibility[:,chosen],K=data['K'],R=data['R'],t_m=data['t_m'],selected_indices=chosen)
    # Review sheet never enters any score or pose-selection path.
    sheet=Image.new('RGB',(720,720),'black')
    for i,view in enumerate((0,20,40,60,80,100,120,140,161)):
        image=_check_npz(folder/'views'/f'{view:03d}.npz',identity)['rgb']
        sheet.paste(Image.fromarray(image).resize((240,240)),((i%3)*240,(i//3)*240))
    sheet.save(folder/'render_contact_sheet.png')
    progress['active_runtime_s']+=time.perf_counter()-start
    result={'status':'complete','identity':identity,'files':{'geometry.npz':sha256(folder/'geometry.npz'),
            'sampling_and_cameras.npz':sha256(initialization),'render_contact_sheet.png':sha256(folder/'render_contact_sheet.png')},
            'view_file_hashes':progress['views'],'runtime_s':progress['active_runtime_s'],
            'initial_points':count,'eligible_points':len(eligible),'retained_points':retained,'views':162,
            'minimum_visibility':int(visibility[:,chosen].sum(0).min()),'appearance':model.get('appearance','unrecorded'),
            'pose_direction':'original_native_CAD_to_OpenCV_camera','internal_units':'metres'}
    atomic_json(complete,result)
    return result


def onboard_features(model,geometry_folder,folder,recipe,backends,implementation,*,minimum_free_gb=15,heartbeat=None):
    """Cache each real frozen giant feature map; no random/placeholder fallback."""
    geometry_folder,folder=Path(geometry_folder),Path(folder)
    dino,gedi=backends
    geometry=read_json(geometry_folder/'complete.json')
    scientific_backends={'dino':dict(dino.provenance),'gedi':dict(gedi.provenance)}
    for description in scientific_backends.values():description.pop('resources',None)
    identity={'schema':'frozen-CAD-feature-onboarding-v2','geometry_identity':digest(geometry['identity']),
              'geometry_sha256':geometry['files']['geometry.npz'],'feature_recipe':recipe['features'],
              'backends':scientific_backends,'seed':recipe['seed'],'implementation':implementation}
    ensure_identity(folder,identity)
    complete=folder/'complete.json'
    if complete.exists():
        result=read_json(complete)
        if result['identity']!=identity:raise ValueError('Feature cache method or weights differ')
        for name,checksum in result['files'].items():
            if sha256(folder/name)!=checksum:raise ValueError(f'Feature cache corruption: {folder/name}')
        return result
    require_disk(folder,minimum_free_gb)
    start=time.perf_counter()
    progress=read_json(folder/'progress.json') if (folder/'progress.json').exists() else {'active_runtime_s':0.,'maps':{}}
    data=np.load(geometry_folder/'geometry.npz',allow_pickle=False)
    points=data['points_m'];total=np.zeros((len(points),1536),np.float64);counts=np.zeros(len(points),np.int32)
    for view,visible in enumerate(data['visibility']):
        if heartbeat:heartbeat(stage='CAD_visual_features',view=view,views=162)
        path=folder/'visual_maps'/f'{view:03d}.npz'
        if str(view) in progress['maps']:
            if sha256(path)!=progress['maps'][str(view)]['sha256']:raise ValueError('Frozen CAD view-map corruption')
            fmap=_check_npz(path,identity,map=(16,16,1536))['map']
        else:
            image_path=geometry_folder/'views'/f'{view:03d}.npz'
            if sha256(image_path)!=geometry['view_file_hashes'][str(view)]['sha256']:raise ValueError('Rendered RGB view changed before feature extraction')
            image=_check_npz(image_path,geometry['identity'])['rgb']
            fmap=dino.extract(image)
            if fmap.shape!=(16,16,1536) or not np.isfinite(fmap).all():raise ValueError('Actual giant feature-map contract failed')
            atomic_npz(path,identity=np.array(digest(identity)),map=fmap)
            progress['maps'][str(view)]={'sha256':sha256(path)}
            progress['active_runtime_s']+=time.perf_counter()-start;start=time.perf_counter()
            atomic_json(folder/'progress.json',progress)
        uv,_=project(transform_points(points,data['R'][view],data['t_m'][view]),data['K'])
        total[visible]+=sample_feature_map(fmap,uv[visible],image_size=(480,480));counts[visible]+=1
    if np.min(counts)<recipe['cad']['min_visible_views']:raise ValueError('Missing visible CAD feature views')
    visual=(total/counts[:,None]).astype(np.float32)
    if heartbeat:heartbeat(stage='CAD_PCA')
    pca=PCATransform(recipe['features']['pca_components']).fit(visual)
    geometry_file=folder/'geometric.npz'
    if geometry_file.exists():
        geometric=_check_npz(geometry_file,identity,geometric=(len(points),64))['geometric']
    else:
        if heartbeat:heartbeat(stage='CAD_GeDi')
        geometric=gedi.compute(points,data['support_points_m'],model['diameter_m'])
        atomic_npz(geometry_file,identity=np.array(digest(identity)),geometric=geometric)
    fused=fuse_features(pca.transform(visual),geometric)
    atomic_npz(folder/'pca.npz',mean=pca.mean_,components=pca.components_,n_components=np.int64(pca.n_components),fit_scope=np.array('cad_only'),whiten=np.array(False))
    atomic_npz(folder/'features.npz',identity=np.array(digest(identity)),fused=fused,geometric=geometric,visual_mean=visual,view_counts=counts)
    progress['active_runtime_s']+=time.perf_counter()-start
    result={'status':'complete','identity':identity,'files':{name:sha256(folder/name) for name in ('features.npz','pca.npz','geometric.npz')},
            'runtime_s':progress['active_runtime_s'],'feature_shape':list(fused.shape),'PCA_fit_scope':'this CAD only, no observations/test GT',
            'maps_completed':len(progress['maps']),'backend_frozen':True,
            'runtime_resources':{'dino':dino.provenance.get('resources',{}),'gedi':gedi.provenance.get('resources',{})}}
    atomic_json(complete,result)
    return result


def load_bundle(model,geometry_folder,feature_folder):
    geometry_folder,feature_folder=Path(geometry_folder),Path(feature_folder)
    points=np.load(geometry_folder/'geometry.npz',allow_pickle=False)['points_m']
    features=np.load(feature_folder/'features.npz',allow_pickle=False)['fused']
    if features.shape!=(len(points),128) or not np.isfinite(features).all():raise ValueError('Invalid final frozen CAD feature cache')
    return {'model':model,'points_m':points,'features':features,'pca':PCATransform.load(feature_folder/'pca.npz'),
            'feature_identity':digest(read_json(feature_folder/'complete.json')['identity'])}
