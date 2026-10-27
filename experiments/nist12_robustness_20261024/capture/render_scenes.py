"""Fresh static supported CAD scenes; inference inputs and simulator audits are separate."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
REPO = HERE.parents[2]
PRIOR = REPO/'experiments/nist12_fullscene_benchmarks_20261023'
CAMERA = 'observed_camera'
SIZE = 2560
WORKSPACE_LO = np.array([-.32, -.24])
WORKSPACE_HI = np.array([.32, .24])


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8-sig'))


def save(file, value):
    file=Path(file)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(file):
    with Path(file).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(file):
    return Path(file).resolve().relative_to(REPO).as_posix()


def path(value):
    value=str(value).replace('\\','/')
    retired = 'C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to'
    for prefix in (retired, '/mnt/c' + retired[2:]):
        if value == prefix or value.startswith(prefix + '/'):
            suffix = value[len(prefix):].lstrip('/')
            if '..' in suffix.split('/') or ':' in suffix:
                raise ValueError('Unsafe retired-workspace path suffix')
            return REPO / 'outputs/datasets/nist_task_board/legacy_workspace_inputs' / suffix
    if sys.platform!='win32' and len(value)>1 and value[1]==':':
        value='/mnt/'+value[0].lower()+value[2:]
    item=Path(value)
    return item if item.is_absolute() else REPO/item


def load_geometry_helpers():
    sys.path.insert(0, str(REPO))
    from simulation import nist_capture
    return nist_capture


def compile_scene(root):
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    return model,data


def geom_vertices(model,data,gid):
    mid=model.geom_dataid[gid]
    vertices=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
    return vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]


def inventory(catalog):
    geometry=load_geometry_helpers()
    items={}
    for entry in catalog:
        mesh=path(entry['mesh'])
        root=ET.Element('mujoco')
        asset=ET.SubElement(root,'asset')
        ET.SubElement(asset,'mesh',name='part',file=str(mesh),scale='1 1 1')
        world=ET.SubElement(root,'worldbody')
        ET.SubElement(world,'geom',type='mesh',mesh='part',contype='0',conaffinity='0')
        model,data=compile_scene(root)
        points=geom_vertices(model,data,0)
        target=entry['target']
        if target.startswith(('KET','RGOCG')):
            hull=ConvexHull(points)
            center=geometry.hull_center(points,hull)
            options=[]
            for name,rotation in [('upright',np.eye(3)),
                    ('lying_x',Rotation.from_euler('x',90,degrees=True).as_matrix()),
                    ('lying_y',Rotation.from_euler('y',90,degrees=True).as_matrix())]:
                audit=geometry.supported_pose(points,rotation,center)
                if audit is not None:
                    options.append(dict(name=name,rotation=rotation,audit=audit))
            assert options,target
        else:
            options=geometry.rest_options(points,target)
        assert np.allclose(np.ptp(points,axis=0),entry['native_extent_m'],atol=1e-7),target
        items[target]=dict(entry=entry,vertices=points,options=options)
    return items


def separated(box,others,gap=.025):
    return all(np.linalg.norm(np.maximum(0,np.maximum(other[0]-box[1],box[0]-other[1])))>=gap
               for other in others)


def generate_layout(index,seed,items,clearance_m=0.):
    """Placement uses CAD geometry only; no segmentation or classifier feedback."""
    rng=np.random.default_rng(seed)
    order=sorted(items,key=lambda name:-np.prod(np.sort(np.ptp(items[name]['vertices'],axis=0))[-2:]))
    boxes=[]
    result=[]
    for name in order:
        item=items[name]
        oid=item['entry']['object_id']
        choice=item['options'][(index+oid-1)%len(item['options'])]
        yaw=float(rng.uniform(-180,180))
        rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@choice['rotation']
        vertices=item['vertices']@rotation.T
        lo,hi=vertices.min(0),vertices.max(0)
        lower=WORKSPACE_LO+(hi[:2]-lo[:2])/2
        upper=WORKSPACE_HI-(hi[:2]-lo[:2])/2
        if np.any(lower>=upper):raise ValueError(f'{name} exceeds fixed workspace')
        for _ in range(20000):
            center=rng.uniform(lower,upper)
            position=np.r_[center-(lo[:2]+hi[:2])/2,clearance_m-lo[2]]
            box=np.array([lo[:2]+position[:2],hi[:2]+position[:2]])
            if separated(box,boxes):break
        else:raise ValueError(f'Unable to separate {name} in seed {seed}')
        boxes.append(box)
        result.append(dict(target=name,object_id=oid,instance_id=name+'__0',
            position_world_m=position.tolist(),rotation_world_from_cad=rotation.tolist(),
            quaternion_wxyz=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]].tolist(),
            minimum_world_z_m=float((vertices+position)[:,2].min()),
            bbox_world_xy_m=box.tolist(),rest_option=choice['name'],
            support_audit=choice['audit'],yaw_deg=yaw))
    return result


def camera_pose(index,far=False):
    # One calibrated pinhole camera per capture. No multiview fusion is implied.
    tilt=[12.,20.,28.,20.][index%4]
    azimuth=[-90.,-70.,-110.,-90.][index%4]
    distance=[.64,.67,.70,.66][index%4]*(1.35 if far else 1.)
    az,el=np.deg2rad([azimuth,90-tilt])
    target=np.zeros(3)
    position=distance*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    z=(position-target)/np.linalg.norm(position-target)
    right=np.cross([0.,0.,1.],z);right/=np.linalg.norm(right)
    up=np.cross(z,right)
    return position,np.column_stack([right,up,z])


def make_scene(layout,index,catalog,condition='nominal',target_id=None):
    catalog_by_id={o['object_id']:o for o in catalog}
    root=ET.Element('mujoco',model='classification_robustness')
    ET.SubElement(root,'compiler',angle='radian')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(SIZE),offheight=str(SIZE))
    ET.SubElement(visual,'quality',offsamples='0',shadowsize='4096')
    ET.SubElement(visual,'map',znear='.001')
    shadow=condition=='light_shadow'
    ET.SubElement(visual,'headlight',active='1',ambient='.2 .2 .2' if shadow else '.6 .6 .6',
        diffuse='.15 .15 .15' if shadow else '.25 .25 .25',specular='.02 .02 .02')
    asset=ET.SubElement(root,'asset')
    for entry in catalog:
        ET.SubElement(asset,'mesh',name='cad_'+str(entry['object_id']),file=str(path(entry['mesh'])),scale='1 1 1')
        ET.SubElement(asset,'material',name='part_material_'+str(entry['object_id']),
            rgba=' '.join(map(str,entry['color_rgb']+[1.])),
            specular='.8' if condition=='finish' else '.08',
            shininess='.8' if condition=='finish' else '.05')
    ET.SubElement(asset,'material',name='table',rgba='.28 .32 .39 1' if condition=='background' else '.46 .49 .48 1',
        specular='.08',shininess='.05')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',name='table_surface',type='box',size='1 1 .025',pos='0 0 -.025',
        material='table',contype='0',conaffinity='0',group='1')
    position,rotation=camera_pose(index,far=condition=='camera_far')
    ET.SubElement(world,'camera',name=CAMERA,pos=' '.join(map(str,position)),
        quat=' '.join(map(str,Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]])),fovy='65')
    directions=[[.7,.35,-1],[-.7,.35,-1],[0,-.7,-1]] if not shadow else [[1.,.3,-.35]]
    for number,direction in enumerate(directions):
        vector=np.asarray(direction,float);vector/=np.linalg.norm(vector)
        vector=vector@rotation.T
        ET.SubElement(world,'light',name=f'light_{number}',pos='0 0 1',dir=' '.join(map(str,vector)),
            type='directional',ambient='0 0 0',diffuse='.75 .75 .75' if shadow else '.12 .12 .12',
            specular='.05 .05 .05',castshadow='true' if shadow else 'false')
    instances=copy.deepcopy(layout)
    if condition=='absent':instances=[p for p in instances if p['object_id']!=target_id]
    if condition=='duplicate':
        original=next(p for p in instances if p['object_id']==target_id)
        duplicate=copy.deepcopy(original);duplicate['instance_id']=original['target']+'__1'
        vertices=np.asarray(duplicate['bbox_world_xy_m'])
        old_center=vertices.mean(0)
        half_size=np.ptp(vertices,axis=0)/2
        rng=np.random.default_rng(2026102400+index*151+target_id)
        for _ in range(20000):
            center=rng.uniform(WORKSPACE_LO+half_size,WORKSPACE_HI-half_size)
            shift=center-old_center
            box=vertices+shift
            if separated(box,[np.asarray(p['bbox_world_xy_m']) for p in instances]):break
        else:
            # Large USB+cable footprints may need unused tabletop outside the
            # initial placement rectangle. Keep the camera and original parts fixed.
            cv_rotation=rotation@np.diag([1.,-1.,-1.])
            limit=np.tan(np.deg2rad(65/2))-5/SIZE
            for _ in range(120000):
                center=rng.uniform(np.array([-.46,-.38])+half_size,np.array([.46,.38])-half_size)
                shift=center-old_center
                box=vertices+shift
                if not separated(box,[np.asarray(p['bbox_world_xy_m']) for p in instances]):continue
                corners=np.array([[x,y,z] for x in box[:,0] for y in box[:,1] for z in (0.,.08)])
                camera=(corners-position)@cv_rotation
                if np.all(camera[:,2]>0) and np.max(np.abs(camera[:,:2]/camera[:,2:3]))<limit:break
            else:raise ValueError('No free camera-visible duplicate location')
        duplicate['position_world_m'][:2]=(np.asarray(duplicate['position_world_m'][:2])+shift).tolist()
        duplicate['bbox_world_xy_m']=box.tolist()
        instances.append(duplicate)
    for instance in instances:
        entry=catalog_by_id[instance['object_id']]
        body=ET.SubElement(world,'body',name=instance['instance_id'],
            pos=' '.join(map(str,instance['position_world_m'])),quat=' '.join(map(str,instance['quaternion_wxyz'])))
        kwargs=dict(name=instance['instance_id']+'_visual',type='mesh',mesh='cad_'+str(entry['object_id']),
            contype='0',conaffinity='0',group='1',material='part_material_'+str(entry['object_id']))
        ET.SubElement(body,'geom',**kwargs)
    if condition.startswith('occlusion_'):
        original=next(p for p in instances if p['object_id']==target_id)
        points=np.asarray(original['bbox_world_xy_m'])
        center=points.mean(0)
        # A real rendered fixture blocks a fixed portion from this observation direction.
        # Its achieved visible fraction is measured only in the post-render audit.
        width=.012 if condition=='occlusion_light' else .024
        toward_camera=position[:2]/np.linalg.norm(position[:2])
        corners=np.array([[x,y] for x in points[:,0] for y in points[:,1]])
        half_extent=np.max((corners-center)@toward_camera)
        fixture_xy=center+toward_camera*(half_extent+.008)
        fixture_yaw=np.arctan2(toward_camera[1],toward_camera[0])-np.pi/2
        fixture_quat=Rotation.from_euler('z',fixture_yaw).as_quat()[[3,0,1,2]]
        ET.SubElement(world,'geom',name='occlusion_fixture',type='box',
            size=f'{width} .004 .1',pos=f'{fixture_xy[0]} {fixture_xy[1]} .1',
            quat=' '.join(map(str,fixture_quat)),
            rgba='.28 .21 .16 1',contype='0',conaffinity='0',group='1')
    return root,instances


def render_scene(scene_id,layout,index,catalog,condition='nominal',target_id=None):
    folder=TRIAL/'inputs'/scene_id
    audit=TRIAL/'audit'/scene_id
    completion=folder/'capture_completion.json'
    if completion.exists():return read(completion)['record']
    folder.mkdir(parents=True,exist_ok=True);audit.mkdir(parents=True,exist_ok=True)
    root,instances=make_scene(layout,index,catalog,condition,target_id)
    model,data=compile_scene(root)
    size=1280 if condition=='resolution_low' else SIZE
    options=mujoco.MjvOption();options.geomgroup[:]=[0,1,1,0,0,0];options.sitegroup[:]=0
    with mujoco.Renderer(model,height=size,width=size) as renderer:
        renderer.update_scene(data,camera=CAMERA,scene_option=options)
        rgb=renderer.render().copy()
        Image.fromarray(rgb).save(folder/'rgb.png')
        renderer.enable_depth_rendering();renderer.update_scene(data,camera=CAMERA,scene_option=options)
        depth=renderer.render().copy()
        np.save(folder/'depth.npy',depth)
        renderer.disable_depth_rendering();renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera=CAMERA,scene_option=options)
        segmentation=renderer.render().copy()
    cid=model.camera(CAMERA).id
    focal=size/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
    K=np.array([[focal,0,size/2],[0,focal,size/2],[0,0,1.]])
    np.save(folder/'intrinsics.npy',K)
    T=np.eye(4);T[:3,:3]=data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);T[:3,3]=data.cam_xpos[cid]
    np.save(audit/'world_T_camera.npy',T)
    ET.ElementTree(root).write(audit/'scene.xml')
    labels=np.zeros((size,size),np.uint16)
    truths=[]
    for number,instance in enumerate(instances,1):
        gid=model.geom(instance['instance_id']+'_visual').id
        mask=(segmentation[:,:,0]==gid)&(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        labels[mask]=number
        yy,xx=np.nonzero(mask)
        truths.append(dict(instance_id=instance['instance_id'],target=instance['target'],object_id=instance['object_id'],
            label_id=number,geom_id=gid,bbox_xyxy=None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
            visible_pixels=len(xx),pose_audit_only=instance))
    np.savez_compressed(audit/'instance_labels.npz',labels=labels)
    save(audit/'truth.json',dict(scene_id=scene_id,instances=truths,world_T_camera=relative(audit/'world_T_camera.npy'),
        expected_instance_counts={str(o['object_id']):sum(p['object_id']==o['object_id'] for p in instances) for o in catalog},
        condition_target_object_id=target_id,static_supported_pose=True,dynamic_settling=False,robot_geometries_present=False))
    record=dict(scene_id=scene_id,base_scene_id=f'R{index:02d}',condition=condition,
        rgb=relative(folder/'rgb.png'),depth=relative(folder/'depth.npy'),intrinsics=relative(folder/'intrinsics.npy'),
        depth_units='meters',image_shape=[size,size],robot_geometries_present=False,
        query_object_ids=[o['object_id'] for o in catalog])
    hashes={relative(folder/name):digest(folder/name) for name in ('rgb.png','depth.npy','intrinsics.npy')}
    save(completion,dict(record=record,input_sha256=hashes,labels_consumed_by_inference=False,
        generator_sha256=digest(__file__)))
    print(f'{scene_id}: {condition}, {len(instances)} rendered CAD instances, {size}px',flush=True)
    return record


def main():
    protocol=read(TRIAL/'protocol.json')
    config=protocol['capture']
    catalog=read(path(config['catalog_file']))['objects'] if config.get('catalog_file') else read(PRIOR/'inputs.json')['objects']
    items=inventory(catalog)
    rows=[]
    for index in range(config['layout_count']):
        seed=int(config['seed_base'])+index*151
        layout=generate_layout(index,seed,items,clearance_m=config.get('support_clearance_m',0.))
        save(TRIAL/'audit'/f'R{index:02d}'/'layout.json',dict(seed=seed,poses=layout,generator_sha256=digest(__file__)))
        rows.append(render_scene(f'R{index:02d}',layout,index,catalog))
        for condition in config['paired_conditions']:
            if index not in config['paired_layout_indices']:continue
            target_id=config['condition_target_ids'][index%len(config['condition_target_ids'])]
            rows.append(render_scene(f'R{index:02d}__{condition}',layout,index,catalog,condition,target_id))
        save(TRIAL/'capture_manifest.json',dict(scenes=rows,complete=False,generator_sha256=digest(__file__)))
    save(TRIAL/'capture_manifest.json',dict(scenes=rows,complete=True,generator_sha256=digest(__file__),
        protocol_sha256=digest(TRIAL/'protocol.json'),labels_available_only_under='audit/'))


if __name__=='__main__':main()
