"""Offline native CAD/STL registration and nominal NIST assembly references.

No grasp sampling, dynamics, robot commands or model calls. Native geometry
correspondence is checked against saved SolidWorks display-coordinate triples;
this reader does not claim to decode native B-rep topology or mate constraints.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from experiments.nist_official_reference_20261029.native_cad_audit import modern_streams
from experiments.nist_mating_goals_20261028.cad_tools import read_native_mesh

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ASSETS=ROOT/'outputs/simulation/nist_task_board_1/assets'
OUTPUT=ROOT/'outputs/nist_scene_corrections_20261029'
AUDIT=ROOT/'outputs/nist_official_reference_20261029/native_cad/taskboard_1_cad__GMC_Competition_Task_Board_Virtual.SLDASM/decoded_instance_tree.json'
BOARD_SIZE_M=np.array([.384,.384,.0089916])
BOARD_NATIVE_TO_WORKING=np.diag([1.,-1.,-1.,1.])
WORLD_T_BOARD=np.eye(4)
WORLD_T_BOARD[:3,3]=[0,.065,.020+BOARD_SIZE_M[2]/2]
PIN_DEPTH_M=.020

# Existing mesh names -> exact model ids in the official newer assembly XML.
MODELS={
 'GMC_Laser_Plate_Virtual':'5','KET12_Square_12mm':'8','KET16_Square_16mm':'11',
 'KET8_Square_8mm':'17','RGOCG12-50_12mm':'20','RGOCG16-50_16mm':'23','RGOCG8-50_8mm':'29',
 'Gear_Plate':'35','Gear_Small':'38','Gear_Medium':'41','Gear_Large':'44','Gear_Shaft':'47',
 'Waterproof_Female':'52','USB_Female':'55','DB_Housing':'58','RJ45_Housing':'61',
 'BNC_Female':'64','M4_Screw':'67','M8_Screw':'70','M12_Screw':'73','M16_Screw':'76',
 'M16_Hex_Nut':'85','RJ45_Female':'97','DSUB_Female':'139',
 'BNC_Male':'154','Waterproof_Male':'157','USB_Male':'160','DSUB_Male':'166',
}
PARTS=('BNC_Male','DSUB_Male','Gear_Large','KET12_Square_12mm','KET16_Square_16mm',
       'KET8_Square_8mm','M16_Hex_Nut','RGOCG12-50_12mm','RGOCG16-50_16mm',
       'RGOCG8-50_8mm','USB_Male','Waterproof_Male')
FIXTURE_ORDER=('Gear_Plate','Gear_Shaft','Gear_Shaft','Gear_Shaft','M4_Screw','M8_Screw',
               'M12_Screw','M16_Screw','BNC_Female','USB_Female','RJ45_Housing',
               'RJ45_Female','DB_Housing','DSUB_Female','Waterproof_Female')
PIN_HOLES_MM={'KET12_Square_12mm':[41.4496135243,197.1757772952],
 'KET16_Square_16mm':[41.4496135243,347.1757772952],
 'KET8_Square_8mm':[266.4496135243,272.1757772952],
 'RGOCG12-50_12mm':[41.4496135243,122.1757772952],
 'RGOCG16-50_16mm':[266.4496135243,122.1757772952],
 'RGOCG8-50_8mm':[116.4496135243,347.1757772952]}


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transform(points,T):
    T=np.asarray(T,float)
    return np.asarray(points,float)@T[:3,:3].T+T[:3,3]


def _rotations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.),repeat=3):
            R=np.eye(3)[:,perm]*np.array(signs)
            if np.linalg.det(R)>.5:yield R


def display_coordinate_candidates(data,bbox):
    """Find plausible native XYZ triples in the saved display cache.

    The stream's complete topology is not decoded. Candidates are float64
    triples at each byte alignment, bounded by the model's native bounds.
    Only candidates independently matching actual STL vertices count as
    geometric evidence; arbitrary triples/zeros do not certify a transform.
    """
    bbox=np.asarray(bbox,float);pieces=[]
    for width in (8,4):
        for offset in range(width):
            a=np.frombuffer(data[offset:offset+(len(data)-offset)//width*width],dtype=f'<f{width}')
            if len(a)<3:continue
            triples=np.lib.stride_tricks.sliding_window_view(a,3)
            keep=np.isfinite(triples).all(1)&(triples>=bbox[:3]-1e-5).all(1)&(triples<=bbox[3:]+1e-5).all(1)
            pieces.append(triples[keep].astype(float))
    return np.unique(np.round(np.concatenate(pieces),12),axis=0) if pieces else np.empty((0,3))


def extract_registration(mesh_dir=None):
    """One offline extraction; normal scene loading uses compact config.json."""
    mesh_dir=Path(mesh_dir or ASSETS/'meshes')
    audit=json.loads(AUDIT.read_text());models={m['id']:m for m in audit['models']}
    refs=audit['recursive_components_relative_to_board']
    by_model={}
    for r in refs:by_model.setdefault(r['model_id'],[]).append(r)
    native_points={};streams_cache={}
    with zipfile.ZipFile(ASSETS/'cad.zip') as archive:
        names={x.lower():x for x in archive.namelist()}
        def model_streams(mid):
            model=models[mid];filename=model['file_reference']['swPath'].replace('\\','/').split('/')[-1]
            member=names.get(('taskboard_1_cad/'+filename).lower(),names.get(filename.lower()))
            if member is None:raise ValueError(f'Native file missing: {filename}')
            if member not in streams_cache:streams_cache[member]=modern_streams(archive.read(member))[0]
            return member,streams_cache[member]
        def points(mid):
            if mid in native_points:return native_points[mid]
            model=models[mid]
            if model['references']:
                pieces=[]
                for child in model['references']:
                    try:part_points=points(child['model_id'])
                    except ValueError as error:
                        if 'Native file missing:' not in str(error):raise
                        continue
                    pieces.append(_transform(part_points,child['transform_child_native_to_parent_m']))
                value=np.unique(np.round(np.concatenate(pieces),12),axis=0)
            else:
                _,streams=model_streams(mid)
                value=display_coordinate_candidates(streams.get('Contents/DisplayLists',b''),model['native_bbox_m'])
            native_points[mid]=value
            return value
        registrations={}
        for name,mid in MODELS.items():
            mesh=mesh_dir/f'{name}_m.stl';vertices,_=read_native_mesh(mesh)
            original=mesh_dir/f'{name}.STL'
            if name.endswith('_Screw'):
                diameter=int(name.split('_')[0][1:])*.001
                head=vertices[np.linalg.norm(vertices[:,:2],axis=1)>diameter*.525]
                registrations[name]=dict(model_id=mid,native_name=models[mid]['name'],
                    native_to_mesh_m=None,status='existing_STL_axis_and_head_shoulder_datums',
                    reason='Native cosmetic-thread envelope is not the triangulated manufacturing solid. Mount directly from the existing STL shank axis and measured head shoulder.',
                    mesh_path=str(mesh.relative_to(ROOT)),mesh_sha256=_hash(mesh),
                    shoulder_z_mesh_m=float(head[:,2].max()),head_top_z_mesh_m=float(head[:,2].min()),
                    shaft_direction_mesh=[0.,0.,1.],native_references=by_model[mid])
                print(name,'uses measured STL shank/head shoulder datum')
                continue
            p=points(mid);bbox=np.array(models[mid]['native_bbox_m']);center=(bbox[:3]+bbox[3:])/2
            native_ref=np.array(by_model[mid][0]['transform_component_native_to_centered_board_m'])
            local_export=name.startswith(('KET','RGOCG','Waterproof','DSUB')) or name=='GMC_Laser_Plate_Virtual'
            S=np.eye(3) if local_export else np.diag([1.,-1.,-1.])
            preferred=np.eye(3) if local_export else S@native_ref[:3,:3]
            candidates=[preferred,np.eye(3),*[r for r in _rotations()],native_ref[:3,:3],S@native_ref[:3,:3]]
            unique=[]
            for R in candidates:
                if not any(np.allclose(R,x,atol=1e-10) for x in unique):unique.append(R)
            tree=cKDTree(vertices);records=[]
            mesh_center=(vertices.min(0)+vertices.max(0))/2
            for i,R in enumerate(unique):
                translation=mesh_center-R@center
                projected=p@R.T+translation
                distance,indices=tree.query(projected)
                good=distance<=2e-6
                matched=np.unique(indices[good])
                ranges=np.ptp(projected[good],axis=0) if good.any() else np.zeros(3)
                records.append((len(matched),int(good.sum()),i,R,translation,good,distance,indices,ranges))
            best=max(records,key=lambda x:(x[0],x[1],-x[2]))
            # Exact export/symmetry alternatives are preserved, not silently
            # called observed orientation. Prefer known export convention only
            # when supported by almost all the best correspondence evidence.
            expected=records[0]
            if expected[0]>=.98*best[0] and expected[0]>=8:best=expected
            matched_count,native_count,which,R,t,good,dist,idx,span=best
            point_rank=np.linalg.matrix_rank(p[good]-p[good].mean(0),tol=1e-6) if good.any() else 0
            if matched_count<4 or point_rank<2:
                if name in PARTS:
                    registrations[name]=dict(model_id=mid,native_name=models[mid]['name'],
                        native_to_mesh_m=None,status=('native_cached_geometry_revision_mismatch'
                            if name=='KET12_Square_12mm' else 'native_cached_registration_unavailable'),
                        reason=('Native part display geometry is 12x12x50mm; official STL and assembly bbox are12x8x50mm.' if name=='KET12_Square_12mm' else 'Cached native geometry does not supply sufficient exact native/STL vertex matches; target must use explicitly labeled STL geometric datums rather than an assumed rigid registration.'),
                        mesh_path=str(mesh.relative_to(ROOT)),mesh_sha256=_hash(mesh),native_references=by_model[mid])
                    print(name,'native registration unavailable; use explicit STL task datums')
                    continue
                raise ValueError(f'{name}: insufficient spatial native/STL correspondence ({matched_count})')
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
            member,_=model_streams(mid)
            alternatives=[dict(native_to_mesh_rotation=x[3].tolist(),matching_STL_vertices=x[0]) for x in records
                          if x[0]>=.98*matched_count and x[0]>=4]
            witness=np.flatnonzero(good)
            # Spread evidence across the full retained set, not just first bytes.
            witness=witness[np.linspace(0,len(witness)-1,min(16,len(witness))).astype(int)]
            registrations[name]=dict(model_id=mid,native_name=models[mid]['name'],native_archive_member=member,
                native_to_mesh_m=T.tolist(),mesh_path=str(mesh.relative_to(ROOT)),mesh_sha256=_hash(mesh),
                original_STL_sha256=_hash(original) if original.is_file() else None,
                native_display_candidates=len(p),matching_STL_vertices=matched_count,
                matching_native_coordinate_triples=native_count,match_tolerance_m=2e-6,
                matched_spatial_span_m=span.tolist(),matched_point_rank=int(point_rank),
                witness_native_points_m=p[witness].tolist(),witness_mesh_points_m=vertices[idx[witness]].tolist(),
                witness_max_error_m=float(dist[witness].max()),
                supported_rotation_alternatives=alternatives,
                supported_rotation_alternative_count=len(alternatives),
                supported_native_to_mesh_rotation_alternatives=[x['native_to_mesh_rotation'] for x in alternatives],
                mapping_note='Candidate frames use native/export metadata and bounds; accepted map independently matches spatial native display-geometry coordinate triples to existing STL vertices. Native B-rep topology is not decoded.',
                native_references=by_model[mid])
            print(name,matched_count,'matched STL vertices; alternatives',len(alternatives))
    return dict(schema=1,units='m',CAD_archive_sha256=_hash(ASSETS/'cad.zip'),
        STL_archive_sha256=_hash(ASSETS/'stl.zip'),
        board_size_m=BOARD_SIZE_M.tolist(),world_T_board=WORLD_T_BOARD.tolist(),
        native_board_to_working_board=BOARD_NATIVE_TO_WORKING.tolist(),
        board_T_plate_mesh=BOARD_NATIVE_TO_WORKING.tolist(),
        board_top_board_m=float(BOARD_SIZE_M[2]/2),support_z_board_m=float(-BOARD_SIZE_M[2]/2-.020),
        frame_note='Board origin is plate bbox center. Rx(pi) rotates native working face -Z to board +Z and flips Y consistently, including plate mesh and all fixtures.',
        registrations=registrations)


def load_native_registration():
    return json.loads((HERE/'config.json').read_text())


def _board_mesh_transform(registration,record,reference=None):
    ref=reference or record['native_references'][0]
    return (np.array(registration['native_board_to_working_board'])@
            np.array(ref['transform_component_native_to_centered_board_m'])@
            np.linalg.inv(np.array(record['native_to_mesh_m'])))


def fixture_transforms(mesh_dir=None):
    """Existing converted STL meshes, centered board frame with +Z working face."""
    data=load_native_registration();mesh_dir=Path(mesh_dir or ASSETS/'meshes');count={};out=[]
    for index,name in enumerate(FIXTURE_ORDER):
        record=data['registrations'][name];occurrence=count.get(name,0);count[name]=occurrence+1
        ref=record['native_references'][occurrence]
        source_T=None
        if name.endswith('_Screw'):
            T=np.eye(4)
            native=np.array(ref['transform_component_native_to_centered_board_m'])
            T[:2,3]=(BOARD_NATIVE_TO_WORKING@native)[:2,3]
            T[2,3]=-BOARD_SIZE_M[2]/2-record['shoulder_z_mesh_m']
            correction=dict(method='existing STL head shoulder meets board underside',
                measured_head_shoulder_mesh_z_m=record['shoulder_z_mesh_m'],
                target_head_shoulder_board_z_m=float(-BOARD_SIZE_M[2]/2))
        else:
            T=_board_mesh_transform(data,record,ref);source_T=T.copy()
            correction={}
            if name in ('Waterproof_Female','DSUB_Female','DB_Housing'):
                T[1,3]+=.00025
                correction=dict(native_board_Y_shift_m=-.00025,
                    working_board_Y_shift_m=.00025,
                    reason='Remove saved assembly 0.25mm Y offset against the plate mounting-hole pattern')
        out.append(dict(name=name,instance=f'nist_fixture_{index}',board_T_mesh=T.tolist(),
            mesh_path=str(mesh_dir/f'{name}_m.stl'),source='official saved native assembly placement with geometry-verified native/STL map',
            mesh_sha256=record['mesh_sha256'],source_board_T_mesh=None if source_T is None else source_T.tolist(),
            applied_mount_correction=correction,reference_path=ref['path'],manufactured_fit_verified=False))
    return out


corrected_fixture_specs=fixture_transforms


def _paired_export_relation(part,receiver,mesh_dir):
    """Catalog pairs with independently inspected coaxial source geometry."""
    centers=[]
    for name in (part,receiver):
        v,_=read_native_mesh(mesh_dir/f'{name}.STL')
        centers.append((v.min(0)+v.max(0))/2)
    T=np.eye(4);T[:3,3]=np.diag([1.,-1.,-1.])@(centers[0]-centers[1])*.001
    return T


def _frame(origin,direction):
    z=np.asarray(direction,float);z/=np.linalg.norm(z)
    x=np.eye(3)[np.argmin(np.abs(z))];x-=z*(x@z);x/=np.linalg.norm(x)
    T=np.eye(4);T[:3,:3]=np.column_stack((x,np.cross(z,x),z));T[:3,3]=origin
    return T


def _protected_bore(vertices,faces,direction):
    """Protect actual inner bore/thread and seating triangles, no ray sampling."""
    from experiments.cad_grasp_selection_20261028.task_geometry import _bore_lip_centers
    tri=vertices[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    norm=np.linalg.norm(cross,axis=1);normal=cross/np.maximum(norm[:,None],1e-30)
    axis=int(np.argmax(np.abs(direction)));center=(vertices.min(0)+vertices.max(0))/2
    lips=_bore_lip_centers(vertices,faces,tri,axis,center)
    transverse=[i for i in range(3) if i!=axis]
    if len(lips)!=2:raise ValueError('Two inner bore-mouth loops are required')
    center[transverse]=np.mean([x['center_transverse_m'] for x in lips],axis=0)
    radius=max(x['radius_m'] for x in lips)
    radial=tri.mean(1)-center;radial[:,axis]=0
    # Thread roots lie outside their mouth radius. A conservative envelope
    # halfway to the exterior separates inner bore from outer teeth/hex sides.
    outer=np.min(np.maximum(np.abs(vertices[:,transverse].min(0)-center[transverse]),
                            np.abs(vertices[:,transverse].max(0)-center[transverse])))
    limit=(radius+outer)/2
    bore=(np.einsum('ij,ij->i',radial,normal)<0)&(np.linalg.norm(radial,axis=1)<=limit)
    level=vertices[:,axis].min() if direction[axis]<0 else vertices[:,axis].max()
    seat=np.all(np.abs(tri[:,:,axis]-level)<1e-7,axis=1)&(normal@direction>.99)
    ids=np.flatnonzero(bore|seat)
    if not bore.any() or not seat.any():raise ValueError('Missing bore or seating surfaces')
    return center,dict(type='face_ids',protected_face_ids=ids.tolist()),dict(
        method='two circular bore-mouth datums; inward radial bore/thread triangles and terminal seating plane',
        bore_mouth_fits=lips,protected_bore_faces=int(bore.sum()),protected_seat_faces=int(seat.sum()),
        bore_radial_envelope_m=float(limit),seat_plane_coordinate_m=float(level))


def build_corrected_native_goals(pin_depth_m=PIN_DEPTH_M,mesh_dir=None):
    """Build static geometric task inputs; never sample or execute grasps."""
    if not np.isfinite(pin_depth_m) or pin_depth_m<=0:raise ValueError('Positive metric pin depth required')
    from simulation.nist_board_geometry import prepare_finished_board
    from experiments.cad_grasp_selection_20261028.task_geometry import _mesh,_connector_geometry
    from experiments.nist_mating_goals_20261028.connectors import frozen_usb_mapping,USB_FROZEN
    data=load_native_registration();mesh_dir=Path(mesh_dir or ASSETS/'meshes')
    fixtures=fixture_transforms(mesh_dir);byname={}
    for f in fixtures:byname.setdefault(f['name'],[]).append(f)
    plate,plate_receipt=prepare_finished_board(mesh_dir)
    board_receiver=dict(name='board',mesh_path=str(plate.relative_to(ROOT)),mesh_sha256=_hash(plate),
                        board_T_mesh=data['board_T_plate_mesh'],role='finished plate')
    def receiver(f):return dict(name=f['instance']+'_'+f['name'],
        mesh_path=Path(f['mesh_path']).relative_to(ROOT).as_posix(),mesh_sha256=f['mesh_sha256'],
        board_T_mesh=f['board_T_mesh'],role='intended mating receiver')
    rows=[]
    for part in PARTS:
        path=mesh_dir/f'{part}_m.stl';v,f=read_native_mesh(path)
        record=data['registrations'][part];native_status=record.get('status','verified_display_geometry_correspondence')
        evidence={};associated=[];corrections=[];status='nominal_CAD_goal_geometry_only'
        if part in PIN_HOLES_MM:
            longaxis=int(np.argmax(np.ptp(v,axis=0)))
            if longaxis!=2:raise ValueError('Current pin STL must have its long CAD axis Z')
            tip=float(v[:,2].max());length=float(np.ptp(v[:,2]))
            T=BOARD_NATIVE_TO_WORKING.copy()
            if part=='KET12_Square_12mm':
                Rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
                T[:3,:3]=BOARD_NATIVE_TO_WORKING[:3,:3]@Rz
            native_xy=(np.array(PIN_HOLES_MM[part])-192)*.001
            T[:2,3]=(BOARD_NATIVE_TO_WORKING[:2,:2]@native_xy)
            T[2,3]=BOARD_SIZE_M[2]/2-pin_depth_m+tip
            axis=np.array([0.,0.,1.]);point=(v.min(0)+v.max(0))/2
            # Protect exactly the part volume below the corrected top plane.
            normal=-T[2,:3];offset=BOARD_SIZE_M[2]/2-T[2,3]
            protected=dict(type='halfspace',normal_cad=normal.tolist(),offset_m=float(offset),
                inequality='normal@point+offset>=0',label='20mm inserted end, including chamfer')
            measured=BOARD_SIZE_M[2]/2-float(_transform(v,T)[:,2].min())
            if abs(measured-pin_depth_m)>1e-9:raise ValueError('Pin tip depth reconstruction failed')
            evidence=dict(method='existing STL longitudinal axis, positive-Z chamfered tip, and official designated hole axes',
                source_hole_xy_mm=PIN_HOLES_MM[part],pin_length_m=length,
                pin_tip_cad_z_m=tip,measured_tip_depth_below_board_top_m=measured,
                rectangular_yaw='exact hole alignment; removed small saved CAD pin yaw offsets',
                native_cached_registration_status=native_status)
        else:
            if part=='Gear_Large':
                mate=byname['Gear_Shaft'][0]
                T=np.array(mate['board_T_mesh'])@_paired_export_relation(part,'Gear_Shaft',mesh_dir)
                associated=[receiver(mate),receiver(byname['Gear_Plate'][0])]
                evidence['goal_source']='inspected original gear/shaft paired export, composed with geometry-registered native shaft placement'
            elif part=='M16_Hex_Nut':
                mate=byname['M16_Screw'][0]
                T=np.array(mate['board_T_mesh'])@_paired_export_relation(part,'M16_Screw',mesh_dir)
                associated=[receiver(mate)]
                status='nominal_thread_reference_phase_and_engagement_unverified'
                evidence['goal_source']='original nut/screw paired export with corrected STL head-shoulder bolt mount'
                evidence['thread_warning']='Static source phase retained. Thread phase/rigid mesh overlap is not resolved or validated as a physical threaded assembly.'
            elif part=='USB_Male':
                mate=byname['USB_Female'][0]
                T=np.array(mate['board_T_mesh'])@_paired_export_relation(part,'USB_Female',mesh_dir)
                mapping,mapping_receipt=frozen_usb_mapping()
                T=T@np.linalg.inv(mapping);path=USB_FROZEN;v,f=read_native_mesh(path)
                evidence['catalog_to_current_CAD_mapping']=mapping_receipt
                evidence['goal_source']='inspected catalog USB male/female source relation; exact catalog-to-current frozen connector mapping'
                associated=[receiver(mate)]
            else:
                T=_board_mesh_transform(data,record)
                family=part.replace('_Male','_Female');mate=byname[family][0]
                if part in ('DSUB_Male','Waterproof_Male'):
                    T[1,3]+=.00025
                    corrections.append(dict(working_board_Y_shift_m=.00025,native_Y_shift_m=-.00025,
                        reason='Same correction as its receiver mounting footprint'))
                associated=[receiver(mate)]
                if part=='DSUB_Male':associated.append(receiver(byname['DB_Housing'][0]))
                evidence['goal_source']='official saved male assembly placement composed with independently matched native/STL geometry'
            desired=np.array([0.,0.,-1.]);axis=T[:3,:3].T@desired
            # All current part insertion axes are aligned to a CAD coordinate
            # axis. Preserve a numerical failure rather than silently tilt it.
            index=int(np.argmax(np.abs(axis)));canonical=np.zeros(3);canonical[index]=np.sign(axis[index])
            if np.linalg.norm(axis-canonical)>1e-5:raise ValueError(f'{part}: native goal axis is not CAD-coordinate aligned')
            axis=canonical
            if part in ('Gear_Large','M16_Hex_Nut'):
                point,protected,details=_protected_bore(v,f,axis)
                # Set the fitted CAD bore line exactly onto the measured shaft
                # axis while preserving source seated height and orientation.
                rv,rf=read_native_mesh(ROOT/associated[0]['mesh_path'])
                rx=(rv.min(0)+rv.max(0))/2
                target=_transform(rx[None],np.array(associated[0]['board_T_mesh']))[0]
                placed=_transform(point[None],T)[0]
                delta=target[:2]-placed[:2];T[:2,3]+=delta
                evidence['coaxial_translation_correction_m']=delta.tolist()
                evidence['task_geometry']=details
                actual_part=_transform(v,T)
                if part=='Gear_Large':
                    plate_spec=associated[1];pv,_=read_native_mesh(ROOT/plate_spec['mesh_path'])
                    plate_points=_transform(pv,np.array(plate_spec['board_T_mesh']))
                    seat_error=float(actual_part[:,2].min()-plate_points[:,2].max())
                    if abs(seat_error)>2e-6:raise ValueError('Gear seating face does not meet its actual plate surface')
                    evidence['measured_gear_to_plate_seating_gap_m']=seat_error
                else:
                    evidence['measured_nut_to_board_seating_gap_m']=float(actual_part[:,2].min()-BOARD_SIZE_M[2]/2)
            else:
                vv,ff,tri,normals,areas=_mesh(v,f)
                point,feature_axis,_,details=_connector_geometry(part,vv,ff,tri,normals,areas)
                if float(feature_axis@axis)<.999:
                    raise ValueError(f'{part}: native goal direction disagrees with independently measured mating shoulder/open end')
                envelope=details['protected_region']
                protected=dict(type='halfspace',normal_cad=envelope['normal'],offset_m=envelope['offset_m'],
                    inequality='normal@point+offset>=0',label='measured mating shoulder toward plug end')
                evidence['task_geometry']=details
        direction=T[:3,:3]@axis
        mating=T@_frame(point,axis)
        allreceivers=[board_receiver,*associated]
        rows.append(dict(part=part,part_mesh=path.relative_to(ROOT).as_posix(),part_mesh_sha256=_hash(path),
            board_T_part=T.tolist(),review_board_T_part=T.tolist(),world_T_cad_final=(WORLD_T_BOARD@T).tolist(),
            axis_point_cad_m=np.asarray(point).tolist(),axis_direction_cad=axis.tolist(),
            axis_direction_board=direction.tolist(),part_mating_frame_cad=_frame(point,axis).tolist(),
            receiver_mating_frame_board=mating.tolist(),protected_contact_region=protected,
            receivers=allreceivers,status=status,geometry_only=True,
            pin_tip_depth_below_board_top_m=pin_depth_m if part in PIN_HOLES_MM else None,
            native_registration_status=native_status,applied_mount_corrections=corrections,evidence=evidence))
    return dict(schema=1,units='m',world_T_board=WORLD_T_BOARD.tolist(),
        board_size_m=BOARD_SIZE_M.tolist(),board_top_board_m=float(BOARD_SIZE_M[2]/2),
        support_z_board_m=float(-BOARD_SIZE_M[2]/2-.020),board_T_plate_mesh=BOARD_NATIVE_TO_WORKING.tolist(),
        pin_tip_depth_below_board_top_m=float(pin_depth_m),parts=rows,
        fixture_transforms=fixtures,finished_board_receipt=plate_receipt,
        scope='static corrected CAD geometry inputs; no grasp generation, VLM, pickup or assembly execution',
        CAD_archive_sha256=data['CAD_archive_sha256'],STL_archive_sha256=data['STL_archive_sha256'])


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if '--goals' in sys.argv:
        goals=build_corrected_native_goals()
        (OUTPUT/'assembly_goals.json').write_text(json.dumps(goals,indent=2)+'\n')
        print('saved',len(goals['parts']),'static CAD goals')
    else:
        data=extract_registration()
        (OUTPUT/'native_registration.json').write_text(json.dumps(data,indent=2)+'\n')
        compact=json.loads(json.dumps(data))
        for record in compact['registrations'].values():
            for key in ['witness_native_points_m','witness_mesh_points_m','supported_rotation_alternatives','mapping_note']:
                record.pop(key,None)
        (HERE/'config.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
