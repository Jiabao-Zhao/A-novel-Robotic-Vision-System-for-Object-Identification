"""White 3-by-4 CAD contact-pair sheets, with no gripper or comparison panels.

Lines join original sampled surface pairs of retained family representatives.
Independent finite-pad contact witnesses are never connected here.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.cad_grasp_geometry_20261026 import gallery as previous

ROOT = previous.ROOT
BASE = ROOT/'outputs/cad_grasp_method_comparison_20261026/contact_atlas'
CAPTION = ('Lines show sampled surface pairs for retained poses; height is not shown. '
           'VLM selection pending.')
ANGLE_TOLERANCE_DEG = 1e-5
LINE_OFFSET_TOLERANCE_M = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pair(grasp):
    pair = np.asarray(grasp['contacts_world_m'],dtype=float)
    pose = np.asarray(grasp['world_T_grip'],dtype=float)
    if pair.shape!=(2,3) or pose.shape!=(4,4) or not np.isfinite(pair).all() or not np.isfinite(pose).all():
        raise ValueError('A source pair or gripper pose is missing or nonfinite')
    difference = pair[1,:2]-pair[0,:2]
    axis = pose[:2,0]
    norm = np.linalg.norm(difference)
    if norm<1e-9 or np.linalg.norm(axis)<1e-9:
        raise ValueError('A retained pair has no visible top-down closing direction')
    direction = difference/norm
    axis = axis/np.linalg.norm(axis)
    cross = float(direction[0]*axis[1]-direction[1]*axis[0])
    angle = float(np.degrees(np.arcsin(np.clip(abs(cross),0,1))))
    offset = pose[:2,3]-pair[:,:2].mean(axis=0)
    distance = abs(float(offset[0]*axis[1]-offset[1]*axis[0]))
    if angle>ANGLE_TOLERANCE_DEG or distance>LINE_OFFSET_TOLERANCE_M:
        raise ValueError(f'Source pair does not lie on projected hand closing axis: {angle} deg, {distance} m')
    return dict(angle_deg=angle,line_offset_m=distance)


def validate_records(records):
    if len(records)!=96:
        raise ValueError('The atlas requires all 96 saved placements, including zeros')
    scenes = sorted({r['scene'] for r in records})
    parts = sorted({r['part'] for r in records})
    if scenes!=[f'N{i:02d}' for i in range(8)] or len(parts)!=12:
        raise ValueError('Expected N00-N07 with the same twelve CAD parts')
    if len({(r['scene'],r['part']) for r in records})!=96:
        raise ValueError('A scene/part placement is repeated or missing')
    checks = []
    for record in records:
        for grasp in record['grasps']:
            if not grasp.get('valid',True):
                raise ValueError('Rejected poses may not be shown as retained')
            alternatives = grasp.get('jaw_aliases') or [grasp]
            for candidate in alternatives:
                contacts = candidate.get('actual_contacts',[])
                if len(contacts)!=2 or any(c['normal_alignment']<np.cos(np.deg2rad(1.))-1e-10 for c in contacts):
                    raise ValueError('The atlas requires actual native contact normals aligned within 1 degree')
            checks.append(dict(scene=record['scene'],part=record['part'],
                family_id=grasp['family_id'],id=grasp['id'],**check_pair(grasp)))
    return scenes,parts,checks


def render(records,meshes,scenes,parts,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    lookup = {(r['scene'],r['part']):r for r in records}
    inventory = []
    for scene in scenes:
        fig,axes = plt.subplots(3,4,figsize=(16,11),constrained_layout=True)
        for ax,part in zip(axes.flat,parts):
            record = lookup[(scene,part)]
            vertices,faces = meshes[part]
            pose = np.asarray(record['world_T_cad'])
            vertices = vertices@pose[:3,:3].T+pose[:3,3]
            lower,upper = vertices.min(axis=0),vertices.max(axis=0)
            center = (lower+upper)/2
            points = (vertices-center)*1000.
            ax.add_collection(PolyCollection(points[faces,:2],facecolors='#cbd2d8',
                edgecolors='none',antialiaseds=False))
            grasps = record['grasps']
            colors = plt.get_cmap('jet')(np.linspace(.12,.92,len(grasps))) if grasps else []
            for grasp,color in zip(grasps,colors):
                pair = (np.asarray(grasp['contacts_world_m'])-center)*1000.
                ax.plot(pair[:,0],pair[:,1],color=color,lw=.7,alpha=.95,zorder=4)
                ax.scatter(pair[:,0],pair[:,1],color=[color],s=8,edgecolors='none',zorder=5)
            radius = max(float(np.ptp(points[:,:2],axis=0).max())*.62,1.)
            ax.set_xlim(-radius,radius)
            ax.set_ylim(-radius,radius)
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(part.replace('_',' ')+f'\n{len(grasps)} retained '+('pose' if len(grasps)==1 else 'poses'),
                fontsize=10,color='#b33d32' if not grasps else '#263846',pad=7)
        fig.suptitle(scene+'\n'+CAPTION,fontsize=12)
        path = output/(scene+'.png')
        fig.savefig(path,dpi=120,facecolor='white')
        plt.close(fig)
        inventory.append(dict(scene=scene,image=path.name,tiles=12,
            retained_poses=sum(len(lookup[(scene,p)]['grasps']) for p in parts),
            empty_tiles=sum(not lookup[(scene,p)]['grasps'] for p in parts)))
    return inventory


def main(results_path):
    source = previous.local_path(results_path)
    protected = [source,ROOT/'docs/cad_view_prompt.txt',
        ROOT/'outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/results.json',
        ROOT/'outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z/gallery/index.html']
    if (source.parent/'manifest.json').exists():
        protected.append(source.parent/'manifest.json')
    protected.extend(sorted((ROOT/'outputs/cad_grasp_method_comparison_20261026/retained_pose_gallery').glob('*/index.html')))
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    records = json.loads(source.read_text())
    scenes,parts,checks = validate_records(records)
    meshes = {}
    mesh_hashes = {}
    for part in parts:
        record = next(r for r in records if r['part']==part)
        path = previous.local_path(record['mesh_path'])
        meshes[part] = previous.read_stl(path)
        mesh_hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha(path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/stamp
    output.mkdir(parents=True,exist_ok=False)
    print('OUTPUT='+str(output),flush=True)
    inventory = render(records,meshes,scenes,parts,output)
    (output/'index.html').write_text(HTML,encoding='utf-8')
    (output/'verify_viewer.cjs').write_text(VERIFY,encoding='utf-8')
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    audit = dict(source_results=str(source.relative_to(ROOT)),source_manifest=str((source.parent/'manifest.json').relative_to(ROOT)),
        default_filter='1 degree actual native contact-normal alignment',
        scenes=scenes,parts=parts,placements=len(records),
        retained_families=len(checks),empty_placements=sum(not r['grasps'] for r in records),
        no_arbitrary_pose_cap=True,no_ranking=True,no_omitted_empty_tiles=True,
        line_endpoints='original contacts_world_m source surface pair, never independent actual first-contact witnesses',
        top_down_pair_parallel_to_gripper_axis=True,
        top_down_pair_line_contains_gripper_reference_after_x_only_centering=True,
        maximum_pair_axis_error_deg=max((c['angle_deg'] for c in checks),default=0.),
        maximum_pair_axis_line_offset_m=max((c['line_offset_m'] for c in checks),default=0.),
        angle_tolerance_deg=ANGLE_TOLERANCE_DEG,line_offset_tolerance_m=LINE_OFFSET_TOLERANCE_M,
        native_CAD_mesh_hashes=mesh_hashes,no_CAD_simplification=True,
        no_gripper_or_cameras_or_table=True,no_metrics_or_comparison_panels=True,
        no_candidate_generation=True,no_VLM_selection=True,no_physics_or_robot_execution=True,
        source_sha256=sha(__file__),display_helper_sha256=sha(previous.__file__),
        protected_unchanged=before==after,protected_sha256=after,images=inventory)
    (output/'atlas_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'pair_axis_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if before!=after:
        raise ValueError('Protected evidence changed while making the atlas')
    print(json.dumps(dict(output=str(output),scenes=8,placements=len(records),retained_families=len(checks),
        protected_unchanged=before==after)),flush=True)


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retained contact pairs</title>
<style>:root{font:15px system-ui;color:#263846;background:white}body{margin:0}main{max-width:1600px;margin:18px auto;padding:0 18px}label{display:flex;gap:10px;align-items:center;font-weight:600}select{font:inherit;padding:7px 12px;border:1px solid #aebbc5;border-radius:4px;background:white}img{display:block;width:100%;height:auto;margin-top:8px}</style>
<main><label>Scene<select id="scene"><option>N00</option><option>N01</option><option>N02</option><option>N03</option><option>N04</option><option>N05</option><option>N06</option><option>N07</option></select></label><img id="sheet" src="N00.png" alt="N00: twelve top-down CAD silhouettes with retained sampled surface pairs"></main>
<script>const scene=document.getElementById('scene'),sheet=document.getElementById('sheet');function showScene(){sheet.src=scene.value+'.png';sheet.alt=scene.value+': twelve top-down CAD silhouettes with retained sampled surface pairs'}scene.onchange=showScene;showScene();</script></html>'''


VERIFY = r'''const fs=require('fs'),path=require('path'),vm=require('vm');const file=process.argv[2],html=fs.readFileSync(file,'utf8');const elements={scene:{value:'N00'},sheet:{}};const sandbox={document:{getElementById:id=>elements[id]}};vm.createContext(sandbox);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],sandbox);const inventory=[];for(let i=0;i<8;i++){const name='N'+String(i).padStart(2,'0');elements.scene.value=name;elements.scene.onchange();if(elements.sheet.src!==name+'.png'||!elements.sheet.alt.startsWith(name))throw Error('Wrong scene sheet');const image=path.join(path.dirname(file),elements.sheet.src);const bytes=fs.readFileSync(image);if(bytes.subarray(1,4).toString()!=='PNG')throw Error('Missing PNG');inventory.push(name)}const audit=JSON.parse(fs.readFileSync(path.join(path.dirname(file),'atlas_audit.json')));if(audit.placements!==96||audit.images.length!==8||audit.images.some(i=>i.tiles!==12))throw Error('Incomplete atlas');const result={scene_selector:'passed',all_scene_images:'passed',scenes:inventory,placements:96,tiles_per_scene:12,retained_families:audit.retained_families,source_pair_axis_checks:'passed'};fs.writeFileSync(path.join(path.dirname(file),'viewer_verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));'''


if __name__=='__main__':
    main(sys.argv[1])
