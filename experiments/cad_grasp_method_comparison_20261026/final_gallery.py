"""Minimal retained-pose display; preserves all frozen experiment evidence."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026 import gallery as previous

ROOT = previous.ROOT
RUN = ROOT/'outputs/cad_grasp_method_comparison_20261026/comparison/20261026T203838Z'
BASE = ROOT/'outputs/cad_grasp_method_comparison_20261026/retained_pose_gallery'
METHOD = 'one_budget512'
PROTOCOL = 'panda_table_proj15'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand_model():
    hand = json.loads((ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    for component in hand['components']:
        vertices = np.asarray(component['vertices_zero_m'])
        hull = ConvexHull(vertices)
        used = np.unique(hull.simplices)
        mapping = np.full(len(vertices),-1,dtype=int)
        mapping[used] = np.arange(len(used))
        component['vertices_zero_m'] = vertices[used].tolist()
        component['triangles'] = mapping[hull.simplices].tolist()
    return hand


def records_for_display(raw,hand):
    records = []
    excluded = []
    for record in raw:
        if record['method']!=METHOD:
            continue
        if not record['grasps']:
            excluded.append(dict(scene=record['scene'],part=record['part'],status=record['status']))
            continue
        grasps = []
        for grasp in record['grasps']:
            if PROTOCOL not in grasp['protocols'] or not grasp['valid']:
                raise ValueError('Only retained primary family representatives may be displayed')
            grasps.append(dict(id=grasp['id'],family_id=grasp['family_id'],
                world_T_grip=grasp['world_T_grip'],
                contact_opening_m=previous.closure_opening(grasp,hand)))
        records.append(dict(scene=record['scene'],part=record['part'],
            world_T_cad=record['world_T_cad'],grasps=grasps))
    records.sort(key=lambda r:(r['part'],r['scene']))
    if len(records)!=95 or len({r['part'] for r in records})!=12:
        raise ValueError('Unexpected retained-pose scope; expected 95 placements of 12 parts')
    if len(excluded)!=1 or excluded[0]['scene']!='N06' or excluded[0]['part']!='RGOCG8-50_8mm':
        raise ValueError('Unexpected unavailable placement')
    return records,excluded


def preview(records,meshes,hand,output):
    """Static 3D review of retained examples; absent from the minimal viewer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    chosen = ['BNC_Male','Gear_Large','KET16_Square_16mm','RGOCG8-50_8mm']
    fig = plt.figure(figsize=(12,11),constrained_layout=True)
    for index,part in enumerate(chosen,1):
        record = next(r for r in records if r['part']==part and r['scene']=='N00')
        grasp = record['grasps'][0]
        pose = np.asarray(record['world_T_cad'])
        cad = np.asarray(meshes[part]['vertices'])@pose[:3,:3].T+pose[:3,3]
        objects = [(cad,np.asarray(meshes[part]['triangles']),'#a6b0b8',.98)]
        pose = np.asarray(grasp['world_T_grip'])
        for component in hand['components']:
            if component['name'].endswith('hand_collision'):
                continue
            vertices = np.asarray(component['vertices_zero_m'])+grasp['contact_opening_m']*np.asarray(component['translation_per_opening'])
            vertices = vertices@pose[:3,:3].T+pose[:3,3]
            objects.append((vertices,np.asarray(component['triangles']),
                '#d89436' if component['is_pad'] else '#377fac',
                .13 if component['name'].endswith('hand_collision') else 1.))
        all_vertices = np.concatenate([v for v,_,_,_ in objects])
        middle = (all_vertices.min(axis=0)+all_vertices.max(axis=0))/2
        radius = max(np.ptp(all_vertices,axis=0))*.54
        ax = fig.add_subplot(2,2,index,projection='3d')
        for vertices,faces,color,alpha in objects:
            ax.add_collection3d(Poly3DCollection((vertices-middle)[faces],facecolors=color,
                edgecolors='none',alpha=alpha))
        ax.set(xlim=(-radius,radius),ylim=(-radius,radius),zlim=(-radius,radius))
        ax.view_init(elev=55,azim=-55)
        ax.set_box_aspect((1,1,1))
        ax.set_axis_off()
        ax.set_title(part.replace('_',' ')+' / '+record['scene'],fontsize=13)
    fig.suptitle('Retained poses; VLM selection pending.',fontsize=17)
    fig.savefig(output/'preview.png',dpi=125,facecolor='white')
    plt.close(fig)


def main():
    protected = [RUN/'results.json',RUN/'manifest.json',
        ROOT/'docs/cad_view_prompt.txt',ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json',
        RUN/'gallery/index.html',RUN/'gallery/gallery_audit.json']
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw = json.loads((RUN/'results.json').read_text())
    hand = hand_model()
    records,excluded = records_for_display(raw,hand)
    mesh_paths = {r['part']:r['mesh_path'] for r in raw if r['method']==METHOD and r['grasps']}
    meshes = {part:previous.display_mesh(previous.local_path(path)) for part,path in mesh_paths.items()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/stamp
    output.mkdir(parents=True,exist_ok=False)
    payload = dict(records=records,meshes=meshes,hand=hand)
    encoded = json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (output/'index.html').write_text(HTML.replace('__DATA__',encoded),encoding='utf-8')
    (output/'verify_viewer.cjs').write_text(VERIFY,encoding='utf-8')
    preview(records,meshes,hand,output)
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    audit = dict(method=METHOD,protocol=PROTOCOL,source_run=str(RUN.relative_to(ROOT)),
        source_placements=96,displayed_placements=len(records),parts=len(meshes),
        displayed_family_representatives=sum(len(r['grasps']) for r in records),
        excluded_placements=excluded,no_jaw_alias_selector=True,
        no_other_methods=True,no_failed_case_cards=True,no_results_tables=True,
        no_metrics_or_raw_results=True,no_points_normals_arrows_or_table_plane=True,
        display_fingers_and_pads_only=True,full_hand_geometry_preserved_in_payload=True,
        no_new_candidate_generation=True,no_new_selection=True,no_vlm=True,
        no_robot_execution=True,no_assembly_views=True,
        source_sha256=sha(__file__),display_helper_sha256=sha(previous.__file__),
        protected_unchanged=before==after,protected_sha256=after)
    (output/'gallery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if before!=after:
        raise ValueError('Protected evidence changed while producing the display')
    print(json.dumps(dict(output=str(output),placements=len(records),parts=len(meshes),
        poses=audit['displayed_family_representatives'],protected_unchanged=before==after)),flush=True)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retained gripper poses</title><style>
:root{font:15px system-ui;color:#233442;background:#f5f7f9}body{margin:0}main{max-width:1600px;margin:auto;padding:20px 24px}h1{font-size:23px;margin:0 0 7px}p{margin:0 0 18px;color:#566b7a}.controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin-bottom:12px}label{font-size:13px;font-weight:600;display:flex;flex-direction:column;gap:4px}select,button{font:inherit;background:white;padding:8px 10px;border:1px solid #adbdc9;border-radius:5px}button{cursor:pointer}canvas{width:100%;height:calc(100vh - 186px);min-height:480px;display:block;background:white;border:1px solid #dce4ea;border-radius:8px;touch-action:none;cursor:grab}@media(max-width:800px){main{padding:16px}canvas{height:65vh;min-height:400px}.controls{gap:8px}select,button{padding:7px}}
</style><main><h1>Retained gripper poses</h1><p>Retained poses; VLM selection pending.</p><div class="controls"><label>Part<select id="part"></select></label><label>Scene<select id="scene"></select></label><label>Pose<select id="pose"></select></label><button id="prev">Previous</button><button id="next">Next</button><button id="top">Top</button><button id="side">Side</button><button id="angled">Angled</button></div><canvas id="view" aria-label="Orbitable three-dimensional aligned CAD and retained gripper pose"></canvas></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function option(select,value,label){const item=document.createElement('option');item.value=value;item.textContent=label;select.append(item)}
[...new Set(D.records.map(r=>r.part))].forEach(p=>option($('part'),p,p.replaceAll('_',' ')));
let record=null,grasp=null,world=[],center=[0,0,0],span=.15,az=-.8,el=.95,zoom=1,drag=null;
const point=(p,T)=>T.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
function rebuildPart(){const previous=$('scene').value;$('scene').replaceChildren();D.records.filter(r=>r.part===$('part').value).forEach(r=>option($('scene'),r.scene,r.scene));if([...$('scene').options].some(o=>o.value===previous))$('scene').value=previous;rebuildScene()}
function rebuildScene(){record=D.records.find(r=>r.part===$('part').value&&r.scene===$('scene').value);$('pose').replaceChildren();record.grasps.forEach((g,i)=>option($('pose'),i,'Pose '+String(i+1).padStart(2,'0')));rebuildPose()}
function addMesh(vertices,triangles,color,kind,alpha=1){for(const triangle of triangles)world.push({p:triangle.map(i=>vertices[i]),color,kind,alpha})}
function rebuildPose(){grasp=record.grasps[Number($('pose').value)];world=[];const mesh=D.meshes[record.part],cad=mesh.vertices.map(p=>point(p,record.world_T_cad));addMesh(cad,mesh.triangles,'#a7b1ba','cad');let all=[...cad];for(const c of D.hand.components){if(c.name.endsWith('hand_collision'))continue;const vertices=c.vertices_zero_m.map(p=>point(p.map((v,k)=>v+grasp.contact_opening_m*c.translation_per_opening[k]),grasp.world_T_grip));addMesh(vertices,c.triangles,c.is_pad?'#d89436':'#377fac','gripper',c.name.endsWith('hand_collision')?.13:1);all.push(...vertices)}const lo=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...all.map(p=>p[k])));center=lo.map((v,k)=>(v+hi[k])/2);span=Math.max(.06,...hi.map((v,k)=>v-lo[k]));draw()}
function project(p){const q=p.map((v,k)=>v-center[k]),ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),x=ca*q[0]+sa*q[1],y=-sa*q[0]+ca*q[1];return[x,ce*q[2]-se*y,se*q[2]+ce*y]}
function draw(){const canvas=$('view'),ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);const scale=Math.min(w,h)/span*.73*zoom,screen=p=>{const q=project(p);return[w/2+scale*q[0],h/2-scale*q[1],q[2]]};const faces=world.map(f=>({...f,sp:f.p.map(screen)}));faces.sort((a,b)=>a.sp.reduce((sum,p)=>sum+p[2],0)-b.sp.reduce((sum,p)=>sum+p[2],0));for(const f of faces){ctx.beginPath();ctx.moveTo(f.sp[0][0],f.sp[0][1]);ctx.lineTo(f.sp[1][0],f.sp[1][1]);ctx.lineTo(f.sp[2][0],f.sp[2][1]);ctx.closePath();ctx.fillStyle=f.color;ctx.globalAlpha=f.alpha;ctx.fill();ctx.globalAlpha=1}}
$('part').onchange=rebuildPart;$('scene').onchange=rebuildScene;$('pose').onchange=rebuildPose;function shift(delta){const select=$('pose');select.selectedIndex=(select.selectedIndex+delta+select.options.length)%select.options.length;rebuildPose()}$('prev').onclick=()=>shift(-1);$('next').onclick=()=>shift(1);$('top').onclick=()=>{el=Math.PI/2;az=0;draw()};$('side').onclick=()=>{el=0;az=0;draw()};$('angled').onclick=()=>{el=.95;az=-.8;draw()};$('view').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('view').setPointerCapture(e.pointerId)};$('view').onpointerup=()=>drag=null;$('view').onpointermove=e=>{if(drag){az+=(e.clientX-drag[0])*.01;el=Math.max(-1.55,Math.min(1.55,el+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw()}};$('view').onwheel=e=>{e.preventDefault();zoom=Math.max(.35,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw()};window.addEventListener('resize',draw);rebuildPart();
</script></html>'''


VERIFY = r'''const fs=require('fs'),vm=require('vm'),path=require('path');
const file=process.argv[2],html=fs.readFileSync(file,'utf8'),payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1],code=html.slice(html.lastIndexOf('<script>')+8).split('</script>')[0];
const elements=new Map(),ctx=new Proxy({}, {get:(o,k)=>k in o?o[k]:(()=>{})});
class Element{constructor(tag='div'){this.tag=tag;this.children=[];this._value='';this.clientWidth=1200;this.clientHeight=800;}append(...items){this.children.push(...items);if(this.tag==='select'&&this.children.length===items.length)this._value=String(this.children[0]?.value??'');}replaceChildren(...items){this.children=[];this._value='';this.append(...items);}get value(){return this._value;}set value(v){this._value=String(v);}get options(){return this.children;}get selectedIndex(){return this.children.findIndex(c=>String(c.value)===this._value);}set selectedIndex(i){this._value=String(this.children[i]?.value??'');}getContext(){return ctx;}setPointerCapture(){}}
for(const id of ['part','scene','pose'])elements.set(id,new Element('select'));elements.set('data',new Element());elements.get('data').textContent=payload;
const sandbox={document:{getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag)},window:{devicePixelRatio:1,addEventListener(){}},console};vm.createContext(sandbox);vm.runInContext(code,sandbox,{timeout:30000});
const result=vm.runInContext(`(()=>{draw=()=>{};const assert=(ok,msg)=>{if(!ok)throw Error(msg)};let parts=0,placements=0,poses=0;for(const part of [...new Set(D.records.map(r=>r.part))]){$('part').value=part;rebuildPart();parts++;const expected=D.records.filter(r=>r.part===part);assert($('scene').options.length===expected.length,'scene count mismatch');if(part==='RGOCG8-50_8mm')assert(![...$('scene').options].some(o=>o.value==='N06'),'unavailable scene exposed');for(const r of expected){$('scene').value=r.scene;rebuildScene();placements++;assert(record===r,'wrong record');assert(record.grasps.length>0,'empty record');assert($('pose').options.length===r.grasps.length,'pose count mismatch');for(let i=0;i<r.grasps.length;i++){$('pose').value=i;rebuildPose();poses++;assert(grasp===r.grasps[i],'wrong pose');assert(world.some(f=>f.kind==='cad')&&world.some(f=>f.kind==='gripper'),'CAD or gripper missing');assert(world.every(f=>f.p.every(p=>p.every(Number.isFinite))),'nonfinite display geometry');assert(grasp.contact_opening_m>=0,'negative opening');}const before=$('pose').selectedIndex;shift(1);assert($('pose').selectedIndex===(before+1)%r.grasps.length,'next failed');shift(-1);assert($('pose').selectedIndex===before,'previous failed');for(const id of ['top','side','angled'])$(id).onclick();}}assert(parts===12&&placements===95,'wrong scope');return{parts,placements,poses,all_selectors:'passed',previous_next:'passed',view_buttons:'passed',finite_CAD_and_gripper_geometry:'passed',unavailable_scene_omitted:'passed',no_aliases_or_other_methods_in_payload:'passed'};})()`,sandbox,{timeout:120000});fs.writeFileSync(path.join(path.dirname(file),'viewer_verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
'''


if __name__=='__main__':
    main()
