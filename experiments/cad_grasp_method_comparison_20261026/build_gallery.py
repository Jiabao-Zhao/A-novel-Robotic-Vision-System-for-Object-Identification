"""Inspect nine previous-zero placements without changing comparison evidence.

Display only retained ``panda_table_proj15`` families and their valid jaw
alternatives. Source pairs, finite-pad witnesses and native triangle normals
have distinct display marks. This file never generates or ranks grasps.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026 import gallery as base
from experiments.cad_grasp_geometry_20261026 import run_trial as geometry

ROOT = base.ROOT
CASES = [('N00','BNC_Male'),('N01','BNC_Male'),('N03','BNC_Male'),
         ('N04','BNC_Male'),('N06','BNC_Male'),('N01','Gear_Large'),
         ('N03','Gear_Large'),('N05','Gear_Large'),('N07','Gear_Large')]
METHODS = ['one_default','one_budget512','graspkit_512']
LABELS = {'one_default':'one · original 20 mm spacing',
          'one_budget512':'one · controlled 512 surface samples',
          'graspkit_512':'GraspKit · 512 starting rays'}
PROTOCOL = 'panda_table_proj15'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_hand():
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


def display_record(record,reference,hand):
    item = deepcopy(record)
    item['reference_zero'] = dict(status=reference['status'],
        generated_count=reference['generated_count'],valid_count=reference['valid_count'],
        grouped_count=reference['grouped_count'],
        rejection_counts=reference.get('rejection_counts',{}),
        preparation_rejection_counts=reference.get('preparation_rejection_counts',{}))
    if reference.get('grasps') or reference.get('valid_count'):
        raise ValueError('The requested frozen reference placement is not a zero result')
    vertices,faces,_ = geometry.load_target(record)
    for family in item['grasps']:
        if PROTOCOL not in family['protocols'] or not family['valid']:
            raise ValueError('Gallery contains a non-primary or rejected family')
        for grasp in [family]+family.get('jaw_aliases',[]):
            if PROTOCOL not in grasp['protocols'] or not grasp['valid']:
                raise ValueError('Gallery contains a non-primary or rejected jaw alternative')
            grasp['display_contact_opening_m'] = base.closure_opening(grasp,hand)
            normals = []
            errors = []
            axis = np.asarray(grasp['world_T_grip'])[:3,0]
            for contact in grasp['actual_contacts']:
                triangle = vertices[faces[contact['first_contact']['triangle_index']]]
                normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
                normal /= np.linalg.norm(normal)
                alignment = abs(float(normal@axis))
                if abs(alignment-contact['normal_alignment'])>1e-6:
                    raise ValueError('Displayed normal differs from the validated native triangle normal')
                normals.append(normal.tolist())
                errors.append(float(np.degrees(np.arccos(np.clip(alignment,-1,1)))))
            grasp['display_native_normals_world'] = normals
            grasp['display_native_normal_errors_deg'] = errors
    return item


def build_review_sheets(records,meshes,hand,output):
    """Show one explicitly unranked canonical example per comparison cell."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    for part in ('BNC_Male','Gear_Large'):
        placements = [(s,p) for s,p in CASES if p==part]
        fig,axes = plt.subplots(len(placements),len(METHODS),
            figsize=(15,3.3*len(placements)),squeeze=False,constrained_layout=True)
        mesh = meshes[part]
        for row,(scene,_) in enumerate(placements):
            for column,method in enumerate(METHODS):
                ax = axes[row,column]
                record = next(r for r in records if r['scene']==scene and r['part']==part and r['method']==method)
                pose = np.asarray(record['world_T_cad'])
                cad = np.asarray(mesh['vertices'])@pose[:3,:3].T+pose[:3,3]
                middle = (cad.min(axis=0)+cad.max(axis=0))/2
                ax.add_collection(PolyCollection(((cad-middle)*1000)[np.asarray(mesh['triangles']),:2],
                    facecolors='#a5b0b9',edgecolors='none',alpha=.9))
                extent = max(np.ptp(cad[:,:2],axis=0).max()*.65,.018)
                count = len(record['grasps'])
                title = f'{scene} / {LABELS[method]}\nPrevious: 0; new: {count} families'
                if count:
                    family = record['grasps'][0]
                    grasp = next((g for g in family.get('jaw_aliases',[]) if g['id']==family['id']),family)
                    hand_pose = np.asarray(grasp['world_T_grip'])
                    opening = grasp['display_contact_opening_m']
                    for component in hand['components']:
                        if component['name'].endswith('hand_collision'):
                            continue
                        local = np.asarray(component['vertices_zero_m'])+opening*np.asarray(component['translation_per_opening'])
                        vertices = local@hand_pose[:3,:3].T+hand_pose[:3,3]
                        projected = (vertices-middle)[:,:2]*1000
                        if component['is_pad']:
                            ax.add_collection(PolyCollection(projected[np.asarray(component['triangles'])],
                                facecolors='#d89436',edgecolors='none',alpha=.9))
                        else:
                            outline = projected[ConvexHull(projected).vertices]
                            ax.plot(*np.vstack([outline,outline[0]]).T,color='#377fac',lw=1.1)
                        extent = max(extent,float(np.abs(vertices[:,:2]-middle[:2]).max()*1.06))
                    source = (np.asarray(grasp['contacts_world_m'])-middle)*1000
                    witnesses = (np.asarray(grasp['contact_points_world_m'])-middle)*1000
                    ax.plot(source[:,0],source[:,1],'o--',color='#d96b98',lw=1,ms=3)
                    axis = hand_pose[:3,0]
                    midpoint = witnesses.mean(axis=0)
                    length = max(12.,min(35.,grasp['width_m']*650))
                    start,end = midpoint-length*axis,midpoint+length*axis
                    ax.annotate('',xy=end[:2],xytext=start[:2],arrowprops=dict(arrowstyle='->',color='#333',lw=1.2))
                    for point,normal in zip(witnesses,grasp['display_native_normals_world']):
                        ax.annotate('',xy=point[:2]+6*np.asarray(normal[:2]),xytext=point[:2],
                            arrowprops=dict(arrowstyle='->',color='#16523a',lw=1.1))
                    ax.scatter(witnesses[:,0],witnesses[:,1],s=30,c='#147e58',edgecolors='white',linewidths=.5,zorder=8)
                    title += f'\nShown: {family["family_id"]} / {grasp["id"]} (unranked example)'
                else:
                    title += '\nNo passing family from this sampler setting'
                ax.set_xlim(-extent*1000,extent*1000)
                ax.set_ylim(-extent*1000,extent*1000)
                ax.set_aspect('equal')
                ax.grid(alpha=.13)
                ax.set_title(title,fontsize=10)
                ax.set_xlabel('World X relative to CAD center (mm)',fontsize=8)
                ax.set_ylabel('World Y relative to CAD center (mm)',fontsize=8)
                ax.tick_params(labelsize=8)
        legend = [Line2D([],[],color='#377fac',label='Finger outline'),
                  Line2D([],[],color='#d89436',lw=5,label='Inner pad'),
                  Line2D([],[],color='#d96b98',ls='--',marker='o',label='Source pair'),
                  Line2D([],[],color='#147e58',ls='none',marker='o',label='Actual witness dots'),
                  Line2D([],[],color='#333',label='Closing axis'),
                  Line2D([],[],color='#16523a',label='Native CAD face normals')]
        fig.legend(handles=legend,loc='outside lower center',ncol=3,fontsize=10)
        fig.suptitle(part.replace('_',' ')+': previous-zero placements, viewed from above\n'
            'One canonical example per cell. The interactive viewer includes every retained family and valid jaw alternative.\n'
            'No VLM ranking, pickup execution or assembly view. Green dots are separate witnesses, not a line or closing direction.',fontsize=12)
        fig.savefig(output/(part+'_examples.png'),dpi=110,facecolor='white')
        plt.close(fig)


def main(run_path):
    run = base.local_path(run_path)
    result_path = run/'results.json'
    manifest = json.loads((run/'manifest.json').read_text())
    raw = json.loads(result_path.read_text())
    expected_total = manifest['cases_per_method']*len(manifest['methods'])
    if len(raw)!=expected_total:
        raise ValueError(f'Comparison is still running: {len(raw)}/{expected_total} records')
    protected = [result_path,run/'manifest.json',ROOT/'docs/cad_view_prompt.txt',
                 ROOT/manifest['reference_run']/'results.json',
                 ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json']
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    reference = json.loads((ROOT/manifest['reference_run']/'results.json').read_text())
    reference = {(r['scene'],r['part']):r for r in reference}
    lookup = {(r['scene'],r['part'],r['method']):r for r in raw}
    hand = prepare_hand()
    records = [display_record(lookup[(scene,part,method)],reference[(scene,part)],hand)
               for scene,part in CASES for method in METHODS]
    meshes = {part:base.display_mesh(base.local_path(next(r['mesh_path'] for r in records if r['part']==part)))
              for part in sorted({part for _,part in CASES})}
    payload = dict(records=records,meshes=meshes,hand=hand,methods=METHODS,method_labels=LABELS,
        cases=[dict(scene=s,part=p,key=s+'__'+p) for s,p in CASES],protocol=PROTOCOL,
        source_run=manifest['created_utc'],reference_run=Path(manifest['reference_run']).name)
    output = run/'gallery'
    output.mkdir(parents=True,exist_ok=True)
    build_review_sheets(records,meshes,hand,output)
    encoded = json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    document = HTML.replace('__DATA__',encoded)
    sheets = '<h2>Static examples from above</h2><p>One canonical example per sampler and placement, without ranking. Use the selectors above to review every retained family.</p>'
    for part in ('BNC_Male','Gear_Large'):
        name = part+'_examples.png'
        sheets += f'<details><summary>{part.replace("_"," ")}: unranked examples</summary><a href="{name}"><img loading="lazy" src="{name}" style="width:100%" alt="Unranked {part} contact examples"></a></details>'
    marker = '<h2>Which pieces came from open source?</h2>'
    document = document.replace(marker,sheets+marker)
    (output/'index.html').write_text(document,encoding='utf-8')
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    audit = dict(cases=len(CASES),records=len(records),source_records=len(raw),protocol=PROTOCOL,
        retained_families=sum(len(r['grasps']) for r in records),
        retained_jaw_alternatives=sum(len(g.get('jaw_aliases',[g])) for r in records for g in r['grasps']),
        previous_zero_verified=True,all_retained_primary_families_in_selector=True,
        source_pairs_are_not_first_contact_witnesses=True,witnesses_not_connected_by_lines=True,
        static_sheets_show_one_unranked_example_per_sampler_and_placement=True,
        native_contact_normals_checked_against_saved_alignment=True,
        no_new_candidates=True,no_new_ranking=True,no_vlm=True,no_execution=True,
        no_assembly_views=True,protected_unchanged=before==after,protected_sha256=after,
        gallery_source_sha256=sha(__file__),display_helper_sha256=sha(base.__file__),
        cad_display_meshes={p:{k:v for k,v in m.items() if 'triangle_count' in k} for p,m in meshes.items()})
    (output/'gallery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'verify_viewer.cjs').write_text(VERIFY,encoding='utf-8')
    if before!=after:
        raise ValueError('A protected source/result changed while building the display')
    print(json.dumps(dict(gallery=str(output/'index.html'),**{k:audit[k] for k in ('cases','records','retained_families','retained_jaw_alternatives','protected_unchanged')})))


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nine previous-zero placements — upstream sampler comparison</title>
<style>
:root{font:15px system-ui;color:#233442;background:#f1f4f6}body{margin:0}header{padding:24px max(22px,calc((100vw - 1480px)/2));background:#fff;border-bottom:1px solid #ccd6dd}h1{font-size:25px;margin:0 0 10px}h2{font-size:19px}h3{font-size:15px}p{line-height:1.5;margin:8px 0}.muted{color:#526879}.wrap{max-width:1480px;margin:20px auto;padding:0 20px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{font-weight:600;display:flex;flex-direction:column;gap:5px}select,button{font:inherit;padding:8px;background:white;border:1px solid #a9bbc8;border-radius:5px}button{cursor:pointer}.layout{display:grid;grid-template-columns:minmax(550px,1.8fr) minmax(340px,1fr);gap:16px;margin-top:16px}.panel{background:#fff;padding:17px;border:1px solid #d5dfe5;border-radius:8px}canvas{width:100%;height:580px;display:block;background:#fbfcfd;cursor:grab;touch-action:none}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:13px;margin:12px 0}.legend span:before{content:'';display:inline-block;width:11px;height:11px;background:var(--c);margin-right:5px}table{border-collapse:collapse;width:100%;margin:8px 0}td,th{text-align:left;padding:7px 5px;border-bottom:1px solid #e1e7ec;font-size:13px}th{background:#f6f8fa}tbody tr[data-case]{cursor:pointer}tbody tr[data-case]:hover{background:#f0f7fb}.good{color:#23704b}.bad{color:#a83228}.selected{background:#e8f3fa}details{margin:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;background:#f4f6f8;padding:10px;font:12px ui-monospace}.scroll{overflow-x:auto}a{color:#205e91}@media(max-width:1000px){.layout{grid-template-columns:1fr}canvas{height:470px}}
</style><header><h1>What changed for the nine placements that previously had no grasp?</h1>
<p>Five BNC placements and four large-gear placements. Compare the frozen strict result with actual open-source contact samplers followed by our common Panda hand checks.</p>
<p class="muted">Every displayed option passed the saved <b>panda_table_proj15</b> geometry checks. These views do not establish pickup success, resistance to slipping or assembly accuracy. No robot execution, physics, VLM selection or final assembly view is included.</p></header>
<main class="wrap"><section class="panel"><h2>Previous strict run versus the new samplers</h2><p class="muted">Cells show retained hand-placement families. Click a cell to inspect every family and its valid jaw alternatives. A family ID is an identifier, not a quality rank.</p><div class="scroll"><table><thead><tr><th>Saved placement</th><th>Previous strict method</th><th>one · 20 mm</th><th>one · 512 samples</th><th>GraspKit · 512 rays</th></tr></thead><tbody id="comparison"></tbody></table></div></section>
<div class="controls" style="margin-top:18px"><label>Placement<select id="placement"></select></label><label>Contact sampler<select id="method"></select></label><label>Retained family<select id="family"></select></label><label>Valid jaw alternative<select id="alias"></select></label>
<label>Hand opening<select id="opening"><option value="contact">At modeled first contact</option><option value="open">Open before approach</option></select></label><label>Hand display<select id="handDisplay"><option value="full">Full hand</option><option value="fingers">Fingers and pads</option><option value="none">Contact points only</option></select></label><label>Contact normals<select id="normals"><option value="show">Show native CAD normals</option><option value="hide">Hide native CAD normals</option></select></label><button id="prev">Previous family</button><button id="next">Next family</button><button id="top">View from above</button><button id="side">View from side</button><button id="oblique">Angled view</button></div>
<div class="layout"><section class="panel"><canvas id="view"></canvas><div class="legend"><span style="--c:#a5b0b9">Target CAD</span><span style="--c:#377fac">Panda hand</span><span style="--c:#d89436">Inner pads</span><span style="--c:#d96b98">Source pair · dashed</span><span style="--c:#147e58">Actual pad-contact witnesses · dots</span><span style="--c:#16523a">CAD face normals · arrows</span><span style="--c:#333">Finger closing axis</span></div><p class="muted">Pink connects the sampler's two proposed points. Each green dot is one native-CAD witness found when a finite pad first meets the part. The green dots may have different heights or positions along the pad; they are deliberately <b>not connected</b>. Dark green arrows are the normals of those native CAD triangles. The black arrow is the gripper's closing axis, independent of the witness positions.</p><p class="muted">Drag to turn; scroll to zoom. The brown downward arrow shows the straight-down approach. The table is drawn using the saved observed-depth plane. Distances in saved data are meters.</p><p id="meshNote" class="muted"></p></section>
<aside class="panel"><h2 id="caseTitle"></h2><p id="caseStatus"></p><table id="metrics"></table><details><summary>Why new proposals were rejected</summary><table id="rejects"></table></details><details><summary>Why the previous strict run had none</summary><table id="oldRejects"></table><p class="muted">Counts belong to the frozen strict run. Different candidate proposals and conversion rules are compared here; this is not a one-rule-only test.</p></details><details><summary>Saved selected alternative</summary><pre id="raw"></pre></details></aside></div>
<section class="panel" style="margin-top:18px"><h2>Which pieces came from open source?</h2><p><b>one</b>: the original surface sampling, ray intersection and antipodal-pair function bodies from the pinned <a href="https://github.com/wanweiwei07/one/tree/ec1bdae83ec33ca6af9de42904d9851647fe012c">one source</a>. The original 20 mm spacing and a declared 512-sample control use the same function. Ray queries are batched to limit memory. The full one planning framework, its hand-roll expansion and its ranking are not run.</p><p><b>GraspKit</b>: the original pinned <a href="https://github.com/DexForce/EmbodiChain/tree/bb2b65e0de1ccc486e022e57a6a56d8ddd78e969">GraspKit antipodal sampler</a> with 512 starting rays. This is its contact sampler, not its full framework.</p><p><b>Common adapter and validation</b>: source pairs are transformed by the saved observed CAD pose, filtered for the Panda opening and a closing-direction tilt of at most 15°, then reduced with the original Fabrica 5 mm pair-spacing function. Our adapter keeps the approach straight down, projects the closing direction horizontally, sets finger height from the actual Panda pads and table, and centers the hand using modeled first contact. Both jaw assignments are checked. Full finite hand/target/table geometry is checked. The old shared-contact-region, cylinder-classification and strict 1° flat-surface rules are absent.</p><p>The displayed native triangle normals come from the same cleaned metric CAD and triangle IDs used by the validator. A display-only mesh reduction may simplify the gray CAD drawing. It does not change the checks. The default alternative follows a saved canonical jaw rule, not a learned or weighted score.</p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function option(select,value,label){const item=document.createElement('option');item.value=value;item.textContent=label;select.append(item)}
D.cases.forEach(c=>option($('placement'),c.key,c.scene+' / '+c.part.replaceAll('_',' ')));D.methods.forEach(m=>option($('method'),m,D.method_labels[m]));
let record=null,family=null,grasp=null,world=[],lines=[],dots=[],center=[0,0,0],span=.05,az=-.8,el=.65,zoom=1,drag=null;
const point=(p,T)=>T.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
const plus=(p,q,scale=1)=>p.map((v,k)=>v+scale*q[k]);
function addMesh(v,f,color,kind){for(const t of f)world.push({p:t.map(i=>v[i]),color,kind})}
function rebuildCase(){const key=$('placement').value;record=D.records.find(r=>r.scene+'__'+r.part===key&&r.method===$('method').value);family=null;$('family').replaceChildren();(record?.grasps||[]).forEach((g,i)=>option($('family'),i,g.family_id+' · '+g.id));if(!record?.grasps?.length)option($('family'),-1,'No retained family');rebuildGrasp();for(const row of $('comparison').children)row.className=row.dataset?.case===key?'selected':''}
function rebuildGrasp(){world=[];lines=[];dots=[];const nextFamily=record?.grasps?.[Number($('family').value)]||null;if(family!==nextFamily){family=nextFamily;$('alias').replaceChildren();const members=family?.jaw_aliases?.length?family.jaw_aliases:(family?[family]:[]);members.forEach((g,i)=>option($('alias'),i,g.id+' · '+(g.jaw_sign>0?'positive':'negative')+' jaw assignment'));if(members.length)$('alias').value=Math.max(0,members.findIndex(g=>g.id===family.id));else option($('alias'),-1,'No valid jaw alternative')};grasp=family?.jaw_aliases?.[Number($('alias').value)]||family;const mesh=D.meshes[record?.part];if(mesh&&record){const v=mesh.vertices.map(p=>point(p,record.world_T_cad));addMesh(v,mesh.triangles,'#a5b0b9','cad');const lo=[0,1,2].map(k=>Math.min(...v.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...v.map(p=>p[k])));center=lo.map((v,k)=>(v+hi[k])/2);span=Math.max(.025,...hi.map((v,k)=>v-lo[k]));$('meshNote').textContent=`CAD drawing: ${mesh.display_triangle_count.toLocaleString()} triangles; native CAD: ${mesh.native_triangle_count.toLocaleString()}. Drawing reduction only.`}
if(grasp){const opening=$('opening').value==='open'?grasp.opening_m:grasp.display_contact_opening_m;for(const c of D.hand.components){if($('handDisplay').value==='none'||($('handDisplay').value==='fingers'&&c.name.includes('hand_collision')))continue;const v=c.vertices_zero_m.map(p=>point(plus(p,c.translation_per_opening,opening),grasp.world_T_grip));addMesh(v,c.triangles,c.is_pad?'#d89436':'#377fac','hand')};const sampled=grasp.contacts_world_m,witnesses=grasp.contact_points_world_m;lines.push({kind:'source_pair',p:sampled,color:'#d96b98',width:1.4,dashed:true,dots:true});witnesses.forEach(p=>dots.push({kind:'contact_witness',p,color:'#147e58',radius:5}));const middle=witnesses[0].map((v,k)=>(v+witnesses[1][k])/2),axis=grasp.world_T_grip.slice(0,3).map(r=>r[0]),length=Math.max(.013,Math.min(.04,grasp.width_m*.65));lines.push({kind:'closing_axis',p:[plus(middle,axis,-length),plus(middle,axis,length)],color:'#333',width:1.8,arrow:true,label:'Closing axis'});lines.push({kind:'approach',p:[plus(middle,[0,0,1],.04),plus(middle,[0,0,1],.012)],color:'#a45120',width:2.4,arrow:true});if($('normals').value==='show')witnesses.forEach((p,i)=>lines.push({kind:'cad_normal',p:[p,plus(p,grasp.display_native_normals_world[i],.006)],color:'#16523a',width:2,arrow:true}));}
updateInfo();draw()}
const mm=v=>Number.isFinite(v)?(1000*v).toFixed(3)+' mm':'unavailable';
function rows(id,values){$(id).replaceChildren();for(const [a,b]of values){const tr=document.createElement('tr'),x=document.createElement('td'),y=document.createElement('td');x.textContent=a;y.textContent=b;tr.append(x,y);$(id).append(tr)}}
function friendly(s){return s.replaceAll('_',' ')}
function updateInfo(){if(!record)return;$('caseTitle').textContent=record.scene+' / '+friendly(record.part);const n=record.grasps.length;$('caseStatus').textContent=n?`${n} retained families; ${record.valid_count} valid jaw configurations. Previous strict run: 0.`:'No retained family in this sampler setting. Previous strict run: 0.';$('caseStatus').className=n?'good':'bad';const data=[['Sampler',D.method_labels[record.method]],['Raw native-CAD contact pairs',record.native_pairs],['Pairs after opening check',record.width_qualified_pairs],['Pairs within 15° of horizontal',record.horizontal_projection_qualified_pairs],['Pairs after 5 mm spacing',record.sparse_pairs],['Primary jaw configurations checked',record.primary_generated_count],['Primary configurations that passed',record.valid_count],['Retained distinct families',n],['CAD-cache generation (once per part)',record.intrinsic_record.sourcegen_seconds.toFixed(3)+' s'],['Saved total runtime (both adapter controls)',record.timing_seconds.toFixed(3)+' s']];if(grasp){data.push(['Displayed family',family.family_id],['Saved jaw alternative',grasp.id],['Valid alternatives in this family',family.jaw_aliases?.length??1],['Source closing-direction tilt',grasp.native_pair_projection_deg.toFixed(4)+'°'],['Actual approach','Straight down'],['Opening before approach',mm(grasp.opening_m)],['Opening at first pad contact',mm(grasp.display_contact_opening_m)],['Difference in first-contact openings',mm(grasp.contact_opening_difference_m)],['Minimum modeled table gap',mm(grasp.table_clearance_m)],['Native face normal vs closing axis',grasp.display_native_normal_errors_deg.map(v=>v.toFixed(4)+'°').join(' / ')],['Centering shift along closing axis',mm(grasp.center_projection?.translation_along_closing_m)])}rows('metrics',data);rows('rejects',Object.entries(record.rejection_counts||{}).map(([k,v])=>[friendly(k),v]));const old=record.reference_zero;rows('oldRejects',[['Previous checked configurations',old.generated_count],...Object.entries(old.rejection_counts).map(([k,v])=>[friendly(k),v]),...Object.entries(old.preparation_rejection_counts).map(([k,v])=>['Before hand checks: '+friendly(k),v])]);$('raw').textContent=JSON.stringify(grasp?{family_id:family.family_id,representative_rule:family.representative_rule,grouped_candidate_ids:family.grouped_candidate_ids,selected_alias:grasp}:record,null,2)}
function project(p){const q=p.map((v,k)=>v-center[k]),ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),x=ca*q[0]+sa*q[1],y=-sa*q[0]+ca*q[1];return[x,ce*q[2]-se*y,se*q[2]+ce*y]}
function draw(){const canvas=$('view'),ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);const scale=Math.min(w,h)/(Math.max(span,.105)*1.8)*zoom,screen=p=>{const q=project(p);return[w/2+scale*q[0],h*.65-scale*q[1],q[2]]};const plane=record?.table_plane_world||[0,0,1,0],s=Math.max(span*.8,.045),corners=[[-s,-s],[s,-s],[s,s],[-s,s]].map(q=>{const x=center[0]+q[0],y=center[1]+q[1];return screen([x,y,-(plane[0]*x+plane[1]*y+plane[3])/plane[2]])});ctx.fillStyle='#e9eff2';ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();const faces=world.map(f=>({...f,sp:f.p.map(screen)}));faces.sort((a,b)=>a.sp.reduce((s,p)=>s+p[2],0)-b.sp.reduce((s,p)=>s+p[2],0));for(const f of faces){ctx.beginPath();ctx.moveTo(f.sp[0][0],f.sp[0][1]);ctx.lineTo(f.sp[1][0],f.sp[1][1]);ctx.lineTo(f.sp[2][0],f.sp[2][1]);ctx.closePath();ctx.fillStyle=f.color;ctx.globalAlpha=f.kind==='cad'?.93:.85;ctx.fill();ctx.globalAlpha=1}for(const l of lines){const p=l.p.map(screen);ctx.strokeStyle=l.color;ctx.lineWidth=l.width;ctx.setLineDash(l.dashed?[5,4]:[]);ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);ctx.lineTo(p[1][0],p[1][1]);ctx.stroke();if(l.dots)for(const q of p){ctx.fillStyle=l.color;ctx.beginPath();ctx.arc(q[0],q[1],3.5,0,2*Math.PI);ctx.fill()}if(l.arrow){ctx.setLineDash([]);const a=Math.atan2(p[1][1]-p[0][1],p[1][0]-p[0][0]);ctx.beginPath();for(const side of [-.4,.4]){ctx.moveTo(p[1][0],p[1][1]);ctx.lineTo(p[1][0]-8*Math.cos(a+side),p[1][1]-8*Math.sin(a+side))}ctx.stroke()}if(l.label){ctx.fillStyle=l.color;ctx.font='12px system-ui';ctx.fillText(l.label,p[1][0]+5,p[1][1]-5)}}ctx.setLineDash([]);for(const d of dots){const p=screen(d.p);ctx.fillStyle=d.color;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],d.radius,0,2*Math.PI);ctx.fill();ctx.stroke()}ctx.fillStyle='#344b5b';ctx.font='13px system-ui';ctx.fillText(D.method_labels[record?.method]||'',14,h-16);if(!grasp){ctx.fillStyle='#a83228';ctx.font='18px system-ui';ctx.fillText('No retained family for this sampler setting',15,30)}}
for(const c of D.cases){const tr=document.createElement('tr');tr.dataset.case=c.key;const label=document.createElement('td');label.textContent=c.scene+' / '+friendly(c.part);tr.append(label);const old=document.createElement('td');old.textContent='0';old.className='bad';tr.append(old);for(const method of D.methods){const r=D.records.find(r=>r.scene===c.scene&&r.part===c.part&&r.method===method),td=document.createElement('td');td.textContent=r.grasps.length+' families / '+r.valid_count+' jaw configurations';td.className=r.grasps.length?'good':'bad';td.style.cursor='pointer';td.onclick=()=>{$('placement').value=c.key;$('method').value=method;rebuildCase();window.scrollTo({top:350,behavior:'smooth'})};tr.append(td)}$('comparison').append(tr)}
$('placement').onchange=rebuildCase;$('method').onchange=rebuildCase;for(const id of ['family','alias','opening','handDisplay','normals'])$(id).onchange=rebuildGrasp;function shift(delta){const select=$('family');select.selectedIndex=(select.selectedIndex+delta+select.options.length)%select.options.length;rebuildGrasp()}$('prev').onclick=()=>shift(-1);$('next').onclick=()=>shift(1);$('top').onclick=()=>{el=Math.PI/2;az=0;draw()};$('side').onclick=()=>{el=0;az=0;draw()};$('oblique').onclick=()=>{el=.65;az=-.8;draw()};$('view').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('view').setPointerCapture(e.pointerId)};$('view').onpointerup=()=>drag=null;$('view').onpointermove=e=>{if(drag){az+=(e.clientX-drag[0])*.01;el=Math.max(-1.5,Math.min(1.57,el+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw()}};$('view').onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};window.addEventListener('resize',draw);rebuildCase();
</script></html>'''


VERIFY = r'''const fs=require('fs'),vm=require('vm'),path=require('path');
const file=process.argv[2],html=fs.readFileSync(file,'utf8');
const payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.slice(html.lastIndexOf('<script>')+8).split('</script>')[0];
const elements=new Map(),context2d=new Proxy({}, {get:(o,k)=>k in o?o[k]:(()=>{})});
class Element{constructor(tag='div'){this.tag=tag;this.children=[];this._value='';this.textContent='';this.clientWidth=900;this.clientHeight=580;this.style={};this.dataset={};}append(...items){this.children.push(...items);if(this.tag==='select'&&this.children.length===items.length)this._value=String(this.children[0]?.value??'');}replaceChildren(...items){this.children=[];this._value='';this.append(...items);}get value(){return this._value;}set value(v){this._value=String(v);}get options(){return this.children;}get selectedIndex(){return this.children.findIndex(c=>String(c.value)===this._value);}set selectedIndex(i){this._value=String(this.children[i]?.value??'');}getContext(){return context2d;}setPointerCapture(){}}
for(const id of ['placement','method','family','alias','opening','handDisplay','normals'])elements.set(id,new Element('select'));
for(const [id,value]of [['opening','contact'],['handDisplay','full'],['normals','show']])elements.get(id).value=value;
elements.set('data',new Element());elements.get('data').textContent=payload;
const sandbox={document:{getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag)},window:{devicePixelRatio:1,addEventListener(){},scrollTo(){}},console};vm.createContext(sandbox);vm.runInContext(code,sandbox,{timeout:30000});
const result=vm.runInContext(`(()=>{draw=()=>{};const assert=(ok,msg)=>{if(!ok)throw Error(msg)};let records=0,families=0,aliases=0,zero=0;for(const r of D.records){$('placement').value=r.scene+'__'+r.part;$('method').value=r.method;rebuildCase();assert(record===r,'record mismatch');assert(r.reference_zero.valid_count===0,'reference not zero');records++;if(!r.grasps.length){zero++;assert(grasp===null,'zero case displays grasp');continue;}for(let i=0;i<r.grasps.length;i++){$('family').value=i;rebuildGrasp();families++;const f=r.grasps[i],members=f.jaw_aliases?.length?f.jaw_aliases:[f];assert(grasp.id===f.id,'canonical default mismatch');assert($('alias').options.length===members.length,'alias count');for(let j=0;j<members.length;j++){$('alias').value=j;rebuildGrasp();aliases++;assert(grasp.id===members[j].id,'alias mismatch');assert(grasp.valid&&grasp.protocols.includes(D.protocol),'invalid or wrong protocol');assert(lines.filter(l=>l.kind==='source_pair').length===1,'source pair missing');assert(dots.filter(d=>d.kind==='contact_witness').length===2,'witness dots missing');assert(!lines.some(l=>l.kind==='contact_witness'),'witnesses incorrectly connected');assert(lines.filter(l=>l.kind==='closing_axis').length===1,'closing axis absent');$('normals').value='show';rebuildGrasp();assert(lines.filter(l=>l.kind==='cad_normal').length===2,'native normals missing');assert(grasp.display_native_normal_errors_deg.every(v=>v<=15.00001),'normal above validation bound');$('normals').value='hide';rebuildGrasp();assert(!lines.some(l=>l.kind==='cad_normal'),'native normal toggle failed');for(const mode of ['contact','open']){$('opening').value=mode;rebuildGrasp();}for(const mode of ['full','fingers','none']){$('handDisplay').value=mode;rebuildGrasp();}}}}return{records,families,jaw_alternatives:aliases,zero_sampler_cases:zero,javascript_bootstrap:'passed',all_family_and_alias_selectors:'passed',canonical_default:'passed',source_pair_vs_actual_witness_display:'passed',native_normal_toggle:'passed',opening_and_hand_controls:'passed'};})()`,sandbox,{timeout:120000});
fs.writeFileSync(path.join(path.dirname(file),'viewer_verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
'''


if __name__=='__main__':
    main(sys.argv[1])
