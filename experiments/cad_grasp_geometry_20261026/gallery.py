"""Review every retained CAD grasp before VLM selection; no inference or physics."""
from collections import Counter
from pathlib import Path
import html
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
DISPLAY_TRIANGLES = 6000


def local_path(value):
    text = str(value).replace('\\', '/')
    if sys.platform == 'win32' and text.startswith('/mnt/'):
        text = text[5].upper() + ':/' + text[7:]
    elif sys.platform != 'win32' and len(text) > 2 and text[1] == ':':
        text = '/mnt/' + text[0].lower() + text[2:]
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_stl(path):
    data = Path(path).read_bytes()
    count = int.from_bytes(data[80:84], 'little') if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        corners = np.frombuffer(data, dtype=dtype, offset=84, count=count)['vertices'].astype(float)
    else:
        points = [list(map(float, line.split()[1:4])) for line in data.decode().splitlines()
                  if line.strip().startswith('vertex ')]
        corners = np.asarray(points).reshape(-1, 3, 3)
    vertices, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def display_mesh(path):
    vertices, triangles = read_stl(path)
    original = len(triangles)
    if original > DISPLAY_TRIANGLES:
        try:
            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                            o3d.utility.Vector3iVector(triangles))
            mesh = mesh.simplify_quadric_decimation(DISPLAY_TRIANGLES)
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        except ImportError:
            # Visualization only: keep all native faces when Open3D is unavailable.
            pass
    return dict(vertices=np.round(vertices, 7).tolist(), triangles=triangles.tolist(),
                native_triangle_count=original, display_triangle_count=len(triangles))


def closure_opening(grasp, hand):
    if 'contact_opening_m' in grasp:
        return float(grasp['contact_opening_m'])
    if 'first_contact_openings_m' in grasp:
        values = grasp['first_contact_openings_m']
        values = list(values.values()) if isinstance(values, dict) else values
        finite = [float(v) for v in values if v is not None and np.isfinite(v)]
        if finite:
            return float(np.mean(finite))
    pads = [c for c in hand['components'] if c['is_pad']]
    pads.sort(key=lambda c: np.asarray(c['vertices_zero_m'])[:, 0].mean())
    gap_zero = (np.asarray(pads[-1]['vertices_zero_m'])[:, 0].min()
                - np.asarray(pads[0]['vertices_zero_m'])[:, 0].max())
    return float(grasp['width_m']) - gap_zero


def case_key(record):
    return str(record['scene']) + '__' + str(record['part'])


def modeled_contacts(grasp):
    return np.asarray(grasp.get('contact_points_world_m', grasp['contacts_world_m']))


def build_overview(records, meshes, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    image_paths = {}
    scene_names = sorted({r['scene'] for r in records})
    part_names = list(dict.fromkeys(r['part'] for r in records))
    for scene in scene_names:
        scene_records = {r['part']: r for r in records if r['scene'] == scene}
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
        for ax, part in zip(axes.flat, part_names):
            record = scene_records.get(part)
            ax.set_aspect('equal')
            ax.set_axis_off()
            if record is None:
                ax.set_title(part + '\nNo record')
                continue
            mesh = meshes.get(part)
            if mesh and record.get('world_T_cad') is not None:
                transform = np.asarray(record['world_T_cad'])
                vertices = np.asarray(mesh['vertices']) @ transform[:3, :3].T + transform[:3, 3]
                center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
                points = (vertices - center) * 1000
                triangles = np.asarray(mesh['triangles'])
                ax.add_collection(PolyCollection(points[triangles, :2], facecolors='#c9d1d8',
                                                 edgecolors='none', alpha=.8))
                grasps = record.get('grasps', [])
                colors = plt.get_cmap('turbo')(np.linspace(.08, .92, max(1, len(grasps))))
                for color, grasp in zip(colors, grasps):
                    contacts = modeled_contacts(grasp)
                    if contacts.shape != (2, 3):
                        continue
                    contacts = (contacts - center) * 1000
                    ax.plot(contacts[:, 0], contacts[:, 1], color=color, lw=.7, alpha=.6)
                    ax.scatter(contacts[:, 0], contacts[:, 1], color=color, s=5, zorder=4)
                size = max(np.ptp(points[:, :2], axis=0).max() * .7, 8)
                ax.set_xlim(-size, size)
                ax.set_ylim(-size, size)
            count = len(record.get('grasps', []))
            status = 'No retained grasp' if not count else f'{count} retained grasps'
            ax.set_title(part.replace('_', ' ') + '\n' + status, fontsize=10,
                         color='#a93333' if not count else '#243342')
        for ax in axes.flat[len(part_names):]:
            ax.set_axis_off()
        fig.suptitle(f'{scene}: all retained contact pairs, viewed from above\n'
                     'Colored endpoints = modeled first finger contacts; each line = one retained grasp. '
                     'No VLM choice or pickup execution.', fontsize=13)
        path = output / (scene + '_contacts.png')
        fig.savefig(path, dpi=130, facecolor='white')
        plt.close(fig)
        image_paths[scene] = path.name
    return image_paths


def build_hand_examples(records, meshes, hand, output):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    paths = {}
    for record in records:
        examples = record.get('grasps', [])[:3]
        mesh = meshes.get(record['part'])
        if not examples or mesh is None:
            continue
        transform = np.asarray(record['world_T_cad'])
        cad = np.asarray(mesh['vertices']) @ transform[:3, :3].T + transform[:3, 3]
        cad_faces = np.asarray(mesh['triangles'])
        center = (cad.min(axis=0) + cad.max(axis=0)) / 2
        fig, axes = plt.subplots(3, len(examples), figsize=(max(12, 5 * len(examples)), 11), squeeze=False,
                                 constrained_layout=True)
        for column, grasp in enumerate(examples):
            pose = np.asarray(grasp['world_T_grip'])
            opening = grasp['display_contact_opening_m']
            objects = [(cad, cad_faces, '#aebbc4')]
            for component in hand['components']:
                local = (np.asarray(component['vertices_zero_m'])
                         + opening * np.asarray(component['translation_per_opening']))
                vertices = local @ pose[:3, :3].T + pose[:3, 3]
                objects.append((vertices, np.asarray(component['triangles']),
                                '#d89436' if component['is_pad'] else '#377fac'))
            for row, coordinates in enumerate(((0, 1), (0, 2))):
                ax = axes[row, column]
                all_points = []
                for vertices, faces, color in objects:
                    points = (vertices - center) * 1000
                    projected = points[:, coordinates]
                    if color == '#377fac':
                        outline = projected[ConvexHull(projected).vertices]
                        outline = np.vstack((outline, outline[0]))
                        ax.plot(outline[:, 0], outline[:, 1], color=color, lw=1.15)
                    else:
                        ax.add_collection(PolyCollection(projected[faces], facecolors=color,
                                                         edgecolors='none', alpha=.85))
                    all_points.append(projected)
                sampled = (np.asarray(grasp['contacts_world_m']) - center) * 1000
                ax.plot(sampled[:, coordinates[0]], sampled[:, coordinates[1]], 'o--',
                        color='#d96b98', ms=3, lw=.8, alpha=.7, zorder=9)
                contacts = (modeled_contacts(grasp) - center) * 1000
                ax.plot(contacts[:, coordinates[0]], contacts[:, coordinates[1]], 'o-',
                        color='#147e58', ms=4, lw=1.5, zorder=10)
                combined = np.concatenate(all_points)
                lower, upper = combined.min(axis=0), combined.max(axis=0)
                midpoint = (lower + upper) / 2
                extent = max(np.ptp(combined, axis=0).max() * .58, 10)
                ax.set_xlim(midpoint[0] - extent, midpoint[0] + extent)
                ax.set_ylim(midpoint[1] - extent, midpoint[1] + extent)
                ax.set_aspect('equal')
                ax.grid(alpha=.16)
                ax.set_xlabel('World X relative to CAD center (mm)')
                ax.set_ylabel('World ' + ('Y' if row == 0 else 'Z') + ' relative to CAD center (mm)')
                view = 'above' if row == 0 else 'side'
                ax.set_title(f"{grasp['id']} — view from {view}\n"
                             f"finger depth {grasp['pad_depth_m'] * 1000:+.1f} mm")
                if row == 1:
                    ax.axhline((record['table_z_m'] - center[2]) * 1000,
                               color='#687e8e', lw=1.5, label='Table')
            ax = axes[2, column]
            for vertices, faces, color in objects:
                local = (vertices - pose[:3, 3]) @ pose[:3, :3]
                projected = local[:, (0, 2)] * np.array([1000., -1000.])
                if color == '#377fac':
                    outline = projected[ConvexHull(projected).vertices]
                    outline = np.vstack((outline, outline[0]))
                    ax.plot(outline[:, 0], outline[:, 1], color=color, lw=1.15)
                else:
                    ax.add_collection(PolyCollection(projected[faces], facecolors=color,
                                                     edgecolors='none', alpha=.85))
            sampled_local = (np.asarray(grasp['contacts_world_m']) - pose[:3, 3]) @ pose[:3, :3]
            sampled_projected = sampled_local[:, (0, 2)] * np.array([1000., -1000.])
            ax.plot(sampled_projected[:, 0], sampled_projected[:, 1], 'o--',
                    color='#d96b98', ms=3, lw=.8, alpha=.7, zorder=9)
            local_contacts = (modeled_contacts(grasp) - pose[:3, 3]) @ pose[:3, :3]
            projected_contacts = local_contacts[:, (0, 2)] * np.array([1000., -1000.])
            ax.plot(projected_contacts[:, 0], projected_contacts[:, 1], 'o-',
                    color='#147e58', ms=5, lw=1.7, zorder=10)
            half_width = max(abs(projected_contacts[:, 0]).max() + 8, 14)
            contact_height = projected_contacts[:, 1].mean()
            ax.set_xlim(-half_width, half_width)
            ax.set_ylim(contact_height - 14, contact_height + 14)
            ax.set_aspect('equal')
            ax.grid(alpha=.16)
            ax.axhline((record['table_z_m'] - pose[2, 3]) * 1000,
                       color='#687e8e', lw=1.5)
            ax.set_title('Finger contact close-up\nview across the finger opening')
            ax.set_xlabel('Across finger opening relative to hand frame (mm)')
            ax.set_ylabel('Height relative to hand frame (mm)')
        fig.suptitle(f"{record['scene']} / {record['part'].replace('_', ' ')}\n"
                     'First retained IDs shown as unranked examples; every ID is available in the interactive viewer.\n'
                     'Blue outlines = hand; orange = pads; gray = CAD; green = modeled first contacts; pink dashed = sampled pair.', fontsize=12)
        path = output / (case_key(record) + '_hand_examples.png')
        fig.savefig(path, dpi=115, facecolor='white')
        plt.close(fig)
        paths[case_key(record)] = path.name
    return paths


def main(run_path):
    run = local_path(run_path)
    raw = json.loads((run / 'results.json').read_text())
    records = raw if isinstance(raw, list) else raw['results']
    hand = json.loads((ROOT / 'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    output = run / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for record in records:
        if record.get('mesh_path') and record['part'] not in meshes:
            meshes[record['part']] = display_mesh(local_path(record['mesh_path']))
    for component in hand['components']:
        points = np.asarray(component['vertices_zero_m'])
        hull = ConvexHull(points)
        used = np.unique(hull.simplices)
        mapping = np.full(len(points), -1, dtype=int)
        mapping[used] = np.arange(len(used))
        component['vertices_zero_m'] = np.round(points[used], 8).tolist()
        component['triangles'] = mapping[hull.simplices].tolist()
    for record in records:
        for grasp in record.get('grasps', []):
            grasp['display_contact_opening_m'] = closure_opening(grasp, hand)
    images = build_overview(records, meshes, output)
    examples = build_hand_examples(records, meshes, hand, output)
    payload = dict(records=records, meshes=meshes, hand=hand, images=images, examples=examples)
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    document = HTML.replace('__DATA__', encoded)
    (output / 'index.html').write_text(document, encoding='utf-8')
    status_counts = Counter(str(r.get('status', 'unknown')) for r in records)
    (output / 'gallery_audit.json').write_text(json.dumps(dict(
        cases=len(records), retained_grasps=sum(len(r.get('grasps', [])) for r in records),
        scenes=sorted(images), statuses=dict(status_counts), static_hand_example_sheets=len(examples),
        cad_display_meshes={p: {k: v for k, v in m.items() if 'triangle_count' in k}
                            for p, m in meshes.items()},
        note='Every retained grasp is available in the selector. Display mesh reduction does not alter collision checks.'), indent=2))
    print(str(output / 'index.html'))


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIST scenes — CAD grasp geometry before VLM selection</title>
<style>
:root{font:15px system-ui;color:#213243;background:#f1f4f6}body{margin:0}header{padding:24px max(24px,calc((100vw - 1450px)/2));background:#fff;border-bottom:1px solid #d7dfe4}h1{font-size:26px;margin:0 0 8px}p{line-height:1.5;margin:7px 0}.muted{color:#586c7b}.wrap{max-width:1450px;margin:20px auto;padding:0 20px}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:end}label{font-weight:600;display:flex;flex-direction:column;gap:5px}select,button{font:inherit;padding:8px;border:1px solid #aabac5;border-radius:5px;background:#fff}button{cursor:pointer}.layout{display:grid;grid-template-columns:minmax(550px,1.7fr) minmax(300px,1fr);gap:18px;margin-top:16px}.panel{background:#fff;padding:18px;border:1px solid #d8e0e5;border-radius:9px}canvas{width:100%;height:540px;display:block;touch-action:none;cursor:grab;background:#fbfcfd;border-radius:5px}.legend{display:flex;gap:20px;font-size:13px;margin:10px 0}.legend span:before{content:'';display:inline-block;width:12px;height:12px;margin-right:5px;background:var(--c)}table{width:100%;border-collapse:collapse;margin:10px 0}td,th{text-align:left;border-bottom:1px solid #e3e9ed;padding:7px 5px;font-size:13px}th{font-weight:600}h2{font-size:18px;margin:6px 0 12px}h3{font-size:15px}pre{font:12px ui-monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:10px;max-height:230px;overflow:auto}#overview{width:100%;display:block}.bad{color:#ad342a}.good{color:#226346}details{margin-top:15px}summary{cursor:pointer;font-weight:600}a{color:#185e9b}@media(max-width:950px){.layout{grid-template-columns:1fr}canvas{height:470px}}
</style>
<header><h1>CAD grasps before VLM selection</h1><p>Eight saved NIST layouts, 12 parts per layout. Review the modeled hand and every retained grasp.</p>
<p class="muted">These are geometry checks against the aligned target CAD and table. No robot pickup, slipping test, assembly execution or VLM selection was performed.</p></header>
<div class="wrap"><div class="controls">
<label>Scene<select id="scene"></select></label><label>Part<select id="part"></select></label><label>Retained grasp<select id="grasp"></select></label>
<label>Hand opening<select id="opening"><option value="contact">At nominal finger contact</option><option value="open">Open for approach</option></select></label>
<label>Hand display<select id="handDisplay"><option value="full">Full hand</option><option value="fingers">Fingers and pads</option><option value="none">Contact points only</option></select></label>
<button id="prev">Previous grasp</button><button id="next">Next grasp</button><button id="top">View from above</button><button id="side">View from side</button><button id="oblique">Angled view</button></div>
<div class="layout"><section class="panel"><canvas id="view"></canvas><div class="legend"><span style="--c:#a2adb6">Target CAD</span><span style="--c:#347dac">Panda hand and fingers</span><span style="--c:#d3872b">Inner contact pads</span><span style="--c:#147e58">Modeled first contacts</span><span style="--c:#d96b98">Sampled pair</span></div><p class="muted">Drag to turn the view; use the mouse wheel to zoom. A downward arrow shows the approach direction. Each grasp ID is a review option, not a quality rank. The modeled first contacts may differ from the sampled pair because the finger pads cover a finite area.</p><p id="meshNote" class="muted"></p></section>
<aside class="panel"><h2 id="caseTitle"></h2><p id="caseStatus"></p><table id="metrics"></table><h3>Why other candidates were rejected</h3><table id="rejects"></table><details><summary>Saved grasp information</summary><pre id="raw"></pre></details><details><summary>What this view means</summary><p>Sampled pair: the two starting contact points proposed by the sampler. Modeled first contacts: the points where each finite finger pad first touches the CAD during modeled closing. Finger depth: where the sampled pair sits along the fingers. Table clearance: the smallest modeled gap above the table. A retained grasp passes the declared geometric checks; it is not proof that the part can be lifted without moving inside the fingers.</p><p>Only the target part, modeled hand and table are evaluated. Other parts, robot arm reach and robot self-contact are outside this test. CAD identity and its corresponding observed crop are supplied assumptions.</p></details></aside></div>
<details id="examplesPanel" class="panel"><summary>Static full-hand examples for this placement</summary><p>These are the first retained IDs, shown for inspection. They have no quality order.</p><a id="examplesLink"><img id="examples" style="width:100%" alt="Unranked modeled hand examples, viewed from above and from the side"></a></details>
<section class="panel" style="margin-top:18px"><h2>All retained modeled contact pairs in this scene</h2><p class="muted">Lines connect modeled first finger contacts. Overlapping lines may represent different finger depths. No example is automatically selected as the winner.</p><a id="overviewLink"><img id="overview" alt="All retained grasp contact pairs for the selected NIST scene"></a></section>
<section class="panel" style="margin-top:18px"><h2>All 96 placements</h2><p id="totals"></p><table><thead><tr><th>Scene</th><th>Part</th><th>Generated</th><th>Passed geometry</th><th>Retained after grouping</th><th>Status</th></tr></thead><tbody id="allCases"></tbody></table></section></div>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const scenes=[...new Set(D.records.map(r=>r.scene))].sort(), parts=[...new Set(D.records.map(r=>r.part))];
function option(select,value,label){let o=document.createElement('option');o.value=value;o.textContent=label;select.append(o)}
scenes.forEach(s=>option($('scene'),s,s));parts.forEach(p=>option($('part'),p,p.replaceAll('_',' ')));
let record=null, grasp=null, az=-.8,el=.65,zoom=1,drag=null,world=[],lines=[],center=[0,0,0],span=.15;
const rotate=(p,R)=>R.slice(0,3).map(row=>row[0]*p[0]+row[1]*p[1]+row[2]*p[2]+row[3]);
function worldPoint(p,T){return rotate(p,T)}
function rebuildCase(){record=D.records.find(r=>r.scene===$('scene').value&&r.part===$('part').value);$('grasp').innerHTML='';(record?.grasps||[]).forEach((g,i)=>option($('grasp'),i,String(g.id??('G'+i))));if(!record?.grasps?.length)option($('grasp'),-1,'No retained grasp');const image=D.images[$('scene').value];$('overview').src=image;$('overviewLink').href=image;const example=D.examples[record?.scene+'__'+record?.part];$('examplesPanel').hidden=!example;if(example){$('examples').src=example;$('examplesLink').href=example}rebuildGrasp()}
function addMesh(vertices,triangles,color,kind){for(const f of triangles)world.push({p:f.map(i=>vertices[i]),color,kind})}
function rebuildGrasp(){world=[];lines=[];grasp=record?.grasps?.[Number($('grasp').value)]||null;let mesh=D.meshes[record?.part];
if(mesh&&record?.world_T_cad){const verts=mesh.vertices.map(p=>worldPoint(p,record.world_T_cad));addMesh(verts,mesh.triangles,'#a5b0b9','cad');let mn=[0,1,2].map(k=>Math.min(...verts.map(p=>p[k]))),mx=[0,1,2].map(k=>Math.max(...verts.map(p=>p[k])));center=mn.map((x,k)=>(x+mx[k])/2);span=Math.max(...mx.map((x,k)=>x-mn[k]),.025);$('meshNote').textContent=`CAD display: ${mesh.display_triangle_count.toLocaleString()} triangles (${mesh.native_triangle_count.toLocaleString()} in the native file). Any reduction is for display only; validation uses the native CAD.`;}
if(grasp){const opening=$('opening').value==='open'?grasp.opening_m:grasp.display_contact_opening_m;for(const comp of D.hand.components){if($('handDisplay').value==='none')continue;if($('handDisplay').value==='fingers'&&comp.name.includes('hand_collision'))continue;let vertices=comp.vertices_zero_m.map(p=>p.map((v,k)=>v+opening*comp.translation_per_opening[k])).map(p=>worldPoint(p,grasp.world_T_grip));addMesh(vertices,comp.triangles,comp.is_pad?'#d89436':'#377fac','hand')};const sampled=grasp.contacts_world_m,contacts=grasp.contact_points_world_m||sampled;if(sampled?.length===2)lines.push({p:sampled,color:'#d96b98',width:1.5,dots:true,dashed:true});if(contacts?.length===2){lines.push({p:contacts,color:'#147e58',width:3,dots:true});const mid=contacts[0].map((v,k)=>(v+contacts[1][k])/2);lines.push({p:[[mid[0],mid[1],mid[2]+.035],[mid[0],mid[1],mid[2]+.007]],color:'#a44c18',width:3,arrow:true})}}
updateInfo();draw()}
function mm(value){return Number.isFinite(Number(value))?(Number(value)*1000).toFixed(2)+' mm':'unavailable'}
function setRows(target,rows){$(target).replaceChildren();for(const [key,value] of rows){const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td');a.textContent=key;b.textContent=value;tr.append(a,b);$(target).append(tr)}}
function updateInfo(){if(!record)return;$('caseTitle').textContent=record.scene+' / '+record.part.replaceAll('_',' ');const count=record.grasps?.length||0;$('caseStatus').textContent=count?`${count} retained options. All options remain available for later VLM review.`:`No retained grasp: ${record.status||'see saved result'}`;$('caseStatus').className=count?'good':'bad';let timing=record.timing_seconds;let total=typeof timing==='number'?timing:timing?.total??timing?.total_seconds??record.total_seconds;const rows=[['Proposed hand poses',record.generated_count??'unavailable'],['Passed geometry',record.valid_count??'unavailable'],['After grouping',record.grouped_count??count],['Recorded generation/check time',Number.isFinite(total)?total.toFixed(3)+' s':'see saved result']];if(grasp)rows.push(['Displayed option',grasp.id??$('grasp').value],['Sampled contact separation',mm(grasp.width_m)],['Opening before closing',mm(grasp.opening_m)],['Opening at contact',mm(grasp.display_contact_opening_m)],['Finger depth',mm(grasp.pad_depth_m)],['Table clearance',mm(grasp.table_clearance_m)]);setRows('metrics',rows);setRows('rejects',Object.entries(record.rejection_counts||{}).map(([k,v])=>[k.replaceAll('_',' '),v]));$('raw').textContent=JSON.stringify(grasp||record,null,2)}
function projection(p){let q=p.map((v,k)=>v-center[k]),ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);let x=ca*q[0]+sa*q[1],y=-sa*q[0]+ca*q[1];return [x,ce*q[2]-se*y,se*q[2]+ce*y]}
function draw(){let canvas=$('view'),ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;let ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);let scale=Math.min(w,h)/(Math.max(span,.11)*1.8)*zoom;const screen=p=>{let v=projection(p);return [w/2+v[0]*scale,h*.62-v[1]*scale,v[2]]};let table=Number(record?.table_z_m)||0,s=span*.7;let corners=[[-s,-s],[s,-s],[s,s],[-s,s]].map(p=>screen([center[0]+p[0],center[1]+p[1],table]));ctx.fillStyle='#e9eff2';ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();let faces=world.map(f=>({...f,sp:f.p.map(screen)}));faces.sort((a,b)=>a.sp.reduce((s,p)=>s+p[2],0)-b.sp.reduce((s,p)=>s+p[2],0));for(const f of faces){ctx.beginPath();ctx.moveTo(f.sp[0][0],f.sp[0][1]);ctx.lineTo(f.sp[1][0],f.sp[1][1]);ctx.lineTo(f.sp[2][0],f.sp[2][1]);ctx.closePath();ctx.fillStyle=f.color;ctx.globalAlpha=f.kind==='cad'?.92:.86;ctx.fill();ctx.globalAlpha=1}for(const line of lines){const p=line.p.map(screen);ctx.strokeStyle=line.color;ctx.lineWidth=line.width;ctx.setLineDash(line.dashed?[4,4]:[]);ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);ctx.lineTo(p[1][0],p[1][1]);ctx.stroke();if(line.dots){for(const q of p){ctx.fillStyle=line.color;ctx.beginPath();ctx.arc(q[0],q[1],5,0,2*Math.PI);ctx.fill()}}if(line.arrow){let a=Math.atan2(p[1][1]-p[0][1],p[1][0]-p[0][0]);ctx.beginPath();ctx.moveTo(p[1][0],p[1][1]);ctx.lineTo(p[1][0]-10*Math.cos(a-.4),p[1][1]-10*Math.sin(a-.4));ctx.moveTo(p[1][0],p[1][1]);ctx.lineTo(p[1][0]-10*Math.cos(a+.4),p[1][1]-10*Math.sin(a+.4));ctx.stroke()}}ctx.fillStyle='#344b5b';ctx.font='13px system-ui';ctx.fillText('Table plane · all distances in saved data are meters',14,h-15);if(!grasp){ctx.fillStyle='#a7352e';ctx.font='18px system-ui';ctx.fillText('No retained hand pose for this placement',18,30)}}
$('scene').onchange=rebuildCase;$('part').onchange=rebuildCase;$('grasp').onchange=rebuildGrasp;$('opening').onchange=rebuildGrasp;$('handDisplay').onchange=rebuildGrasp;
function changeGrasp(delta){let s=$('grasp');s.selectedIndex=(s.selectedIndex+delta+s.options.length)%s.options.length;rebuildGrasp()}$('prev').onclick=()=>changeGrasp(-1);$('next').onclick=()=>changeGrasp(1);
$('top').onclick=()=>{el=Math.PI/2;az=0;draw()};$('side').onclick=()=>{el=0;az=0;draw()};$('oblique').onclick=()=>{el=.65;az=-.8;draw()};
$('view').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('view').setPointerCapture(e.pointerId)};$('view').onpointerup=()=>drag=null;$('view').onpointermove=e=>{if(drag){az+=(e.clientX-drag[0])*.01;el=Math.max(-1.5,Math.min(1.57,el+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw()}};$('view').onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};window.addEventListener('resize',draw);
let total=D.records.reduce((sum,r)=>sum+(r.grasps?.length||0),0),available=D.records.filter(r=>r.grasps?.length).length;$('totals').textContent=`${available} of ${D.records.length} placements have at least one retained grasp; ${total} retained options in total. Select a row to review it.`;
for(const r of D.records){const tr=document.createElement('tr');for(const value of [r.scene,r.part.replaceAll('_',' '),r.generated_count??'—',r.valid_count??'—',r.grouped_count??r.grasps?.length??0,r.status??'']){let td=document.createElement('td');td.textContent=value;tr.append(td)}tr.style.cursor='pointer';tr.onclick=()=>{$('scene').value=r.scene;$('part').value=r.part;rebuildCase();window.scrollTo({top:100,behavior:'smooth'})};$('allCases').append(tr)}
rebuildCase();
</script></html>'''


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_geometry_20261026.gallery RUN_PATH')
    main(sys.argv[1])
