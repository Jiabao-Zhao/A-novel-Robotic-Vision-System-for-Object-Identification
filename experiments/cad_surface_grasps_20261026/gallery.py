"""Review surface-derived grasps without changing saved generation evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026 import gallery as base
from experiments.cad_grasp_regions_20261026.gallery import compact_html

VIEWER_CHECK = r'''const fs=require('fs'),vm=require('vm'),path=require('path');
const file=process.argv[2]||path.join(__dirname,'index.html'),html=fs.readFileSync(file,'utf8');
const payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.slice(html.lastIndexOf('<script>')+8).split('</script>')[0];
const elements=new Map(),context2d=new Proxy({},{get:(o,k)=>k in o?o[k]:(()=>{})});
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this._value='';this.textContent='';this.clientWidth=900;this.clientHeight=540;this.style={};}
 append(...a){this.children.push(...a);if(this.tag==='select'&&this.children.length===a.length)this._value=this.children[0]?.value??'';}
 replaceChildren(...a){this.children=[];this.append(...a);}
 set innerHTML(v){this.children=[];this._value='';}get innerHTML(){return '';}
 set value(v){this._value=String(v);}get value(){return this._value;}
 get options(){return this.children;}get selectedIndex(){return this.children.findIndex(c=>c.value===this._value);}
 set selectedIndex(i){this._value=this.children[i]?.value??'';}getContext(){return context2d;}setPointerCapture(){}
}
for(const id of ['scene','part','grasp','alias','opening','handDisplay','cadAxes','contactEvidence'])elements.set(id,new Element('select'));
for(const [id,value] of [['opening','contact'],['handDisplay','full'],['cadAxes','hide'],['contactEvidence','paired']])elements.get(id).value=value;
elements.set('data',new Element());elements.get('data').textContent=payload;
const document={getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag)};
const sandbox={document,window:{devicePixelRatio:1,addEventListener(){},scrollTo(){}},console};
vm.createContext(sandbox);vm.runInContext(code,sandbox,{timeout:30000});
const result=vm.runInContext(`(()=>{
 draw=()=>{};
 const assert=(ok,message)=>{if(!ok)throw Error(message)};
 let cases=0,families=0,aliases=0,zeroCases=0,surfaceAliases=0,shaftAliases=0;
 for(const expected of D.records){
  $('scene').value=expected.scene;$('part').value=expected.part;$('cadAxes').value='hide';$('contactEvidence').value='paired';rebuildCase();
  assert(record===expected,'case mismatch');cases++;
  assert(!lines.some(l=>l.label?.startsWith('CAD ')),'native CAD axes should be optional');
  if(!expected.grasps.length){zeroCases++;assert(grasp===null,'zero case has a hand');continue;}
  for(let i=0;i<expected.grasps.length;i++){
   $('grasp').value=i;rebuildGrasp();families++;
   const familyRecord=expected.grasps[i],members=familyRecord.jaw_aliases?.length?familyRecord.jaw_aliases:[familyRecord];
   assert($('alias').options.length===members.length,'alias option count');
   assert(grasp.id===familyRecord.id,'default differs from canonical ID');
   for(let j=0;j<members.length;j++){
    $('alias').value=j;$('contactEvidence').value='paired';rebuildGrasp();aliases++;
    assert(grasp.id===members[j].id,'wrong hand alternative');
    assert(lines.some(l=>l.label==='Finger closing'),'closing direction missing');
    const pair=grasp.paired_surface_points_world_m;
    assert(pair?.length===2,'paired surface points missing');
    assert(JSON.stringify(grasp.contact_points_world_m)===JSON.stringify(pair),'display still uses arbitrary witnesses');
    const delta=pair[1].map((v,k)=>v-pair[0][k]),T=grasp.world_T_grip;
    for(const k of [1,2])assert(Math.abs(delta.reduce((s,v,l)=>s+v*T[l][k],0))<1e-8,'paired contact Y/Z mismatch');
    assert(grasp.actual_pad_normal_errors_deg?.length===2,'pad normal evidence missing');
    const regions=grasp.pad_surface_regions||[];
    assert(regions.length===2&&regions.every(p=>p.polygons?.length),'two pad regions missing');
    assert(world.some(f=>f.kind==='nearContact'),'near-contact polygons absent');surfaceAliases++;
    if(cylinderFor(grasp)?.shaft_axis_world){assert(lines.some(l=>l.label==='Physical shaft'),'physical shaft overlay absent');shaftAliases++;}
    $('contactEvidence').value='witnesses';rebuildGrasp();
    assert(lines.filter(l=>l.color==='#9555ba').length===(grasp.raw_first_contact_witnesses_world_m||[]).length,'raw witnesses missing');
   }
   for(const mode of ['open','contact']){$('opening').value=mode;rebuildGrasp();}
   for(const mode of ['full','fingers','none']){$('handDisplay').value=mode;rebuildGrasp();}
   $('handDisplay').value='full';
  }
  $('cadAxes').value='show';rebuildGrasp();assert(lines.filter(l=>l.label?.startsWith('CAD ')).length===3,'optional CAD reference missing');
 }
 return {cases,contact_families:families,checked_hand_alternatives:aliases,zero_family_cases:zeroCases,aliases_with_surface_regions:surfaceAliases,aliases_with_shaft_overlay:shaftAliases,javascript_bootstrap:'passed',all_selectors:'passed',canonical_defaults:'passed',paired_points_and_surface_regions:'passed',native_axes_optional:'passed',raw_witness_toggle:'passed',hand_modes:'passed'};
})()`,sandbox,{timeout:180000});
fs.writeFileSync(path.join(path.dirname(file),'viewer_verification.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
'''


def presentation_records(records, hand):
    """Paired surface points replace arbitrary witnesses only in this viewer."""
    result = deepcopy(records)
    for record in result:
        for family in record.get('grasps', []):
            for grasp in [family]+family.get('jaw_aliases', []):
                witnesses = grasp.get('contact_points_world_m')
                if witnesses is not None:
                    grasp['raw_first_contact_witnesses_world_m'] = witnesses
                pair = grasp.get('paired_surface_points_world_m')
                if pair is not None:
                    grasp['contact_points_world_m'] = pair
                grasp['display_contact_opening_m'] = base.closure_opening(grasp, hand)
    return result


def surface_html():
    source = compact_html(base.HTML)
    replacements = [
      ('NIST scenes — compact CAD contact families', 'NIST scenes — surface-derived CAD grasps'),
      ('<h1>Compact CAD contact families</h1>', '<h1>Surface-derived CAD grasp families</h1>'),
      ('<label>CAD frame<select id="cadAxes"><option value="show">Show CAD X / Y / Z</option><option value="hide">Hide CAD axes</option></select></label>',
       '<label>CAD reference<select id="cadAxes"><option value="hide">Hide native CAD axes</option><option value="show">Show native CAD X / Y / Z</option></select></label><label>Contact evidence<select id="contactEvidence"><option value="paired">Paired points and surface regions</option><option value="witnesses">Also show raw first-touch witnesses</option></select></label>'),
      ('<span style="--c:#147e58">Modeled first contacts</span>', '<span style="--c:#147e58">Paired points on opposed surfaces</span><span style="--c:#00b59a">25 µm near-contact surface band</span>'),
      ('<span style="--c:#d96b98">Region contact pair</span>', '<span style="--c:#d96b98">Proposed region pair</span><span style="--c:#9555ba">Optional raw first-touch witnesses</span>'),
      ('Each family is a distinct final hand placement after merging equivalent regions, not a score rank. Every checked jaw assignment and starting opening remains available. The approach stays straight down; the black closing axis follows the horizontal projection of the labeled CAD axis, with its measured deviation shown. CAD axes are drawn at the original CAD frame origin. Green points are witnesses from finite contact regions; their connecting line is not the hand closing axis.',
       'Each family is an unranked review option. Closing directions come from actual surfaces or a fitted round-section shaft, not arbitrary native CAD X/Y/Z labels. The approach stays straight down. Green paired points share the same location across the finger pads and lie on the original CAD triangles. Turquoise regions lie within a 25 µm geometric band of the pads; they are not measured loaded contact areas. Optional purple markers show the original first-touch witnesses without treating their separation as the closing direction.'),
      ('Region contact pair: contacts recomputed at the chosen region center.', 'Proposed region pair: contacts recomputed during hand placement.'),
      ('Modeled first contacts: the points where each finite finger pad first touches the CAD during modeled closing.', 'Green paired surface points are selected from the common opposed near-contact region of both pads. Raw first-touch witnesses remain in the saved information and optional purple markers.'),
      ('Derived contact offset: one finger position calculated from pad centering and table/part constraints. It is not a sweep through six depth labels. A family contains valid positive/negative jaw assignments and checked starting openings; the default uses the positive CAD direction when valid and the smallest checked starting opening, not a highest-score winner.',
       'Contact offset: the position of the proposed contact along the fingers. Every saved checked hand alternative is available; the initial display is the saved canonical representative, not a VLM winner. Surface-normal errors are reported for each pad. Fitted-cylinder tangency and original triangle-normal alignment are separate checks.'),
      (' positive CAD direction', ' positive surface direction'),
      ("+' CAD direction · '", "+' surface direction · '"),
      ("$('cadAxes').onchange=rebuildGrasp;", "$('cadAxes').onchange=rebuildGrasp;$('contactEvidence').onchange=rebuildGrasp;"),
      ("ctx.globalAlpha=f.kind==='cad'?.92:.86;", "ctx.globalAlpha=f.kind==='nearContact'?.6:(f.kind==='cad'?.92:.86);"),
      ('Lines connect modeled first finger contacts.', 'Lines connect paired points on the opposed native surfaces.'),
      ('Contact points only', 'Contact regions only'),
    ]
    for old, new in replacements:
        if old not in source:
            if old==' positive CAD direction':
                continue
            raise ValueError('Gallery source changed: '+old[:100])
        source = source.replace(old, new)
    start = source.index('function updateInfo(){')
    stop = source.index('\nfunction projection(', start)
    source = source[:start]+r'''function degrees(values){return Array.isArray(values)?values.map(v=>Number(v).toFixed(4)+'°').join(' / '):'unavailable'}
function directionFor(g){return g?.direction||g?.direction_record||(record?.directions||[]).find(d=>d.id===g?.direction_id||d.label===g?.axis_label||d.label===g?.direction_label)||{}}
function cylinderFor(g){const d=directionFor(g);return g?.cylinder||d.cylinder||d.cylinder_model||g?.cylinder_model||null}
function updateInfo(){if(!record)return;$('caseTitle').textContent=record.scene+' / '+record.part.replaceAll('_',' ');const count=record.grasps?.length||0;$('caseStatus').textContent=count?`${count} distinct contact families. Every valid checked hand alternative is available.`:`No retained contact family: ${record.status||'see saved result'}`;$('caseStatus').className=count?'good':'bad';let timing=record.timing_seconds;let total=typeof timing==='number'?timing:timing?.total??timing?.total_seconds??record.total_seconds;const rows=[['Surface direction families',(record.directions||record.axes||[]).filter(d=>d.usable!==false).length],['Hand configurations checked',record.generated_count??'unavailable'],['Passed configurations',record.valid_count??'unavailable'],['Retained contact families',record.grouped_count??count],['Generation/check time',Number.isFinite(total)?total.toFixed(3)+' s':'see saved result']];if(grasp){const d=directionFor(grasp);rows.push(['Displayed family',family.family_id??family.id],['Source region IDs',(family.source_family_ids||[family.family_id]).join(', ')],['Checked hand alternative',grasp.id],['Valid hand alternatives in family',family.jaw_aliases?.length??1],['Closing direction source',grasp.direction_label??grasp.axis_label??d.label??'unavailable'],['Source geometry',grasp.direction_source??d.source??d.kind??'see saved information'],['Actual touched geometry',grasp.contact_geometry??'see saved information'],['Original triangle normal errors (two pads)',degrees(grasp.actual_pad_normal_errors_deg)],['Allowed original triangle normal error',Number.isFinite(grasp.surface_normal_tolerance_deg)?grasp.surface_normal_tolerance_deg.toFixed(3)+'°':'unavailable'],['Fitted round-side tangency errors',degrees(grasp.fitted_cylinder_tangent_errors_deg)],['Fitted tangency allowances (base + mesh sector)',degrees(grasp.fitted_cylinder_tangent_allowances_deg)],['Native triangle radial sector widths',degrees(grasp.mesh_contact_sector_widths_deg)],['Departure from perpendicular to shaft',Number.isFinite(grasp.cylinder_closing_perpendicular_error_deg)?grasp.cylinder_closing_perpendicular_error_deg.toFixed(4)+'°':'not a round-side contact'],['Near-contact band','25 µm geometric band; not loaded area'],['Approach','Straight down'],['Hand reference height',mm(grasp.world_T_grip?.[2]?.[3])],['Contact offset along fingers',mm(grasp.pad_depth_m)],['Proposed contact separation',mm(grasp.width_m)],['Opening before closing',mm(grasp.opening_m)],['Opening at first contact',mm(grasp.display_contact_opening_m)],['Table clearance',mm(grasp.table_clearance_m)])};setRows('metrics',rows);setRows('rejects',Object.entries(record.rejection_counts||{}).map(([k,v])=>[k.replaceAll('_',' '),v]));setRows('preparationRejects',Object.entries(record.preparation_rejection_counts||{}).map(([k,v])=>[k.replaceAll('_',' '),v]));$('raw').textContent=JSON.stringify(grasp?{family_id:family.family_id,representative_rule:family.representative_rule,valid_alias_ids:family.grouped_candidate_ids,selected_alias:grasp}:record,null,2)}''' + source[stop:]
    source = source.replace("setRows('metrics',rows);", "if(grasp?.center_projection){const shift=grasp.center_projection.translation_along_closing_m;rows.push(['Geometric hand-centering shift',Number.isFinite(shift)?(1000*shift).toFixed(4)+' mm along the positive surface direction':'unavailable'])}setRows('metrics',rows);")
    marker = 'updateInfo();draw()}'
    extra = r'''if(grasp){for(const pad of grasp.pad_surface_regions||[]){for(const region of pad.polygons||[]){const points=(region.polygon_local_m||[]).map(p=>worldPoint(p,grasp.world_T_grip));if(points.length>=3){const triangles=[];for(let i=1;i<points.length-1;i++)triangles.push([0,i,i+1]);addMesh(points,triangles,'#00b59a','nearContact')}}}if($('contactEvidence').value==='witnesses'){for(const p of grasp.raw_first_contact_witnesses_world_m||[])lines.push({p:[p,p],color:'#9555ba',width:1,dots:true})}const cylinder=cylinderFor(grasp);if(cylinder?.shaft_axis_world){const c=cylinder.point_on_axis_world||grasp.contacts_world_m[0].map((v,k)=>(v+grasp.contacts_world_m[1][k])/2),s=cylinder.shaft_axis_world,length=Math.max(span*.55,.02);lines.push({p:[c.map((v,k)=>v-length*s[k]),c.map((v,k)=>v+length*s[k])],color:'#944a16',width:2,arrow:true,label:'Physical shaft'})}}updateInfo();draw()}'''
    if marker not in source:
        raise ValueError('Missing gallery rebuild hook')
    source = source.replace(marker, extra)
    return source


def surface_world(grasp):
    pose = np.asarray(grasp['world_T_grip'])
    return [np.asarray(region['polygon_local_m'])@pose[:3,:3].T+pose[:3,3]
            for pad in grasp.get('pad_surface_regions', []) for region in pad['polygons']]


def build_overview(records, meshes, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    paths={}
    parts=list(dict.fromkeys(r['part'] for r in records))
    for scene in sorted({r['scene'] for r in records}):
        cases={r['part']:r for r in records if r['scene']==scene}
        fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
        for ax,part in zip(axes.flat,parts):
            record=cases.get(part)
            ax.set_aspect('equal')
            ax.set_axis_off()
            if record is None:
                ax.set_title(part+' — no record')
                continue
            mesh=meshes.get(part)
            if mesh:
                pose=np.asarray(record['world_T_cad'])
                vertices=np.asarray(mesh['vertices'])@pose[:3,:3].T+pose[:3,3]
                center=(vertices.min(axis=0)+vertices.max(axis=0))/2
                points=(vertices-center)*1000
                ax.add_collection(PolyCollection(points[np.asarray(mesh['triangles']),:2],facecolors='#c9d1d8',edgecolors='none',alpha=.85))
                grasps=record.get('grasps',[])
                colors=plt.get_cmap('turbo')(np.linspace(.08,.92,max(1,len(grasps))))
                for color,grasp in zip(colors,grasps):
                    pair=(np.asarray(grasp['contact_points_world_m'])-center)*1000
                    ax.plot(pair[:,0],pair[:,1],'o-',color=color,ms=3,lw=.8,zorder=5)
                extent=max(float(np.ptp(points[:,:2],axis=0).max())*.7,8)
                ax.set_xlim(-extent,extent)
                ax.set_ylim(-extent,extent)
            count=len(record.get('grasps',[]))
            ax.set_title(part.replace('_',' ')+f'\n{count} contact families',fontsize=10,color='#243342' if count else '#ad342a')
        for ax in axes.flat[len(parts):]:
            ax.set_axis_off()
        fig.suptitle(f'{scene}: all retained surface-contact families, viewed from above\n'
                     'Each line joins paired points on opposed native surfaces. No VLM choice or pickup execution.',fontsize=13)
        path=output/(scene+'_contacts.png')
        fig.savefig(path,dpi=130,facecolor='white')
        plt.close(fig)
        paths[scene]=path.name
    return paths


def build_examples(records, meshes, hand, output):
    """Static pad close-ups use the same paired points and near-contact regions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    paths = {}
    for record in records:
        examples = record.get('grasps', [])[:3]
        mesh = meshes.get(record['part'])
        if not examples or mesh is None:
            continue
        transform = np.asarray(record['world_T_cad'])
        cad = np.asarray(mesh['vertices'])@transform[:3,:3].T+transform[:3,3]
        center = (cad.min(axis=0)+cad.max(axis=0))/2
        fig, axes = plt.subplots(3,len(examples),figsize=(max(11,5*len(examples)),10),squeeze=False,constrained_layout=True)
        for column,grasp in enumerate(examples):
            pose = np.asarray(grasp['world_T_grip'])
            objects = [(cad,np.asarray(mesh['triangles']),'#aebbc4',False)]
            for component in hand['components']:
                local = np.asarray(component['vertices_zero_m'])+grasp['display_contact_opening_m']*np.asarray(component['translation_per_opening'])
                vertices = local@pose[:3,:3].T+pose[:3,3]
                objects.append((vertices,np.asarray(component['triangles']),'#d89436' if component['is_pad'] else '#377fac',not component['is_pad']))
            contacts = np.asarray(grasp['contact_points_world_m'])
            patches = surface_world(grasp)
            for row in range(3):
                ax = axes[row,column]
                if row==0:
                    project = lambda p: (np.asarray(p)-center)[:,:2]*1000
                else:
                    project = lambda p: ((np.asarray(p)-pose[:3,3])@pose[:3,:3])[:,[0,2]]*[1000,-1000]
                all_points=[]
                for vertices,faces,color,outline in objects:
                    points = project(vertices)
                    if outline:
                        indices = ConvexHull(points).vertices
                        p = points[np.r_[indices,indices[0]]]
                        ax.plot(p[:,0],p[:,1],color=color,lw=1.1)
                    else:
                        ax.add_collection(PolyCollection(points[faces],facecolors=color,edgecolors='none',alpha=.8))
                    all_points.append(points)
                for patch in patches:
                    ax.add_collection(PolyCollection([project(patch)],facecolors='#00b59a',edgecolors='#00a68f',linewidths=.35,alpha=.65,zorder=8))
                pair = project(contacts)
                ax.plot(pair[:,0],pair[:,1],'o-',color='#147e58',lw=1.4,ms=4,zorder=10)
                if row==2:
                    middle = pair.mean(axis=0)
                    half_width = max(abs(pair[:,0]).max()+7,13)
                    ax.set_xlim(-half_width,half_width)
                    ax.set_ylim(middle[1]-13,middle[1]+13)
                else:
                    combined=np.concatenate(all_points)
                    mid=(combined.min(axis=0)+combined.max(axis=0))/2
                    extent=max(float(np.ptp(combined,axis=0).max())*.56,12)
                    ax.set_xlim(mid[0]-extent,mid[0]+extent)
                    ax.set_ylim(mid[1]-extent,mid[1]+extent)
                if row:
                    ax.axhline((record['table_z_m']-pose[2,3])*1000,color='#687e8e',lw=1.3)
                ax.set_aspect('equal')
                ax.grid(alpha=.15)
                ax.set_xlabel('World X from CAD center (mm)' if row==0 else 'Across finger opening (mm)')
                ax.set_ylabel('World Y from CAD center (mm)' if row==0 else 'Height from hand reference (mm)')
                angle_text=' / '.join(f'{a:.4f}°' for a in grasp.get('actual_pad_normal_errors_deg', []))
                title=f"{grasp['id']} — "+('above' if row==0 else 'side' if row==1 else 'pad close-up')
                title+=f'\nTriangle normal errors: {angle_text or "unavailable"}'
                title+=f" (limit {grasp.get('surface_normal_tolerance_deg','?')}°)"
                if row==2 and grasp.get('fitted_cylinder_tangent_errors_deg'):
                    error=' / '.join(f'{a:.2f}°' for a in grasp['fitted_cylinder_tangent_errors_deg'])
                    allowance=' / '.join(f'{a:.2f}°' for a in grasp.get('fitted_cylinder_tangent_allowances_deg',[]))
                    title+=f'\nFitted tangent: {error}; allowances: {allowance}'
                ax.set_title(title,fontsize=9)
        fig.suptitle(f"{record['scene']} / {record['part'].replace('_',' ')} — unranked examples, all families in viewer\n"
                     'Green = paired points on opposed native surfaces; turquoise = 25 µm geometric band, not loaded contact area.\n'
                     'Blue = hand outlines; orange = pads; gray = CAD. No VLM choice or pickup execution.',fontsize=11)
        path=output/(base.case_key(record)+'_hand_examples.png')
        fig.savefig(path,dpi=115,facecolor='white')
        plt.close(fig)
        paths[base.case_key(record)]=path.name
    return paths


def main(run_path):
    run=base.local_path(run_path)
    source=run/'results.json'
    frozen_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    raw=json.loads(source.read_text())
    hand=json.loads((base.ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json').read_text())
    records=presentation_records(raw if isinstance(raw,list) else raw['results'],hand)
    output=run/'gallery'
    output.mkdir(parents=True,exist_ok=True)
    meshes={}
    for record in records:
        if record.get('mesh_path') and record['part'] not in meshes:
            meshes[record['part']]=base.display_mesh(base.local_path(record['mesh_path']))
    for component in hand['components']:
        points=np.asarray(component['vertices_zero_m'])
        hull=ConvexHull(points)
        used=np.unique(hull.simplices)
        mapping=np.full(len(points),-1,dtype=int)
        mapping[used]=np.arange(len(used))
        component['vertices_zero_m']=np.round(points[used],8).tolist()
        component['triangles']=mapping[hull.simplices].tolist()
    images=build_overview(records,meshes,output)
    examples=build_examples(records,meshes,hand,output)
    payload=dict(records=records,meshes=meshes,hand=hand,images=images,examples=examples)
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (output/'index.html').write_text(surface_html().replace('__DATA__',encoded),encoding='utf-8')
    (output/'verify_viewer.cjs').write_text(VIEWER_CHECK,encoding='utf-8')
    audit=dict(cases=len(records),contact_families=sum(len(r.get('grasps',[])) for r in records),
        checked_hand_alternatives=sum(len(g.get('jaw_aliases',[g])) for r in records for g in r.get('grasps',[])),
        static_example_sheets=len(examples),paired_points_used_for_green_lines=True,
        near_contact_band_m=.000025,near_contact_band_is_loaded_area=False,
        original_witnesses_retained=True,generation_results_sha256=frozen_hash,
        generation_results_unchanged=frozen_hash==hashlib.sha256(source.read_bytes()).hexdigest(),
        gallery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'gallery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(output/'index.html')


if __name__=='__main__':
    main(sys.argv[1])
