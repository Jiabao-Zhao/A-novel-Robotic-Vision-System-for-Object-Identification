"""Compact contact-family review, reusing rendering without editing old sources."""
import hashlib
import json
from pathlib import Path
import sys

from experiments.cad_grasp_geometry_20261026 import gallery as previous


def compact_html(source):
    replacements = [
      ('NIST scenes — CAD grasp geometry before VLM selection','NIST scenes — compact CAD contact families'),
      ('<h1>CAD grasps before VLM selection</h1>','<h1>Compact CAD contact families</h1>'),
      ('Review the modeled hand and every retained grasp.','Review one distinct final hand placement after merging equivalent regions, with checked hand alternatives.'),
      ('<label>Retained grasp<select id="grasp"></select></label>',
       '<label>Contact family<select id="grasp"></select></label><label>Checked hand alternative<select id="alias"></select></label><label>CAD frame<select id="cadAxes"><option value="show">Show CAD X / Y / Z</option><option value="hide">Hide CAD axes</option></select></label>'),
      ('Previous grasp','Previous family'),('Next grasp','Next family'),
      ('At nominal finger contact','At modeled first contact'),
      ('display:flex;gap:20px;font-size:13px','display:flex;gap:20px;flex-wrap:wrap;font-size:13px'),
      ('<span style="--c:#d96b98">Sampled pair</span>',
       '<span style="--c:#d96b98">Region contact pair</span><span style="--c:#444">Finger closing axis</span><span style="--c:#c83c3c">CAD X</span><span style="--c:#209c52">CAD Y</span><span style="--c:#325cdd">CAD Z</span>'),
      ('Each grasp ID is a review option, not a quality rank. The modeled first contacts may differ from the sampled pair because the finger pads cover a finite area.',
       'Each family is a distinct final hand placement after merging equivalent regions, not a score rank. Every checked jaw assignment and starting opening remains available. The approach stays straight down; the black closing axis follows the horizontal projection of the labeled CAD axis, with its measured deviation shown. CAD axes are drawn at the original CAD frame origin. Green points are witnesses from finite contact regions; their connecting line is not the hand closing axis.'),
      ('<h3>Why other candidates were rejected</h3><table id="rejects"></table>',
       '<h3>Why hand configurations were rejected</h3><table id="rejects"></table><h3>Why contact regions could not produce a centered hand placement</h3><table id="preparationRejects"></table>'),
      ('Sampled pair: the two starting contact points proposed by the sampler.',
       'Region contact pair: contacts recomputed at the chosen region center.'),
      ('Finger depth: where the sampled pair sits along the fingers.',
       'Derived contact offset: one finger position calculated from pad centering and table/part constraints. It is not a sweep through six depth labels. A family contains valid positive/negative jaw assignments and checked starting openings; the default uses the positive CAD direction when valid and the smallest checked starting opening, not a highest-score winner.'),
      ('These are the first retained IDs, shown for inspection. They have no quality order.',
       'These are representative contact-family IDs, shown for inspection with one derived placement per family. The viewer also exposes each valid jaw alternative; neither is a VLM choice.'),
      ('All retained modeled contact pairs in this scene','All retained contact families in this scene'),
      ('Overlapping lines may represent different finger depths. No example is automatically selected as the winner.',
       'Each line represents one contact family. A single pad placement is derived for that region; jaw alternatives are shown under the same family. No VLM winner is selected.'),
      ('<th>Generated</th><th>Passed geometry</th><th>Retained after grouping</th>',
       '<th>Hand configurations checked</th><th>Passed configurations</th><th>Contact families</th>'),
      ('let record=null, grasp=null,','let record=null, family=null, grasp=null,'),
      ("String(g.id??('G'+i))", "String(g.family_id??g.id??('F'+i))"),
      ("if(!record?.grasps?.length)option($('grasp'),-1,'No retained grasp');",
       "if(!record?.grasps?.length)option($('grasp'),-1,'No retained contact family');"),
      ("function rebuildGrasp(){world=[];lines=[];grasp=record?.grasps?.[Number($('grasp').value)]||null;let mesh=D.meshes[record?.part];",
       """function rebuildGrasp(){world=[];lines=[];const selectedFamily=record?.grasps?.[Number($('grasp').value)]||null;if(family!==selectedFamily){family=selectedFamily;$('alias').innerHTML='';const alternatives=family?.jaw_aliases?.length?family.jaw_aliases:(family?[family]:[]);alternatives.forEach((g,i)=>option($('alias'),i,(g.jaw_sign>0?'Positive':'Negative')+' CAD direction · '+g.id+' · starts '+(1000*g.opening_m).toFixed(1)+' mm open'));if(!alternatives.length)option($('alias'),-1,'No valid jaw alternative');else $('alias').value=Math.max(0,alternatives.findIndex(g=>g.id===family.id))}grasp=family?.jaw_aliases?.[Number($('alias').value)]||family;if(grasp)grasp.display_contact_opening_m=grasp.contact_opening_m??family.display_contact_opening_m;let mesh=D.meshes[record?.part];"""),
      ("if(grasp){const opening=$('opening').value==='open'?grasp.opening_m:grasp.display_contact_opening_m;",
       """if(record?.world_T_cad&&$('cadAxes').value==='show'){const T=record.world_T_cad,origin=T.slice(0,3).map(r=>r[3]),length=Math.max(.015,Math.min(span*.45,.035));['X','Y','Z'].forEach((label,k)=>{const end=origin.map((v,j)=>v+length*T[j][k]);lines.push({p:[origin,end],color:['#c83c3c','#209c52','#325cdd'][k],width:2,arrow:true,label:'CAD '+label})})}
if(grasp){const opening=$('opening').value==='open'?grasp.opening_m:grasp.display_contact_opening_m;"""),
      ("ctx.stroke()}}ctx.fillStyle='#344b5b';",
       "ctx.stroke()}if(line.label){ctx.setLineDash([]);ctx.fillStyle=line.color;ctx.font='bold 13px system-ui';ctx.fillText(line.label,p[1][0]+5,p[1][1]-5)}}ctx.fillStyle='#344b5b';"),
      ("$('grasp').onchange=rebuildGrasp;", "$('grasp').onchange=rebuildGrasp;$('alias').onchange=rebuildGrasp;$('cadAxes').onchange=rebuildGrasp;"),
      ('No retained hand pose for this placement','No retained contact family for this placement'),
      ('retained options in total. Select a row to review it.','contact families in total. Select a row to review it.'),
      ('placements have at least one retained grasp;','placements have at least one contact family;'),
    ]
    for old,new in replacements:
        if old not in source:
            raise ValueError('Previous gallery markup changed; cannot apply compact wrapper: '+old[:100])
        source = source.replace(old,new)
    start = source.index('function updateInfo(){')
    stop = source.index('\nfunction projection(',start)
    source = source[:start]+r'''function updateInfo(){if(!record)return;$('caseTitle').textContent=record.scene+' / '+record.part.replaceAll('_',' ');const count=record.grasps?.length||0;$('caseStatus').textContent=count?`${count} distinct contact families. Valid hand alternatives are nested under each family.`:`No retained contact family: ${record.status||'see saved result'}`;$('caseStatus').className=count?'good':'bad';let timing=record.timing_seconds;let total=typeof timing==='number'?timing:timing?.total??timing?.total_seconds??record.total_seconds;const rows=[['Opposing regions after merging',record.merged_contact_regions??'unavailable'],['Hand configurations checked',record.generated_count??'unavailable'],['Passed configurations',record.valid_count??'unavailable'],['Retained contact families',record.grouped_count??count],['Generation/check time',Number.isFinite(total)?total.toFixed(3)+' s':'see saved result']];if(grasp){rows.push(['Displayed family',family.family_id??family.id],['Source region IDs',(family.source_family_ids||[family.family_id]).join(', ')],['Checked hand alternative',grasp.id],['Valid hand alternatives in family',family.jaw_aliases?.length??1],['CAD closing axis',grasp.axis_label??'unavailable'],['Deviation from native CAD axis',Number.isFinite(grasp.cad_axis_projection_error_deg)?grasp.cad_axis_projection_error_deg.toFixed(4)+'°':'unavailable'],['Approach','Straight down'],['Derived hand reference height',mm(grasp.world_T_grip?.[2]?.[3])],['Derived contact offset along fingers',mm(grasp.pad_depth_m)],['Region contact separation',mm(grasp.width_m)],['Opening before closing',mm(grasp.opening_m)],['Opening at first contact',mm(grasp.display_contact_opening_m)],['Table clearance',mm(grasp.table_clearance_m)])};setRows('metrics',rows);setRows('rejects',Object.entries(record.rejection_counts||{}).map(([k,v])=>[k.replaceAll('_',' '),v]));setRows('preparationRejects',Object.entries(record.preparation_rejection_counts||{}).map(([k,v])=>[k.replaceAll('_',' '),v]));$('raw').textContent=JSON.stringify(grasp?{family_id:family.family_id,representative_rule:family.representative_rule,valid_alias_ids:family.grouped_candidate_ids,selected_alias:grasp}:record,null,2)}''' + source[stop:]
    source = source.replace('updateInfo();draw()}', r'''if(grasp){const points=grasp.contact_points_world_m||grasp.contacts_world_m;if(points?.length===2){const mid=points[0].map((v,k)=>(v+points[1][k])/2),length=Math.max(grasp.opening_m*.65,.015),axis=grasp.world_T_grip.slice(0,3).map(row=>row[0]);lines.push({p:[mid.map((v,k)=>v-length*axis[k]),mid.map((v,k)=>v+length*axis[k])],color:'#444',width:1.5,dashed:true,arrow:true,label:'Finger closing'})}}updateInfo();draw()}''')
    return source


def main(run_path):
    original = previous.HTML
    try:
        previous.HTML = compact_html(original)
        previous.main(run_path)
    finally:
        previous.HTML = original
    run = previous.local_path(run_path)
    records = json.loads((run/'results.json').read_text())
    audit = dict(cases=len(records),contact_families=sum(len(r['grasps']) for r in records),
        displayed_jaw_aliases=sum(len(g.get('jaw_aliases',[g])) for r in records for g in r['grasps']),
        cad_axes_overlay=True,derived_height_per_family=True,
        base_gallery_source_sha256=hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        old_gallery_sources_modified=False)
    (run/'gallery/compact_gallery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_regions_20261026.gallery RUN_PATH')
    main(sys.argv[1])
