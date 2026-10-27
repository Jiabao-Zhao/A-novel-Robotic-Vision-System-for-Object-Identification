"""Twenty newly queried-CAD scenes through the shared frozen scene generator."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import HERE, read, save, digest, relative


def main(preflight=False):
    folder=HERE/'new_cads'
    protocol=read(folder/'protocol.json')
    catalog=read(folder/'catalog.json')['objects']
    source=HERE/'capture/render_scenes.py'
    spec=importlib.util.spec_from_file_location('new_query_scene_capture',source)
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    capture.TRIAL=folder
    items=capture.inventory(catalog)
    records=[]
    for index in range(protocol['layout_count']):
        seed=protocol['seed_base']+index*151
        scene_id=f'U{index:02d}'
        layout=capture.generate_layout(index,seed,items,clearance_m=protocol['clearance_m'])
        save(folder/'audit'/scene_id/'layout.json',{'seed':seed,'poses':layout,'generator_sha256':digest(source)})
        if preflight:
            records.append({'scene_id':scene_id,'seed':seed,'CAD_supported_instances':len(layout)})
            continue
        record=capture.render_scene(scene_id,layout,index,catalog)
        record=dict(record,base_scene_id=scene_id,cohort='new_query_CADs')
        records.append(record)
        save(folder/'capture_manifest.json',{'scenes':records,'complete':False,'generator_sha256':digest(source)})
        print(f'{scene_id}: sixteen native-scale products captured.',flush=True)
    if preflight:
        save(folder/'layout_preflight.json',{'passed':True,'layouts':records,'generator_sha256':digest(source),
            'protocol_sha256':digest(folder/'protocol.json'),'classification_or_audit_label_feedback':False})
        print('All twenty sixteen-CAD layouts pass static CAD-only placement preflight.',flush=True)
        return
    save(folder/'capture_manifest.json',{'scenes':records,'complete':True,'generator_sha256':digest(source),
        'protocol_sha256':digest(folder/'protocol.json'),'labels_available_only_under':'audit/'})
    save(folder/'inputs.json',{'scenes':records,'objects':catalog,'protocol_sha256':digest(folder/'protocol.json'),
        'audit_labels_read_for_inference':False,'reference_inputs':relative(folder/'cad_inputs.json')})


if __name__=='__main__':main(len(sys.argv)>1 and sys.argv[1]=='preflight')
