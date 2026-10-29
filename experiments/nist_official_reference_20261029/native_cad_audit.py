"""Read official native CAD metadata without a CAD installation or execution.

Modern SolidWorks stream framing follows the MIT-licensed openswx project's
documented container layout, inspected at:
https://github.com/schwitters/openswx/blob/main/libopenswx/src/internal/modern_parser.cc
This small metadata reader checks decoded size and CRC. It does not interpret
Parasolid B-rep or binary mate constraints and never modifies source archives.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'outputs/simulation/nist_task_board_1/assets'
OUTPUT=ROOT/'outputs/nist_official_reference_20261029'
MARKER=bytes.fromhex('140006000800')


def instance_tree(xml_bytes):
    """Read the saved configuration's component transforms, not binary mates.

    XML stores row-vector homogeneous matrices; transpose to the conventional
    column-vector form. The union of transformed child bounding boxes is an
    independent check against each assembly model's saved bounding box.
    """
    tree=ET.fromstring(xml_bytes)
    models={x.get('id'):x for x in tree.iter() if x.tag.rsplit('}',1)[-1]=='swModel'}
    filemap={x.get('id'):dict(x.attrib) for x in tree.iter()
             if x.tag.rsplit('}',1)[-1]=='swFile'}
    output=[]
    for key,model in models.items():
        bbox=np.fromstring(model.get('swBoundingBox'),sep=' ')
        refs=[];corners=[]
        for ref in model:
            if ref.tag.rsplit('}',1)[-1]!='swReference': continue
            child=models[ref.get('swModelRef')]
            child_bbox=np.fromstring(child.get('swBoundingBox'),sep=' ')
            transform=np.fromstring(ref.get('swTransform'),sep=' ').reshape(4,4).T
            center=(child_bbox[:3]+child_bbox[3:])/2
            points=np.array(list(itertools.product(*zip(child_bbox[:3],child_bbox[3:]))))
            transformed=points@transform[:3,:3].T+transform[:3,3]
            corners.append(transformed)
            refs.append(dict(name=ref.get('swName'),reference_number=ref.get('swReferenceNumber'),
                model_id=ref.get('swModelRef'),hidden=ref.get('swHidden'),
                suppressed=ref.get('swSuppressed'),
                transform_child_native_to_parent_m=transform.tolist(),
                child_native_bbox_m=child_bbox.tolist(),
                child_native_bbox_center_m=center.tolist(),
                child_bbox_center_in_parent_m=(transform[:3,:3]@center+transform[:3,3]).tolist()))
        item=dict(id=key,name=model.get('swName'),native_bbox_m=bbox.tolist(),
            file_reference=filemap.get(model.get('swFileRef')),references=refs)
        if corners:
            points=np.concatenate(corners)
            union=np.r_[points.min(0),points.max(0)]
            item['transformed_children_bbox_union_m']=union.tolist()
            item['bbox_union_max_abs_error_m']=float(np.max(np.abs(union-bbox)))
        output.append(item)
    configuration=next(x for x in tree.iter() if x.tag.rsplit('}',1)[-1]=='swConfiguration')
    root=next(x for x in output if x['id']==configuration.get('swModelRef'))
    board=next((x for x in root['references'] if x['name']=='GMC_Laser_Plate_Virtual'),None)
    result=dict(matrix_storage='XML row-vector 4x4 transposed to column-vector homogeneous transform',
        units='m',models=output,root_model_id=root['id'])
    if board:
        bbox=np.array(board['child_native_bbox_m']);center=(bbox[:3]+bbox[3:])/2
        center_shift=np.eye(4);center_shift[:3,3]=-center
        assembly_to_board=center_shift@np.linalg.inv(np.array(board['transform_child_native_to_parent_m']))
        result['board_frame']=dict(origin='board native bounding-box center',
            axes='native plate XYZ; no simulation-frame rotation inferred',
            size_m=(bbox[3:]-bbox[:3]).tolist(),
            native_bbox_center_m=center.tolist(),
            transform_assembly_to_centered_native_board_m=assembly_to_board.tolist(),
            working_face_note='Receiver openings and assembled components extend toward native board -Z; the current simulation uses +Z. Do not copy matrices without an explicit frame conversion.')
        components=[]
        for ref in root['references']:
            native=assembly_to_board@np.array(ref['transform_child_native_to_parent_m'])
            c=np.array(ref['child_native_bbox_center_m'])
            local=np.eye(4);local[:3,3]=c
            centered=native@local
            components.append(dict(name=ref['name'],reference_number=ref['reference_number'],
                model_id=ref['model_id'],hidden=ref['hidden'],suppressed=ref['suppressed'],
                transform_component_native_to_centered_board_m=native.tolist(),
                transform_component_bbox_center_frame_to_centered_board_m=centered.tolist(),
                component_bbox_center_in_centered_board_mm=(centered[:3,3]*1000).tolist(),
                component_bbox_center_xy_in_board_stl_0_to_384_frame_mm=(centered[:2,3]*1000+192).tolist()))
        result['top_level_components_relative_to_board']=components
        byid={x['id']:x for x in output}
        flattened=[]
        def visit(model_id,parent_transform,path):
            for ref in byid[model_id]['references']:
                transform=parent_transform@np.array(ref['transform_child_native_to_parent_m'])
                name=f"{ref['name']}#{ref['reference_number']}"
                trail=path+[name]
                bbox=np.array(ref['child_native_bbox_m'])
                center=(bbox[:3]+bbox[3:])/2
                point=transform[:3,:3]@center+transform[:3,3]
                halfextent=np.abs(transform[:3,:3])@((bbox[3:]-bbox[:3])/2)
                flattened.append(dict(path=trail,name=ref['name'],model_id=ref['model_id'],
                    is_subassembly=bool(byid[ref['model_id']]['references']),
                    transform_component_native_to_centered_board_m=transform.tolist(),
                    component_bbox_center_in_centered_board_mm=(point*1000).tolist(),
                    component_bbox_center_xy_in_board_stl_0_to_384_frame_mm=(point[:2]*1000+192).tolist(),
                    component_bbox_Z_range_in_centered_board_mm=((point[2]+np.array([-halfextent[2],halfextent[2]]))*1000).tolist()))
                visit(ref['model_id'],transform,trail)
        visit(root['id'],assembly_to_board,[])
        result['recursive_components_relative_to_board']=flattened
    return result


def modern_streams(data):
    """Decode named raw-deflate streams, retaining only size/CRC-valid data."""
    position=0; streams={}; records=[]; rejected=0
    if data[4:8]!=b'\0\0\0\4': raise ValueError('Not inspected modern SW container')
    rotation=data[7]%8
    while True:
        offset=data.find(MARKER,position)
        if offset<0: break
        position=offset+len(MARKER)
        if offset+26>len(data): rejected+=1; continue
        type_id,crc,compressed,uncompressed,nlen=struct.unpack_from('<IIIII',data,offset+6)
        if not 0<nlen<512 or compressed>64*1024**2 or uncompressed>128*1024**2:
            rejected+=1; continue
        encoded=data[offset+26:offset+26+nlen]
        name=bytes(((v>>rotation)|(v<<(8-rotation)))&255 for v in encoded)
        if any(v<32 or v>126 for v in name): rejected+=1; continue
        begin=offset+26+nlen; end=begin+compressed
        if end>len(data): rejected+=1; continue
        try: payload=zlib.decompress(data[begin:end],-15)
        except zlib.error: rejected+=1; continue
        if len(payload)!=uncompressed or zlib.crc32(payload)!=crc:
            rejected+=1; continue
        name=name.decode('ascii')
        duplicate=name in streams
        streams.setdefault(name,payload)
        records.append(dict(name=name,offset=offset,type_id=type_id,
                            bytes=len(payload),crc32=f'{crc:08x}',duplicate=duplicate))
        position=end
    return streams,records,rejected


def offline_checks():
    """Small parser/transform checks; no simulation or robot execution."""
    payload=b'<metadata>test</metadata>';name=b'metadata.xml'
    compressor=zlib.compressobj(wbits=-15)
    compressed=compressor.compress(payload)+compressor.flush()
    encoded=bytes(((v>>4)|(v<<4))&255 for v in name)
    frame=(b'\0\0\0\0\0\0\0\4'+MARKER+
           struct.pack('<IIIII',1,zlib.crc32(payload),len(compressed),len(payload),len(name))+
           encoded+compressed)
    streams,_,_=modern_streams(frame)
    assert streams['metadata.xml']==payload
    damaged=bytearray(frame);damaged[18]^=1
    assert not modern_streams(bytes(damaged))[0], 'Corrupt CRC must not be accepted'
    # Deliberately noncommuting rotation/translation, with the expected point
    # calculated by hand: (0.4,0,0) -> (0,0.4,0)+(0.1,0.2,0.3).
    xml=b'''<swSolidWorks><swModelList>
      <swModel id="2" swName="root" swBoundingBox="-1 -1 -1 1 1 1">
        <swReference swName="GMC_Laser_Plate_Virtual" swModelRef="3" swReferenceNumber="1" swTransform="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
        <swReference swName="sub" swModelRef="4" swReferenceNumber="1" swTransform="0 1 0 0 -1 0 0 0 0 0 1 0 0.1 0.2 0.3 1"/>
      </swModel>
      <swModel id="3" swName="GMC_Laser_Plate_Virtual" swBoundingBox="-1 -1 -1 1 1 1"/>
      <swModel id="4" swName="sub" swBoundingBox="0.3 -0.1 -0.1 0.5 0.1 0.1">
        <swReference swName="child" swModelRef="5" swReferenceNumber="1" swTransform="1 0 0 0 0 1 0 0 0 0 1 0 0.4 0 0 1"/>
      </swModel>
      <swModel id="5" swName="child" swBoundingBox="-0.1 -0.1 -0.1 0.1 0.1 0.1"/>
      </swModelList><swConfiguration swModelRef="2"/></swSolidWorks>'''
    parsed=instance_tree(xml)
    child=next(x for x in parsed['recursive_components_relative_to_board'] if x['name']=='child')
    np.testing.assert_allclose(child['component_bbox_center_in_centered_board_mm'],[100,600,300],atol=1e-10)
    for model in parsed['models']:
        assert model.get('bbox_union_max_abs_error_m',0)<1e-12
    current=json.loads((OUTPUT/'native_cad_audit.json').read_text())
    count=0
    for doc in current['inspected_documents']:
        for model in doc.get('assembly_metadata',{}).get('models',[]):
            assert model.get('bbox_union_max_abs_error_m',0)<1e-12
            for ref in model['references']:
                transform=np.array(ref['transform_child_native_to_parent_m'])
                np.testing.assert_allclose(transform[3],[0,0,0,1],atol=1e-12)
                np.testing.assert_allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-10)
                np.testing.assert_allclose(np.linalg.det(transform[:3,:3]),1,atol=1e-10)
                count+=1
    return dict(named_stream_roundtrip=True,CRC_corruption_rejected=True,
        row_to_column_transpose_and_parent_chain=True,bbox_union_reconstruction=True,
        checked_native_component_transforms=count,max_allowed_bbox_error_m=1e-12)


def audit():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    archive=ASSETS/'cad.zip'
    findings=[]
    with zipfile.ZipFile(archive) as source:
        inventory=[dict(name=x.filename,bytes=x.file_size,
                        sha256=hashlib.sha256(source.read(x)).hexdigest())
                   for x in source.infolist() if x.filename.lower().endswith(('.sldasm','.sldprt','.slddrw'))]
        selected=[x['name'] for x in inventory if x['name'].lower().endswith('.sldasm')]
        selected+=['taskboard_1_cad/MCON-E10-SP.SLDPRT','taskboard_1_cad/MCON-E10-SS.SLDPRT']
        for filename in selected:
            data=source.read(filename)
            streams,records,rejected=modern_streams(data)
            folder=OUTPUT/'native_cad'/filename.replace('/','__')
            folder.mkdir(parents=True,exist_ok=True)
            saved=[]
            for name,payload in streams.items():
                if name.endswith('.xml') or name.startswith('swXmlContents/'):
                    target=folder/(name.replace('/','__')+'.xml' if not name.endswith('.xml') else name.replace('/','__'))
                    target.write_bytes(payload); saved.append(str(target.relative_to(ROOT)))
                if 'PreviewPNG' in name:
                    at=payload.find(b'\x89PNG\r\n\x1a\n')
                    if at>=0:
                        target=folder/(name.replace('/','__')+'.png')
                        target.write_bytes(payload[at:]); saved.append(str(target.relative_to(ROOT)))
            item=dict(archive_member=filename,sha256=hashlib.sha256(data).hexdigest(),
                      parsed_streams=len(streams),valid_records=records,
                      other_markers_not_decoded=rejected,saved_metadata_and_previews=saved,
                      native_Brep_decoded=False,binary_mates_decoded=False)
            tree=streams.get('swXmlContents/COMPINSTANCETREE')
            if tree:
                xml=ET.fromstring(tree)
                item['XML_tag_counts']={tag:sum(1 for x in xml.iter() if x.tag.rsplit('}',1)[-1]==tag)
                    for tag in sorted({x.tag.rsplit('}',1)[-1] for x in xml.iter()})}
                item['model_version']=xml.get('swVersion')
                item['assembly_metadata']=instance_tree(tree)
                target=folder/'decoded_instance_tree.json'
                target.write_text(json.dumps(item['assembly_metadata'],indent=2)+'\n')
                saved.append(str(target.relative_to(ROOT)))
            findings.append(item)
    result=dict(archive=str(archive.relative_to(ROOT)),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        inventory=inventory,inspected_documents=findings,
        method='Python stdlib named-stream decoding with raw DEFLATE, decoded-size and CRC32 checks; metadata only',
        format_reference='https://github.com/schwitters/openswx/blob/main/libopenswx/src/internal/modern_parser.cc',
        installed_native_CAD_reader_available=False,
        no_global_install=True,no_source_archive_changes=True,no_grasp_VLM_or_execution=True)
    full=[x for x in findings if x['archive_member'].endswith('GMC_Competition_Task_Board_Virtual.SLDASM')]
    if len(full)==2:
        first={ (x['name'],x['reference_number']):x for x in full[0]['assembly_metadata']['top_level_components_relative_to_board']}
        second={ (x['name'],x['reference_number']):x for x in full[1]['assembly_metadata']['top_level_components_relative_to_board']}
        differences=[]
        for key in sorted(first.keys()&second.keys()):
            delta=np.max(np.abs(np.array(first[key]['transform_component_native_to_centered_board_m'])-
                               np.array(second[key]['transform_component_native_to_centered_board_m'])))
            differences.append(dict(name=key[0],reference_number=key[1],max_abs_transform_difference=float(delta)))
        result['full_assembly_revision_comparison']=dict(common_component_count=len(differences),
            common_component_transform_differences=differences,
            only_root_archive_components=[list(x) for x in sorted(first.keys()-second.keys())],
            only_nested_archive_components=[list(x) for x in sorted(second.keys()-first.keys())])
    result['limits']=[
        'Saved configuration placements are recovered; mate constraints, contact engagement, native B-rep and final seated correctness are not decoded.',
        'Bounding-box-centered component frames are mathematical reference frames. They are not guaranteed to equal the existing converted STL frame when the original STL export rotated the part.',
        'Added male components in the newer assembly can be separated or illustrative positions. Their native placements are not automatically verified assembly goals.',
        'Board-STL XY values add 192 mm to the bounding-box-centered native board XY. The plate size and mounting patterns agree, but a native-to-existing-simulation 3D frame conversion must be explicit.',
    ]
    (OUTPUT/'native_cad_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    if '--check' in sys.argv:
        checks=offline_checks()
        (OUTPUT/'native_cad'/'offline_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(checks))
    else:
        d=audit()
        for row in d['inspected_documents']:
            print(row['archive_member'],row['parsed_streams'],row.get('XML_tag_counts'))
