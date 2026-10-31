import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import visualize_own_diagnostics as m


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def case(tmp_path):
    own=tmp_path/'own';assets=tmp_path/'assets';view=assets/'bop_view';public=tmp_path/'public'
    folder=own/'datasets/lmo/targets/s000001_i000002_o000003';folder.mkdir(parents=True)
    data=tmp_path/'physical';data.mkdir()
    rgb_source=data/'rgb.png';Image.new('RGB',(64,48),(60,90,100)).save(rgb_source)
    camera_source=data/'scene_camera.json';dump(camera_source,{'2':{'cam_K':[50,0,32,0,50,24,0,0,1],'depth_scale':1.}})
    scene=view/'lmo/test/000001';(scene/'rgb').mkdir(parents=True)
    rgb=scene/'rgb/000002.png';rgb.symlink_to(rgb_source)
    camera=scene/'scene_camera.json';camera.symlink_to(camera_source)
    asset={'GT_pose_access':False,'GT_annotation_links':0,'datasets':{'lmo':{'frames':[{'scene_id':1,'im_id':2,
        'rgb':{'alias_path':str(rgb),'source_path':str(rgb_source),'source_sha256':m.sha(rgb_source)},
        'camera_source_path':str(camera_source),'camera_sha256':m.sha(camera_source)}]}}}
    dump(assets/'assets_manifest.json',asset)
    projection=public/'freezev2/geometry.py';projection.parent.mkdir(parents=True)
    projection.write_text('import numpy as np\ndef project_points(points,K,R,t):\n p=points@R.T+t\n h=p@K.T\n return h[:,:2]/h[:,2:3],p[:,2]\n')
    config={'paths':{'output':str(own),'assets_output':str(assets),'bop_root':str(view),'features_dir':str(own/'features'),'public_source':str(public)},
            'public_revision':'pinned-public-revision','method_id':'fixed-fixture-method'}
    config_path=tmp_path/'configuration.json';dump(config_path,config)
    query=own/'features/lmo_obj_000003_query.npz';query.parent.mkdir();np.savez(query,query_points=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]))
    pins={'configuration_sha256':m.digest(config),'assets_manifest_sha256':m.sha(assets/'assets_manifest.json')}
    dump(own/'stages/lmo/obj_000003/fuse/receipt.json',{'status':'complete','source_pins':pins,'outputs':{str(query):m.sha(query)}})
    dump(own/'run_manifest.json',{'public_source_identity':{'tracked_files':{'freezev2/geometry.py':m.sha(projection)}}})
    candidates=[];selected=[];outputs={}
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for index in range(2):
        mask=folder/f'candidate_{index:03d}_mask.png';array=np.zeros((48,64),dtype=np.uint8);array[10:35,20+index*10:36+index*10]=255;Image.fromarray(array).save(mask)
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=[index*10,0,50]
        coarse=folder/f'candidate_{index:03d}_coarse.npz';fine=folder/f'candidate_{index:03d}_fine.npz'
        np.savez(coarse,coarse_pose=T,coarse_score=np.float64(.4))
        np.savez(fine,fine_pose=T,coarse_feature_score=np.float64(.4),fine_feature_score=np.float64(.3),icp_score=np.float64(.5),final_score=np.float64(.06))
        row={'candidate_index':index,'status':'valid','nms_status':'selected','nms_selected':True,
            'mask_path':str(mask),'coarse_cache_path':str(coarse),'fine_cache_path':str(fine),
            'coarse_score':.4,'fine_feature_score':.3,'icp_score':.5,'final_score':.06,
            'R':R.tolist(),'t_mm':T[:3,3].tolist()}
        candidates.append(row);selected.append({k:row[k] for k in ('candidate_index','final_score','R','t_mm')})
        for path in (mask,coarse,fine):outputs[str(path)]=m.sha(path)
    result={'dataset':'lmo','scene_id':1,'im_id':2,'obj_id':3,'candidates':candidates,'selected':selected}
    result_path=folder/'result.json';dump(result_path,result);outputs[str(result_path)]=m.sha(result_path)
    receipt={'status':'complete','target':{'scene_id':1,'im_id':2,'obj_id':3,'inst_count':2},
        'source_pins':pins,'selected_count':2,'original_result':str(result_path),'outputs':outputs}
    receipt_path=folder/'receipt.json';dump(receipt_path,receipt)
    return {'own':own,'assets':assets,'config':config,'config_path':config_path,'receipt':receipt,'receipt_path':receipt_path,
            'result':result,'result_path':result_path,'output':own/'auxiliary_diagnostics/test_case','rgb':rgb,'rgb_source':rgb_source,
            'query':query,'projection':projection}


def render(c):return m.visualize_case(c['config_path'],c['receipt_path'],c['output'])


def save_result(c):
    dump(c['result_path'],c['result']);c['receipt']['outputs'][str(c['result_path'])]=m.sha(c['result_path']);dump(c['receipt_path'],c['receipt'])


def test_multiple_selected_poses_native_units_and_candidate_stages(case):
    result=render(case)
    assert result['selected_count']==2 and result['candidate_count']==2
    assert result['state']=='complete_visual_diagnostic_only' and not result['GT_access']
    assert not result['GPU_or_model_inference'] and not result['pose_or_score_changes']
    assert all(r['projection']['in_frame_count']==4 for r in result['selected'])
    assert result['selected'][1]['t_mm']==[10.,0.,50.]
    assert len(result['figures'])==3 and len(result['candidates'])==2
    assert result['candidates'][0]['stages']['coarse']['saved_scalars']['coarse_score']==.4
    assert result['candidates'][0]['saved_scores']['final_score']==.06
    assert 'Unoccluded all-points' in result['visibility']
    assert all(Path(f['path']).is_relative_to(case['own']/'auxiliary_diagnostics') for f in result['figures'])


def test_completed_geometry_is_unchanged(case):
    before={str(p):m.sha(p) for p in case['receipt_path'].parent.iterdir() if p.is_file()}
    render(case)
    assert all(m.sha(p)==value for p,value in before.items())


def test_no_pose_failed_case_does_not_invent_prediction(case):
    case['result']['selected']=[];case['receipt'].update(selected_count=0,status='failed',reason='original GeDi failure')
    save_result(case)
    result=render(case)
    assert result['selected']==[] and result['selected_count']==0 and result['target_status']=='failed'
    assert len(result['candidates'])==2 and result['candidates'][0]['mask_pixels']>0


def test_blocked_case_with_missing_result_and_features_is_partial(case):
    case['receipt'].update(selected_count=0,status='blocked',reason='onboarding failed',original_result=None)
    dump(case['receipt_path'],case['receipt']);case['query'].unlink()
    result=render(case)
    assert result['state']=='partial' and result['selected_count']==0 and result['candidate_count']==0
    assert any(r['role']=='original_result' and r['state']=='not_saved' for r in result['missing'])


def test_changed_completed_candidate_hash_marks_partial(case):
    mask=Path(case['result']['candidates'][0]['mask_path']);Image.new('L',(64,48),0).save(mask)
    result=render(case)
    assert result['state']=='partial'
    assert any('Completed target output checksum differs' in w for w in result['warnings'])


def test_changed_feature_cache_marks_partial(case):
    with np.load(case['query']) as data:points=data['query_points'].copy()
    np.savez(case['query'],query_points=points+1)
    result=render(case)
    assert result['state']=='partial' and any('completed feature-stage' in w for w in result['warnings'])


def test_changed_projection_source_marks_partial(case):
    case['projection'].write_text(case['projection'].read_text()+'\n# changed\n')
    result=render(case)
    assert result['state']=='partial' and any('projection source hash differs' in w for w in result['warnings'])


def test_authorized_alias_cannot_be_redirected(case,tmp_path):
    case['rgb'].unlink();other=tmp_path/'unapproved.png';Image.new('RGB',(64,48)).save(other);case['rgb'].symlink_to(other)
    with pytest.raises(ValueError,match='authorized physical source'):render(case)


def test_asset_manifest_must_match_completed_target_pin(case):
    path=case['assets']/'assets_manifest.json';value=m.read(path);value['extra']='changed';dump(path,value)
    with pytest.raises(ValueError,match='manifest differs'):render(case)


def test_running_receipt_is_not_a_representative_complete_case(case):
    case['receipt']['status']='running';dump(case['receipt_path'],case['receipt'])
    with pytest.raises(ValueError,match='Only terminal'):render(case)
    assert m.select_cases(case['own'])==[]


def test_auxiliary_outputs_cannot_touch_primary_files(case):
    with pytest.raises(ValueError,match='escaped'):
        m.visualize_case(case['config_path'],case['receipt_path'],case['own']/'datasets/lmo/overlays')


def test_GT_named_candidate_path_is_rejected(case):
    case['result']['candidates'][0]['mask_path']=str(case['receipt_path'].parent/'mask_visib/candidate.png')
    save_result(case)
    with pytest.raises(ValueError,match='GT annotation'):render(case)


def test_selection_uses_fixed_receipt_order_not_scores(case):
    base=case['own']/'datasets/lmo/targets'
    second=base/'s000001_i000003_o000003/receipt.json'
    receipt=dict(case['receipt']);receipt['target']={'scene_id':1,'im_id':3,'obj_id':3,'inst_count':1};receipt['selected_count']=0;receipt['status']='failed';dump(second,receipt)
    later=base/'s000001_i000009_o000003/receipt.json';receipt=dict(receipt);receipt['target']={**receipt['target'],'im_id':9};dump(later,receipt)
    choices=m.select_cases(case['own'])
    assert len(choices)==2 and choices[0]['receipt_path']==str(case['receipt_path'])
    assert choices[0]['reasons']==['first_available','first_with_saved_pose','first_multi_instance']
    assert choices[1]['receipt_path']==str(second) and choices[1]['reasons']==['first_no_pose','first_failed_or_blocked']


def test_snapshot_records_predeclared_rule_and_missing_datasets(case):
    result=m.snapshot(case['config_path'],case['own']/'auxiliary_diagnostics/snapshot')
    assert result['rule']['rule']==m.RULE and not result['case_selection_by_test_scores']
    assert len(result['cases'])==1 and 'tless' in result['datasets_without_available_terminal_receipts']
    assert result['cases'][0]['receipt_sha256']==m.sha(case['receipt_path'])


def test_projection_uses_CAD_to_camera_direction_in_millimetres(case):
    projector=m.projection_api(case['projection'])
    points=np.array([[1.,0.,0.]])
    K=np.array([[50.,0.,32.],[0.,50.,24.],[0.,0.,1.]])
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    uv,data=m.project(points,K,{'R':R,'t_mm':[0,0,50]},projector,(64,48))
    np.testing.assert_allclose(uv,[[32.,25.]])
    assert data['t_mm']==[0.,0.,50.]

def test_first_failed_receipt_still_selects_first_saved_pose_example(case):
    first=m.read(case['receipt_path']);first.update(status='blocked',selected_count=0)
    dump(case['receipt_path'],first)
    later=case['receipt_path'].parent.parent/'s000001_i000003_o000003/receipt.json'
    saved=dict(case['receipt']);saved['target']={**saved['target'],'im_id':3};dump(later,saved)
    chosen=m.select_cases(case['own'])
    assert len(chosen)==2
    assert 'first_with_saved_pose' not in chosen[0]['reasons']
    assert chosen[1]['reasons']==['first_with_saved_pose']


def test_nonfinite_saved_diagnostic_scalar_is_explicit_not_fabricated(case):
    fine=Path(case['result']['candidates'][0]['fine_cache_path'])
    with np.load(fine,allow_pickle=False) as values:arrays={key:values[key] for key in values.files}
    arrays['icp_rmse']=np.float64(np.inf);np.savez(fine,**arrays)
    case['receipt']['outputs'][str(fine)]=m.sha(fine);dump(case['receipt_path'],case['receipt'])
    result=render(case)
    assert result['candidates'][0]['stages']['refined']['saved_scalars']['icp_rmse']=={'nonfinite_saved_value':'inf'}
    assert Path(result['helper']['path']).is_file() and m.sha(result['helper']['path'])==result['helper']['sha256']

def test_missing_invalid_candidate_pose_headers_are_explicit(case,monkeypatch):
    candidate=case['result']['candidates'][0]
    for name in ('coarse_cache_path','fine_cache_path'):
        Path(candidate[name]).unlink()
    candidate.pop('R');candidate.pop('t_mm');candidate.update(status='invalid',error='Original RANSAC did not produce a finite pose')
    case['result']['selected']=case['result']['selected'][1:];case['receipt']['selected_count']=1
    save_result(case)
    titles=[];original=m.figure
    def capture(left,right,headings,lines,path):
        titles.append(tuple(headings));return original(left,right,headings,lines,path)
    monkeypatch.setattr(m,'figure',capture)
    result=render(case)
    assert titles[0]==('Candidate0: coarse pose unavailable','Candidate0: refined pose unavailable')
    assert titles[1]==('Candidate1: saved coarse pose','Candidate1: saved refined pose')
    assert result['candidates'][0]['stages']=={'coarse':{'state':'unavailable'},'refined':{'state':'unavailable'}}