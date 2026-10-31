from contextlib import contextmanager
import csv
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import suite_runner as s


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Durable:
    atomic_json=staticmethod(dump)
    def require_disk(self,path,size):assert size==15
    @contextmanager
    def worker_lock(self,path):yield object()


@pytest.fixture
def ctx(tmp_path,monkeypatch):
    config={'public_revision':s.ref.PUBLIC_REVISION,'datasets':list(s.DATASETS),'settings':dict(s.FIXED),
        'resources':{'cpu_threads':4,'inference_workers':1,'disk_reserve_GiB':15},
        'no_test_GT_tuning':True,'no_author_CSV_inference_inputs':True,
        'paths':{name:str(tmp_path/name) for name in ('output','assets_output','bop_root','geometry_root','features_dir','public_source',
                    'inference_python','evaluation_python','dinov2_root','gedi_root','gedi_checkpoint','torch_home','compute_lock')}}
    config['paths']['inference_python']=str(tmp_path/'venv/bin/python')
    identity={'revision':s.ref.PUBLIC_REVISION,'tracked_tree_sha256':'unchanged-source'}
    monkeypatch.setattr(s.ref,'git_identity',lambda *a:identity)
    class Owned:
        calls=0;returncode=0
        @staticmethod
        def start_ticks(pid):return 'ticks'
        def run_owned(self,command,environment,folder,lock,heartbeat):
            assert lock is not None;self.calls+=1;heartbeat(1234);return self.returncode
    result={'root':tmp_path,'output':tmp_path/'output','config':config,'assets':{'datasets':{'lmo':{'masks':{'path':str(tmp_path/'masks.json')}}}},
        'pins':{'configuration_sha256':'config','assets_manifest_sha256':'assets','public_revision':s.ref.PUBLIC_REVISION,'public_tracked_tree_sha256':'unchanged-source'},
        'manifest':{'public_source_identity':identity},'durable':Durable(),'owned':Owned()}
    return result


def test_fixed_all_seven_protocol_rejects_variant_and_tuning(ctx):
    assert s.validate_configuration(ctx['config']) is ctx['config']
    changed=json.loads(json.dumps(ctx['config']));changed['settings']['iterations']=100
    with pytest.raises(ValueError,match='settings'):s.validate_configuration(changed)
    changed=json.loads(json.dumps(ctx['config']));changed['datasets']=['lmo']
    with pytest.raises(ValueError,match='seven'):s.validate_configuration(changed)
    changed=json.loads(json.dumps(ctx['config']));changed['no_test_GT_tuning']=False
    with pytest.raises(ValueError,match='boundary'):s.validate_configuration(changed)


def test_original_feature_commands_keep_gedi_on_retained_visual_points(ctx):
    visual=s.feature_command(ctx,'lmo',1,'visual');gedi=s.feature_command(ctx,'lmo',1,'gedi');fuse=s.feature_command(ctx,'lmo',1,'fuse')
    assert visual[3]=='extract-query-visual' and visual[visual.index('--layer')+1]=='30'
    assert gedi[gedi.index('--onboarding-cache')+1].endswith('lmo_obj_000001_visual.npz')
    assert fuse[fuse.index('--visual')+1].endswith('lmo_obj_000001_visual.npz')
    assert fuse[fuse.index('--geometric')+1].endswith('lmo_obj_000001_gedi.npz')
    assert '--gt-id' not in visual+gedi+fuse


def test_localization_is_original_complete_candidate_pipeline(ctx):
    row={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':2};folder=s.target_folder(ctx,'lmo',row)
    command=s.localization_command(ctx,'lmo',row,folder)
    assert command[3]=='run-localization'
    assert s.ref.read(folder/'target.json')==[row]
    assert command[command.index('--iterations')+1]=='10000'
    assert command[command.index('--nms-translation-threshold-mm')+1]=='10.0'
    assert not any('gt' in token.lower() for token in command[4:] if token.startswith('--'))


def test_environment_keeps_single_gpu_fp32_offline_and_no_allocator_override(ctx,monkeypatch):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF','obsolete')
    env=s.environment(ctx)
    assert env['CUDA_VISIBLE_DEVICES']=='0' and env['NVIDIA_TF32_OVERRIDE']=='0'
    assert env['HF_HUB_OFFLINE']=='1' and env['OMP_NUM_THREADS']=='4'
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in env


def test_stage_receipt_reuses_only_identical_source_command_and_output(ctx,tmp_path):
    output=tmp_path/'result';output.write_text('measured')
    row={'status':'complete','source_pins':ctx['pins'],'command':['original'],'outputs':{str(output):s.ref.sha256(output)}}
    assert s.receipt_reusable(row,ctx['pins'],['original'])
    with pytest.raises(s.IdentityError,match='identity'):s.receipt_reusable(row,ctx['pins'],['modified'])
    output.write_text('tampered')
    with pytest.raises(s.IdentityError,match='changed'):s.receipt_reusable(row,ctx['pins'],['original'])


def test_owned_stage_keeps_failures_without_relaunching(ctx,tmp_path):
    ctx['owned'].returncode=2
    folder=tmp_path/'failed'
    with pytest.raises(RuntimeError):s.owned_stage(ctx,folder,['original'],object(),lambda:({},{}))
    assert s.ref.read(folder/'receipt.json')['status']=='failed'
    with pytest.raises(RuntimeError,match='Preserved prior'):s.owned_stage(ctx,folder,['original'],object(),lambda:({},{}))
    assert ctx['owned'].calls==1


def test_owned_stage_reuses_verified_result_and_source_changed_is_failure(ctx,tmp_path,monkeypatch):
    folder=tmp_path/'success'
    first=s.owned_stage(ctx,folder,['original'],object(),lambda:({}, {'measured':True}))
    assert first['status']=='complete'
    assert s.owned_stage(ctx,folder,['original'],object(),lambda:({},{}))==first
    assert ctx['owned'].calls==1
    monkeypatch.setattr(s.ref,'git_identity',lambda *args:{'changed':True})
    with pytest.raises(s.IdentityError,match='source changed'):s.owned_stage(ctx,tmp_path/'tampered',['original'],object(),lambda:({},{}))


def test_blocked_cad_records_exact_target_without_gpu(ctx):
    target={'scene_id':2,'im_id':3,'obj_id':27,'inst_count':3}
    row=s.target_stage(ctx,'lmo',target,object(),'original GeDi nonfinite')
    assert row['status']=='blocked' and row['target']==target and row['poses']==[]
    assert ctx['owned'].calls==0


def make_target(ctx,target):
    folder=s.target_folder(ctx,'lmo',target);folder.mkdir(parents=True)
    R=np.eye(3).tolist();t=[1.,2.,500.];score=.123456789123
    selected={'final_score':score,'R':R,'t_mm':t}
    result=folder/'original'/f'scene_{target["scene_id"]:06d}_im_{target["im_id"]:06d}_obj_{target["obj_id"]:06d}'/'result.json'
    dump(result,{**target,'instance_count':target['inst_count'],'selected':[selected],'candidate_count':2,'valid_candidate_count':1})
    with (folder/'original.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=s.FIELDS);writer.writeheader()
        writer.writerow({**{k:target[k] for k in ('scene_id','im_id','obj_id')},'score':format(score,'.9g'),
            'R':'1 0 0 0 1 0 0 0 1','t':'1 2 500','time':'2.5'})
    return folder,result


def test_original_selected_pose_and_csv_must_match_exact_serialization(ctx):
    target={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':1};folder,result=make_target(ctx,target)
    outputs,details=s.validate_target(folder,target)
    assert details['selected_count']==1 and details['poses'][0]['t_mm']==[1.,2.,500.]
    bad=s.ref.read(result);bad['selected'][0]['t_mm'][0]=20;dump(result,bad)
    with pytest.raises(s.IdentityError,match='CSV pose differs'):s.validate_target(folder,target)


def test_incremental_exports_retain_failed_targets_full_instances_and_image_time(ctx):
    good={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':1}
    failed={'scene_id':2,'im_id':3,'obj_id':27,'inst_count':3}
    folder,_=make_target(ctx,good);outputs,details=s.validate_target(folder,good)
    a={'status':'complete','target':good,'runtime_s':9,**details,'source_pins':ctx['pins']}
    dump(folder/'receipt.json',a)
    b=s.target_stage(ctx,'lmo',failed,object(),'original GeDi nonfinite');b['runtime_s']=1;dump(s.target_folder(ctx,'lmo',failed)/'receipt.json',b)
    partial=s.export_dataset(ctx,'lmo',[good,failed],[a])
    assert partial['status']=='in_progress' and partial['expected_object_image_targets']==2 and partial['requested_instances']==4
    final=s.export_dataset(ctx,'lmo',[good,failed],[a,b])
    assert final['status']=='all_targets_recorded' and final['counts']=={'complete':1,'failed':0,'blocked':1}
    assert final['targets'][0]['poses'][0]['time']==3.5
    assert s.read_csv(final['csv'])[0]['time']=='3.5'
    assert final['targets'][1]['selected_count']==0 and final['targets'][1]['inst_count']==3


def test_target_export_rejects_duplicate_receipts(ctx):
    target={'scene_id':2,'im_id':3,'obj_id':27,'inst_count':3}
    record=s.target_stage(ctx,'lmo',target,object(),'failure')
    with pytest.raises(s.IdentityError,match='duplicate'):s.export_dataset(ctx,'lmo',[target],[record,record])


def test_exact_targets_never_accept_duplicate_keys(tmp_path):
    path=tmp_path/'targets.json';target={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':2}
    dump(path,[target,target])
    with pytest.raises(ValueError):s.exact_targets(path)


def test_gedi_validation_rejects_misaligned_or_nonfinite_rows(ctx):
    files=s.feature_paths(ctx,'lmo',1);files['visual'].parent.mkdir(parents=True)
    points=np.array([[0,0,0],[1,2,3]],dtype=np.float32)
    np.savez(files['visual'],query_points=points)
    np.savez(files['gedi'],query_points=points[::-1],geometric_features=np.ones((2,64)))
    with pytest.raises(s.IdentityError,match='alignment'):s.validate_npz(ctx,'lmo',1,'gedi')
    np.savez(files['gedi'],query_points=points,geometric_features=np.full((2,64),np.nan))
    with pytest.raises(ValueError,match='descriptors'):s.validate_npz(ctx,'lmo',1,'gedi')


def test_adoption_accepts_reordered_original_flags_and_default_facet_only(ctx,tmp_path):
    files=s.feature_paths(ctx,'lmo',1);files['visual'].parent.mkdir(parents=True)
    np.savez(files['visual'],query_points=np.ones((2,3)),visual_features=np.ones((2,1536)),
             dino_layer=30,dino_model='dinov2_vitg14',depth_tolerance=1.,min_views=18)
    command=s.feature_command(ctx,'lmo',1,'visual')
    flags=list(zip(command[4::2],command[5::2]));flags=[pair for pair in flags if pair[0]!='--facet']
    actual=command[:4]+[token for pair in reversed(flags) for token in pair]
    path=tmp_path/'probe/command.json';dump(path,actual)
    dump(path.parent/'receipt.json',{'status':'complete','returncode':0,'command':actual,'runtime_s':12.5,
        'source_pins':{'configuration':ctx['config'],'public_source':ctx['manifest']['public_source_identity'],'assets_manifest_sha256':'assets'},
        'outputs':{str(files['visual']):s.ref.sha256(files['visual'])}})
    record=s.adopt_stage(ctx,'lmo',1,'visual',path,12.5)
    assert record['adopted_original_probe'] and record['runtime_s']==12.5
    assert record['command']==command


def test_adoption_rejects_different_layer_before_cache_access(ctx,tmp_path):
    command=s.feature_command(ctx,'lmo',1,'visual');command[command.index('--layer')+1]='31'
    path=tmp_path/'command.json';dump(path,command)
    with pytest.raises(ValueError,match='differs'):s.adopt_stage(ctx,'lmo',1,'visual',path,10)


def test_fatal_target_identity_error_is_never_saved_as_an_individual_failure(ctx,monkeypatch):
    target={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':1}
    def fail(*args,**kwargs):raise s.IdentityError('source changed')
    monkeypatch.setattr(s,'owned_stage',fail)
    with pytest.raises(s.IdentityError,match='source changed'):s.target_stage(ctx,'lmo',target,object())
    assert not (s.target_folder(ctx,'lmo',target)/'receipt.json').exists()


def test_incremental_export_uses_validated_poses_without_reopening_prior_artifacts(ctx,monkeypatch):
    target={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':1}
    folder,_=make_target(ctx,target);_,details=s.validate_target(folder,target)
    record={'status':'complete','target':target,'runtime_s':9,**details,'source_pins':ctx['pins'],'_receipt_sha256':'previously-verified-sha'}
    def forbidden(*args,**kwargs):raise AssertionError('Prior artifact reopened during incremental export')
    monkeypatch.setattr(s,'read_csv',forbidden)
    real_sha=s.ref.sha256
    monkeypatch.setattr(s.ref,'sha256',lambda path:forbidden(path) if str(path).endswith('receipt.json') else real_sha(path))
    result=s.export_dataset(ctx,'lmo',[target],[record])
    assert result['targets'][0]['target_receipt_sha256']=='previously-verified-sha'
    assert result['pose_count']==1


def test_fatal_cad_identity_error_halts_dataset_before_blocked_zero_pose_rows(ctx,monkeypatch):
    target={'scene_id':2,'im_id':3,'obj_id':1,'inst_count':1}
    targets=s.path_of(ctx['config'],'bop_root')/'lmo/test_targets_bop19.json';dump(targets,[target])
    masks=Path(ctx['assets']['datasets']['lmo']['masks']['path']);masks.write_text('[]')
    ctx['assets']['datasets']['lmo'].update(target_sha256=s.ref.sha256(targets),required_object_ids=[1])
    ctx['assets']['datasets']['lmo']['masks']['actual_sha256']=s.ref.sha256(masks)
    def fail(*args,**kwargs):raise s.IdentityError('CAD cache source differs')
    monkeypatch.setattr(s,'geometry_stage',fail)
    with pytest.raises(s.IdentityError,match='cache source'):s.run_dataset(ctx,'lmo',object())
    assert not (ctx['output']/'datasets/lmo/target_ledger.json').exists()
