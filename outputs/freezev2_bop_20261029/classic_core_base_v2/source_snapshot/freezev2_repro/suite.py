"""Single-GPU, resumable BOP-Classic-Core base inference and official evaluation."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import copy
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback

from .data import BOPDataset,sha256
from .durable import atomic_json,digest,ensure_identity,read_json,require_disk,worker_lock
from .pipeline import local_path,load_feature_backends,source_hashes
from .masks import load_localization_targets
from .mask_index import MaskIndex
from .onboarding import onboard_geometry,onboard_features,load_bundle
from .case_inference import infer_image,InvalidCUDAContext
from .evaluation import PosePrediction,write_bop_csv,build_evaluation_command
from .owned_process import run_owned

DATASETS=('lmo','tless','tudl','icbin','itodd','hb','ycbv')


def implementation_identity():
    return source_hashes()


class CheckedDataset(BOPDataset):
    """Bind sensor/calibration bytes; camera extrinsics remain inaccessible."""
    def __init__(self,root,name,originals):
        super().__init__(root,name);self.originals=originals
    def frame(self,scene_id,image_id):
        original=self.originals[(self.dataset,scene_id,image_id)]
        camera=(self.root/original['camera_path']).resolve()
        if not camera.is_relative_to(self.root) or sha256(camera)!=original['camera_sha256']:
            raise ValueError('Camera calibration changed from immutable run snapshot')
        frame=super().frame(scene_id,image_id)
        if (Path(frame['source']['camera']).resolve()!=camera
                or frame['source']['camera_sha256']!=original['camera_sha256']):
            raise ValueError('Camera calibration changed from immutable run snapshot')
        for actual,expected in (('rgb_sha256','sha256'),('depth_sha256','depth_sha256')):
            if frame['source'][actual]!=original[expected]:raise ValueError('Actual sensor file changed from original pinned input manifest')
        return frame


def read_ledgers(root,dataset):
    records=[]
    for folder in sorted((Path(root)/'datasets'/dataset/'images').glob('*')):
        file=folder/'complete.json'
        if not file.exists():file=folder/'partial.json'
        if file.exists():records.append(read_json(file))
    return records


def summarize(root,assets,states,protocol):
    root=Path(root);rows={}
    for name in protocol['datasets']:
        asset=assets['datasets'][name];records=read_ledgers(root,name)
        completed=failed=blocked=predictions=0;instance_expected=asset['requested_instances']
        for image in records:
            for target in image['targets']:
                completed+=target['status']=='completed';failed+=target['status']=='failed';blocked+=target['status']=='blocked'
            predictions+=len(image['predictions'])
        expected=asset['target_records'];pending=expected-completed-failed-blocked
        if pending<0:raise ValueError('More checkpoint targets than official expected targets')
        state=states.get(name,{})
        if state.get('status','').startswith('blocked'):
            blocked+=pending;pending=0
        model_root=root/'datasets'/name/'onboarding'
        geometries=list(model_root.glob('obj_*/geometry/complete.json'))
        features=list(model_root.glob('obj_*/features/complete.json'))
        metrics_file=root/'datasets'/name/'evaluation_receipt.json'
        evaluation=read_json(metrics_file) if metrics_file.exists() else {'status':'pending','official_pose_metrics':None}
        if evaluation['status']=='complete':
            csv=root/'datasets'/name/'predictions'/f'freezev2base_{name}-test.csv'
            score_file=Path(evaluation.get('score_file','__missing_score__'))
            valid=(csv.is_file() and evaluation.get('csv_sha256')==sha256(csv)
                   and evaluation.get('target_sha256')==asset['targets']['sha256']
                   and score_file.is_file() and evaluation.get('score_sha256')==sha256(score_file))
            if not valid:evaluation={**evaluation,'status':'stale_prediction_or_metric_receipt','official_pose_metrics':None}
        rows[name]={'dataset':name,'task':'BOP19_model_based_6D_localization','split':'test',
                    'expected_images':asset['official_target_images'],'expected_targets':expected,'expected_instances':instance_expected,
                    'completed_targets':completed,'failed_targets':failed,'blocked_targets':blocked,'pending_targets':pending,
                    'attempted_images':len(records),'valid_pose_predictions':predictions,
                    'onboarding_expected_CADs':asset['cad_objects'],'geometry_CADs':len(geometries),'frozen_feature_CADs':len(features),
                    'onboarding_runtime_s':sum(read_json(p)['runtime_s'] for p in geometries+features),
                    'inference_runtime_s':sum(image['runtime_s'] for image in records),
                    'mask_source':asset['masks']['source_kind']+':'+asset['masks']['segmentor'],
                    'mask_sha256':asset['masks']['sha256'],'official_targets_sha256':asset['targets']['sha256'],
                    'status':state.get('status','pending'),'blocker':state.get('reason'),
                    'validation':state.get('validation'),'evaluation':evaluation,
                    'predictions_origin':'our_frozen_model_inference','authors_predictions_evaluated':False}
    compatible=(len(rows)==7 and set(rows)==set(DATASETS)
                and all(r['pending_targets']==0 and r['blocked_targets']==0 and r['evaluation']['status']=='complete' for r in rows.values()))
    aggregate=None
    if compatible:
        aggregate={'scope':'all_seven_BOP_Classic_Core_equal_dataset_macro',
                   'bop19_average_recall':sum(r['evaluation']['official_pose_metrics']['bop19_average_recall'] for r in rows.values())/7}
    result={'schema':'freezev2-suite-results-v2','method':protocol['method'],'mask_configuration':protocol['mask_configuration'],
            'task':'BOP19_model_based_6D_localization','official_GT_selection':'official inst_count most-visible GT; evaluator visib_gt_min=-1',
            'suite_complete_and_evaluated':compatible,'all_seven_aggregate':aggregate,
            'expected_images':sum(r['expected_images'] for r in rows.values()),
            'expected_targets':sum(r['expected_targets'] for r in rows.values()),
            'datasets':rows,'GT_pose_access_in_inference':False,
            'claim':'Own inference results only. A complete seven-dataset aggregate is withheld until every compatible official evaluation completes.'}
    atomic_json(root/'results_summary.json',result)
    text=['# FreeZeV2 base: BOP-Classic-Core run', '',f"Method: `{protocol['method']}`; mask configuration: `{protocol['mask_configuration']}`.",
          '', 'Task: BOP19 model-based 6D localization, official object/count targets, official k-most-visible GT evaluation. GT poses are evaluation-only.',
          '', '| Dataset | Expected targets | Completed | Failed | Blocked | Pending | Poses | CADs with features | Official AR |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        metric=r['evaluation'].get('official_pose_metrics')
        score=f"{100*metric['bop19_average_recall']:.4f}%" if metric else 'unavailable'
        text.append(f"| {name} | {r['expected_targets']} | {r['completed_targets']} | {r['failed_targets']} | {r['blocked_targets']} | {r['pending_targets']} | {r['valid_pose_predictions']} | {r['frozen_feature_CADs']}/{r['onboarding_expected_CADs']} | {score} |")
    text+=['','A completed target produced its requested number of distinct valid poses. A failed target was attempted but produced fewer poses; any surviving predictions remain in the CSV and the full official target list remains the evaluator denominator.',
           '', 'Onboarding and inference runtimes, exact mask/target hashes, metric receipts and per-dataset blockers are in results_summary.json. Inference timing includes per-image candidate work and excludes CAD onboarding/evaluation.',
           '', '## Resume', '', 'Use the immutable source snapshot and the recorded inference interpreter:', '```bash',
           f"{protocol['paths']['inference_python']} {root/'source_snapshot'/'run_suite.py'} --config {root/'protocol.json'} --resume", '```',
           '', 'No placeholder descriptors, pose constraints, GT initialization, parameter tuning on test GT, or author-prediction substitution is used. Source/recipe/model identities are sealed per run.',
           '', 'Status: '+('all seven evaluated under one compatible protocol.' if compatible else 'SUITE INCOMPLETE; no complete BOP-Classic-Core aggregate is reported.')]
    (root/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


class SuiteRunner:
    def __init__(self,protocol):
        self.protocol=protocol;self.base=protocol['base'];self.root=local_path(protocol['paths']['suite_output_root'])
        self.assets=read_json(local_path(protocol['paths']['assets_manifest']))
        if set(protocol['datasets'])!=set(DATASETS) or len(protocol['datasets'])!=7:raise ValueError('Suite protocol must name exactly all seven datasets')
        if set(protocol['execution_order'])!=set(DATASETS) or len(protocol['execution_order'])!=7:raise ValueError('Execution order must contain every dataset exactly once')
        self.code=implementation_identity();self.method_identity={'method':protocol['method'],'base':self.base,'code':self.code,'mask_configuration':protocol['mask_configuration'],'task':'BOP19_localization'}
        self.method_digest=digest(self.method_identity)
        ensure_identity(self.root,self.method_identity)
        self.states=read_json(self.root/'dataset_states.json') if (self.root/'dataset_states.json').exists() else {}
        self.backends=None;self.last_heartbeat=0.;self.current={}
        dataset_root=local_path(protocol['paths']['dataset_root']).resolve()
        original=read_json(dataset_root/'full_manifest.json')
        # The retained acquisition manifest pins RGB/depth but not calibration.
        # Seal every unique original camera file before inference, not only the
        # subset of scenes reached by a first small run. Hash whole files while
        # continuing to whitelist only intrinsics/depth_scale in BOPDataset.
        cameras={}
        for row in original['images']:
            path=(dataset_root/row['camera_path']).resolve()
            if not path.is_relative_to(dataset_root) or path.name!='scene_camera.json':
                raise ValueError('Original camera path must identify scene_camera.json inside the dataset root')
            key=path.relative_to(dataset_root).as_posix()
            if key not in cameras:cameras[key]=sha256(path)
        self.camera_sha256_by_relative_path=cameras
        self.camera_snapshot_digest=digest(cameras)
        self.originals={}
        for row in original['images']:
            copied=dict(row)
            key=(dataset_root/row['camera_path']).resolve().relative_to(dataset_root).as_posix()
            copied['camera_sha256']=cameras[key]
            self.originals[(row['dataset'],row['scene_id'],row['im_id'])]=copied
        self.input_manifest_sha=sha256(dataset_root/'full_manifest.json')
        manifest=self.root/'run_manifest.json'
        if not manifest.exists():atomic_json(manifest,{'method_identity':self.method_identity,'protocol':protocol,'python':sys.executable,'python_version':platform.python_version(),'original_input_manifest_sha256':self.input_manifest_sha,'camera_sha256_by_relative_path':cameras,'camera_snapshot_digest':self.camera_snapshot_digest,'started_utc':time.time()})
        else:
            prior=read_json(manifest)
            if prior['original_input_manifest_sha256']!=self.input_manifest_sha:raise ValueError('Retained input manifest changed during run')
            if (prior.get('camera_sha256_by_relative_path')!=cameras
                    or prior.get('camera_snapshot_digest')!=self.camera_snapshot_digest):
                raise ValueError('Camera calibration changed or immutable run snapshot is absent; new run required')

    def heartbeat(self,force=False,**fields):
        self.current.update(fields)
        now=time.time()
        if force or now-self.last_heartbeat>=2:
            import psutil
            process=psutil.Process()
            status={'pid':os.getpid(),'create_time':process.create_time(),'cmdline':process.cmdline(),'updated_epoch':now,
                    'method_digest':self.method_digest,'current':self.current,'disk_free_bytes':require_disk(self.root,self.protocol['resources']['minimum_free_disk_gb'])}
            if Path('/proc/self/stat').exists():status['linux_start_ticks']=Path('/proc/self/stat').read_text().split()[21]
            atomic_json(self.root/'status.json',status);self.last_heartbeat=now

    def state(self,name,**fields):
        self.states.setdefault(name,{}).update(fields)
        atomic_json(self.root/'dataset_states.json',self.states)
        summarize(self.root,self.assets,self.states,self.protocol)

    def get_backends(self):
        if self.backends is None:
            self.heartbeat(stage='loading_real_frozen_models',force=True)
            config=copy.deepcopy(self.base);config['paths']=self.protocol['paths']
            config['resources']=self.protocol['resources']
            self.backends=load_feature_backends(config)
            actual={'dino':self.backends[0].provenance,'gedi':self.backends[1].provenance}
            scientific=copy.deepcopy(actual)
            for item in scientific.values():item.pop('resources',None)
            file=self.root/'frozen_models.json'
            if file.exists() and read_json(file)['scientific_identity']!=scientific:raise ValueError('Frozen model/source identity changed; new run required')
            atomic_json(file,{'scientific_identity':scientific,'actual_provenance':actual})
        return self.backends

    def release_backends(self):
        if self.backends is not None:
            dino,gedi=self.backends
            dino.model.cpu();gedi.backend.gedi_net.cpu()
            self.backends=None;del dino,gedi
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def dataset(self,name):return CheckedDataset(local_path(self.protocol['paths']['dataset_root']),name,self.originals)

    def targets(self,name):
        asset=self.assets['datasets'][name]
        path=local_path(asset['targets']['path'])
        if sha256(path)!=asset['targets']['sha256']:raise ValueError('Official target list checksum changed')
        targets=load_localization_targets(path)
        if len(targets)!=asset['target_records']:raise ValueError('Official target count differs from asset manifest')
        for t in targets:
            if (name,t.scene_id,t.image_id) not in self.originals:raise FileNotFoundError('An official target image is absent from retained sensor inputs')
        return targets

    def index(self,name):
        source=self.assets['datasets'][name]['masks']
        entry={'path':str(local_path(source['path'])),'segmentor':'cnos-fastsam','source_id':'public_BOP23_default_CNOS_FastSAM:'+source['sha256'],
               'sha256':source['sha256'],'source_kind':'predicted'}
        return MaskIndex(self.root/'datasets'/name/'mask_index.sqlite',[entry])

    def identity(self,name):
        asset=self.assets['datasets'][name]
        return {'method_digest':self.method_digest,'dataset':name,'targets_sha256':asset['targets']['sha256'],
                'mask_sha256':asset['masks']['sha256'],'original_input_manifest_sha256':self.input_manifest_sha,
                'camera_snapshot_digest':self.camera_snapshot_digest}

    def onboard(self,name,object_id):
        model=self.dataset(name).model(object_id)
        root=self.root/'datasets'/name/'onboarding'/f'obj_{object_id:06d}'
        self.heartbeat(dataset=name,object_id=object_id,force=True)
        geometry=onboard_geometry(model,root/'geometry',self.base,local_path(self.protocol['paths']['cnos_poses']),self.code,
                                  minimum_free_gb=self.protocol['resources']['minimum_free_disk_gb'],heartbeat=self.heartbeat)
        features=onboard_features(model,root/'geometry',root/'features',self.base,self.get_backends(),self.code,
                                  minimum_free_gb=self.protocol['resources']['minimum_free_disk_gb'],heartbeat=self.heartbeat)
        return load_bundle(model,root/'geometry',root/'features')

    def validation_targets(self,name,targets):
        if name=='tless':
            keys={(r['scene_id'],r['im_id'],1) for r in self.protocol['validation']['tless_fixed_cohort']}
            return [t for t in targets if (t.scene_id,t.image_id,t.object_id) in keys]
        # Fixed metadata-only selection, before any pose/GT outcome is available.
        object_id=targets[0].object_id
        return [t for t in targets if t.object_id==object_id][:self.protocol['validation']['other_dataset_target_rows']]

    def validate_dataset(self,name):
        previous=self.states.get(name,{}).get('validation')
        if previous and previous.get('status')=='passed':return True
        targets=self.targets(name);sample=self.validation_targets(name,targets)
        self.state(name,status='validating',validation={'status':'running','target_rows':[asdict(t) for t in sample]})
        bundles={};blocked={}
        for object_id in sorted({t.object_id for t in sample}):
            try:bundles[object_id]=self.onboard(name,object_id)
            except (ValueError,RuntimeError,FileNotFoundError) as exc:
                blocked[object_id]=str(exc)
                atomic_json(self.root/'datasets'/name/'validation'/f'CAD_{object_id:06d}_blocker.json',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
                if any(k in str(exc).lower() for k in ('illegal memory access','device-side assert')):raise InvalidCUDAContext(str(exc)) from exc
        index=self.index(name);images=defaultdict(list)
        for t in sample:images[(t.scene_id,t.image_id)].append(t)
        records=[]
        try:
            for ordinal,((scene,image),rows) in enumerate(images.items(),1):
                records.append(infer_image(self.dataset(name),scene,image,rows,bundles,index,self.base,self.get_backends(),
                    self.root/'datasets'/name/'validation'/'images'/f'{scene:06d}_{image:06d}',self.identity(name),self.protocol['resources'],self.heartbeat))
        finally:index.close()
        poses=[p for r in records for p in r['predictions']]
        resource_blocks=sum(t['status']=='blocked' for image in records for t in image['targets'])
        result={'status':'passed' if poses and not blocked and not resource_blocks else 'failed_or_blocked','fixed_targets':[asdict(t) for t in sample],
                'valid_pose_predictions':len(poses),'CAD_blockers':blocked,'ground_truth_pose_access':False,
                'runtime_blocked_targets':resource_blocks,
                'scope':'real frozen feature extraction, global rigid registration/refinement/scoring and output validity; no accuracy tuning'}
        if poses:
            csv=self.root/'datasets'/name/'validation'/f'freezev2base_{name}-test.csv'
            if not csv.exists():write_bop_csv([PosePrediction(**{k:p[k] for k in ('scene_id','image_id','object_id','score','R_m2c','t_m','image_time_s')}) for p in poses],csv)
            result['official_format_check']=self.check_csv(csv)
        atomic_json(self.root/'datasets'/name/'validation'/'gate.json',result)
        self.state(name,status='validated' if result['status']=='passed' else 'blocked_validation',validation=result,
                   reason=None if result['status']=='passed' else 'Fixed real end-to-end validation did not pass; see gate.json')
        return result['status']=='passed'

    def check_csv(self,csv):
        runtime=read_json(local_path(self.protocol['paths']['evaluation_runtime']))
        python=runtime['python']
        code="import json,sys;sys.path.insert(0,sys.argv[1]);from bop_toolkit_lib import inout;ok,msg=inout.check_bop_results(sys.argv[2],version='bop19');print(json.dumps({'valid':bool(ok),'message':msg}));sys.exit(0 if ok else 2)"
        process=subprocess.run([python,'-c',code,str(local_path(self.protocol['paths']['bop_toolkit'])),str(csv)],capture_output=True,text=True)
        if process.returncode:raise RuntimeError('Official BOP format validation failed: '+process.stderr+process.stdout)
        result=json.loads(process.stdout.strip().splitlines()[-1])
        if not result['valid']:raise ValueError('Official BOP CSV rejected')
        return result

    def export_dataset(self,name,targets):
        records=read_ledgers(self.root,name)
        expected={(t.scene_id,t.image_id,t.object_id) for t in targets}
        found=set();poses=[]
        for image in records:
            key=(image['identity']['scene_id'],image['identity']['image_id'])
            for row in image['targets']:
                target_key=(*key,row['object_id'])
                if target_key not in expected or target_key in found:raise ValueError('Export target coverage/uniqueness violation')
                found.add(target_key)
            poses.extend(image['predictions'])
        folder=self.root/'datasets'/name/'predictions';folder.mkdir(parents=True,exist_ok=True)
        file=folder/f'freezev2base_{name}-test.csv'
        rows=[PosePrediction(**{k:p[k] for k in ('scene_id','image_id','object_id','score','R_m2c','t_m','image_time_s')}) for p in poses]
        # Dataset export is rebuilt from durable unique-image receipts, never an
        # append log. Atomic replacement cannot duplicate rows after a restart.
        temporary=folder/f'.export_{name}-test.csv'
        if temporary.exists():temporary.unlink()
        if rows:write_bop_csv(rows,temporary)
        else:temporary.write_text('scene_id,im_id,obj_id,score,R,t,time\n',encoding='utf-8')
        os.replace(temporary,file)
        receipt={'source':'our_inference','expected_targets':len(expected),'ledger_targets':len(found),'poses':len(rows),
                 'exact_target_coverage':found==expected,
                 'all_targets_retained_for_evaluation':True,'missing_targets_remain_in_official_denominator':True,
                 'target_sha256':self.assets['datasets'][name]['targets']['sha256'],'csv_sha256':sha256(file),
                 'empty_predictions':not rows,'format':self.check_csv(file)}
        atomic_json(folder/'export_receipt.json',receipt)
        return file

    def infer_dataset(self,name):
        self.state(name,status='onboarding_all_CADs',reason=None)
        targets=self.targets(name);bundles={};blocks={}
        for object_id in sorted({t.object_id for t in targets}):
            try:bundles[object_id]=self.onboard(name,object_id)
            except (ValueError,RuntimeError,FileNotFoundError) as exc:
                blocks[object_id]=str(exc)
                atomic_json(self.root/'datasets'/name/'onboarding'/f'obj_{object_id:06d}'/'blocker.json',{'reason':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()})
                if any(k in str(exc).lower() for k in ('illegal memory access','device-side assert')):raise InvalidCUDAContext(str(exc)) from exc
        index=self.index(name);images=defaultdict(list)
        for t in targets:images[(t.scene_id,t.image_id)].append(t)
        self.state(name,status='inferring_full_official_targets',CAD_blockers=blocks)
        try:
            for ordinal,((scene,image),rows) in enumerate(images.items(),1):
                self.heartbeat(dataset=name,scene_id=scene,image_id=image,force=True)
                infer_image(self.dataset(name),scene,image,rows,bundles,index,self.base,self.get_backends(),
                            self.root/'datasets'/name/'images'/f'{scene:06d}_{image:06d}',self.identity(name),self.protocol['resources'],self.heartbeat)
                if ordinal%self.protocol['resources']['export_interval_images']==0:
                    self.export_dataset(name,targets)
                    self.state(name,status='inferring_full_official_targets')
        finally:index.close()
        csv=self.export_dataset(name,targets)
        runtime_blocks=sum(t['status']=='blocked' for image in read_ledgers(self.root,name) for t in image['targets'])
        export=read_json(csv.parent/'export_receipt.json')
        complete=not blocks and runtime_blocks==0 and export.get('exact_target_coverage',export['ledger_targets']==export['expected_targets'])
        self.state(name,status='inference_complete' if complete else 'blocked_some_targets',reason=None if complete else str({'CADs':blocks,'runtime_blocked_targets':runtime_blocks}))
        return csv,complete

    def evaluate_dataset(self,name,csv,inference_complete):
        runtime=read_json(local_path(self.protocol['paths']['evaluation_runtime']))
        dataset=self.assets['datasets'][name]
        metadata=dataset.get('evaluation',{})
        receipt=self.root/'datasets'/name/'evaluation_receipt.json'
        if receipt.exists():
            prior=read_json(receipt)
            score=Path(prior.get('score_file','__missing_score__'))
            if (prior.get('csv_sha256')==sha256(csv) and prior.get('target_sha256')==dataset['targets']['sha256']
                    and prior['status']=='complete' and score.is_file() and prior.get('score_sha256')==sha256(score)):return prior
        if not inference_complete:
            result={'status':'blocked_incomplete_inference','official_pose_metrics':None,'reason':'Some target CADs remain blocked; successful cases are not the full dataset'}
        elif not metadata.get('ready',metadata.get('status')=='ready'):
            result={'status':'blocked_evaluation_assets','official_pose_metrics':None,'reason':metadata.get('reason',metadata.get('blockers','Official evaluation metadata is not ready'))}
        else:
            self.release_backends()
            evaluation_root=self.root/'datasets'/name/'official_evaluation'
            command,environment=build_evaluation_command(toolkit_root=local_path(self.protocol['paths']['bop_toolkit']),results_file=csv,
                datasets_root=local_path(runtime['dataset_view']),evaluation_root=evaluation_root,targets_filename=runtime.get('targets_filename','test_targets_bop19.json'),
                python_executable=runtime['python'],renderer_type='vispy',num_workers=1)
            environment.update(runtime.get('environment',{}))
            environment.update({'OMP_NUM_THREADS':str(self.protocol['resources']['cpu_workers']),'OPENBLAS_NUM_THREADS':str(self.protocol['resources']['cpu_workers'])})
            self.heartbeat(stage='official_BOP_evaluation',dataset=name,force=True)
            evaluation_root.mkdir(parents=True,exist_ok=True)
            start=time.perf_counter()
            returncode=run_owned(command,environment,evaluation_root,self.lock_stream,
                lambda pid:self.heartbeat(stage='official_BOP_evaluation',dataset=name,evaluator_guard_pid=pid))
            score_path=evaluation_root/csv.stem/'scores_bop19.json'
            if returncode==0 and score_path.exists():
                scores=read_json(score_path)
                if not all(k in scores for k in ('bop19_average_recall','bop19_average_recall_vsd','bop19_average_recall_mssd','bop19_average_recall_mspd')):raise ValueError('Official final metric file is incomplete')
                result={'status':'complete','official_pose_metrics':scores,'command':command,'runtime_s':time.perf_counter()-start,
                        'score_file':str(score_path),'score_sha256':sha256(score_path),'toolkit_revision':runtime['toolkit_revision'],
                        'GT_selection':'official inst_count k-most-visible; visib_gt_min=-1','predictions_origin':'our_inference'}
            else:result={'status':'failed_official_evaluator','official_pose_metrics':None,'returncode':returncode,'command':command,'log':str(evaluation_root/'execution.log')}
        result['csv_sha256']=sha256(csv);result['target_sha256']=dataset['targets']['sha256'];atomic_json(receipt,result)
        return result

    def run(self,validation_only=False):
        with worker_lock(self.root/'worker.lock') as lock_stream:
            self.lock_stream=lock_stream
            self.heartbeat(stage='suite_start',force=True)
            atomic_json(self.root/'protocol.json',self.protocol)
            for name in self.protocol['execution_order']:
                try:self.validate_dataset(name)
                except (ValueError,RuntimeError,FileNotFoundError) as exc:
                    self.state(name,status='blocked_validation',reason=f'{type(exc).__name__}:{exc}')
                    atomic_json(self.root/'datasets'/name/'validation_exception.json',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
                    if isinstance(exc,InvalidCUDAContext):raise
            for name in ([] if validation_only else self.protocol['execution_order']):
                if self.states.get(name,{}).get('validation',{}).get('status')!='passed':continue
                try:
                    csv,complete=self.infer_dataset(name)
                    metric=self.evaluate_dataset(name,csv,complete)
                    self.state(name,status='complete' if complete and metric['status']=='complete' else ('inference_complete_metrics_unavailable' if complete else 'blocked_some_CADs'))
                except (ValueError,RuntimeError,FileNotFoundError) as exc:
                    self.state(name,status='blocked_runtime',reason=f'{type(exc).__name__}:{exc}')
                    atomic_json(self.root/'datasets'/name/'runtime_exception.json',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
                    if isinstance(exc,InvalidCUDAContext):raise
            self.release_backends()
            final=summarize(self.root,self.assets,self.states,self.protocol)
            self.heartbeat(stage='suite_pass_finished',suite_complete=final['suite_complete_and_evaluated'],force=True)
            return final
