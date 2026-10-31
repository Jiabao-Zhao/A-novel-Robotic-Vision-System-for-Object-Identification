"""Run the first actual public-code pose; all algorithms execute from the pinned checkout."""
import argparse, json, os, sys, time
from pathlib import Path
import reference_gate as ref

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfgpath=ref.local(a.config);cfg=ref.read(cfgpath);paths=cfg['paths'];s=cfg['settings']
    root=Path(__file__).resolve().parents[2];out=Path(paths['output']);base=Path(paths['assets_output'])
    source=Path(paths['public_source']);durable,owned=ref.pinned_guards(root)
    work=out/'first_execution';work.mkdir(parents=True,exist_ok=True)
    assets=ref.read(base/'assets_manifest.json')
    pins={'configuration_sha256':ref.sha256(cfgpath),'configuration':cfg,
          'public_source':ref.git_identity(source,ref.PUBLIC_REVISION),
          'toolkit_source':ref.git_identity(ref.local(ref.read(root/'outputs/freezev2_bop_20261029/evaluation_runtime.json')['toolkit_root']),ref.TOOLKIT_REVISION),
          'assets_manifest_sha256':ref.sha256(base/'assets_manifest.json'),
          'bootstrap_sha256':ref.sha256(__file__),'prediction_origin':'our generated inference; never authors CSVs'}
    durable.atomic_json(work/'identity.json',pins)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=paths['torch_home'],CUDA_VISIBLE_DEVICES='0',NVIDIA_TF32_OVERRIDE='0',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):env.pop(k,None)
    geom=Path(paths['geometry_root'])/'lmo_obj_000001_raw5000'/'geometry'
    if ref.read(geom.parent/'geometry_receipt.json')['status']!='complete':raise RuntimeError('Original geometry incomplete')
    features=Path(paths['features_dir']);features.mkdir(parents=True,exist_ok=True)
    v=features/'lmo_obj_000001_visual.npz';g=features/'lmo_obj_000001_gedi.npz';q=features/'lmo_obj_000001_query.npz'
    common=[paths['inference_python'],'-B',str(source/'run_bop.py')]
    visual=common+['extract-query-visual','--dataset','lmo','--obj-id','1','--layer',str(s['dino_layer']),
      '--depth-tolerance',str(s['depth_tolerance_mm']),'--min-views',str(s['min_views']),'--depth-sampling',s['depth_sampling'],
      '--device','cuda','--dino-model',s['dino_model'],'--dinov2-root',paths['dinov2_root'],
      '--onboarding-cache',str(geom/'onboarding.npz'),'--rgb-dir',str(geom/'rgb'),'--output',str(v)]
    gedi=common+['extract-query-gedi','--dataset','lmo','--obj-id','1','--bop-root',paths['bop_root'],
      '--gedi-root',paths['gedi_root'],'--checkpoint',paths['gedi_checkpoint'],'--seed',str(s['seed']),
      '--onboarding-cache',str(v),'--output',str(g)]
    fuse=common+['fuse-query-features','--dataset','lmo','--obj-id','1','--visual',str(v),'--geometric',str(g),
      '--pca-dim',str(s['pca_dim']),'--output',str(q)]
    masks=assets['datasets']['lmo']['masks']
    def find_path(value):
        if isinstance(value,dict):
            for key in ('path','source','json_path','file'):
                if key in value and isinstance(value[key],str) and value[key].endswith('.json'):return value[key]
            for value2 in value.values():
                try:return find_path(value2)
                except ValueError:pass
        if isinstance(value,list):
            for item in value:
                try:return find_path(item)
                except ValueError:pass
        raise ValueError('No original detection JSON path in mask receipt')
    maskpath=str(ref.local(find_path(masks)))
    poseout=work/'first_own_pose.json'
    pose=common+['estimate-multi-mask','--dataset','lmo','--scene-id','2','--im-id','3','--obj-id','1',
      '--bop-root',paths['bop_root'],'--split','test','--query-cache',str(q),'--detection-json',maskpath,
      '--targets-json',str(base/'bop_view/lmo/test_targets_bop19.json'),'--nms-translation-threshold-mm',str(s['nms_translation_threshold_mm']),
      '--dinov2-root',paths['dinov2_root'],'--gedi-root',paths['gedi_root'],'--checkpoint',paths['gedi_checkpoint'],
      '--device','cuda','--grid-size',str(s['grid_size']),'--dense-size',str(s['dense_size']),
      '--top-k',str(s['top_k']),'--iterations',str(s['iterations']),'--seed',str(s['seed']),
      '--edge-similarity-threshold',str(s['edge_similarity_threshold']),'--icp-max-iterations',str(s['icp_max_iterations']),
      '--work-dir',str(work/'pose_candidates'),'--output',str(poseout)]
    commands=[('visual',visual,v),('gedi',gedi,g),('fuse',fuse,q),('pose',pose,poseout)]
    with durable.worker_lock(paths['compute_lock']) as lock:
        for name,command,expected in commands:
            folder=work/name;folder.mkdir(parents=True,exist_ok=True)
            durable.require_disk(out,15)
            receipt={'status':'running','source_pins':pins,'command':command,'started_epoch':time.time()}
            durable.atomic_json(folder/'receipt.json',receipt);durable.atomic_json(folder/'command.json',command)
            started=time.perf_counter()
            def heartbeat(guard_pid):
                durable.atomic_json(work/'status.json',{'stage':name,'pid':os.getpid(),'start_ticks':owned.start_ticks(os.getpid()),
                    'guard_pid':guard_pid,'epoch':time.time(),'prediction_origin':'own generated'})
            code=owned.run_owned(command,env,folder,lock,heartbeat)
            receipt.update(returncode=code,runtime_s=time.perf_counter()-started,
                           status='complete' if code==0 and expected.is_file() else 'failed')
            if expected.is_file():receipt['outputs']={str(expected):ref.sha256(expected)}
            durable.atomic_json(folder/'receipt.json',receipt)
            print(json.dumps({'stage':name,'status':receipt['status'],'runtime_s':receipt['runtime_s']}),flush=True)
            if receipt['status']!='complete':return 1
    durable.atomic_json(work/'finished.json',{'status':'first_generated_pose_finished','configuration_sha256':pins['configuration_sha256'],
        'pose_result':str(poseout),'claim':'Real own inference; no benchmark performance claim from one case'})
    return 0
if __name__=='__main__':raise SystemExit(main())
