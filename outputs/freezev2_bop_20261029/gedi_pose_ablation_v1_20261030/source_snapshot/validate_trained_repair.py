"""Bounded original-checkpoint GeDi repair diagnostic; no pose inference/GT."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="2"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gedi_valid_rows import extract_valid_rows

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
ROOT=Path('/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification')
BASE=ROOT/'outputs/freezev2_bop_20261029'
RUN=BASE/'classic_core_base_v2'
OUT=BASE/'gedi_valid_rows_v1_20261030'
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(RUN/'source_snapshot'))
from freezev2_repro.features import FrozenGeDi
from freezev2_repro.geometry import depth_to_metres, observed_clouds

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

recipe=json.loads((RUN/'protocol.json').read_text())['base']['features']
model=FrozenGeDi(BASE/'external/gedi',BASE/'external/gedi/data/chkpts/3dmatch/chkpt.tar',
 expected_source_sha256=recipe['gedi_source_sha256'],expected_revision=recipe['gedi_source_revision'],
 checkpoint_sha256=recipe['gedi_checkpoint_sha256'],forward_microbatch_size=32)
assert not model.backend.gedi_net.training
assert all(not p.requires_grad for p in model.backend.gedi_net.parameters())
source_before=sha(BASE/'external/gedi/gedi.py')
report={'method':'GeDi-valid-prepared-rows-v1-diagnostic','independent_repair_not_author_parity':True,
 'scope':'Geometric descriptor diagnostic only. No DINO extraction/PCA/refitting/poses/GT/evaluation.',
 'source_sha256':source_before,'checkpoint_sha256':model.provenance['checkpoint_sha256'],
 'sampling_batch':500,'LRF_samples':4000,'network_samples':512,'forward_microbatch':32,
 'radii_diameter_fractions':[.3,.4],'seed':0,'torch':torch.__version__,'cases':{},
 'PCA_policy':'Not touched by this geometric-only diagnostic; preserve existing visual PCA in any minimal later pose comparison.',
 'support_policy':'All original support points retained; query validity applied only after both scale passes.',
 'invalid_policy':'Nonfinite prepared rows never enter descriptor network. Internal sentinel rows are excluded by explicit original-index mask before descriptor output.',
 'limitations':['Does not repair empty-neighborhood exceptions.','Does not create features for a genuinely uninformative flat patch.','No pose accuracy improvement measured.','A later pose runner must apply the same original indices to sparse/CAD points and visual descriptors, with explicit treatment of full CAD geometry for ICP.']}
started=time.perf_counter()
info=json.loads(Path('/mnt/d/BOP_SAM3_Pilot/data/tless/models_cad/models_info.json').read_text())

def process(name,points,support,diameter,expected_valid,source_bindings):
    t0=time.perf_counter()
    result=extract_valid_rows(model.backend,torch.from_numpy(points.astype(np.float32)),
       torch.from_numpy(support.astype(np.float32)),[.3*diameter,.4*diameter],seed=0)
    assert result['valid_rows']==expected_valid,(name,result['valid_rows'],expected_valid)
    assert result['descriptors'].shape==(expected_valid,64)
    assert np.isfinite(result['descriptors']).all()
    assert all(len(s['nonfinite_neural_indices'])==0 for s in result['scales'])
    path=OUT/(name+'.npz')
    np.savez_compressed(path,original_indices=result['original_indices'],geometric=result['descriptors'])
    item={k:v for k,v in result.items() if k!='descriptors'}
    item.update({'descriptor_shape':list(result['descriptors'].shape),'all_descriptors_finite':True,
        'sources':source_bindings,'cache_path':str(path),'cache_sha256':sha(path),'seconds':time.perf_counter()-t0})
    if name=='cad_000001_control':
        old=RUN/'datasets/tless/onboarding/obj_000001/features/geometric.npz'
        with np.load(old,allow_pickle=False)as d:reference=d['geometric']
        difference=float(np.max(np.abs(result['descriptors']-reference)))
        np.testing.assert_allclose(result['descriptors'],reference,rtol=1e-5,atol=1e-6)
        item['old_healthy_cache_comparison']={'path':str(old),'sha256':sha(old),'max_absolute_difference':difference,'rtol':1e-5,'atol':1e-6,'passed':True}
    report['cases'][name]=item
    print(json.dumps({'case':name,'input':result['input_rows'],'valid':result['valid_rows'],'invalid':result['invalid_rows'].tolist(),'all_descriptors_finite':True,'seconds':item['seconds']}),flush=True)
    (OUT/'progress.json').write_text(json.dumps(report,indent=2,default=jsonable)+'\n')

for obj,expected in [(1,5000),(27,4988),(28,4990),(30,4994)]:
    path=RUN/f'datasets/tless/onboarding/obj_{obj:06d}/geometry/geometry.npz'
    with np.load(path,allow_pickle=False)as d:
        points=np.asarray(d['points_m'],dtype=np.float32);support=np.asarray(d['support_points_m'],dtype=np.float32)
    process(f'cad_{obj:06d}'+('_control'if obj==1 else ''),points,support,info[str(obj)]['diameter']*.001,expected,{'geometry':str(path),'geometry_sha256':sha(path)})
folder=RUN/'datasets/tless/images/000003_000476/obj_000008/cnos-fastsam_007747'
diagnostic=json.loads((folder/'diagnostic.json').read_text());src=diagnostic['source']
camera=json.loads(Path(src['camera']).read_text())['476']
depth=depth_to_metres(np.asarray(Image.open(src['depth'])),camera['depth_scale'])
mask=np.asarray(Image.open(folder/'mask.png'))>0
K=np.asarray(camera['cam_K']).reshape(3,3)
cloud=observed_clouds(depth,mask,K,grid=16,dense_count=3000,seed=0)
assert len(cloud['sparse'])==91 and len(cloud['dense'])==3000
process('observed_000003_000476_obj8',cloud['sparse'],cloud['dense'],info['8']['diameter']*.001,90,
 {'diagnostic':str(folder/'diagnostic.json'),'diagnostic_sha256':sha(folder/'diagnostic.json'),'mask_sha256':sha(folder/'mask.png')})
assert sha(BASE/'external/gedi/gedi.py')==source_before
report['seconds']=time.perf_counter()-started
report['peak_allocated_CUDA_bytes']=torch.cuda.max_memory_allocated()
report['original_source_unchanged']=True
report['repair_adapter_sha256']=sha(Path(__file__).with_name('gedi_valid_rows.py'))
report['validation_script_sha256']=sha(__file__)
report['status']='passed_trained_descriptor_diagnostic_not_pose_benchmark'
(OUT/'trained_validation.json').write_text(json.dumps(report,indent=2,default=jsonable)+'\n')
print(json.dumps({'status':report['status'],'audit':str(OUT/'trained_validation.json'),'seconds':report['seconds'],'peak_allocated_CUDA_bytes':report['peak_allocated_CUDA_bytes']}),flush=True)
