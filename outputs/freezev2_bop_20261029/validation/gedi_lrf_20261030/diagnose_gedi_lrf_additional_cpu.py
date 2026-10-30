"""CPU-only original LRF planar witness and one failed observed candidate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import open3d.ml.torch as ml3d
torch.set_num_threads(2);torch.set_num_interop_threads(2)
BASE=Path('/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029')
RUN=BASE/'classic_core_base_v2';SOURCE=BASE/'external/gedi/gedi.py'
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert source_sha=='50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249'
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LRF')
namespace={'torch':torch,'nn':nn,'np':np}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
LRF=namespace['LRF']
diagnostic_source=Path(__file__).with_name('diagnose_gedi_lrf_cpu.py')
node=next(n for n in ast.parse(diagnostic_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='detail')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(diagnostic_source),'exec'),namespace)
detail=namespace['detail']
report={'source_path':str(SOURCE),'source_sha256':source_sha,'CPU_threads':2,
        'CUDA_VISIBLE_DEVICES':'','GT_access':False,'neural_model_loaded':False,'active_run_modified':False}
started=time.time()
axis=torch.linspace(-.02,.02,64)
xx,yy=torch.meshgrid(axis,axis,indexing='ij')
plane=torch.stack([xx.flatten()[:4000],yy.flatten()[:4000],torch.ones(4000)],dim=0)
queries=torch.tensor([[0.,0.,1.],[.005,0.,1.]])
patches=plane[None].repeat(2,1,1)
np.random.seed(0)
lrf=LRF(patches_per_pair=500,samples_per_patch=512,r_lrf=.05,device='cpu')
with torch.inference_mode():
    output=lrf(queries,patches)
    singular,vi_c,length,norm=detail(queries,patches,.05)
report['synthetic_exact_plane']={'queries':queries.tolist(),'support_z':1.,'lrf_samples':4000,
        'network_samples':512,'vi_c':vi_c.tolist(),'norm':length.tolist(),
        'normal_offsets_all_zero':bool((norm==0).all()),'smallest_covariance_singular':singular[:,-1].tolist(),
        'nonfinite_patch_count':int((~torch.isfinite(output).all(dim=(1,2))).sum()),
        'original_eps':lrf.eps,'eps_is_not_used_in_original_forward':True}
sys.path.insert(0,str(RUN/'source_snapshot'))
from freezev2_repro.geometry import depth_to_metres,observed_clouds
folder=RUN/'datasets/tless/images/000003_000476/obj_000008/cnos-fastsam_007747'
diagnostic=json.loads((folder/'diagnostic.json').read_text())
src=diagnostic['source']
camera=json.loads(Path(src['camera']).read_text())['476']
depth=depth_to_metres(np.asarray(Image.open(src['depth'])),camera['depth_scale'])
mask=np.asarray(Image.open(folder/'mask.png'))>0
K=np.asarray(camera['cam_K']).reshape(3,3)
cloud=observed_clouds(depth,mask,K,grid=16,dense_count=3000,seed=0)
query=torch.from_numpy(np.asarray(cloud['sparse'],dtype=np.float32))
support=torch.from_numpy(np.asarray(cloud['dense'],dtype=np.float32))
diameter=json.loads(Path('/mnt/d/BOP_SAM3_Pilot/data/tless/models_cad/models_info.json').read_text())['8']['diameter']*.001
observed={'diagnostic_path':str(folder/'diagnostic.json'),'diagnostic_sha256':hashlib.sha256((folder/'diagnostic.json').read_bytes()).hexdigest(),
          'query_count':len(query),'support_count':len(support),'diameter_m':diameter,
          'input_finite':bool(torch.isfinite(query).all() and torch.isfinite(support).all()),'scales':[]}
np.random.seed(0)
for fraction in (.3,.4):
    radius=fraction*diameter
    search=ml3d.ops.radius_search(support,query,radius*torch.ones(len(query)),
        points_row_splits=torch.LongTensor([0,len(support)]),queries_row_splits=torch.LongTensor([0,len(query)]))
    array=np.empty((len(query),3,4000));neighbors_counts=[]
    for i in range(len(query)):
        neighbors=search[0][search[1][i]:search[1][i+1]];neighbors_counts.append(len(neighbors))
        try:inds=np.random.choice(neighbors.numpy(),size=4000,replace=False)
        except Exception:inds=np.r_[neighbors,np.random.choice(neighbors.numpy(),4000-len(neighbors))]
        array[i]=support[inds].T
    patches=torch.Tensor(array);lrf=LRF(patches_per_pair=500,samples_per_patch=512,r_lrf=radius,device='cpu')
    with torch.inference_mode():
        output=lrf(query,patches);singular,vi_c,length,norm=detail(query,patches,radius)
    bad=(~torch.isfinite(output).all(dim=(1,2)))
    zero=(vi_c==0).all(1)
    scale={'fraction':fraction,'radius_m':radius,'nonfinite_patch_indices':torch.nonzero(bad).flatten().tolist(),
        'zero_vi_c_indices':torch.nonzero(zero).flatten().tolist(),
        'all_nonfinite_explained_by_zero_norm':bool(torch.equal(bad,length==0)),
        'neighbor_counts':neighbors_counts,'examples':[]}
    for i in torch.nonzero(bad).flatten().tolist()[:5]:
        scale['examples'].append({'query_index':i,'point_m':query[i].tolist(),'neighbors':neighbors_counts[i],
            'vi_c':vi_c[i].tolist(),'vi_c_norm':float(length[i]),'covariance_singular_values':singular[i].tolist(),
            'max_abs_normal_offset_m':float(norm[i].abs().max()),
            'sample_coordinate_ranges_m':(patches[i].max(1).values-patches[i].min(1).values).tolist()})
    observed['scales'].append(scale)
    print(json.dumps({'observed_fraction':fraction,'queries':len(query),'bad':scale['nonfinite_patch_indices'],'zero':scale['zero_vi_c_indices'],'neighbors_min':min(neighbors_counts)}),flush=True)
report['observed_candidate']=observed
report['cuda_initialized']=torch.cuda.is_initialized();report['runtime_s']=time.time()-started
target=Path(__file__).with_name('gedi_lrf_cpu_additional_diagnosis_20261030.json')
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'synthetic':report['synthetic_exact_plane'],'audit':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'cuda_initialized':report['cuda_initialized']}))
