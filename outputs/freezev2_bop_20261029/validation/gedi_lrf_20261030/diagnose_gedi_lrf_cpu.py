"""Read-only CPU replay of original GeDi sampling/LRF; no descriptor model."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import open3d.ml.torch as ml3d

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
BASE=Path('/mnt/d/GitHub/A-novel-Robotic-Vision-System-for-Object-Identification/outputs/freezev2_bop_20261029')
RUN=BASE/'classic_core_base_v2'
SOURCE=BASE/'external/gedi/gedi.py'
source=SOURCE.read_text()
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert source_sha=='50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249'
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='LRF')
namespace={'torch':torch,'nn':nn,'np':np}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
LRF=namespace['LRF']
model_info=json.loads(Path('/mnt/d/BOP_SAM3_Pilot/data/tless/models_cad/models_info.json').read_text())

def detail(query,patches,radius):
    # Diagnostic instrumentation mirrors original CPU _forward intermediates.
    x=query.unsqueeze(2).contiguous()-patches.contiguous()
    covariance=torch.bmm(x,x.transpose(1,2))/patches.shape[2]
    _,singular,v=torch.svd(covariance)
    signs=torch.ones((len(patches),1))-2*((v[...,-1].unsqueeze(1)@x).sum(2)<0)
    zp=(signs*v[...,-1]).unsqueeze(1)
    x*=-1
    norm=(zp@x).transpose(1,2)
    vi=x-(norm*zp).transpose(1,2)
    alpha=(radius-torch.sqrt((x**2).sum(dim=1,keepdim=True)))**2
    beta=(norm*norm).transpose(1,2)
    vi_c=(alpha*beta*vi).sum(2)
    length=torch.sqrt((vi_c**2).sum(1))
    return singular,vi_c,length,norm

started=time.time()
report={'source_path':str(SOURCE),'source_sha256':source_sha,'seed':0,
        'sampling_batch':500,'lrf_samples':4000,'network_samples':512,
        'threads':2,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
        'neural_model_loaded':False,'GPU_operations':False,'GT_access':False,
        'active_run_modified':False,'objects':{}}
for object_id in (27,28,30,1):
    path=RUN/f'datasets/tless/onboarding/obj_{object_id:06d}/geometry/geometry.npz'
    with np.load(path) as data:
        points=np.asarray(data['points_m'],dtype=np.float32)
        support=np.asarray(data['support_points_m'],dtype=np.float32)
    query=torch.from_numpy(points);cloud=torch.from_numpy(support)
    diameter=model_info[str(object_id)]['diameter']*.001
    item={'geometry_path':str(path),'geometry_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'diameter_m':diameter,'queries':len(points),'support':len(support),
          'input_finite':bool(np.isfinite(points).all() and np.isfinite(support).all()),'scales':[]}
    np.random.seed(0)
    for fraction in (.3,.4):
        radius=float(fraction*diameter)
        search=ml3d.ops.radius_search(cloud,query,radius*torch.ones(len(points)),
            points_row_splits=torch.LongTensor([0,len(cloud)]),
            queries_row_splits=torch.LongTensor([0,len(query)]))
        lrf=LRF(patches_per_pair=500,samples_per_patch=512,r_lrf=radius,device='cpu')
        bad=[];zero=[];underflow=[];examples=[];counts=[];min_positive=float('inf')
        for start in range(0,len(query),500):
            end=min(start+500,len(query))
            array=np.empty((end-start,3,4000))
            for j,index in enumerate(range(start,end)):
                neighbors=search[0][search[1][index]:search[1][index+1]]
                counts.append(len(neighbors))
                try:
                    inds=np.random.choice(neighbors.numpy(),size=4000,replace=False)
                except Exception:
                    inds=np.r_[neighbors,np.random.choice(neighbors.numpy(),4000-len(neighbors))]
                array[j]=cloud[inds].T
            patches=torch.Tensor(array)
            with torch.inference_mode():
                actual=lrf(query[start:end],patches)
                singular,vi_c,length,norm=detail(query[start:end],patches,radius)
            actual_bad=~torch.isfinite(actual).all(dim=(1,2))
            zero_vector=(vi_c==0).all(1)
            zero_length=length==0
            bad.extend((torch.nonzero(actual_bad).flatten()+start).tolist())
            zero.extend((torch.nonzero(zero_vector).flatten()+start).tolist())
            underflow.extend((torch.nonzero(zero_length & ~zero_vector).flatten()+start).tolist())
            positive=length[length>0]
            if len(positive):min_positive=min(min_positive,float(positive.min()))
            for local in torch.nonzero(actual_bad).flatten().tolist():
                if len(examples)>=5:break
                index=start+local
                patch=patches[local]
                examples.append({'query_index':index,'point_m':points[index].tolist(),
                    'neighbors':counts[index],'sample_coordinate_ranges_m':(patch.max(1).values-patch.min(1).values).tolist(),
                    'covariance_singular_values':singular[local].tolist(),
                    'vi_c':vi_c[local].tolist(),'vi_c_norm':float(length[local]),
                    'max_abs_normal_offset_m':float(norm[local].abs().max()),
                    'exactly_planar_in_computed_LRF':bool((norm[local]==0).all())})
        scale={'fraction':fraction,'radius_m':radius,'nonfinite_patch_count':len(bad),
               'nonfinite_patch_indices':bad,'zero_vi_c_count':len(zero),
               'zero_vi_c_indices':zero,'normalization_underflow_count':len(underflow),
               'all_nonfinite_patches_explained_by_zero_norm':sorted(bad)==sorted(zero+underflow),
               'smallest_positive_norm':min_positive,'neighbor_min':min(counts),'neighbor_max':max(counts),
               'examples':examples,'original_adapter_reaches_scale':fraction==.3 or item['scales'][0]['nonfinite_patch_count']==0}
        item['scales'].append(scale)
        print(json.dumps({'object':object_id,'fraction':fraction,'bad':len(bad),'zero':len(zero),'underflow':len(underflow),'bad_equals_zero_norm':scale['all_nonfinite_patches_explained_by_zero_norm']}),flush=True)
    report['objects'][str(object_id)]=item
report['runtime_s']=time.time()-started
report['cuda_initialized']=torch.cuda.is_initialized()
target=Path(__file__).with_name('gedi_lrf_cpu_diagnosis_20261030.json')
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'audit':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'cuda_initialized':report['cuda_initialized'],'runtime_s':report['runtime_s']}))
