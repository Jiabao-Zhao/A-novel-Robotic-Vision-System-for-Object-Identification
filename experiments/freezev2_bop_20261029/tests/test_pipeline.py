import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from freezev2_repro.pipeline import load_config, config_hash, load_feature_backends
from freezev2_repro.data import BOPDataset
from freezev2_repro.rendering import camera_poses


def test_reader_does_not_expose_gt_or_camera_extrinsics(tmp_path):
    root=tmp_path/'tless'
    (root/'models_cad').mkdir(parents=True)
    (root/'models_cad'/'models_info.json').write_text('{"1":{"diameter":60}}')
    scene=root/'test_primesense'/'000005'
    (scene/'rgb').mkdir(parents=True)
    (scene/'depth').mkdir()
    Image.fromarray(np.zeros((3,4,3),np.uint8)).save(scene/'rgb'/'000007.png')
    Image.fromarray(np.full((3,4),1000,np.uint16)).save(scene/'depth'/'000007.png')
    (scene/'scene_camera.json').write_text(json.dumps({'7':{'cam_K':[100,0,2,0,100,1,0,0,1],
        'depth_scale':.1,'cam_R_w2c':'forbidden','cam_t_w2c':'forbidden'}}))
    # A malformed GT file would fail immediately if inference tried to read it.
    (scene/'scene_gt.json').write_text('DO NOT READ')
    frame=BOPDataset(tmp_path,'tless').frame(5,7)
    assert np.allclose(frame['depth_m'],.1)
    assert not any(key.startswith('cam_') for key in frame)


def test_exact_cnos_rotation_and_native_cad_projection():
    p=np.array([[-.01,-.02,-.03],[.01,.02,.03]])
    cams=np.tile(np.eye(4),(162,1,1))
    cams[:,:3,:3]=np.diag([-1,-1,1])
    cams[:,2,3]=-1000
    before=p.copy()
    K,poses=camera_poses(p,camera_to_model=cams)
    assert np.array_equal(p,before)
    R,t=poses[0]
    assert np.array_equal(R,cams[0,:3,:3].T)
    assert t[2]>np.max(p[:,2])
    assert np.isclose(np.linalg.det(R),1)


def test_wrong_configuration_and_missing_features_fail_closed():
    config=load_config(Path(__file__).parents[1]/'config.json')
    changed=json.loads(json.dumps(config))
    changed['seed']=1
    assert config_hash(config)!=config_hash(changed)
    config['paths']['dino_checkpoint']='definitely_missing_giant.pth'
    with pytest.raises(FileNotFoundError,match='no descriptor substitution'):
        load_feature_backends(config)


def test_nonpointing_cnos_camera_rejected():
    p=np.array([[-.01,-.02,-.03],[.01,.02,.03]])
    cams=np.tile(np.eye(4),(162,1,1))
    cams[:,2,3]=1000
    with pytest.raises(ValueError,match='face CAD origin'):
        camera_poses(p,camera_to_model=cams)
