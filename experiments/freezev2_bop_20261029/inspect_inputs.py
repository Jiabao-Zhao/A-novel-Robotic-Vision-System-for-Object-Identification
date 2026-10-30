"""Read-only BOP/predicted-mask geometry diagnostic; no trained pose inference."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from freezev2_repro.pipeline import load_config, local_path, config_hash
from freezev2_repro.data import BOPDataset, write_json, sha256
from freezev2_repro.masks import load_localization_targets, load_mask_candidates, select_localization_candidates
from freezev2_repro.geometry import observed_clouds, project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    args=parser.parse_args()
    cfg=load_config(args.config)
    output=local_path(cfg['paths']['output_root'])/'input_geometry_diagnostic'
    if output.exists():
        raise ValueError('Preserve previous diagnostics; choose separate configured output_root')
    dataset=BOPDataset(local_path(cfg['paths']['dataset_root']),cfg['dataset'])
    keys={(r['scene_id'],r['im_id'],cfg['object_id']) for r in cfg['cohort']}
    targets=[t for t in load_localization_targets(local_path(cfg['paths']['targets'])) if (t.scene_id,t.image_id,t.object_id) in keys]
    masks=[]
    for source in cfg['masks']:
        masks.extend(load_mask_candidates(local_path(source['path']),segmentor=source['segmentor'],source_id=source['source_id'],target_keys=keys))
    report={'scope':'observed_RGBD_and_external_mask_geometry_only','trained_features_or_pose_inference':False,
            'config_sha256':config_hash(cfg),'GT_pose_access':False,'target_sha256':sha256(local_path(cfg['paths']['targets'])), 'images':[]}
    for target in targets:
        frame=dataset.frame(target.scene_id,target.image_id)
        pool=select_localization_candidates(masks,scene_id=target.scene_id,image_id=target.image_id,object_id=target.object_id,instance_count=target.instance_count)
        image_report={'scene_id':target.scene_id,'im_id':target.image_id,'obj_id':target.object_id,'instance_count':target.instance_count,
                      'source':frame['source'],'K':frame['K'].tolist(),'depth_scale_mm_per_raw':frame['depth_scale_mm'],'candidates':[]}
        for candidate in pool:
            folder=output/f'{target.scene_id:06d}_{target.image_id:06d}'/f'candidate_{candidate.record_index:06d}'
            folder.mkdir(parents=True)
            record={'mask':candidate.provenance(),'mask_area':int(candidate.mask.sum())}
            Image.fromarray(candidate.mask.astype(np.uint8)*255).save(folder/'mask.png')
            try:
                clouds=observed_clouds(frame['depth_m'],candidate.mask,frame['K'],grid=cfg['observed']['grid'],dense_count=cfg['observed']['dense_points'],seed=cfg['seed'])
                Image.fromarray(clouds['crop'].extract_rgb(frame['rgb'])).save(folder/'crop.png')
                np.savez_compressed(folder/'observed_clouds.npz',sparse_m=clouds['sparse'],dense_m=clouds['dense'],patch_indices=clouds['patch_indices'],pixel_uv=clouds['pixel_uv'],feature_uv=clouds['feature_uv'])
                uv,valid=project(clouds['sparse'],frame['K'])
                residual=float(np.max(np.abs(uv-clouds['pixel_uv'])))
                if residual>1e-6 or not valid.all():
                    raise ValueError('Observed projection/backprojection failed')
                image=Image.fromarray(frame['rgb'].copy())
                overlay=np.asarray(image).copy()
                overlay[candidate.mask]=(overlay[candidate.mask]*.5+np.array([20,190,230])*.5).astype(np.uint8)
                image=Image.fromarray(overlay)
                draw=ImageDraw.Draw(image)
                for x,y in clouds['pixel_uv']:
                    draw.ellipse((x-1,y-1,x+1,y+1),fill='yellow')
                draw.text((8,8),'public predicted mask + valid sparse depth; no pose/GT',fill='red')
                image.save(folder/'sparse_overlay.png')
                record.update(status='valid_observed_geometry',sparse_count=len(clouds['sparse']),dense_count=len(clouds['dense']),
                              available_dense_points=clouds['valid_dense_before_subsampling'],median_depth_m=float(np.median(clouds['dense'][:,2])),
                              reprojection_max_error_px=residual)
            except ValueError as exc:
                record.update(status='unavailable',reason=str(exc))
            image_report['candidates'].append(record)
        report['images'].append(image_report)
    write_json(output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
