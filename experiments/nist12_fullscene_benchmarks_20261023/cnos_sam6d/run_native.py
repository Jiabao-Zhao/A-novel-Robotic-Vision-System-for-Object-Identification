"""Full official CNOS / SAM-6D ISM on raw scenes and a CAD catalogue.

Separate ``propose`` and ``match`` processes keep ViT-H and DINOv2 off the GPU
simultaneously. Official proposal generation, preprocessing, matching and NMS
are called directly. No previously localized scene crop is consumed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
from pathlib import Path
import sys
import types
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import HERE as ROOT, REPO, SOURCES, WEIGHTS, setup, read, save, path, relative, digest
sys.path.insert(0, str(REPO/'outputs/cache/bop_toolkit'))

DINO_HUB = REPO/'outputs/cache/torch_hub/facebookresearch_dinov2_e1277af2ba9496fbadf7aec6eba56e8d882d1e35'
DINO_WEIGHT = REPO/'outputs/cache/torch_hub/checkpoints/dinov2_vitl14_pretrain.pth'
SAM_WEIGHT = WEIGHTS/'sam_vit_h_4b8939.pth'
MICROBATCH = 2
POINTS_PER_BATCH = 8


def write_once(destination, value):
    if Path(destination).exists():
        assert read(destination) == value, f'Preserve frozen output {destination}'
    else:
        save(destination, value)


def official(method):
    from omegaconf import OmegaConf
    if method == 'cnos':
        source = SOURCES/'cnos'
        sys.path.insert(0, str(source))
        from src.model.sam import load_sam, CustomSamAutomaticMaskGenerator
        from src.model.detector import CNOS as Detector
        from src.model.dinov2 import CustomDINOv2
        from src.model.loss import PairwiseSimilarity
        from src.model.utils import Detections
        from src.dataloader.bop import BOPTemplate
        config = OmegaConf.load(source/'configs/model/cnos.yaml')
        sam_config = OmegaConf.load(source/'configs/model/segmentor_model/sam.yaml')
    else:
        source = SOURCES/'sam6d/SAM-6D/Instance_Segmentation_Model'
        sys.path.insert(0, str(source))
        from model.sam import load_sam, CustomSamAutomaticMaskGenerator
        from model.detector import Instance_Segmentation_Model as Detector
        from model.dinov2 import CustomDINOv2
        from model.loss import PairwiseSimilarity
        from model.utils import Detections
        from provider.bop import BOPTemplate
        config = OmegaConf.load(source/'configs/model/ISM_sam.yaml')
        sam_config = OmegaConf.load(source/'configs/model/segmentor_model/sam.yaml')
    return types.SimpleNamespace(**locals())


def check_inputs(manifest):
    for filename, expected in manifest['input_sha256'].items():
        assert digest(path(filename)) == expected, filename


def freeze(method, manifest, native):
    import torch
    receipt = read(SOURCES/f'{method}_provenance.json')
    base = SOURCES/method
    for filename, expected in receipt['source_sha256'].items():
        assert digest(base/filename) == expected, filename
    check_inputs(manifest)
    value = dict(method=method, official_revision=receipt['revision'],
        native_source_receipt_sha256=digest(SOURCES/f'{method}_provenance.json'),
        runner_sha256=digest(__file__), inputs_sha256=digest(ROOT/'inputs.json'),
        reference_bank_sha256=digest(ROOT/'templates42/completion.json'),
        SAM_checkpoint_sha256=digest(SAM_WEIGHT), DINO_checkpoint_sha256=digest(DINO_WEIGHT),
        scene_input='Raw complete RGB-D and intrinsic matrices only; no crops, masks, poses or extrinsics',
        catalogue='All 12 native metric CAD identities supplied simultaneously',
        proposal_config={k:native.sam_config.get(k) for k in
            ('stability_score_thresh','box_nms_thresh','min_mask_region_area','pred_iou_thresh')},
        proposal_width=640, native_points_per_batch=64, actual_points_per_batch=POINTS_PER_BATCH,
        descriptor_batch=MICROBATCH, descriptor='Official frozen non-register DINOv2 ViT-L/14 FP32',
        reference_variant='Released Pyrender route with 42 CAD-only canonical templates; native BOPTemplate preprocessing',
        final_mask_export='Official BOP force_binary_mask default: soft resized mask >0',
        execution_adjustments=[
            'SAM point decoder batches reduced64to8, with unchanged sampling grid/filter thresholds.',
            'Official SAM native-resolution bilinear mask upsampling is executed in chunks and copied to CPU.',
            'Official descriptor preprocessing/encoding in proposal batches2 to avoid replicating full RGB for every proposal.',
            'SAM-6D geometric projection uses official functions on CPU in proposal chunks2; native all-proposal IoU guard is preserved.',
            'A singleton geometry chunk is duplicated solely for the upstream squeeze API, then its duplicate discarded.',
            'Meters provided directly using depth_scale1000; original CAD points are already meters, so no mm division.',
        ],
        torch_version=torch.__version__, GPU=torch.cuda.get_device_name(0),
        simulator_labels_or_poses_read=False, fixed_scene_proposals=False)
    write_once(ROOT/method/'protocol.json', value)


def propose(method):
    manifest = setup()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    torch.manual_seed(2023)
    np.random.seed(2023)
    assert torch.cuda.is_available() and SAM_WEIGHT.exists()
    native = official(method)
    freeze(method, manifest, native)
    sam = native.load_sam('vit_h', str(WEIGHTS)).eval().requires_grad_(False).cuda()
    torch.cuda.reset_peak_memory_stats()
    options = dict(sam=sam, points_per_batch=POINTS_PER_BATCH,
        stability_score_thresh=float(native.sam_config.stability_score_thresh),
        box_nms_thresh=float(native.sam_config.box_nms_thresh), min_mask_region_area=0,
        crop_overlap_ratio=native.sam_config.get('crop_overlap_ratio'), segmentor_width_size=640)
    if method != 'cnos':
        options['pred_iou_thresh'] = float(native.sam_config.pred_iou_thresh)
    generator = native.CustomSamAutomaticMaskGenerator(**options)

    def upsample_cpu(self, detections, orig_size):
        # Identical official interpolation and box scaling, without holding
        # every full-resolution proposal on the 8GB GPU simultaneously.
        original = detections['masks']
        masks = [F.interpolate(original[start:start+4, None].float(), size=orig_size,
                               mode='bilinear', align_corners=False)[:, 0].cpu()
                 for start in range(0, len(original), 4)]
        detections['masks'] = torch.cat(masks) if masks else torch.empty((0,*orig_size))
        boxes = detections['boxes'].float().cpu() * (orig_size[1]/self.segmentor_width_size)
        boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, orig_size[1]-1)
        boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, orig_size[0]-1)
        detections['boxes'] = boxes
        return detections

    generator.postprocess_resize = types.MethodType(upsample_cpu, generator)
    for scene in manifest['scenes']:
        destination = ROOT/method/'raw_proposals'/scene['scene_id']
        if (destination/'completion.json').exists():
            evidence = read(destination/'completion.json')
            assert digest(destination/'soft_masks.npz') == evidence['soft_masks_sha256']
            continue
        started = perf_counter()
        rgb = np.asarray(Image.open(path(scene['rgb'])).convert('RGB'))
        with torch.inference_mode():
            result = generator.generate_masks(rgb)
        masks = result['masks'].cpu().numpy()
        compact = masks.astype(np.float16)
        assert np.array_equal(compact.astype(np.float32), masks), 'Lossless float16 mask storage required'
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/'soft_masks.npz', masks=compact)
        rows = [dict(proposal_id=i, mask_index=i, bbox_xyxy=box.tolist(),
                     positive_mask_pixels=int(np.count_nonzero(masks[i]>0)))
                for i,box in enumerate(result['boxes'].cpu().numpy())]
        save(destination/'proposals.json', rows)
        save(destination/'completion.json', dict(scene_id=scene['scene_id'], proposals=len(rows),
            input_rgb_sha256=digest(path(scene['rgb'])), seconds=perf_counter()-started,
            native_resolution=list(rgb.shape[:2]), own_method_proposals=True,
            soft_masks_sha256=digest(destination/'soft_masks.npz'),
            proposals_sha256=digest(destination/'proposals.json'), soft_mask_storage='lossless float16'))
        print(f'{method} own SAM proposals {scene["scene_id"]}: {len(rows)}, '
              f'peakCUDA={torch.cuda.max_memory_allocated()/1024**3:.2f}GiB', flush=True)
        del result, masks, compact, rows
    save(ROOT/method/'proposal_completion.json', dict(passed=True, scenes=8,
        method=method, source='Official method-specific SAM ViT-H generator', inputs_sha256=digest(ROOT/'inputs.json')))


def build_detector(method, native):
    import torch
    from omegaconf import OmegaConf
    if method == 'cnos':
        encoder = torch.hub.load(str(DINO_HUB), 'dinov2_vitl14', source='local', pretrained=False)
        encoder.load_state_dict(torch.load(DINO_WEIGHT, map_location='cpu', weights_only=True), strict=True)
        descriptor = native.CustomDINOv2(model_name='dinov2_vitl14', model=encoder,
            token_name='x_norm_clstoken', image_size=224, chunk_size=MICROBATCH, descriptor_width_size=640)
    else:
        descriptor = native.CustomDINOv2(model_name='dinov2_vitl14', token_name='x_norm_clstoken',
            image_size=224, chunk_size=MICROBATCH, descriptor_width_size=640,
            checkpoint_dir=str(DINO_WEIGHT.parent), validpatch_thresh=.5)
    descriptor = descriptor.eval().requires_grad_(False).cuda()
    cfg = native.config
    match = OmegaConf.create(OmegaConf.to_container(cfg.matching_config))
    match.metric = None
    # SimpleNamespace preserves the native metric object; OmegaConf would
    # otherwise reject a live torch module as a scalar configuration value.
    matching = types.SimpleNamespace(**{k:v for k,v in match.items() if k!='metric'},
                                      metric=native.PairwiseSimilarity(metric='cosine',chunk_size=16))
    settings = dict(segmentor_model=None, descriptor_model=descriptor,
        onboarding_config=cfg.onboarding_config, matching_config=matching,
        post_processing_config=cfg.post_processing_config, log_interval=5,
        log_dir=str(ROOT/method/'native_logs'))
    if method=='cnos':
        settings['save_mask']=True
    else:
        settings.update(visible_thred=.5,pointcloud_sample_num=2048)
    return native.Detector(**settings).eval()


def reference_features(method, native, detector, manifest):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    templates = ROOT/'templates42'
    evidence = read(templates/'completion.json')
    for filename, expected in evidence['sha256'].items():
        assert digest(path(filename))==expected
    dataset = native.BOPTemplate(template_dir=str(templates),
        obj_ids=[obj['object_id'] for obj in manifest['objects']],
        processing_config=OmegaConf.create({'image_size':224}),level_templates=0,pose_distribution='all')
    metadata = read(templates/'metadata.json')
    np.testing.assert_array_equal(dataset.index_templates, metadata['index_templates'])
    descriptors, appearance = [], []
    with torch.inference_mode():
        for i, obj in enumerate(manifest['objects']):
            item=dataset[i]
            rgb=item['templates'].cuda()
            descriptors.append(detector.descriptor_model.compute_features(rgb,token_name='x_norm_clstoken'))
            if method!='cnos':
                appearance.append(detector.descriptor_model.compute_masked_patch_feature(rgb,item['template_masks'].cuda()))
            print(f'{method} native onboarding: {obj["target"]}',flush=True)
    detector.ref_data={'descriptors':torch.stack(descriptors)}
    if appearance:
        detector.ref_data['appe_descriptors']=torch.stack(appearance)
        import trimesh
        clouds=[]
        for obj in manifest['objects']:
            np.random.seed(2023+obj['object_id'])
            mesh=trimesh.load_mesh(path(obj['mesh']))
            # Source geometry stores BOP CAD millimeters then /1000; our source
            # CAD already uses meters and its origin/scale remain unchanged.
            clouds.append(torch.from_numpy(mesh.sample(2048).astype(np.float32)))
        detector.ref_data['pointcloud']=torch.stack(clouds).cuda()
        detector.ref_data['poses']=torch.from_numpy(np.asarray(metadata['object_poses_camera_m'],np.float32)).cuda()
    cache={k:v.detach().cpu() for k,v in detector.ref_data.items()}
    torch.save(cache,ROOT/method/'reference_features.pt')
    return metadata


def output_detections(method, scene_id, detections, components, image_shape):
    import numpy as np
    destination=ROOT/method/scene_id
    assert not (destination/'completion.json').exists()
    destination.mkdir(parents=True,exist_ok=True)
    masks=(detections.masks.cpu().numpy()>0) if len(detections) else np.zeros((0,*image_shape),bool)
    np.savez_compressed(destination/'masks.npz',masks=masks)
    rows=[]
    for i in range(len(detections)):
        pid=int(detections.proposal_ids[i])
        rows.append(dict(proposal_id=pid, object_id=int(detections.object_ids[i])+1,
            score=float(detections.scores[i]), bbox_xyxy=detections.boxes[i].cpu().tolist(),
            mask_index=i, components=components.get(pid,{})))
    save(destination/'detections.json',rows)
    save(destination/'completion.json',dict(passed=True,method=method,scene_id=scene_id,
        retained_detections=len(rows), own_proposals=True,
        masks_sha256=digest(destination/'masks.npz'),detections_sha256=digest(destination/'detections.json'),
        no_ground_truth_read=True))


def geometric_projection(native, detector, best, objects, batch, masks):
    """Official per-proposal projection, with bounded CPU working memory."""
    import torch
    helper=types.SimpleNamespace(ref_data={k:detector.ref_data[k].cpu() for k in ('poses','pointcloud')})
    helper.Calculate_the_query_translation=types.MethodType(native.Detector.Calculate_the_query_translation,helper)
    outputs=[]
    for start in range(0,len(best),2):
        poses=best[start:start+2].cpu(); ids=objects[start:start+2].cpu()
        subset=masks[start:start+2].clone().cpu()
        singleton=len(subset)==1
        if singleton:
            subset=torch.cat([subset,subset]); poses=torch.cat([poses,poses]); ids=torch.cat([ids,ids])
        projected=native.Detector.project_template_to_image(helper,poses,ids,batch,subset)
        outputs.append(projected[:1] if singleton else projected)
    return torch.cat(outputs)


def match(method):
    manifest=setup()
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4); torch.manual_seed(2023); np.random.seed(2023)
    native=official(method)
    freeze(method,manifest,native)
    detector=build_detector(method,native)
    reference_features(method,native,detector,manifest)
    for scene in manifest['scenes']:
        scene_id=scene['scene_id']
        methods=['cnos'] if method=='cnos' else ['sam6d_rgb','sam6d_rgbd']
        if all((ROOT/m/scene_id/'completion.json').exists() for m in methods):
            continue
        raw_dir=ROOT/method/'raw_proposals'/scene_id
        evidence=read(raw_dir/'completion.json')
        assert digest(raw_dir/'soft_masks.npz')==evidence['soft_masks_sha256']
        raw_masks=np.load(raw_dir/'soft_masks.npz')['masks'].astype(np.float32)
        raw=read(raw_dir/'proposals.json')
        rgb=np.asarray(Image.open(path(scene['rgb'])).convert('RGB'))
        detections=native.Detections(dict(masks=torch.from_numpy(raw_masks),
            boxes=torch.tensor([r['bbox_xyxy'] for r in raw],dtype=torch.float32),
            proposal_ids=torch.arange(len(raw),dtype=torch.long)))
        detections.remove_very_small_detections(native.config.post_processing_config.mask_post_processing)
        del raw_masks
        save(ROOT/method/scene_id/'proposal_filter.json',dict(raw_proposals=len(raw),
            retained_after_native_size_filter=len(detections),
            retained_proposal_ids=detections.proposal_ids.tolist()))
        if not len(detections):
            for output in methods: output_detections(output,scene_id,detections,{},rgb.shape[:2])
            continue
        cls, patches=[],[]
        with torch.inference_mode():
            for start in range(0,len(detections),MICROBATCH):
                subset=native.Detections(dict(masks=detections.masks[start:start+MICROBATCH].cuda(),
                    boxes=detections.boxes[start:start+MICROBATCH].cuda()))
                processed=detector.descriptor_model.process_rgb_proposals(rgb,subset.masks,subset.boxes)
                if method=='cnos':
                    cls.append(detector.descriptor_model.compute_features(processed,token_name='x_norm_clstoken'))
                else:
                    processed_masks=detector.descriptor_model.process_masks_proposals(subset.masks,subset.boxes)
                    processed_masks=processed_masks.reshape(len(subset),224,224)
                    c,p=detector.descriptor_model.compute_cls_and_patch_features(processed,processed_masks)
                    cls.append(c);patches.append(p)
                del subset,processed
            queries=torch.cat(cls)
            if method=='cnos':
                selected,objects,scores,distributions=detector.find_matched_proposals(queries)
                detections.filter(selected.cpu())
                components={int(p):{'class_scores':v.cpu().tolist()} for p,v in zip(detections.proposal_ids,distributions)}
                detections.add_attribute('object_ids',objects.cpu())
                detections.add_attribute('scores',scores.cpu())
                if len(detections):detections.apply_nms_per_object_id(float(native.config.post_processing_config.nms_thresh))
                output_detections('cnos',scene_id,detections,components,rgb.shape[:2])
            else:
                selected,objects,semantic,best=detector.compute_semantic_score(queries)
                detections.filter(selected.cpu())
                if not len(detections):
                    for output in methods:output_detections(output,scene_id,detections,{},rgb.shape[:2])
                    continue
                query_patches=torch.cat(patches)[selected]
                appearance,ref_patches=detector.compute_appearance_score(best,objects,query_patches)
                batch={'depth':torch.from_numpy(np.load(path(scene['depth'])))[None],
                    'cam_intrinsic':torch.from_numpy(np.load(path(scene['intrinsics'])))[None],
                    'depth_scale':torch.tensor([1000.])}
                projected=geometric_projection(native,detector,best,objects,batch,detections.masks)
                # One native full-batch geometry call intentionally preserves
                # the released compute_iou all-batch intersection guard.
                geometry,coverage=detector.compute_geometric_score(projected,detections,query_patches,
                    ref_patches,visible_thred=detector.visible_thred)
                if not torch.is_tensor(geometry):geometry=torch.zeros_like(semantic)+float(geometry)
                geometry=geometry.to(semantic.device)
                values={'sam6d_rgb':(semantic+appearance)/2,
                    'sam6d_rgbd':(semantic+appearance+geometry*coverage)/(2+coverage)}
                components={int(pid):dict(semantic=float(semantic[i]),appearance=float(appearance[i]),
                    geometry_bbox_iou=float(geometry[i]),patch_coverage=float(coverage[i]),
                    best_template_ordinal=int(best[i])) for i,pid in enumerate(detections.proposal_ids)}
                for output,score in values.items():
                    copied=native.Detections(dict(masks=detections.masks,boxes=detections.boxes,
                        proposal_ids=detections.proposal_ids))
                    copied.add_attribute('object_ids',objects.cpu());copied.add_attribute('scores',score.cpu())
                    copied.apply_nms_per_object_id(float(native.config.post_processing_config.nms_thresh))
                    output_detections(output,scene_id,copied,components,rgb.shape[:2])
        print(f'{method} official full identification {scene_id} completed',flush=True)
    for output in (['cnos'] if method=='cnos' else ['sam6d_rgb','sam6d_rgbd']):
        save(ROOT/output/'completion.json',dict(passed=True,scenes=8,method=output,
            own_raw_scene_proposals=True,reference_features_sha256=digest(ROOT/method/'reference_features.pt'),
            runner_sha256=digest(__file__),inputs_sha256=digest(ROOT/'inputs.json')))


if __name__=='__main__':
    method,phase=sys.argv[1:3]
    assert method in ('cnos','sam6d')
    {'propose':propose,'match':match}[phase](method)
