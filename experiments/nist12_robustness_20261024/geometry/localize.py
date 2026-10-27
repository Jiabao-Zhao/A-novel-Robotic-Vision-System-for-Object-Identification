"""Unchanged depth proposal construction for prospectively gripper-free scenes."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import sys
from dataclasses import asdict
from time import perf_counter
import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, REPO, read, save, path, digest, relative
sys.path.insert(0, str(REPO))
from point_cloud_localization import PointCloudConfig, PointCloudLocalization

CONFIG = dict(voxel_size_m=.002, plane_distance_threshold_m=.002,
    dbscan_eps_m=.02, dbscan_min_points=5, cluster_in_table_plane=False,
    min_cluster_points=50, min_cluster_extent_m=.006, max_cluster_aspect_ratio=16.,
    min_roi_width_px=8, min_roi_height_px=8, edge_artifact_margin_px=3,
    bottom_edge_artifact_max_thickness_m=.012, bottom_edge_artifact_max_roi_height_ratio=.16,
    raw_cluster_padding_m=.004, raw_cluster_dbscan_eps_m=.006, raw_cluster_dbscan_min_points=5,
    statistical_outlier_neighbors=10, statistical_outlier_std_ratio=2.,
    radius_outlier_radius_m=.008, radius_outlier_min_neighbors=2)


def localize_scene(scene, destination, geometry_cache):
    start = perf_counter()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes = {key: digest(path(scene[key])) for key in ('rgb', 'depth', 'intrinsics')}
    signature = (source_hashes['depth'], source_hashes['intrinsics'])
    complete = destination/'localization.json'
    if complete.exists():
        saved = read(complete)
        assert saved['input_sha256'] == source_hashes
        geometry_cache.setdefault(signature, destination)
        return saved
    rgb = np.asarray(Image.open(path(scene['rgb'])).convert('RGB'))
    depth = np.load(path(scene['depth']))
    K = np.load(path(scene['intrinsics']))
    height, width = depth.shape
    assert rgb.shape[:2] == depth.shape and K.shape == (3, 3)
    cached = geometry_cache.get(signature)
    if cached is not None:
        saved = read(cached/'localization.json')
        proposals = saved['proposals']
        plane = saved['plane_camera_m']
        masks_file = cached/'masks.npz'
        try:
            os.link(masks_file, destination/'masks.npz')
        except OSError:
            import shutil
            shutil.copyfile(masks_file, destination/'masks.npz')
        rows = []
        for proposal in proposals:
            roi = proposal['roi']
            crop = destination/f"crop_{proposal['proposal_id']:03d}.png"
            Image.fromarray(rgb[roi['y1']:roi['y2']+1, roi['x1']:roi['x2']+1]).save(crop)
            rows.append(dict(proposal, scene_id=scene['scene_id'], crop=relative(crop), crop_sha256=digest(crop)))
        result = dict(saved, scene_id=scene['scene_id'], proposals=rows, input_sha256=source_hashes,
                      geometry_cache_source=relative(cached), seconds=perf_counter()-start)
        save(complete, result)
        return result
    camera = {'width': width, 'height': height, 'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
    localizer = PointCloudLocalization(PointCloudConfig(**CONFIG))
    # Open3D Image accepts float32 depth; scoring below retains the raw array.
    cloud_depth = np.ascontiguousarray(depth, dtype=np.float32)
    cloud = localizer.rgbd_to_pointcloud(rgb, cloud_depth, 1., camera).voxel_down_sample(CONFIG['voxel_size_m'])
    o3d.utility.random.seed(0)
    objects, plane, _ = localizer.segment_table_plane(cloud)
    clusters = localizer.cluster_objects(objects, camera, plane)
    proposals = localizer.localize_clusters(clusters, camera)
    rows, masks = [], []
    for pid, proposal in enumerate(proposals):
        roi = proposal.roi
        x1, y1, x2, y2 = [roi[key] for key in ('x1', 'y1', 'x2', 'y2')]
        yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
        z = depth[y1:y2+1, x1:x2+1]
        signed = (((xx-K[0, 2])*z/K[0, 0])*plane[0]
                  + ((yy-K[1, 2])*z/K[1, 1])*plane[1] + z*plane[2] + plane[3])
        foreground = np.isfinite(z) & (z > 0) & (np.abs(signed)/np.linalg.norm(plane[:3]) > .002)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[y1:y2+1, x1:x2+1] = foreground
        masks.append(mask)
        crop = destination/f'crop_{pid:03d}.png'
        Image.fromarray(rgb[y1:y2+1, x1:x2+1]).save(crop)
        rows.append({'scene_id': scene['scene_id'], 'proposal_id': pid, 'roi': roi,
                     'bbox_xyxy': [x1, y1, x2, y2], 'mask_index': pid,
                     'crop': relative(crop), 'crop_sha256': digest(crop), 'localizer': asdict(proposal)})
    np.savez_compressed(destination/'masks.npz', masks=np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool))
    result = {'scene_id': scene['scene_id'], 'plane_camera_m': list(plane), 'proposals': rows,
              'proposal_count': len(rows), 'input_sha256': source_hashes, 'seconds': perf_counter()-start,
              'geometry_cache_source': None, 'labels_read': False, 'denoising': 'No new cleaning; frozen prior proposal and observed-mask math.'}
    save(complete, result)
    geometry_cache[signature] = destination
    return result


def main():
    manifest = path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'inputs.json'
    assert manifest.exists(), f'New trial manifest is not ready: {manifest}'
    scenes = read(manifest)['scenes']
    output = HERE/'geometry/localization'
    cache, all_rows = {}, []
    protocol_file = HERE/'geometry/localization_protocol.json'
    protocol = {'parameters': CONFIG,
         'source_sha256': digest(REPO/'point_cloud_localization.py'), 'wrapper_sha256': digest(__file__),
         'protocol_sha256': digest(HERE/'protocol.json'), 'world_crop': False, 'denoising_changes': False,
         'gripper_exclusion': 'New capture omits robot hardware; no scene labels used.'}
    if protocol_file.exists():
        previous = read(protocol_file)
        assert previous['parameters'] == CONFIG and previous['source_sha256'] == protocol['source_sha256']
    else:
        save(protocol_file, protocol)
    for scene in scenes:
        record = localize_scene(scene, output/scene['scene_id'], cache)
        all_rows.extend(record['proposals'])
        print(f"Localize {scene['scene_id']}: {len(record['proposals'])} proposals, {record['seconds']:.2f}s", flush=True)
    save(HERE/'geometry/proposals.json', all_rows)
    save(HERE/'geometry/localization_completion.json', {'manifest_sha256': digest(manifest),
         'scenes': len(scenes), 'proposals': len(all_rows), 'labels_read': False})


if __name__ == '__main__':
    main()
