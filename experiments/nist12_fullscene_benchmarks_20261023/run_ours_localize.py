"""Fresh camera-frame depth proposals; never reads old ROIs or audit identities."""
from dataclasses import asdict
import sys
from time import perf_counter
from common import HERE, REPO, setup, path, read, save, digest, relative

inputs = setup()
sys.path.insert(0, str(REPO))
import numpy as np
import open3d as o3d
from PIL import Image
from point_cloud_localization import PointCloudConfig, PointCloudLocalization

CONFIG = dict(voxel_size_m=.002, plane_distance_threshold_m=.002,
              dbscan_eps_m=.02, dbscan_min_points=5, cluster_in_table_plane=False,
              min_cluster_points=50, min_cluster_extent_m=.006, max_cluster_aspect_ratio=16.,
              min_roi_width_px=8, min_roi_height_px=8, edge_artifact_margin_px=3,
              bottom_edge_artifact_max_thickness_m=.012, bottom_edge_artifact_max_roi_height_ratio=.16,
              raw_cluster_padding_m=.004, raw_cluster_dbscan_eps_m=.006,
              raw_cluster_dbscan_min_points=5, statistical_outlier_neighbors=10,
              statistical_outlier_std_ratio=2., radius_outlier_radius_m=.008, radius_outlier_min_neighbors=2)


def main():
    save(HERE / 'ours/localizer_protocol.json', {'parameters': CONFIG, 'world_crop': False,
         'input_manifest_sha256': digest(HERE / 'inputs.json'), 'source_sha256': digest(REPO / 'point_cloud_localization.py'),
         'seed': 0, 'labels_read': False, 'proposal_count_forced': False})
    all_rows = []
    for scene in inputs['scenes']:
        start = perf_counter()
        folder = HERE / 'ours' / scene['scene_id']
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'localization.json').exists():
            all_rows.extend(read(folder / 'localization.json')['proposals'])
            continue
        rgb = np.array(Image.open(path(scene['rgb'])).convert('RGB'))
        depth = np.load(path(scene['depth']))
        K = np.load(path(scene['intrinsics']))
        height, width = depth.shape
        camera = {'width': width, 'height': height, 'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
        localizer = PointCloudLocalization(PointCloudConfig(**CONFIG))
        cloud = localizer.rgbd_to_pointcloud(rgb, depth, 1., camera).voxel_down_sample(CONFIG['voxel_size_m'])
        o3d.utility.random.seed(0)
        objects, plane, _ = localizer.segment_table_plane(cloud)
        clusters = localizer.cluster_objects(objects, camera, plane)
        proposals = localizer.localize_clusters(clusters, camera)
        yy, xx = np.indices(depth.shape)
        distance = np.abs(((xx-K[0, 2])*depth/K[0, 0])*plane[0]
                          + ((yy-K[1, 2])*depth/K[1, 1])*plane[1] + depth*plane[2] + plane[3]) / np.linalg.norm(plane[:3])
        foreground = np.isfinite(depth) & (depth > 0) & (distance > .002)
        rows, masks = [], []
        for index, proposal in enumerate(proposals):
            roi = proposal.roi
            x1, y1, x2, y2 = [roi[k] for k in ('x1', 'y1', 'x2', 'y2')]
            crop = folder / f'crop_{index:03d}.png'
            Image.fromarray(rgb[y1:y2+1, x1:x2+1]).save(crop)
            mask = np.zeros(depth.shape, dtype=bool)
            mask[y1:y2+1, x1:x2+1] = foreground[y1:y2+1, x1:x2+1]
            masks.append(mask)
            rows.append({'scene_id': scene['scene_id'], 'proposal_id': index,
                         'bbox_xyxy': [x1, y1, x2, y2], 'roi': roi, 'mask_index': index,
                         'crop': relative(crop), 'crop_sha256': digest(crop),
                         'localizer': asdict(proposal)})
        np.savez_compressed(folder / 'masks.npz', masks=np.stack(masks) if masks else np.zeros((0,height,width), bool))
        save(folder / 'localization.json', {'plane_camera_m': list(plane), 'proposals': rows,
            'seconds': perf_counter()-start, 'labels_read': False, 'proposal_count': len(rows)})
        all_rows.extend(rows)
        print(f"{scene['scene_id']}: {len(rows)} own depth proposals, {perf_counter()-start:.1f}s", flush=True)
    save(HERE / 'ours/proposals.json', all_rows)


if __name__ == '__main__':
    main()
