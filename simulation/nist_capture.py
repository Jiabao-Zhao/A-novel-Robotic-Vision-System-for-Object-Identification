"""NIST-only supported poses and calibrated depth-crop capture helpers.

Extracted from the prior capture chain; this module runs no scene on import.
Simulator identities are used only after depth localization for capture audits.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .perception_adapter import mask_depth_to_world_workspace, pointcloud_localization_inputs

SIZE = 2560
EXISTING_PARTS = ('Gear_Large', 'M16_Hex_Nut', 'RGOCG16-50_16mm',
                  'KET16_Square_16mm', 'Waterproof_Male', 'USB_Male', 'DSUB_Male', 'BNC_Male')
BOUNDS = ((-.70, -.20, -.01), (.065, .47, .20))
# Same depth-localizer controls recorded in the retained NIST captures.
LOCALIZATION_PARAMETERS = {
    'voxel_size_m': 0.002, 'plane_distance_threshold_m': 0.002,
    'dbscan_eps_m': 0.02, 'dbscan_min_points': 5, 'cluster_in_table_plane': False,
    'min_cluster_points': 50, 'min_cluster_extent_m': 0.006,
    'max_cluster_aspect_ratio': 16.0, 'min_roi_width_px': 8, 'min_roi_height_px': 8,
    'edge_artifact_margin_px': 3, 'bottom_edge_artifact_max_thickness_m': 0.012,
    'bottom_edge_artifact_max_roi_height_ratio': 0.16, 'raw_cluster_padding_m': 0.004,
    'raw_cluster_dbscan_eps_m': 0.006, 'raw_cluster_dbscan_min_points': 5,
    'statistical_outlier_neighbors': 10, 'statistical_outlier_std_ratio': 2.0,
    'radius_outlier_radius_m': 0.008, 'radius_outlier_min_neighbors': 2,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertices(model, data, gid):
    mid = model.geom_dataid[gid]
    v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
    return v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]


def audit_roi(roi, seg, truth):
    x1, y1, x2, y2 = [roi[k] for k in ('x1', 'y1', 'x2', 'y2')]
    matches = []
    for identity, item in truth.items():
        count = int(np.count_nonzero(seg[y1:y2+1, x1:x2+1, 0] == item['geom_id']))
        coverage = count / item['visible_pixels']
        if count >= 20 and coverage > .15:
            matches.append(dict(identity=identity, visible_coverage=coverage, pixels=count))
    matches.sort(key=lambda m: m['pixels'], reverse=True)
    identity = matches[0]['identity'] if len(matches) == 1 else None
    return identity, matches


def visual_ids(model):
    import mujoco
    result = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('nist_part_') and name.endswith('_visual'):
            result[name.removeprefix('nist_part_').removesuffix('_visual')] = gid
    return result


def compile_scene(root):
    import mujoco
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def hull_center(points, hull):
    reference = points.mean(axis=0)
    triangles = points[hull.simplices]
    volume = np.abs(np.einsum('ij,ij->i', triangles[:, 0]-reference,
        np.cross(triangles[:, 1]-reference, triangles[:, 2]-reference))) / 6
    return np.average((triangles.sum(axis=1)+reference)/4, axis=0, weights=volume)


def supported_pose(points, rotation, center):
    rotated = points @ rotation.T
    support = rotated[rotated[:, 2] <= rotated[:, 2].min() + .0002, :2]
    if len(support) < 3:
        return None
    try:
        hull = ConvexHull(support)
    except Exception:
        return None
    projected = (center @ rotation.T)[:2]
    clearance = -np.max(hull.equations[:, :2] @ projected + hull.equations[:, 2])
    if clearance < .0004 or hull.volume < .000015:
        return None
    return dict(support_area_m2=float(hull.volume), com_proxy_margin_m=float(clearance))


def rest_options(points, identity):
    hull = ConvexHull(points)
    center = hull_center(points, hull)
    base = supported_pose(points, np.eye(3), center)
    options = ([dict(rotation=np.eye(3), normal=[0., 0., -1.],
                    audit=base, name='source_rest')] if base else [])
    if identity == 'USB_Male':
        return options
    possibilities = []
    for normal in hull.equations[:, :3]:
        # Keep the gear on either broad face, not balanced on its toothed edge.
        if identity == 'Gear_Large' and abs(normal[2]) < .98:
            continue
        if any(np.dot(normal, n) > .9999 for n, _, _ in possibilities):
            continue
        rotation = Rotation.align_vectors([[0., 0., -1.]], [normal])[0].as_matrix()
        audit = supported_pose(points, rotation, center)
        if audit:
            possibilities.append((normal, rotation, audit))
    possibilities.sort(key=lambda p: -p[2]['support_area_m2'])
    for normal, rotation, audit in possibilities:
        if any(np.dot(normal, p['normal']) > np.cos(np.deg2rad(25)) for p in options):
            continue
        options.append(dict(rotation=rotation, normal=normal.tolist(), audit=audit,
                            name=f'supported_face_{len(options)}'))
        if len(options) == 6:
            break
    if not options:
        raise ValueError(f'No supported resting orientation for {identity}')
    return options


def separated(box, placed):
    # The unchanged depth DBSCAN uses 20 mm, so keep at least 23 mm in XY.
    return all(np.linalg.norm(np.maximum(0., np.maximum(other[0]-box[1], box[0]-other[1])))
               >= .023 for other in placed)


def layout(template, items, index, seed, attempt):
    rng = np.random.default_rng(seed + attempt * 100003)
    root = copy.deepcopy(template)
    bodies = {b.get('name'): b for b in root.find('worldbody').findall('body')}
    poses, boxes = {}, []
    order = sorted(items, key=lambda k: -np.prod(np.sort(items[k]['extent_m'])[-2:]))
    for identity in order:
        item = items[identity]
        choice = item['options'][(index + list(items).index(identity)) % len(item['options'])]
        yaw = float(rng.uniform(-22, 22) if identity == 'USB_Male' else rng.uniform(-180, 180))
        delta = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ choice['rotation']
        points = item['points'] @ delta.T
        lo, hi = points.min(axis=0), points.max(axis=0)
        if identity == 'USB_Male':
            low, high = np.array([-.65, .265]), np.array([-.37, .455])
        else:
            low, high = np.array([-.68, -.17]), np.array([-.335, .245])
        lower, upper = low + (hi[:2]-lo[:2])/2, high - (hi[:2]-lo[:2])/2
        if np.any(lower >= upper):
            raise ValueError(f'{identity} does not fit its fixed pick area')
        for _ in range(4000):
            center = rng.uniform(lower, upper)
            position = np.r_[center-(lo[:2]+hi[:2])/2, .001-lo[2]]
            box = np.array([lo[:2]+position[:2], hi[:2]+position[:2]])
            if separated(box, boxes):
                break
        else:
            raise ValueError(f'Cannot separate {identity} in layout {index}')
        boxes.append(box)
        rotation = delta @ item['base_rotation']
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        body = bodies[item['body']]
        body.set('pos', ' '.join(map(str, position)))
        body.set('quat', ' '.join(map(str, quat)))
        poses[identity] = dict(position_world_m=position.tolist(), quaternion_wxyz=quat.tolist(),
            yaw_change_deg=yaw, rest_option=choice['name'], rest_option_count=len(item['options']),
            supporting_face_normal_before_rotation=choice['normal'],
            support_audit=choice['audit'], minimum_world_z_m=float((points+position)[:, 2].min()),
            extent_world_m=np.ptp(points, axis=0).tolist())
    return root, poses


def localize(state, observation, seg, truth, *, output_dir, raw_dir):
    import open3d as o3d
    from PIL import Image, ImageDraw
    from point_cloud_localization import PointCloudConfig, PointCloudLocalization

    config_values = dict(LOCALIZATION_PARAMETERS)
    output_dir, raw_dir = Path(output_dir), Path(raw_dir)
    folder = output_dir / 'scenes' / state
    config = PointCloudConfig(output_dir=raw_dir/state/'localization', annotation_dir=folder, **config_values)
    loc = PointCloudLocalization(config)
    depth, _ = mask_depth_to_world_workspace(observation, *BOUNDS)
    inputs = pointcloud_localization_inputs(observation, folder/'wrist_rgb.png', raw_dir/state/'depth.npy', depth)
    K = inputs['camera_intrinsics']
    cloud = loc.rgbd_to_pointcloud(observation.rgb, depth, 1., K).voxel_down_sample(config.voxel_size_m)
    o3d.utility.random.seed(0)
    objects, plane, _ = loc.segment_table_plane(cloud)
    clusters = loc.cluster_objects(objects, K, plane)
    items = loc.localize_clusters(clusters, K)
    rows = []
    annotated = Image.fromarray(observation.rgb)
    draw = ImageDraw.Draw(annotated)
    for i, item in enumerate(items):
        roi = item.roi
        x1, y1, x2, y2 = [roi[k] for k in ('x1', 'y1', 'x2', 'y2')]
        crop = Image.fromarray(observation.rgb).crop((x1, y1, x2+1, y2+1))
        native = list(crop.size)
        crop.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        path = folder / 'crops' / f'candidate_{i:03d}.png'
        path.parent.mkdir(exist_ok=True)
        crop.save(path)
        identity, matches = audit_roi(roi, seg, truth)
        rows.append(dict(proposal_id=i, **asdict(item), crop_file=path.relative_to(output_dir).as_posix(),
            crop_sha256=digest(path), native_crop_wh=native, sent_crop_wh=list(crop.size),
            posthoc_identity=identity, audit_matches=matches, assisted_fallback=False,
            localization_method='accepted_depth_rectangle', background_removed=False))
        draw.rectangle((x1, y1, x2, y2), outline='red', width=4)
        draw.text((x1, max(0, y1-24)), str(i), fill='red', stroke_width=1)
    annotated.save(folder / 'localization_boxes.png')
    covered = {r['posthoc_identity'] for r in rows if r['posthoc_identity']}
    missing = sorted(set(truth)-covered)
    save(folder / 'candidates.json', rows)
    save(folder / 'localization_audit.json', dict(accepted_proposals=len(rows), missing_objects=missing,
        merged_or_unresolved=[r['proposal_id'] for r in rows if r['posthoc_identity'] is None],
        plane_model_camera=list(plane), config=config_values, workspace_bounds_world_m=BOUNDS,
        note='Simulator labels are applied after localization; not used for depth clustering or crop selection.'))
    print(f'{state}: {len(rows)} depth proposals; missing={missing}', flush=True)
    return rows

