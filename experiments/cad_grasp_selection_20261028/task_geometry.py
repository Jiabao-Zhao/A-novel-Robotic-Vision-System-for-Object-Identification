"""CAD-feature task inputs for the twelve saved NIST products.

These are explicit part-family rules, not a generic semantic CAD recognizer.
The original nominal goals are preserved. Protected connector half-spaces are
conservative feature envelopes from a measured CAD shoulder toward the open
mating end; they are not measurements of actual engagement depth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from experiments.cad_grasp_geometry_20261026.geometry_kernel import MeshCollider
from experiments.fabrica_candidates_20261027.sampler import _all_ray_hits
from experiments.fabrica_goal_geometry_20261027.task_inputs import geometry_spec
from .geometry import _faces_at_point


PLANE_TOL_M = 1e-7
REGION_TOL_M = 1e-8
POINT_TOL_M = 1e-6
NORMAL_COS = 1. - 1e-6


def _mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Finite metric CAD vertices and integer triangle faces required')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    if np.any(twice <= 0):
        raise ValueError('Feature extraction requires nondegenerate CAD triangles')
    return vertices, faces, triangles, cross/twice[:, None], twice/2


def _planar_patches(vertices, triangles, normals, areas, internal=True):
    """Collect coordinate-aligned CAD planes, using area rather than face count."""
    patches = []
    bounds = np.asarray([vertices.min(0), vertices.max(0)])
    for axis in range(3):
        ids = np.flatnonzero(np.abs(normals[:, axis]) >= NORMAL_COS)
        if not len(ids):
            continue
        levels = triangles[ids].mean(axis=1)[:, axis]
        bins = np.rint(levels/PLANE_TOL_M).astype(np.int64)
        for key in np.unique(bins):
            take = ids[bins == key]
            level = float(np.average(triangles[take].mean(1)[:, axis], weights=areas[take]))
            is_internal = (level > bounds[0, axis]+PLANE_TOL_M
                           and level < bounds[1, axis]-PLANE_TOL_M)
            if internal != is_internal:
                continue
            patches.append(dict(axis=axis, level_m=level, area_m2=float(areas[take].sum()),
                                face_ids=take))
    return sorted(patches, key=lambda item: (-item['area_m2'], item['axis'], item['level_m']))


def _terminal_coverage(vertices, triangles, areas, axis):
    """Open/cavity/pin ends have less planar-cap coverage than a closed back."""
    transverse = [i for i in range(3) if i != axis]
    records = []
    for side in (-1, 1):
        level = float(vertices[:, axis].min() if side < 0 else vertices[:, axis].max())
        near = np.abs(vertices[:, axis]-level) <= PLANE_TOL_M
        points = vertices[near][:, transverse]
        hull_area = float(ConvexHull(points).volume) if len(points) >= 3 else 0.
        take = np.all(np.abs(triangles[:, :, axis]-level) <= PLANE_TOL_M, axis=1)
        cap_area = float(areas[take].sum())
        coverage = cap_area/hull_area if hull_area > 0 else 0.
        records.append(dict(side=side, level_m=level, planar_cap_area_m2=cap_area,
                            projected_end_hull_area_m2=hull_area, coverage=coverage))
    return records


def halfspace_checker(normal, offset, *, label):
    """Reserve n.p + offset >= -tol; certify whole polygons by linear extrema."""
    normal, offset = np.asarray(normal, float), float(offset)
    if normal.shape != (3,) or abs(np.linalg.norm(normal)-1.) > 1e-8:
        raise ValueError('Protected half-space requires a unit normal')
    def check(points, *, contact_polygons, contact_face_ids):
        points = np.asarray(points, float).reshape(-1, 3)
        eligible = points @ normal + offset < -REGION_TOL_M
        polygon_eligible = np.asarray([
            np.all(np.asarray(polygon) @ normal + offset < -REGION_TOL_M)
            for polygon in contact_polygons], dtype=bool)
        return dict(status='available_CAD_derived_'+label, eligible=eligible,
                    polygon_eligible=polygon_eligible)
    return check


def face_checker(vertices, faces, protected_face_ids):
    """Use native face labels for full polygons and BVH point-on-face queries."""
    collider = MeshCollider(vertices, faces)
    protected = np.zeros(len(faces), dtype=bool)
    protected[np.asarray(protected_face_ids, dtype=int)] = True
    def check(points, *, contact_polygons, contact_face_ids):
        points = np.asarray(points, float).reshape(-1, 3)
        eligible = np.ones(len(points), dtype=bool)
        for i, point in enumerate(points):
            ids = _faces_at_point(collider, point, POINT_TOL_M)
            eligible[i] = not np.any(protected[ids])
        ids = np.asarray(contact_face_ids, dtype=int)
        return dict(status='available_CAD_derived_bore_and_seat', eligible=eligible,
                    polygon_eligible=~protected[ids])
    return check


def _first_hits(origins, directions, triangles):
    locations, ray_ids, face_ids, _ = _all_ray_hits(origins, directions, triangles)
    radii = np.linalg.norm(locations-origins[ray_ids], axis=1)
    minimum = np.full(len(origins), np.inf)
    np.minimum.at(minimum, ray_ids, radii)
    return locations, ray_ids, face_ids, radii, minimum


def _bore_lip_centers(vertices, faces, triangles, axis, initial_center):
    """Fit circular INNER boundary loops of the two terminal seating planes.

    A threaded bore's middle cross-section is not a circle centered on its
    screw axis. The two circular mouth loops provide that axis without fitting
    the helix's changing cross-section.
    """
    transverse = [i for i in range(3) if i != axis]
    records = []
    for level in (vertices[:, axis].min(), vertices[:, axis].max()):
        on_plane = np.all(np.abs(triangles[:, :, axis]-level) <= PLANE_TOL_M, axis=1)
        selected = faces[on_plane]
        if not len(selected):
            continue
        edges = np.sort(np.concatenate((selected[:, [0, 1]], selected[:, [1, 2]],
                                        selected[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        if not len(boundary):
            continue
        ids, inverse = np.unique(boundary, return_inverse=True)
        pairs = inverse.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(ids), len(ids)))
        number, labels = connected_components(graph, directed=False)
        loops = []
        for component in range(number):
            points = vertices[ids[labels == component]][:, transverse]
            if len(points) < 6:
                continue
            hull = ConvexHull(points)
            if np.any(hull.equations[:, :2] @ initial_center[transverse] + hull.equations[:, 2] > PLANE_TOL_M):
                continue
            solution = np.linalg.lstsq(np.column_stack((2*points, np.ones(len(points)))),
                                        np.sum(points*points, axis=1), rcond=None)[0]
            radius = float(np.sqrt(solution[2]+np.dot(solution[:2], solution[:2])))
            rms = float(np.sqrt(np.mean((np.linalg.norm(points-solution[:2], axis=1)-radius)**2)))
            if radius > 0 and rms/radius <= .05:
                loops.append(dict(center=solution[:2], radius=radius, rms=rms,
                                  area=float(hull.volume), points=len(points)))
        if loops:
            inner = min(loops, key=lambda item:item['area'])
            records.append(dict(level_m=float(level), center_transverse_m=inner['center'].tolist(),
                                radius_m=inner['radius'], rms_m=inner['rms'],
                                loop_vertex_count=inner['points']))
    return records


def _bore_geometry(vertices, faces, triangles, normals, areas, spec):
    terminal = _planar_patches(vertices, triangles, normals, areas, internal=False)
    if not terminal:
        raise ValueError('No opposed planar end geometry for bore axis')
    axis = terminal[0]['axis']
    transverse = [i for i in range(3) if i != axis]
    center = (vertices.min(0)+vertices.max(0))/2
    angles = np.linspace(0., 2*np.pi, 64, endpoint=False)
    directions = np.zeros((len(angles), 3))
    directions[:, transverse] = np.column_stack((np.cos(angles), np.sin(angles)))
    origins = np.tile(center, (len(angles), 1))
    locations, ray_ids, hit_faces, radii, minimum = _first_hits(origins, directions, triangles)
    if not np.isfinite(minimum).all():
        raise ValueError('Bore cross-section did not enclose all radial probes')
    first = np.zeros((len(angles), 3))
    first_faces = np.zeros(len(angles), dtype=int)
    for i in range(len(angles)):
        eligible = np.flatnonzero(ray_ids == i)
        j = eligible[np.argmin(radii[eligible])]
        first[i], first_faces[i] = locations[j], hit_faces[j]
    if np.any(np.einsum('ij,ij->i', normals[first_faces], directions) >= 0):
        raise ValueError('Radial probes did not start inside an inward-facing bore')
    xy = first[:, transverse]
    solution = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                                np.sum(xy*xy, axis=1), rcond=None)[0]
    center[transverse] = solution[:2]
    radius = float(np.sqrt(solution[2]+np.dot(solution[:2], solution[:2])))
    residual = float(np.sqrt(np.mean((np.linalg.norm(xy-solution[:2], axis=1)-radius)**2)))
    if radius <= 0 or residual/radius > .05:
        raise ValueError('Radial section is not sufficiently circular for a bore-axis fit')
    lips = _bore_lip_centers(vertices, faces, triangles, axis, center)
    if len(lips) == 2:
        lip_centers = np.asarray([item['center_transverse_m'] for item in lips])
        if np.linalg.norm(lip_centers[0]-lip_centers[1]) > .01*radius:
            raise ValueError('The two bore mouth circles are not coaxial')
        center[transverse] = lip_centers.mean(axis=0)
    else:
        raise ValueError('Two circular bore-mouth loops are required to identify its axis')
    direction = np.eye(3)[axis]
    if (np.asarray(spec['goal_T_cad_board'])[:3, :3] @ direction)[2] > 0:
        direction = -direction
    # Classify actual bore triangles by first radial visibility, not just a
    # negative radial normal that could also identify gear-tooth recesses.
    centroids = triangles.mean(1)
    radial = centroids-center
    radial[:, axis] = 0
    candidates = np.flatnonzero(np.einsum('ij,ij->i', radial, normals) < -1e-12)
    ray_origins = np.tile(center, (len(candidates), 1))
    ray_origins[:, axis] = centroids[candidates, axis]
    distance = np.linalg.norm(radial[candidates], axis=1)
    dirs = radial[candidates]/distance[:, None]
    _, _, _, _, first_distance = _first_hits(ray_origins, dirs, triangles)
    bore = candidates[np.abs(first_distance-distance) <= PLANE_TOL_M]
    seat_level = (vertices[:, axis].min() if direction[axis] < 0 else vertices[:, axis].max())
    seat = np.flatnonzero(np.all(np.abs(triangles[:, :, axis]-seat_level) <= PLANE_TOL_M, axis=1)
                          & (normals @ direction > 0))
    protected = np.unique(np.r_[bore, seat])
    if not len(bore) or not len(seat):
        raise ValueError('Could not derive both bore and seating surfaces')
    return center, direction, face_checker(vertices, faces, protected), dict(
        method='area-dominant terminal planes + two circular inner bore-mouth loops',
        axis_coordinate_index=axis, bore_radius_m=radius, bore_fit_rms_m=residual,
        radial_probe_count=len(angles), bore_face_count=len(bore), seat_face_count=len(seat),
        protected_face_ids=protected.tolist(), seat_plane_coordinate_m=float(seat_level),
        circular_fit_relative_rms_limit=.05, bore_mouth_loops=lips,
        midsection_fit_center_is_not_used_for_threaded_axis=True)


def _connector_geometry(part, vertices, faces, triangles, normals, areas):
    if part == 'USB_Male':
        # The native cable model has separate plug-shell, insert, body and cable
        # shells. Derive the plug/body interface from a coincident pair of their
        # transverse end planes; do not use the long cable's centroid or PCA.
        collider = MeshCollider(vertices, faces)
        components = []
        for i in range(collider.component_count):
            vv = vertices[np.unique(faces[collider.component_ids == i])]
            components.append(dict(id=i, bounds=np.asarray([vv.min(0), vv.max(0)]), vertices=vv))
        options = []
        for body in components:
            for plug in components:
                if body['id'] == plug['id']:
                    continue
                # USB's independent thin metal shell is wider than it is thick;
                # its mating axis is the shared plug/body planar interface.
                for axis in range(3):
                    transverse = [i for i in range(3) if i != axis]
                    for sign in (-1, 1):
                        body_level = body['bounds'][0 if sign < 0 else 1, axis]
                        plug_level = plug['bounds'][1 if sign < 0 else 0, axis]
                        if abs(body_level-plug_level) > PLANE_TOL_M:
                            continue
                        body_width = np.ptp(body['bounds'], axis=0)[transverse]
                        plug_width = np.ptp(plug['bounds'], axis=0)[transverse]
                        body_length = np.ptp(body['bounds'], axis=0)[axis]
                        plug_length = np.ptp(plug['bounds'], axis=0)[axis]
                        if (np.any(plug_width <= 0) or np.any(body_width <= plug_width)
                                or body_length <= plug_length):
                            continue
                        score = float(np.prod(plug_width))
                        options.append((score, axis, sign, float((body_level+plug_level)/2), body, plug))
        if not options:
            raise ValueError('No separated USB plug/body shoulder interface found')
        _, axis, sign, level, body, plug = max(options, key=lambda x:x[0])
        transverse = [i for i in range(3) if i != axis]
        center = plug['bounds'].mean(axis=0)
        center[axis] = level
        direction = sign*np.eye(3)[axis]
        evidence = dict(method='separate plug/body shells with coincident interface planes',
                        body_component_id=body['id'], plug_component_id=plug['id'],
                        plug_bounds_m=plug['bounds'].tolist(), interface_coordinate_m=level,
                        interface_axis_index=axis, mating_side_sign=sign)
    else:
        patches = _planar_patches(vertices, triangles, normals, areas, internal=True)
        if not patches:
            raise ValueError('No internal connector shoulder plane found')
        shoulder = patches[0]
        axis, level = shoulder['axis'], shoulder['level_m']
        coverage = _terminal_coverage(vertices, triangles, areas, axis)
        ordered = sorted(coverage, key=lambda item:item['coverage'])
        if ordered[1]['coverage']-ordered[0]['coverage'] < .05:
            raise ValueError('Connector front/back cap coverage is ambiguous')
        sign = ordered[0]['side']
        points = triangles[shoulder['face_ids']].reshape(-1, 3)
        center = (points.min(0)+points.max(0))/2
        center[axis] = level
        direction = sign*np.eye(3)[axis]
        evidence = dict(method='largest internal planar shoulder + open-end cap coverage',
                        shoulder_axis_index=axis, shoulder_coordinate_m=level,
                        shoulder_area_m2=shoulder['area_m2'], mating_side_sign=sign,
                        terminal_cap_coverage=coverage, cap_coverage_difference_min=.05,
                        shoulder_face_ids=shoulder['face_ids'].tolist())
    evidence['protected_region'] = dict(kind='mating_feature_halfspace',
        normal=direction.tolist(), offset_m=float(-direction @ center),
        definition='Reserve the CAD shoulder and everything toward the exposed mating end.',
        conservative_feature_envelope=True, actual_engagement_depth_inferred=False)
    return center, direction, halfspace_checker(direction, -direction @ center,
                                               label='connector_feature_envelope'), evidence


def derive_task(part, vertices, faces):
    """Derive CAD task features, preserving old nominal goals and native scale."""
    try:
        vertices, faces, triangles, normals, areas = _mesh(vertices, faces)
        spec = geometry_spec(part)
        if part.startswith(('RGOCG', 'KET')):
            lengths = np.ptp(vertices, axis=0)
            axis = int(lengths.argmax())
            if lengths[axis] <= 2*np.max(np.delete(lengths, axis)):
                raise ValueError('Pin does not have an unambiguous longitudinal axis')
            center = (vertices.min(0)+vertices.max(0))/2
            direction = np.eye(3)[axis]
            if (np.asarray(spec['goal_T_cad_board'])[:3, :3] @ direction)[2] > 0:
                direction = -direction
            goal = np.asarray(spec['goal_T_cad_board'])
            normal = -goal[2, :3]
            offset = spec['board_top_m']-goal[2, 3]
            checker = halfspace_checker(normal, offset, label='nominal_inserted_pin_segment')
            evidence = dict(method='longitudinal CAD extent + transverse bounding-center symmetry',
                extents_m=lengths.tolist(), axis_coordinate_index=axis,
                protected_region=dict(kind='nominal_goal_board_halfspace', normal=normal.tolist(),
                                      offset_m=float(offset)),
                engagement_source='Existing user-authorized nominal final pose; not inferred from pickup.',
                chamfer_note='Native positive-axis chamfer is preserved; old nominal goal inserts its negative-axis end.')
        elif part in ('Gear_Large', 'M16_Hex_Nut'):
            center, direction, checker, evidence = _bore_geometry(
                vertices, faces, triangles, normals, areas, spec)
        elif part in ('BNC_Male', 'DSUB_Male', 'USB_Male', 'Waterproof_Male'):
            center, direction, checker, evidence = _connector_geometry(
                part, vertices, faces, triangles, normals, areas)
            evidence['goal_warning'] = (
                'Existing goal only touches lower/upper bounding planes with zero assigned engagement. '
                'This CAD-derived feature axis does not validate that nominal mating pose; '
                'the final goal is preserved rather than reoriented or translated.')
            evidence['feature_direction_in_nominal_board_frame'] = (
                np.asarray(spec['goal_T_cad_board'])[:3, :3] @ direction).tolist()
        else:
            raise ValueError('Part is outside the twelve-part CAD feature adapter')
        evidence.update(plane_grouping_tolerance_m=PLANE_TOL_M,
                        protected_halfspace_tolerance_m=REGION_TOL_M,
                        native_frames_and_scale_preserved=True,
                        simulator_object_poses_used=False,
                        known_nominal_final_pose_preserved=True)
        return dict(status='available_CAD_derived_nominal_task', part=part,
                    axis_point_cad_m=np.asarray(center).tolist(),
                    axis_direction_cad=np.asarray(direction).tolist(),
                    protected_contact_check=checker, task_spec=spec, evidence=evidence)
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        return dict(status='unavailable_CAD_task_geometry', part=part, reason=str(error),
                    protected_contact_check=None, evidence=dict(simulator_object_poses_used=False))
