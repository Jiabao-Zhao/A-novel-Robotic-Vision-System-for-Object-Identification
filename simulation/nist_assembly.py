"""Known-fixture assembly pilot. No live object truth is used by the controller."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .nist_task_board_1 import (ASSET_DIR, BOARD_FILE, BOARD_BOTTOM_Z_M,
                               BOARD_CENTER_XY_M, BOARD_SIZE_M, _values)
from .libero_control import move_eef_to_pose, hold_gripper, OPEN_GRIPPER, CLOSE_GRIPPER


def board_openings():
    """Extract actual bottom-face boundary loops from the supplied plate CAD."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(ASSET_DIR / 'meshes' / f'{BOARD_FILE}_m.stl'))
    mesh.remove_duplicated_vertices()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    triangles = triangles[np.all(np.isclose(vertices[triangles, 2], vertices[:, 2].min(), atol=1e-8), axis=1)]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacent = {}
    for first, second in edges[counts == 1]:
        adjacent.setdefault(int(first), set()).add(int(second))
        adjacent.setdefault(int(second), set()).add(int(first))
    seen, openings = set(), []
    for first in adjacent:
        if first in seen:
            continue
        pending, indices = [first], []
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            indices.append(index)
            pending.extend(adjacent[index] - seen)
        xy = vertices[indices, :2]
        hull = ConvexHull(xy)
        center = (xy.min(0) + xy.max(0)) / 2
        openings.append({'center_design_mm': (center * 1000 + 192).tolist(),
                         'extent_mm': (np.ptp(xy, axis=0) * 1000).tolist(),
                         'polygon_world_m': (xy[hull.vertices] + BOARD_CENTER_XY_M).tolist()})
    return openings


def assembly_jobs(catalog):
    import open3d as o3d
    openings = board_openings()
    top = float(BOARD_BOTTOM_Z_M + BOARD_SIZE_M[2])
    definitions = (
        ('RGOCG16-50_16mm', 'round_hole_16mm', [266.45, 122.176], 'Insert the round pin into the matching round hole.', True),
        ('KET16_Square_16mm', 'rectangular_hole_16x10mm', [41.45, 347.176], 'Insert the square pin into the matching rectangular opening.', True),
        ('Gear_Large', 'large_gear_shaft', [141.45, 47.176], 'Fit the large gear onto its matching shaft.', False),
        ('DSUB_Male', 'dsub_socket', [41.45, 272.176], 'Insert the D-sub connector into its matching female connector.', False),
        ('M16_Hex_Nut', 'm16_bolt', [341.45, 47.176], 'Thread the M16 hex nut onto the matching M16 bolt.', False),
    )
    reasons = {
        'Gear_Large': 'CAD bore and shaft are both nominally 10 mm; positive slip-fit clearance is not validated. Bore-preserving collision replaces the solid hull, but does not invent clearance.',
        'DSUB_Male': 'Male and female connector collision hulls fill mating cavities; pin/socket contact geometry and seating pose are not validated.',
        'M16_Hex_Nut': 'Nut and bolt use convex collision hulls without thread engagement; no validated screw motion or torque/contact controller exists.',
    }
    jobs = []
    for name, destination, design_xy, instruction, ready in definitions:
        job = {'target': name, 'destination_id': destination, 'instruction': instruction,
               'ready': ready, 'readiness_reason': reasons.get(name),
               'fixture_pose_source': 'fixed workcell design from supplied NIST CAD; not estimated board perception',
               'entrance_z_m': top, 'cad_extent_m': catalog[name]['extent_m']}
        if ready:
            opening = min(openings, key=lambda entry: np.linalg.norm(np.array(entry['center_design_mm']) - design_xy))
            assert np.linalg.norm(np.array(opening['center_design_mm']) - design_xy) < .01
            job.update(opening)
            job['center_world_m'] = [*((np.array(opening['center_design_mm']) - 192) * .001 + BOARD_CENTER_XY_M), top]
            job['cross_section'] = 'round' if name.startswith('RGOCG') else 'rectangle'
            job['commanded_insertion_depth_m'] = .007
            vertices = np.asarray(o3d.io.read_triangle_mesh(catalog[name]['cad_path']).vertices)
            job['cross_section_polygon_cad_m'] = vertices[ConvexHull(vertices[:, :2]).vertices, :2].tolist()
            job['board_bottom_z_m'] = float(BOARD_BOTTOM_Z_M)
            job['scoring_revision'] = 2
            job['contact_numerical_tolerance_m'] = .000001
        else:
            job['center_world_m'] = [*((np.array(design_xy) - 192) * .001 + BOARD_CENTER_XY_M), top]
        jobs.append(job)
    return jobs


def configure_assembly_contacts(environment):
    """Rigid-part development contact model, not a measured hardware calibration.

    Keep timestep, friction, mass, and actuator forces unchanged. The time constant
    is twice the existing 2ms step, as required by MuJoCo's contact solver.
    """
    model = environment.sim.model
    if model.opt.timestep > .002:
        raise ValueError('Assembly contact preset requires a timestep no larger than 2ms.')
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_solref[active] = [.004, 1.]
    model.geom_solimp[active] = [.99, .999, .0001, .5, 2.]
    return {'solref': [.004, 1.], 'solimp': [.99, .999, .0001, .5, 2.],
            'source': 'simulation contact-compliance correction; not physical calibration',
            'documentation': 'https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters'}


def preserve_gear_bore(model, catalog):
    """Conservative annular rim/hub collision; retains original mass and visuals."""
    import open3d as o3d
    name = 'Gear_Large'
    vertices = np.asarray(o3d.io.read_triangle_mesh(catalog[name]['cad_path']).vertices)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    inner = float(radius.min())
    body = model.worldbody.find(f"./body[@name='nist_part_{name}']")
    for geom in body.findall('geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    for tier, (outer, low, high) in enumerate(((radius.max(), vertices[:, 2].min(), 0.),
                                            (radius[vertices[:, 2] > 1e-6].max(), 0., vertices[:, 2].max()))):
        for index in range(64):
            angles = 2 * np.pi * np.array([index, index + 1]) / 64
            points = [[r * np.cos(a), r * np.sin(a), z] for z in (low, high)
                      for r in (inner / np.cos(np.pi / 64), outer) for a in angles]
            key = f'assembly_gear_{tier}_{index}'
            ET.SubElement(model.asset, 'mesh', name=key, vertex=_values(np.ravel(points)))
            ET.SubElement(body, 'geom', name=key, type='mesh', mesh=key, mass='0',
                          group='0', friction='0.8 0.005 0.0001')


def bind_assembly_context(context, job):
    context['completion_condition'] = 'assembled_and_released'
    context['objects'].append({'object_id': job['destination_id'],
        'description': job['destination_id'].replace('_', ' '), 'world_T_grasp': None,
        'centroid_world_m': job['center_world_m'], 'placement_relations': [],
        'insertion_supported': job['ready'], 'assembly_binding': job})


def insertion_goal(item, destination):
    job = destination['assembly_binding']
    if not destination.get('insertion_supported') or not job['ready']:
        raise ValueError('Assembly destination has not passed simulation readiness.')
    initial = np.asarray(item['world_T_cad'], dtype=float)
    grasp = np.asarray(item['world_T_grasp'], dtype=float)
    if initial.shape != (4, 4) or grasp.shape != (4, 4) or not np.isfinite(initial).all() or not np.isfinite(grasp).all():
        raise ValueError('Insertion needs finite CAD and grasp transforms.')
    yaw = np.arctan2(initial[1, 0], initial[0, 0])
    if job['cross_section'] == 'rectangle':
        yaw = round(yaw / np.pi) * np.pi
    final = np.eye(4)
    final[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    final[:3, 3] = job['center_world_m']
    final[2, 3] += item['cad_extent_m'][2] / 2 - job['commanded_insertion_depth_m']
    return final @ np.linalg.inv(initial) @ grasp


def insert_object(environment, observation, item, destination, callback=None):
    """CAD-bound straight insertion; no object feedback or hidden pose correction."""
    goal = insertion_goal(item, destination)
    above = goal.copy()
    above[2, 3] = max(.27, goal[2, 3] + .12)
    observation = move_eef_to_pose(environment, observation, above, CLOSE_GRIPPER,
                                   'align_above_fixture', callback, tolerance_m=.001)
    for index, offset in enumerate((.035, .015, .008, .005, .002, 0.)):
        pose = goal.copy()
        pose[2, 3] += offset
        observation = move_eef_to_pose(environment, observation, pose, CLOSE_GRIPPER,
            f'insertion_{index}', callback, tolerance_m=.00015,
            orientation_tolerance_rad=.01, max_steps=100, compensate_position_bias=True)
    observation = hold_gripper(environment, observation, OPEN_GRIPPER, 'release_inserted', 20, callback)
    observation = move_eef_to_pose(environment, observation, above, OPEN_GRIPPER, 'retreat_from_fixture', callback)
    return hold_gripper(environment, observation, OPEN_GRIPPER, 'observe_assembly', 20, callback)


def peg_seating_metrics(position, rotation, job):
    """Conservative CAD prism cross-sections inside the actual opening polygon."""
    extent = np.asarray(job['cad_extent_m'])
    if 'cross_section_polygon_cad_m' in job:
        xy = np.asarray(job['cross_section_polygon_cad_m'])
    elif job['cross_section'] == 'round':
        angles = np.arange(96) * 2 * np.pi / 96
        radius = max(extent[:2]) / 2 / np.cos(np.pi / 96)
        xy = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    else:
        xy = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * extent[:2] / 2
    lower = np.column_stack((xy, np.full(len(xy), -extent[2] / 2))) @ rotation.T + position
    upper = lower + rotation[:, 2] * extent[2]
    entrance = job['entrance_z_m']
    tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
    depth = float(entrance - np.max(lower[:, 2]))
    if tilt >= 30.:
        # Extending a nearly horizontal prism to the entrance produces meaningless
        # metre-scale clearance values. This pose already cannot satisfy seating.
        return {'inserted': False, 'minimum_insertion_depth_mm': depth * 1000,
                'minimum_clearance_mm': None, 'tilt_deg': tilt}
    fraction = np.clip((entrance - lower[:, 2]) / (upper[:, 2] - lower[:, 2]), 0., 1.)
    at_entrance = lower + fraction[:, None] * (upper - lower)
    engaged_lower = lower
    if job.get('scoring_revision', 1) >= 2:
        fraction = np.clip((job['board_bottom_z_m'] - lower[:, 2]) / (upper[:, 2] - lower[:, 2]), 0., 1.)
        engaged_lower = lower + fraction[:, None] * (upper - lower)
    points = np.vstack((engaged_lower[:, :2], at_entrance[:, :2]))
    halfplanes = ConvexHull(np.array(job['polygon_world_m'])).equations
    clearance = float(-np.max(points @ halfplanes[:, :2].T + halfplanes[:, 2]))
    allowance = job.get('contact_numerical_tolerance_m', 0.)
    seated = bool(depth >= .005 and np.min(lower[:, 2]) >= -.001 and clearance >= -allowance and tilt < 1.)
    return {'inserted': seated, 'minimum_insertion_depth_mm': depth * 1000,
            'minimum_clearance_mm': clearance * 1000, 'tilt_deg': tilt,
            'contact_numerical_tolerance_mm': allowance * 1000,
            'scoring_revision': job.get('scoring_revision', 1)}


from scripts.run_wrist_cluster import PickupEvaluator


class AssemblyEvaluator(PickupEvaluator):
    """Truth is read only here for evaluation, never passed to planning/control."""

    def __init__(self, environment, name, job):
        super().__init__(environment, name)
        self.job = job
        self.assembly_success = False
        self.released_steps = 0
        self.last_assembly_metrics = None

    def score(self):
        pickup = super().score()
        sim = self.environment.sim
        position = sim.data.body_xpos[self.ids[self.target]].copy()
        rotation = sim.data.body_xmat[self.ids[self.target]].reshape(3, 3).copy()
        if not self.job['ready']:
            return pickup
        metrics = peg_seating_metrics(position, rotation, self.job)
        away = np.linalg.norm(sim.data.site_xpos[self.environment.robots[0].eef_site_id] - position) > .06
        released = metrics['inserted'] and away and not pickup['target_grasped'] and not self.wrong_lifted
        self.released_steps = self.released_steps + 1 if released else 0
        self.assembly_success = bool(self.released_steps >= 20 and self.max_hold_steps >= 3)
        self.last_assembly_metrics = {**metrics, 'released_stable_steps': self.released_steps,
            'success': self.assembly_success, 'object_center_world_m_evaluation_only': position.tolist()}
        return {**pickup, 'assembly': self.last_assembly_metrics}
