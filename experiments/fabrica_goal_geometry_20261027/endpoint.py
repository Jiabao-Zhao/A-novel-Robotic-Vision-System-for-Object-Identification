"""Static, metre-scale solid checks at a supplied final part pose.

No trajectory, finger opening sequence, robot state or simulator is used.
FCL tests boundary intersection; batched containment also detects nested solids.
"""
from __future__ import annotations

from experiments.fabrica_original_20261027 import native
import numpy as np


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def welded_mesh(path):
    """Merge only exactly equal STL vertices, without moving CAD surfaces."""
    up = native.load()
    original = up.trimesh.load_mesh(str(path), process=False, maintain_order=True)
    vertices, inverse = np.unique(original.vertices, axis=0, return_inverse=True)
    return up.trimesh.Trimesh(vertices=vertices, faces=inverse[original.faces], process=False)


def solid_representatives(mesh):
    components = mesh.split(only_watertight=False)
    if len(components) == 0 or any(not c.is_watertight for c in components):
        raise ValueError('Solid containment requires closed connected components')
    if not mesh.is_winding_consistent:
        raise ValueError('Inconsistent mesh winding; do not invent a solid')
    return np.asarray([c.vertices[0] for c in components])


def contains_batched(mesh, points, chunk=2048, query_mesh=None):
    points = np.asarray(points).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(np.all((points > mesh.bounds[0]) & (points < mesh.bounds[1]), axis=1))
    if query_mesh is None:
        query_mesh = mesh.copy()
        query_mesh.apply_scale(100.)
    for start in range(0, len(indices), chunk):
        take = indices[start:start + chunk]
        inside[take] = query_mesh.contains(points[take] * 100.)
    return inside


class EndpointChecker:
    def __init__(self, receivers):
        self.up = native.load()
        raw = self.up.geometry['load_gripper_meshes'](
            'panda', str(native.VENDOR/'assets'), has_ft_sensor=False, visual=False)
        self.hand_meshes = {name: mesh.copy() for name, mesh in raw.items()}
        self.reflection = np.diag([1., -1., 1., 1.])
        for name, mesh in self.hand_meshes.items():
            mesh.vertices *= .01
            if name == 'panda_rightfinger':
                # FCL accepts proper rigid rotations; bake the upstream mirror
                # into vertices rather than passing a reflection as a rotation.
                mesh.apply_transform(self.reflection)
        self.hand_reps = {name: solid_representatives(mesh)
                          for name, mesh in self.hand_meshes.items()}
        self.hand_queries = {name: mesh.copy() for name, mesh in self.hand_meshes.items()}
        for mesh in self.hand_queries.values():
            mesh.apply_scale(100.)
        manager = self.up.trimesh.collision.CollisionManager
        self.hand_manager = manager()
        for name, mesh in self.hand_meshes.items():
            self.hand_manager.add_object(name, mesh)
        self.receivers = []
        for item in receivers:
            mesh = welded_mesh(item['mesh_path'])
            mesh.apply_transform(item['board_T_mesh'])
            reps = solid_representatives(mesh)
            collision = manager()
            collision.add_object(item['name'], mesh)
            query = mesh.copy()
            query.apply_scale(100.)
            self.receivers.append(dict(name=item['name'], mesh=mesh, reps=reps,
                                       collision=collision, query=query))

    def component_transforms(self, board_T_hand, open_ratio):
        transforms = self.up.geometry['get_panda_meshes_transforms'](
            self.hand_meshes, open_ratio)
        for name, local in transforms.items():
            local = local.copy()
            local[:3, 3] *= .01
            if name == 'panda_rightfinger':
                local = local @ self.reflection
            transforms[name] = board_T_hand @ local
        return transforms

    def check(self, board_T_hands, open_ratios):
        count = len(board_T_hands)
        reasons = ['clear_at_nominal_goal'] * count
        components = []
        for i, (hand, opening) in enumerate(zip(board_T_hands, open_ratios)):
            transforms = self.component_transforms(hand, opening)
            components.append(transforms)
            for name, transform in transforms.items():
                self.hand_manager.set_transform(name, transform)
            for receiver in self.receivers:
                if self.hand_manager.in_collision_other(receiver['collision']):
                    reasons[i] = 'hand_surface_intersects:' + receiver['name']
                    break

        # With no boundary intersection, a connected closed component is wholly
        # inside or outside the other solid. Test one vertex per component in
        # BOTH directions, using exact CAD shells rather than convex hulls.
        pending = np.asarray([i for i, reason in enumerate(reasons)
                              if reason == 'clear_at_nominal_goal'], dtype=int)
        for receiver in self.receivers:
            for name, mesh in self.hand_meshes.items():
                if not len(pending):
                    break
                world_points = np.asarray([transform_points(self.hand_reps[name], components[i][name])
                                           for i in pending])
                nested = contains_batched(receiver['mesh'], world_points,
                    query_mesh=receiver['query']).reshape(len(pending), -1).any(axis=1)
                for i in pending[nested]:
                    reasons[i] = 'hand_inside_receiver:' + receiver['name'] + ':' + name
                remaining = pending[~nested]
                if len(remaining):
                    local_points = []
                    for i in remaining:
                        transform = components[i][name]
                        local_points.append((receiver['reps'] - transform[:3, 3]) @ transform[:3, :3])
                    reverse = contains_batched(mesh, local_points,
                        query_mesh=self.hand_queries[name]).reshape(len(remaining), -1).any(axis=1)
                    for i in remaining[reverse]:
                        reasons[i] = 'receiver_inside_hand:' + receiver['name'] + ':' + name
                pending = np.asarray([i for i in pending if reasons[i] == 'clear_at_nominal_goal'], dtype=int)
        return np.asarray([r == 'clear_at_nominal_goal' for r in reasons]), reasons

