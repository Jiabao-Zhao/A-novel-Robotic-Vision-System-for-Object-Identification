"""Convex hand components against the unchanged target triangle mesh, in metres.

No dynamics or triangle-mesh convexification is used for the target. Surface
crossings are found by clipping target triangles against the convex component;
ray parity handles either solid wholly containing the other. A linear sweep of
a translating convex component is the hull of its two endpoint vertex sets.

Positive clearance offsets supporting planes and the AABB. It conservatively
rejects some near-corner placements and is not an exact Euclidean distance.
Numerical contact tolerance is 1e-10 m, separate from any declared clearance.
Solid containment assumes closed, non-self-intersecting connected components.
Disconnected closed CAD components are treated as a union of solids, so their
overlap cannot cancel through whole-mesh parity. Native internal triangles stay
in the surface test; they can conservatively reject a placement inside a union.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


NUMERICAL_TOL_M = 1e-10


def _hull(vertices):
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Convex component needs finite N x 3 vertices.")
    hull = ConvexHull(points)
    equations = np.unique(np.round(hull.equations, 12), axis=0)
    return points[hull.vertices], equations


def _clip_triangle(triangle, planes):
    polygon = np.asarray(triangle, dtype=np.float64)
    for plane in planes:
        if not len(polygon):
            return polygon
        # Classification and edge intersection must use the SAME boundary.
        # Otherwise two positive distances straddling the tolerance can be
        # classified on opposite sides, then extrapolated toward distance zero.
        distances = polygon @ plane[:3] + plane[3] - NUMERICAL_TOL_M
        inside = distances <= 0.
        if inside.all():
            continue
        if not inside.any():
            return np.empty((0, 3))
        output = []
        for index in range(len(polygon)):
            previous = index - 1
            if inside[index] != inside[previous]:
                fraction = np.clip(distances[previous] / (distances[previous] - distances[index]), 0., 1.)
                output.append(polygon[previous] + fraction * (polygon[index] - polygon[previous]))
            if inside[index]:
                output.append(polygon[index])
        polygon = np.asarray(output)
    return polygon


class MeshCollider:
    """Read-only target mesh and exact, finite convex-component intersection tests."""

    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        if not np.isfinite(self.vertices).all() or not len(self.triangles):
            raise ValueError("Target needs finite vertices and nonempty triangles.")
        self.faces = self.vertices[self.triangles]
        self.triangle_min = self.faces.min(axis=1)
        self.triangle_max = self.faces.max(axis=1)
        self.minimum = self.vertices.min(axis=0)
        self.maximum = self.vertices.max(axis=0)
        self._component_topology()
        self._tree = []
        self._build_tree(np.arange(len(self.triangles)))
        self.last_report = None

    def _component_topology(self):
        # Connected components use shared mesh edges, not proximity across
        # distinct intersecting CAD solids. Input vertices must be welded.
        edges = np.sort(np.concatenate((self.triangles[:, [0, 1]],
                                        self.triangles[:, [1, 2]],
                                        self.triangles[:, [2, 0]])), axis=1)
        owners = np.tile(np.arange(len(self.triangles)), 3)
        keys = edges[:, 0].astype(np.uint64)*np.uint64(len(self.vertices)) + edges[:, 1]
        order = np.argsort(keys, kind="stable")
        keys, owners = keys[order], owners[order]
        repeated = keys[1:] == keys[:-1]
        graph = coo_matrix((np.ones(np.count_nonzero(repeated), dtype=np.int8),
                            (owners[:-1][repeated], owners[1:][repeated])),
                           shape=(len(self.triangles), len(self.triangles)))
        self.component_count, self.component_ids = connected_components(graph, directed=False)
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(keys)])
        edge_components = self.component_ids[owners[starts]]
        bad = np.bincount(edge_components[counts != 2], minlength=self.component_count)
        self.closed_components = bad == 0
        self.topology = {"component_count": int(self.component_count),
                         "closed_component_count": int(self.closed_components.sum()),
                         "component_triangle_counts": np.bincount(self.component_ids).tolist(),
                         "component_closed": self.closed_components.tolist(),
                         "boundary_edges": int(np.count_nonzero(counts == 1)),
                         "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
                         "containment_model": "union_of_edge_connected_closed_components"}

    def _build_tree(self, indices):
        lower = self.triangle_min[indices].min(axis=0)
        upper = self.triangle_max[indices].max(axis=0)
        index = len(self._tree)
        self._tree.append(None)
        if len(indices) <= 64:
            self._tree[index] = (lower, upper, -1, -1, indices)
        else:
            centers = (self.triangle_min[indices] + self.triangle_max[indices]) / 2
            axis = int(np.ptp(centers, axis=0).argmax())
            middle = len(indices) // 2
            order = np.argpartition(centers[:, axis], middle)
            left = self._build_tree(indices[order[:middle]])
            right = self._build_tree(indices[order[middle:]])
            self._tree[index] = (lower, upper, left, right, None)
        return index

    def _query_box(self, lower, upper):
        stack, leaves = [0], []
        while stack:
            a, b, left, right, indices = self._tree[stack.pop()]
            if np.any(b < lower) or np.any(a > upper):
                continue
            if left < 0:
                leaves.append(indices)
            else:
                stack.extend((left, right))
        if not leaves:
            return np.empty(0, dtype=np.int64)
        ids = np.concatenate(leaves)
        active = np.all(self.triangle_max[ids] >= lower, axis=1) & np.all(self.triangle_min[ids] <= upper, axis=1)
        return ids[active]

    def _query_ray(self, origin, direction):
        stack, leaves = [0], []
        inverse = 1. / direction
        while stack:
            a, b, left, right, indices = self._tree[stack.pop()]
            t0, t1 = (a - origin) * inverse, (b - origin) * inverse
            near = np.maximum(np.minimum(t0, t1), 0.).max()
            far = np.maximum(t0, t1).min()
            if far < near:
                continue
            if left < 0:
                leaves.append(indices)
            else:
                stack.extend((left, right))
        return np.concatenate(leaves) if leaves else np.empty(0, dtype=np.int64)

    def contains_points(self, points):
        """Union of per-component majority ray parities away from mesh surfaces.

        Three fixed non-axis directions avoid relying on triangle winding. Hits
        sharing the same ray distance count once, including a shared face edge.
        Open components do not establish containment. The function is for solid
        containment, not signed-distance measurement. Nested components are
        treated as solids, not as separately encoded cavity shells.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        inside = np.zeros(len(points), dtype=bool)
        directions = np.asarray([[1., .371390676, .694746591],
                                 [.217986204, 1., .529847612],
                                 [.413270198, .193480271, 1.]])
        for row, point in enumerate(points):
            if np.any(point < self.minimum) or np.any(point > self.maximum):
                continue
            votes = []
            for direction in directions:
                direction = direction / np.linalg.norm(direction)
                candidates = self._query_ray(point, direction)
                faces = self.faces[candidates]
                v0 = faces[:, 0]
                edge1, edge2 = faces[:, 1] - v0, faces[:, 2] - v0
                pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
                determinant = np.einsum("ij,ij->i", edge1, pvec)
                active = np.abs(determinant) > 1e-16
                inverse = np.zeros_like(determinant)
                inverse[active] = 1. / determinant[active]
                tvec = point - v0
                u = np.einsum("ij,ij->i", tvec, pvec) * inverse
                qvec = np.cross(tvec, edge1)
                v = qvec @ direction * inverse
                distance = np.einsum("ij,ij->i", edge2, qvec) * inverse
                hit = active & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance > NUMERICAL_TOL_M)
                labels = self.component_ids[candidates[hit]]
                depths = distance[hit]
                order = np.lexsort((depths, labels))
                labels, depths = labels[order], depths[order]
                unique = np.r_[True, (labels[1:] != labels[:-1]) | (np.diff(depths) > 1e-9)] if len(labels) else np.empty(0, dtype=bool)
                counts = np.bincount(labels[unique], minlength=self.component_count)
                votes.append(counts % 2 == 1)
            component_inside = np.sum(votes, axis=0) >= 2
            inside[row] = np.any(component_inside & self.closed_components)
        return inside

    def intersection_report(self, convex_vertices, clearance_m=0.):
        """Return intersection type and one witness point, if available.

        ``clearance_m=0`` tests solid touching/intersection. A positive value
        uses a conservative supporting-plane clearance guard, not an exact SDF.
        """
        if clearance_m < 0:
            raise ValueError("Clearance must be nonnegative.")
        points, planes = _hull(convex_vertices)
        planes[:, 3] -= clearance_m
        lower = points.min(axis=0) - clearance_m - NUMERICAL_TOL_M
        upper = points.max(axis=0) + clearance_m + NUMERICAL_TOL_M
        candidate = self._query_box(lower, upper)
        result = {"intersects": False, "kind": "separate", "triangle_index": None,
                  "point_world_m": None, "broadphase_triangles": int(len(candidate))}
        if not len(candidate):
            if not (np.any(upper < self.minimum) or np.any(lower > self.maximum)):
                center = points.mean(axis=0)
                if self.contains_points(center)[0]:
                    result.update(intersects=True, kind="hand_inside_target", point_world_m=center.tolist())
            self.last_report = result
            return result
        # Eliminate triangles lying wholly outside any one supporting plane.
        possible = []
        for first in range(0, len(candidate), 4096):
            ids = candidate[first:first + 4096]
            distances = self.faces[ids] @ planes[:, :3].T + planes[:, 3]
            rejected = np.any(np.all(distances > NUMERICAL_TOL_M, axis=1), axis=1)
            possible.extend(ids[~rejected])
        for index in possible:
            polygon = _clip_triangle(self.faces[index], planes)
            if len(polygon):
                result.update(intersects=True, kind="target_surface_intersection", triangle_index=int(index),
                              point_world_m=polygon.mean(axis=0).tolist())
                self.last_report = result
                return result
        # Target vertices wholly inside the hand are also caught by clipping,
        # because each such vertex belongs to a triangle. The remaining case
        # with no surface crossing is a hand entirely within target material.
        center = points.mean(axis=0)
        if self.contains_points(center)[0]:
            result.update(intersects=True, kind="hand_inside_target", point_world_m=center.tolist())
        self.last_report = result
        return result

    def intersects(self, convex_vertices, clearance_m=0.):
        return self.intersection_report(convex_vertices, clearance_m)["intersects"]

    def surface_contacts(self, convex_vertices, clearance_m=0.):
        """Return every clipped native target face, without choosing a winner.

        At a CAD edge, several faces can contact the same finger simultaneously.
        Their ordering has no physical meaning. Callers must examine the whole
        set when checking the closing direction and intended pad region. Face
        normals follow target triangle winding, which the caller must audit.
        """
        if clearance_m < 0:
            raise ValueError("Clearance must be nonnegative.")
        points, planes = _hull(convex_vertices)
        planes[:, 3] -= clearance_m
        lower = points.min(axis=0) - clearance_m - NUMERICAL_TOL_M
        upper = points.max(axis=0) + clearance_m + NUMERICAL_TOL_M
        candidate = self._query_box(lower, upper)
        contacts = []
        for first in range(0, len(candidate), 4096):
            ids = candidate[first:first + 4096]
            distances = self.faces[ids] @ planes[:, :3].T + planes[:, 3]
            rejected = np.any(np.all(distances > NUMERICAL_TOL_M, axis=1), axis=1)
            for index in ids[~rejected]:
                triangle = self.faces[index]
                polygon = _clip_triangle(triangle, planes)
                if not len(polygon):
                    continue
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                magnitude = np.linalg.norm(normal)
                if magnitude <= 1e-20:
                    continue
                contacts.append({"triangle_index": int(index),
                                 "point_world_m": polygon.mean(axis=0).tolist(),
                                 "normal_world": (normal/magnitude).tolist(),
                                 "polygon_world_m": polygon.tolist()})
        return contacts

    def swept_intersects(self, convex_vertices, displacement, clearance_m=0.):
        points = np.asarray(convex_vertices, dtype=np.float64)
        delta = np.asarray(displacement, dtype=np.float64)
        sweep = np.concatenate((points, points + delta)) if np.linalg.norm(delta) > 1e-14 else points
        return self.intersects(sweep, clearance_m)

    def first_contact(self, convex_vertices, displacement, fraction_tolerance=1e-4):
        """Bracket earliest contact along a straight convex-component translation.

        Bisection is on the complete prefix sweep, so it does not skip a thin
        obstacle between clear endpoint samples. Spatial bracket width is at
        most ``norm(displacement) * fraction_tolerance`` metres.
        """
        points = np.asarray(convex_vertices, dtype=np.float64)
        delta = np.asarray(displacement, dtype=np.float64)
        if fraction_tolerance <= 0 or fraction_tolerance >= 1:
            raise ValueError("fraction_tolerance must lie strictly between 0 and 1.")
        initial = self.intersection_report(points)
        if initial["intersects"]:
            return {"fraction": 0., "lower_fraction": 0., "upper_fraction": 0., **initial}
        if not self.swept_intersects(points, delta):
            return None
        lower, upper = 0., 1.
        while upper - lower > fraction_tolerance:
            middle = (lower + upper) / 2
            if self.swept_intersects(points, delta * middle):
                upper = middle
            else:
                lower = middle
        witness = self.intersection_report(points + delta * upper)
        if not witness["intersects"]:
            witness = self.intersection_report(np.concatenate((points + delta*lower, points + delta*upper)))
        return {"fraction": float(upper), "lower_fraction": float(lower), "upper_fraction": float(upper),
                "spatial_bracket_m": float(np.linalg.norm(delta) * (upper - lower)), **witness}

    def first_contact_fraction(self, convex_vertices, displacement, fraction_tolerance=1e-4):
        contact = self.first_contact(convex_vertices, displacement, fraction_tolerance)
        return None if contact is None else contact["fraction"]
