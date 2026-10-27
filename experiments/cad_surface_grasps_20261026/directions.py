"""Small sets of horizontal closing directions from native CAD surfaces.

Directions are proposals, not force/slip guarantees. Flat normals and local
circular-cylinder fits replace imported CAD X/Y/Z labels. A cylinder hypothesis
describes only its supported surface band, not symmetry of an entire connector.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CONFIG = dict(flat_normal_cluster_deg=.25, flat_max_tilt_deg=1.,
              flat_min_opposing_area_m2=16e-6, flat_plane_halfwidth_m=.000025,
              flat_min_support_width_m=.002,direction_merge_deg=.25,
              cylinder_normal_perpendicular_deg=1., cylinder_line_residual_deg=5.,
              cylinder_radius_band_halfwidth_m=.00015,
              cylinder_min_band_area_m2=16e-6, cylinder_min_surface_fraction=.015,
              cylinder_min_axial_span_m=.003,
              cylinder_azimuth_bins=24, cylinder_min_occupied_bins=18,
              cylinder_max_circle_residual_m=.000025,
              cylinder_max_circle_residual_fraction=.002,
              cylinder_vertical_deg=1.)


def canonical(axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    index = int(np.argmax(np.abs(axis)))
    return axis if axis[index] >= 0 else -axis


def radial_basis(axis):
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first)))


def triangle_data(vertices, triangles):
    points = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    twice_area = np.linalg.norm(cross, axis=1)
    good = twice_area > 1e-16
    return points[good], points[good].mean(axis=1), cross[good]/twice_area[good,None], twice_area[good]/2,np.flatnonzero(good)


def _axis_candidates(centers, normals, area):
    center = np.average(centers, axis=0, weights=area)
    normal_scatter = (normals*area[:,None]).T@normals
    centered = centers-center
    point_scatter = (centered*area[:,None]).T@centered
    candidates = []
    for matrix in (normal_scatter, point_scatter):
        _, vectors = np.linalg.eigh(matrix)
        for axis in vectors.T:
            axis = canonical(axis)
            if not any(abs(axis@old)>np.cos(np.deg2rad(.1)) for old in candidates):
                candidates.append(axis)
    return candidates


def _occupied_bins(normals_2d, count):
    angles = np.arctan2(normals_2d[:,1], normals_2d[:,0])
    bins = np.floor((angles+np.pi)/(2*np.pi)*count).astype(int)%count
    return int(len(np.unique(bins)))


def _circle_fit(points):
    origin = points.mean(axis=0)
    centered = points-origin
    solution, _, rank, _ = np.linalg.lstsq(np.c_[2*centered, np.ones(len(centered))],
                                         np.sum(centered**2, axis=1), rcond=None)
    if rank < 3:
        return None
    center = origin+solution[:2]
    radius_squared = solution[2]+np.sum(solution[:2]**2)
    if radius_squared <= 0:
        return None
    radius = float(np.sqrt(radius_squared))
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
    return center, radius, float(np.quantile(residual,.95)), float(residual.max())


def _radius_bands(radii, area, halfwidth, minimum_area):
    """Extract disjoint dense radius bands without prespecifying their number."""
    remaining = np.arange(len(radii))
    result = []
    while len(remaining):
        order = remaining[np.argsort(radii[remaining], kind='stable')]
        values = radii[order]
        left = np.searchsorted(values, values-halfwidth, side='left')
        right = np.searchsorted(values, values+halfwidth, side='right')
        prefix = np.r_[0., np.cumsum(area[order])]
        masses = prefix[right]-prefix[left]
        best = int(np.argmax(masses))
        if masses[best] < minimum_area:
            break
        selected = order[left[best]:right[best]]
        result.append(selected)
        keep = np.ones(len(radii), dtype=bool)
        keep[selected] = False
        remaining = remaining[keep[remaining]]
    return result


def _fit_circular_subbands(face_ids, points, basis, area, config, minimum_area):
    """Split a failed coarse band at a real gap between native vertex radii.

    The initial radius window is only proposal generation. Two nearby coaxial
    cylinders must not be forced into one circle and then rejected together.
    Acceptance tolerances remain unchanged for every refined band.
    """
    pending = [face_ids]
    while pending:
        selected = pending.pop()
        if area[selected].sum() < minimum_area:
            continue
        projected = points[selected]@basis
        circle = _circle_fit(projected.reshape(-1,2))
        if circle is None:
            continue
        center,radius,residual95,_ = circle
        limit = max(config['cylinder_max_circle_residual_m'],
                    config['cylinder_max_circle_residual_fraction']*radius)
        distances = np.linalg.norm(projected-center,axis=2)
        if residual95 > limit:
            median_radius = np.median(distances,axis=1)
            order = np.argsort(median_radius,kind='stable')
            gaps = np.diff(median_radius[order])
            if not len(gaps):
                continue
            split = int(np.argmax(gaps))
            if gaps[split] <= 2*limit:
                continue
            pending.extend((selected[order[:split+1]],selected[order[split+1:]]))
            continue
        # A 95th percentile fit must not label its rejected transitions as
        # cylindrical support. Every saved support triangle has all vertices
        # within the same unchanged residual limit.
        supported = np.max(np.abs(distances-radius),axis=1) <= limit
        selected = selected[supported]
        if area[selected].sum() < minimum_area:
            continue
        circle = _circle_fit((points[selected]@basis).reshape(-1,2))
        if circle is None:
            continue
        center,radius,residual95,residual_max = circle
        limit = max(config['cylinder_max_circle_residual_m'],
                    config['cylinder_max_circle_residual_fraction']*radius)
        if residual_max > limit:
            continue
        yield selected,circle


def cylinder_hypotheses(points, centers, normals, area, config):
    hypotheses = []
    total_area = float(area.sum())
    minimum_area = max(config['cylinder_min_band_area_m2'], total_area*config['cylinder_min_surface_fraction'])
    side_cosine = np.sin(np.deg2rad(config['cylinder_normal_perpendicular_deg']))
    max_line_sine = np.sin(np.deg2rad(config['cylinder_line_residual_deg']))
    for initial in _axis_candidates(centers, normals, area):
        axis = initial
        side = np.abs(normals@axis) <= side_cosine
        if area[side].sum() < minimum_area:
            continue
        # Refine the shaft using only normals nearly perpendicular to it. This
        # estimates a surface axis, never an imported coordinate-frame axis.
        for _ in range(2):
            _, vectors = np.linalg.eigh((normals[side]*area[side,None]).T@normals[side])
            refined = canonical(vectors[:,0])
            if abs(refined@axis) < np.cos(np.deg2rad(3.)):
                break
            axis = refined
            side = np.abs(normals@axis) <= side_cosine
        basis = radial_basis(axis)
        ids = np.flatnonzero(side)
        positions = centers[ids]@basis
        radial_normals = normals[ids]@basis
        radial_normals /= np.linalg.norm(radial_normals, axis=1)[:,None]
        tangent = np.c_[-radial_normals[:,1], radial_normals[:,0]]
        weights = area[ids]
        origin = np.average(positions, axis=0, weights=weights)
        center = np.zeros(2)
        valid = np.ones(len(ids), dtype=bool)
        # Robust line intersection handles different radii on a stepped shaft.
        for _ in range(4):
            matrix = tangent[valid]*np.sqrt(weights[valid,None])
            rhs = np.sum(tangent[valid]*(positions[valid]-origin), axis=1)*np.sqrt(weights[valid])
            if np.linalg.matrix_rank(matrix) < 2:
                break
            center = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            delta = positions-origin-center
            radial_distance = np.linalg.norm(delta, axis=1)
            line_error = np.abs(np.sum(tangent*delta, axis=1))
            valid = (radial_distance>1e-4)&(line_error <= radial_distance*max_line_sine)
            if weights[valid].sum() < minimum_area:
                break
        if weights[valid].sum() < minimum_area:
            continue
        local_ids = np.flatnonzero(valid)
        distances = np.abs(np.sum(radial_normals[valid]*(positions[valid]-origin-center), axis=1))
        for band_local in _radius_bands(distances, weights[valid],
                                       config['cylinder_radius_band_halfwidth_m'], 1e-16):
            selected = local_ids[band_local]
            if _occupied_bins(radial_normals[selected],config['cylinder_azimuth_bins']) < config['cylinder_min_occupied_bins']:
                continue
            for face_ids,circle in _fit_circular_subbands(ids[selected],points,basis,area,config,1e-16):
                selected_normals = normals[face_ids]@basis
                occupied = _occupied_bins(selected_normals,config['cylinder_azimuth_bins'])
                if occupied < config['cylinder_min_occupied_bins']:
                    continue
                selected_points = points[face_ids].reshape(-1,3)
                axial = selected_points@axis
                span = float(np.ptp(axial))
                if span < config['cylinder_min_axial_span_m']:
                    continue
                circle_center, radius, residual95, residual_max = circle
                limit = max(config['cylinder_max_circle_residual_m'],
                            config['cylinder_max_circle_residual_fraction']*radius)
                point_on_axis = basis@circle_center + axis*float(np.average(centers[face_ids]@axis, weights=area[face_ids]))
                if any(abs(axis@h['axis']) > np.cos(np.deg2rad(.25))
                       and abs(radius-h['radius']) <= limit
                       and np.linalg.norm(np.cross(point_on_axis-h['point'],axis)) <= limit
                       for h in hypotheses):
                    continue
                center_delta = centers[face_ids]-point_on_axis
                radial_vectors = center_delta-(center_delta@axis)[:,None]*axis
                radial_vectors /= np.linalg.norm(radial_vectors,axis=1)[:,None]
                tangent_error = np.degrees(np.arccos(np.clip(np.abs(np.sum(radial_vectors*normals[face_ids],axis=1)),0,1)))
                hypotheses.append(dict(axis=axis,point=point_on_axis,radius=radius,
                    triangle_ids=face_ids, area_m2=float(area[face_ids].sum()),
                    direction_area_gate_m2=minimum_area,
                    supports_direction_generation=bool(area[face_ids].sum()>=minimum_area),
                    axial_span_m=span,axial_min_m=float(axial.min()),axial_max_m=float(axial.max()),
                    occupied_azimuth_bins=occupied,circle_vertex_residual95_m=residual95,
                    circle_vertex_residual_max_m=residual_max,
                    normal_to_radial_95_deg=float(np.quantile(tangent_error,.95)),
                    normal_to_radial_max_deg=float(tangent_error.max())))
    strong = [h for h in hypotheses if h['supports_direction_generation']]
    for h in strong:
        h.update(classification_only=False,area_below_direction_gate=False,
                 coaxial_direction_support_index=None)
    kept = list(strong)
    for h in hypotheses:
        if h['supports_direction_generation']:
            continue
        # A small exact band may label real contact geometry only when a larger
        # band already establishes the same shaft. It creates no new heading.
        limit = max(config['cylinder_max_circle_residual_m'],
                    config['cylinder_max_circle_residual_fraction']*h['radius'])
        endpoints = np.array([h['point']+h['axis']*(s-h['point']@h['axis'])
                              for s in (h['axial_min_m'],h['axial_max_m'])])
        support = []
        for index,s in enumerate(strong):
            if abs(h['axis']@s['axis']) < np.cos(np.deg2rad(.25)):
                continue
            error = float(np.linalg.norm(np.cross(endpoints-s['point'],s['axis']),axis=1).max())
            if error <= limit:
                support.append((error,index))
        if support:
            error,index = min(support)
            h.update(classification_only=True,area_below_direction_gate=True,
                     coaxial_direction_support_index=index,coaxial_max_axis_distance_m=error)
            kept.append(h)
    return kept


def _flat_support(ids, points, centers, area, axis, config):
    if not len(ids):
        return []
    offsets = centers[ids]@axis
    support = []
    for band in _radius_bands(offsets,area[ids],config['flat_plane_halfwidth_m'],
                              config['flat_min_opposing_area_m2']):
        band_ids = ids[band]
        # A common normal/plane does not make separated tooth strips one face.
        # Split by shared native triangle edges before computing pad support.
        _,vertices = np.unique(points[band_ids].reshape(-1,3),axis=0,return_inverse=True)
        vertices = vertices.reshape(-1,3)
        edges = np.sort(vertices[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        _,edge_ids = np.unique(edges,axis=0,return_inverse=True)
        incidence = coo_matrix((np.ones(len(edge_ids)),
                              (np.repeat(np.arange(len(band_ids)),3),edge_ids))).tocsr()
        count, labels = connected_components(incidence@incidence.T,directed=False)
        for label in range(count):
            selected = band_ids[labels==label]
            if area[selected].sum() < config['flat_min_opposing_area_m2']:
                continue
            positions = points[selected].reshape(-1,3)@radial_basis(axis)
            weights = np.repeat(area[selected]/3,3)
            centered = positions-np.average(positions,axis=0,weights=weights)
            _,axes = np.linalg.eigh((centered*weights[:,None]).T@centered)
            spans = np.ptp(centered@axes,axis=0)
            if min(spans) < config['flat_min_support_width_m']:
                continue
            support.append(dict(area_m2=float(area[selected].sum()),support_spans_m=spans.tolist(),
                        plane_offset_m=float(np.average(centers[selected]@axis,weights=area[selected])),
                        plane_offset_spread_m=float(np.ptp(centers[selected]@axis)),
                        supporting_triangle_ids=selected.tolist()))
    return support


def _normal_clusters(points,centers,normals, area, config):
    angle_deg = config['flat_normal_cluster_deg']
    minimum_opposing_area = config['flat_min_opposing_area_m2']
    canonical_normals = normals.copy()
    axes = np.argmax(np.abs(canonical_normals),axis=1)
    signs = np.where(canonical_normals[np.arange(len(normals)),axes] >= 0, 1., -1.)
    canonical_normals *= signs[:,None]
    # Coarse unique bins reduce repeated triangles before radius clustering.
    _, inverse = np.unique(np.round(canonical_normals/1e-4).astype(np.int64),axis=0,return_inverse=True)
    count = int(inverse.max()+1)
    masses = np.bincount(inverse,weights=area,minlength=count)
    means = np.column_stack([np.bincount(inverse,weights=canonical_normals[:,i]*area,minlength=count) for i in range(3)])
    means /= np.linalg.norm(means,axis=1)[:,None]
    tree = cKDTree(means)
    remaining = np.ones(count,dtype=bool)
    result = []
    for index in np.argsort(-masses,kind='stable'):
        if not remaining[index]:
            continue
        neighbors = np.asarray(tree.query_ball_point(means[index],2*np.sin(np.deg2rad(angle_deg)/2)),dtype=int)
        neighbors = neighbors[remaining[neighbors]]
        remaining[neighbors] = False
        if masses[neighbors].sum() < 2*minimum_opposing_area:
            continue
        member = np.isin(inverse,neighbors)
        axis = canonical(np.sum(canonical_normals[member]*area[member,None],axis=0))
        positive = _flat_support(np.flatnonzero(member & (normals@axis>=0)),points,centers,area,axis,config)
        negative = _flat_support(np.flatnonzero(member & (normals@axis<0)),points,centers,area,axis,config)
        if positive and negative:
            result.append((axis,positive,negative))
    return result


def surface_directions(vertices_world, triangles, config=None):
    """Return JSON-safe directions plus geometric support and rejected flats.

    No CAD frame, object name, pose labels, learned model or grasp annotation is
    used. Flat projection error is explicit. A fitted local cylinder supports
    one horizontal radial direction for a lying shaft; a nearly vertical shaft
    uses world X/Y representatives of the radial family, not random CAD roll.
    """
    config = dict(CONFIG, **(config or {}))
    points, centers, normals, area, original_triangle_ids = triangle_data(vertices_world,triangles)
    if not len(area) or not np.isfinite(points).all():
        raise ValueError('Finite nondegenerate native CAD triangles required')
    cylinders = cylinder_hypotheses(points,centers,normals,area,config)
    curved = np.zeros(len(area),dtype=bool)
    directions = []
    cylinder_records = []
    def add(axis, source, evidence):
        horizontal = np.asarray(axis,dtype=float).copy()
        horizontal[2] = 0.
        if np.linalg.norm(horizontal)<1e-10:
            return
        horizontal = canonical(horizontal)
        existing = next((d for d in directions if abs(horizontal@np.asarray(d['closing_axis_world'])) >= np.cos(np.deg2rad(config['direction_merge_deg']))),None)
        if existing is not None:
            existing['sources'].append(dict(type=source,**evidence))
            return
        directions.append(dict(label='',source=source,usable=True,cad_index=None,
            closing_axis_world=horizontal.tolist(),projection_error_deg=float(evidence.get('projection_error_deg',0.)),
            sources=[dict(type=source,**evidence)]))
    for index,hypothesis in enumerate(cylinders,1):
        curved[hypothesis['triangle_ids']] = True
        record = {key:value for key,value in hypothesis.items() if key not in ('axis','point','radius','triangle_ids')}
        record.update(id=f'CYL{index:03d}',shaft_axis_world=hypothesis['axis'].tolist(),
                      point_on_axis_world=hypothesis['point'].tolist(),fit_radius_m=hypothesis['radius'],
                      supporting_triangle_ids=original_triangle_ids[hypothesis['triangle_ids']].tolist(),
                      scope='local_circular_surface_band_not_whole_object_symmetry')
        support_index=hypothesis['coaxial_direction_support_index']
        record['direction_support_cylinder_id']=None if support_index is None else f'CYL{support_index+1:03d}'
        cylinder_records.append(record)
        if not hypothesis['supports_direction_generation']:
            continue
        axis = hypothesis['axis']
        evidence = dict(cylinder_id=record['id'],shaft_axis_world=axis.tolist(),
                        point_on_axis_world=hypothesis['point'].tolist(),fit_radius_m=hypothesis['radius'])
        if abs(axis[2]) >= np.cos(np.deg2rad(config['cylinder_vertical_deg'])):
            for heading in (np.array([1.,0.,0.]),np.array([0.,1.,0.])):
                add(heading,'cylinder_radial',dict(evidence,orientation_rule='world_canonical_near_vertical_shaft',
                    radial_perpendicular_error_deg=float(np.degrees(np.arcsin(np.clip(abs(heading@axis),0,1))))))
        else:
            heading = np.cross([0.,0.,1.],axis)
            add(heading,'cylinder_radial',dict(evidence,orientation_rule='world_up_cross_fitted_shaft',
                                               radial_perpendicular_error_deg=0.))
    rejected_flats = []
    flat_ids = ~curved
    if flat_ids.any():
        flat_original_ids = original_triangle_ids[flat_ids]
        for axis,positive,negative in _normal_clusters(points[flat_ids],centers[flat_ids],normals[flat_ids],area[flat_ids],config):
            for patch in positive+negative:
                patch['supporting_triangle_ids'] = flat_original_ids[np.asarray(patch['supporting_triangle_ids'])].tolist()
            error = float(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1))))
            evidence = dict(surface_normal_world=axis.tolist(),positive_area_m2=sum(p['area_m2'] for p in positive),
                            negative_area_m2=sum(p['area_m2'] for p in negative),
                            positive_planar_patches=positive,negative_planar_patches=negative,
                            projection_error_deg=error)
            if error > config['flat_max_tilt_deg']:
                rejected_flats.append(dict(evidence,reason='opposing_flat_normals_not_horizontal_enough'))
            else:
                add(axis,'opposing_flat_normals',evidence)
    directions.sort(key=lambda d:float(np.arctan2(d['closing_axis_world'][1],d['closing_axis_world'][0])%np.pi))
    for index,direction in enumerate(directions,1):
        direction['label'] = f'SURFACE_{index:02d}'
    return dict(directions=directions,cylinder_hypotheses=cylinder_records,rejected_flat_directions=rejected_flats,
                metadata=dict(config=config,triangle_count=len(area),total_area_m2=float(area.sum()),
                              curved_support_area_m2=float(area[curved].sum()),
                              directions=len(directions),uses_native_CAD_frame=False,
                              no_whole_object_symmetry_claim=True))
