"""Opposing contact regions on a finite horizontal-ray grid, in metres.

This is geometric search reduction, not a force or pickup-quality predictor.
Native target triangles and holes are retained. Grid, width-jump, center-jump
and normal tolerances are explicit local settings, not universal constants.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _grid(low, high, spacing):
    if spacing <= 0 or high <= low:
        raise ValueError('Grid needs positive spacing and nonzero target extent.')
    count = max(1, int(np.ceil((high-low)/spacing)))
    step = (high-low)/count
    return low+(np.arange(count)+.5)*step, step


def _cast(result, coordinates):
    yz = np.atleast_2d(np.asarray(coordinates, dtype=np.float64))
    x = np.asarray(result['frame']['closing_axis_world'])
    y = np.asarray(result['frame']['lateral_axis_world'])
    origins = (result['_ray_start_x']*x + yz[:, :1]*y
               + yz[:, 1:]*np.array([0., 0., 1.]))
    other = origins + result['_ray_length']*x
    directions = np.broadcast_to(x, origins.shape)
    first = result['_scene'].cast_rays(o3d.core.Tensor(np.c_[origins, directions].astype(np.float32)))
    second = result['_scene'].cast_rays(o3d.core.Tensor(np.c_[other, -directions].astype(np.float32)))
    d1, d2 = first['t_hit'].numpy().astype(np.float64), second['t_hit'].numpy().astype(np.float64)
    finite = np.isfinite(d1) & np.isfinite(d2)
    pairs = np.full((len(yz), 2, 3), np.nan)
    pairs[finite, 0] = origins[finite] + d1[finite, None]*x
    pairs[finite, 1] = other[finite] - d2[finite, None]*x
    widths = np.full(len(yz), np.nan)
    widths[finite] = (pairs[finite, 1]-pairs[finite, 0])@x
    n1 = first['primitive_normals'].numpy().astype(np.float64)
    n2 = second['primitive_normals'].numpy().astype(np.float64)
    cosine = np.cos(np.deg2rad(result['metadata']['normal_angle_deg']))
    valid = finite & (widths > 1e-6) & (widths <= result['metadata']['maximum_contact_width_m'])
    valid &= np.abs(n1@x) >= cosine
    valid &= np.abs(n2@x) >= cosine
    valid &= np.einsum('ij,ij->i', n1, n2) <= -np.cos(2*np.deg2rad(result['metadata']['normal_angle_deg']))
    return dict(valid=valid, pairs=pairs, widths=widths,
                normals=np.stack((n1,n2),axis=1), coordinates=yz,
                triangle_ids=np.stack((first['primitive_ids'].numpy(),second['primitive_ids'].numpy()),axis=1))


def recast_pair(result, y_m, z_m):
    """Recompute both outer contacts at one requested contact-plane position.

    Returns None if either ray misses or the same normal/width checks fail.
    The caller must still validate the finite pads and the whole hand.
    """
    cast = _cast(result, [[y_m,z_m]])
    if not cast['valid'][0]:
        return None
    return dict(contacts_world_m=cast['pairs'][0].tolist(), width_m=float(cast['widths'][0]),
                normals_world=cast['normals'][0].tolist(), triangle_ids=cast['triangle_ids'][0].tolist(),
                center_y_m=float(y_m), center_z_m=float(z_m))


def contact_regions(vertices_world, triangles, direction_world, max_width_m,
                    grid_spacing_m=.001, normal_angle_deg=15.,
                    neighbor_width_jump_m=.002, neighbor_midpoint_jump_m=.001,
                    neighbor_normal_angle_deg=5.):
    """Find 4-connected qualifying regions before constructing hand poses.

    The contact plane uses hand lateral Y=cross(world-down, closing X), and
    world-up height. Bounds describe whole occupied grid cells; their sampling
    resolution is included in metadata and must not be called exact CAD edges.
    Arrays under ``_`` keys and each region's cells/pairs support representative
    construction; save metadata rather than serializing the Open3D scene.
    """
    vertices = np.asarray(vertices_world, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    direction = np.asarray(direction_world, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(faces)
            or not np.isfinite(vertices).all() or not np.isfinite(direction).all()):
        raise ValueError('Finite vertices, triangles and direction are required.')
    if abs(direction[2]) > 1e-8 or np.linalg.norm(direction) < 1e-10:
        raise ValueError('Closing direction must already be horizontal and nonzero.')
    if max_width_m <= 0 or not 0 < normal_angle_deg < 45:
        raise ValueError('Positive width and normal angle between 0 and 45 degrees required.')
    x = direction/np.linalg.norm(direction)
    y = np.cross([0.,0.,-1.],x)
    px, py, pz = vertices@x, vertices@y, vertices[:,2]
    ys, dy = _grid(float(py.min()),float(py.max()),grid_spacing_m)
    zs, dz = _grid(float(pz.min()),float(pz.max()),grid_spacing_m)
    yy, zz = np.meshgrid(ys,zs,indexing='ij')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.int32))))
    margin = max(float(np.ptp(px)),.01)+.01
    result = dict(regions=[],frame=dict(closing_axis_world=x.tolist(), lateral_axis_world=y.tolist(),
                                       height_axis_world=[0.,0.,1.]),
                  metadata=dict(grid_spacing_requested_m=float(grid_spacing_m),
                     grid_step_y_m=dy,grid_step_z_m=dz,grid_shape=[len(ys),len(zs)],
                     horizontal_lines_cast=len(ys)*len(zs), normal_angle_deg=float(normal_angle_deg),
                     maximum_contact_width_m=float(max_width_m),
                     neighbor_width_jump_m=float(neighbor_width_jump_m),
                     neighbor_midpoint_jump_m=float(neighbor_midpoint_jump_m),
                     neighbor_normal_angle_deg=float(neighbor_normal_angle_deg),
                     qualified_cells=0,region_count=0,region_cap=None),
                  _scene=scene,_ray_start_x=float(px.min()-margin),
                  _ray_length=float(np.ptp(px)+2*margin),_ys=ys,_zs=zs)
    cast = _cast(result,np.c_[yy.ravel(),zz.ravel()])
    active = np.flatnonzero(cast['valid'])
    result['metadata']['qualified_cells'] = int(len(active))
    if not len(active):
        return result
    lookup = np.full(len(ys)*len(zs),-1,dtype=np.int64)
    lookup[active] = np.arange(len(active))
    edges_left, edges_right = [], []
    pair_mid_x = cast['pairs'][active].mean(axis=1)@x
    widths = cast['widths'][active]
    normals = cast['normals'][active]
    cosine = np.cos(np.deg2rad(neighbor_normal_angle_deg))
    for offset, allowed in ((1,active%len(zs) != len(zs)-1),
                            (len(zs),active//len(zs) != len(ys)-1)):
        left = np.flatnonzero(allowed)
        right_flat = active[left]+offset
        right = lookup[right_flat]
        exists = right >= 0
        left,right = left[exists],right[exists]
        compatible = np.abs(widths[left]-widths[right]) <= neighbor_width_jump_m+1e-10
        compatible &= np.abs(pair_mid_x[left]-pair_mid_x[right]) <= neighbor_midpoint_jump_m+1e-10
        compatible &= np.all(np.einsum('nij,nij->ni',normals[left],normals[right]) >= cosine,axis=1)
        edges_left.extend(left[compatible])
        edges_right.extend(right[compatible])
    graph = coo_matrix((np.ones(len(edges_left),dtype=np.int8),(edges_left,edges_right)),
                       shape=(len(active),len(active)))
    count, labels = connected_components(graph,directed=False)
    for label in range(count):
        ids = active[labels==label]
        iy,iz = np.unravel_index(ids,(len(ys),len(zs)))
        coords = cast['coordinates'][ids]
        center = coords.mean(axis=0)
        region = dict(id=f'R{label+1:03d}',cells=len(ids),
          y_min=float(coords[:,0].min()-dy/2), y_max=float(coords[:,0].max()+dy/2),
          z_min=float(coords[:,1].min()-dz/2), z_max=float(coords[:,1].max()+dz/2),
          center_y=float(center[0]),center_z=float(center[1]),
          median_width_m=float(np.median(cast['widths'][ids])),
          min_width_m=float(np.min(cast['widths'][ids])),max_width_m=float(np.max(cast['widths'][ids])),
          grid_indices=np.c_[iy,iz], coordinates_m=coords,
          contact_pairs_world_m=cast['pairs'][ids],normals_world=cast['normals'][ids])
        # The mean can fall in a hole/concavity. Provide an actually occupied
        # cell nearest that mean, but leave exact representative recasting to caller.
        nearest = int(np.argmin(np.linalg.norm(coords-center,axis=1)))
        region['nearest_cell_y_m'] = float(coords[nearest,0])
        region['nearest_cell_z_m'] = float(coords[nearest,1])
        result['regions'].append(region)
    result['metadata']['region_count'] = int(count)
    return result
