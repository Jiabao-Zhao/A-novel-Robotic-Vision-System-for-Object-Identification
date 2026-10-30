"""Deterministic metric software rasterizer; photometry is a recorded assumption.

This is independent code, not a copy of the unlicensed reproduction. Native CAD
vertices are never scaled to unit size. Camera distance sets render occupancy.
"""
from __future__ import annotations

import numpy as np
from .geometry import transform_points


def sample_texture(texture, uv):
    """BOP PLY UV is bottom-left; PNG rows are top-left (official renderer)."""
    coordinates = np.asarray(uv, float)
    image = np.asarray(texture['image_rgb'], float) / 255.
    if texture.get('uv_origin') != 'bottom_left':
        raise ValueError('Explicit BOP bottom-left UV convention required')
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(coordinates).all():
        raise ValueError('Invalid CAD texture or UV')
    # Clamp at the texture edges. Filter and edge behavior are render assumptions.
    h, w = image.shape[:2]
    x = np.clip(coordinates[:, 0], 0, 1) * (w-1)
    y = (1-np.clip(coordinates[:, 1], 0, 1)) * (h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    wx, wy = (x-x0)[:, None], (y-y0)[:, None]
    return (image[y0,x0]*(1-wx)*(1-wy)+image[y0,x1]*wx*(1-wy)
            +image[y1,x0]*(1-wx)*wy+image[y1,x1]*wx*wy)


def icosphere_directions(subdivisions=2):
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([[-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
                         [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
                         [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1]], float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
             [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
             [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    verts = list(vertices)
    for _ in range(subdivisions):
        cache, next_faces = {}, []
        def midpoint(a, b):
            key = tuple(sorted((a,b)))
            if key not in cache:
                p = np.asarray(verts[a]) + np.asarray(verts[b]); p /= np.linalg.norm(p)
                cache[key] = len(verts); verts.append(p)
            return cache[key]
        for a,b,c in faces:
            ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
            next_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        faces = next_faces
    return np.asarray(verts)


def camera_poses(vertices, *, resolution=480, occupancy=.5, directions=None, camera_to_model=None):
    """OpenCV +X right,+Y down,+Z forward; T_camera_from_CAD in metres."""
    directions = icosphere_directions() if directions is None else np.asarray(directions)
    if camera_to_model is not None:
        camera_to_model = np.asarray(camera_to_model, dtype=float)
        if camera_to_model.shape != (162, 4, 4) or not np.isfinite(camera_to_model).all():
            raise ValueError('CNOS camera-to-model poses must be finite162x4x4')
        rotations = camera_to_model[:, :3, :3]
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
            raise ValueError('CNOS rotations must be in SO(3)')
        directions = camera_to_model[:, :3, 3]
    if len(directions) != 162 or not 0 < occupancy < 1:
        raise ValueError('Base configuration requires 162 view directions')
    centre = (vertices.min(0) + vertices.max(0)) / 2
    radius = np.linalg.norm(vertices - centre, axis=1).max()
    f = float(resolution)
    distance = radius * np.sqrt(1 + (2 / occupancy) ** 2)
    K = np.array([[f,0,(resolution-1)/2],[0,f,(resolution-1)/2],[0,0,1.]])
    poses=[]
    for index, direction in enumerate(directions):
        outward = direction / np.linalg.norm(direction)
        if camera_to_model is not None:
            R = camera_to_model[index, :3, :3].T
            if not np.allclose(R @ outward, [0, 0, -1], atol=1e-5):
                raise ValueError('CNOS camera does not face CAD origin')
        else:
            z = -outward
            up = np.array([0.,0.,1.]) if abs(z[2]) < .99 else np.array([0.,1.,0.])
            x = np.cross(z, up); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.stack([x,y,z])
        t = -R @ (centre + distance * outward)
        poses.append((R,t))
    return K,poses


def render_mesh(vertices, triangles, colors, R, t, K, *, height=480, width=480, background=0):
    p = transform_points(vertices,R,t)
    uvh = p @ np.asarray(K).T
    uv = uvh[:,:2] / uvh[:,2:3]
    zbuf = np.full((height,width),np.inf)
    rgb = np.full((height,width,3),background,dtype=np.float64)
    face_ids = np.full((height,width),-1,np.int32)
    for fid,face in enumerate(triangles):
        pc = p[face]
        if np.any(pc[:,2] <= 0):
            continue
        a,b,c = uv[face]
        xmin=max(0,int(np.ceil(min(a[0],b[0],c[0])))); xmax=min(width-1,int(np.floor(max(a[0],b[0],c[0]))))
        ymin=max(0,int(np.ceil(min(a[1],b[1],c[1])))); ymax=min(height-1,int(np.floor(max(a[1],b[1],c[1]))))
        if xmin>xmax or ymin>ymax:
            continue
        denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denom)<1e-12:
            continue
        yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
        w0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denom
        w1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom
        w2=1-w0-w1
        bary=np.stack([w0,w1,w2],-1)
        invz=np.sum(bary/pc[:,2],axis=-1)
        depth=1/np.maximum(invz,1e-30)
        old=zbuf[ymin:ymax+1,xmin:xmax+1]
        hit=(bary.min(-1)>=-1e-8)&(depth<old)
        if not hit.any():
            continue
        weights=bary/pc[:,2]*depth[...,None]
        color=np.einsum('hwc,cd->hwd',weights,np.asarray(colors)[face])
        normal=np.cross(pc[1]-pc[0],pc[2]-pc[0]); normal/=max(np.linalg.norm(normal),1e-12)
        # Two-sided headlight, ambient=.4/diffuse=.6: photometric assumption.
        intensity=.4+.6*abs(normal[2])
        old[hit]=depth[hit]
        rgb[ymin:ymax+1,xmin:xmax+1][hit]=255*np.clip(color[hit]*intensity,0,1)
        face_ids[ymin:ymax+1,xmin:xmax+1][hit]=fid
    zbuf[~np.isfinite(zbuf)]=0
    return dict(rgb=np.clip(rgb,0,255).astype(np.uint8),depth_m=zbuf,face_ids=face_ids)


class RaycastRenderer:
    """Native metric intersections; rays have camera-Z1, t_hit is Z, not range.

    Camera rays are transformed into the original CAD frame. Float32
    intersection arithmetic is an explicit rendering implementation choice.
    """
    def __init__(self, vertices, triangles, colors, *, texture=None, nthreads=4):
        import open3d as o3d
        self.o3d, self.triangles, self.colors = o3d, np.asarray(triangles), np.asarray(colors)
        self.texture = texture
        if texture is not None:
            uv = np.asarray(texture['uv_per_vertex'])
            if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
                raise ValueError('Texture UV count must match original CAD vertices')
        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                                         o3d.core.Tensor(np.asarray(triangles, np.uint32)))
        self.nthreads=int(nthreads)
        if self.nthreads<1:raise ValueError('Explicit bounded positive raycaster worker count required')
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=self.nthreads)
        self.scene.add_triangles(mesh)

    def render(self, R, t, K, *, height=480, width=480):
        yy, xx = np.mgrid[:height, :width]
        camera_rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K).T
        directions = camera_rays @ np.asarray(R)
        origins = np.broadcast_to(-np.asarray(t) @ np.asarray(R), directions.shape)
        hit = self.scene.cast_rays(self.o3d.core.Tensor(np.concatenate((origins, directions), -1).astype(np.float32)),nthreads=self.nthreads)
        depth = hit['t_hit'].numpy().astype(np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        ids = hit['primitive_ids'].numpy().astype(np.int64)
        rgb = np.zeros((height, width, 3), np.float64)
        bary_uv = hit['primitive_uvs'].numpy()[valid]
        bary = np.column_stack((1-bary_uv.sum(1), bary_uv))
        if self.texture is None:
            color = np.einsum('nc,ncd->nd', bary, self.colors[self.triangles[ids[valid]]])
        else:
            uv = np.einsum('nc,ncd->nd', bary, np.asarray(self.texture['uv_per_vertex'])[self.triangles[ids[valid]]])
            color = sample_texture(self.texture, uv)
        normals_camera = hit['primitive_normals'].numpy()[valid] @ np.asarray(R).T
        intensity = .4 + .6 * np.abs(normals_camera[:, 2])
        rgb[valid] = color * intensity[:, None] * 255
        depth[~valid] = 0
        ids[~valid] = -1
        return dict(rgb=np.clip(rgb, 0, 255).astype(np.uint8), depth_m=depth, face_ids=ids)
