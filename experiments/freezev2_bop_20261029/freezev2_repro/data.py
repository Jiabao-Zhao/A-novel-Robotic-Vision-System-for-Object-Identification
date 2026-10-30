"""BOP readers deliberately whitelist inference fields; scene GT stays separate."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from pathlib import PureWindowsPath
import numpy as np
from PIL import Image

from .geometry import depth_to_metres, validate_intrinsics


SPLITS = {'tless': 'test_primesense', 'hb': 'test_primesense'}
DATASETS = ('lmo', 'tless', 'tudl', 'icbin', 'itodd', 'hb', 'ycbv')


def read_ply_header(path):
    """Read only the bounded PLY header, including explicit texture assets.

    This independent parser does not vendor a third-party PLY reader. BOP's
    downloaded meshes use scalar vertex properties and triangular faces. The
    property order is preserved so texture coordinates cannot silently become
    attached to different vertices. No face or model-GT annotation is read.
    """
    path = Path(path)
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY model: {path}')
        result = {'format': None, 'vertex_count': None, 'face_count': None,
                  'vertex_properties': [], 'texture_filename': None}
        element = None
        for _ in range(4096):
            raw = stream.readline(8192)
            if not raw or len(raw) >= 8192 or stream.tell() > 1024 * 1024:
                raise ValueError(f'Malformed or oversized PLY header: {path}')
            line = raw.decode('ascii').strip()
            pieces = line.split()
            if not pieces:
                continue
            if pieces[0] == 'format':
                result['format'] = pieces[1]
            elif pieces[:2] == ['comment', 'TextureFile']:
                name = line.split('TextureFile', 1)[1].strip()
                if not name or result['texture_filename'] is not None:
                    raise ValueError('A textured BOP PLY must name exactly one texture image')
                result['texture_filename'] = name
            elif pieces[0] == 'element':
                element = pieces[1]
                if element == 'vertex':
                    result['vertex_count'] = int(pieces[2])
                elif element == 'face':
                    result['face_count'] = int(pieces[2])
            elif pieces[0] == 'property' and element == 'vertex':
                if len(pieces) != 3:
                    raise ValueError('List-valued PLY vertex properties are unsupported; no UV approximation is made')
                result['vertex_properties'].append((pieces[1], pieces[2]))
            elif pieces[0] == 'end_header':
                result['header_bytes'] = stream.tell()
                break
        else:
            raise ValueError('PLY header has no end_header')
    if (result['format'] not in {'ascii', 'binary_little_endian', 'binary_big_endian'}
            or result['vertex_count'] is None or result['vertex_count'] < 1):
        raise ValueError('Unsupported PLY format or missing/nonpositive vertex count')
    return result


def _texture_vertices(path, header):
    """Read native XYZ and UV in original PLY row order, without faces."""
    properties = header['vertex_properties']
    names = [name for _, name in properties]
    required = ('x', 'y', 'z', 'texture_u', 'texture_v')
    if not set(required) <= set(names):
        raise ValueError('Textured CAD requires explicit per-vertex XYZ and texture_u/texture_v')
    with Path(path).open('rb') as stream:
        stream.seek(header['header_bytes'])
        if header['format'] == 'ascii':
            values = np.loadtxt(stream, dtype=np.float64, max_rows=header['vertex_count'],
                                usecols=[names.index(name) for name in required], ndmin=2)
        else:
            scalar_types = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1',
                            'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
                            'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
                            'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}
            order = '<' if header['format'] == 'binary_little_endian' else '>'
            try:
                dtype = np.dtype([(name, order + scalar_types[kind]) for kind, name in properties])
            except KeyError as exc:
                raise ValueError(f'Unsupported binary PLY scalar type: {exc}') from exc
            rows = np.fromfile(stream, dtype=dtype, count=header['vertex_count'])
            values = np.column_stack([rows[name] for name in required]).astype(np.float64)
    if values.shape != (header['vertex_count'], 5) or not np.isfinite(values).all():
        raise ValueError('Incomplete/nonfinite textured PLY vertex data')
    return values[:, :3], values[:, 3:]


def load_ply_texture(path, vertices_m=None, header=None):
    """Return original texture/UV inputs or None for an untextured model.

    ``image_rgb`` keeps PNG row order (top-left). PLY UV has bottom-left origin,
    matching the pinned official BOP renderer's explicit np.flipud(image).
    The consumer must apply that convention; this function does not bake
    approximate vertex colors. A declared missing texture is an asset blocker.
    """
    path = Path(path)
    header = read_ply_header(path) if header is None else header
    filename = header['texture_filename']
    has_uv = any(name in {'texture_u', 'texture_v'} for _, name in header['vertex_properties'])
    if filename is None:
        if has_uv:
            raise ValueError('CAD has texture UV but no TextureFile declaration; no gray fallback')
        return None
    relative = Path(filename)
    if relative.is_absolute() or PureWindowsPath(filename).drive:
        raise ValueError('CAD texture must be an explicit relative model asset')
    texture_path = (path.parent / relative).resolve()
    if not texture_path.is_relative_to(path.parent.resolve()):
        raise ValueError('CAD texture path escapes the model asset directory')
    if not texture_path.is_file():
        raise FileNotFoundError(f'Required CAD texture is missing: {texture_path}; no gray fallback')
    native_vertices, uv = _texture_vertices(path, header)
    if vertices_m is not None:
        actual = np.asarray(vertices_m)
        if actual.shape != native_vertices.shape or not np.allclose(actual, native_vertices * 1e-3, atol=1e-9, rtol=1e-7):
            raise ValueError('CAD loader reordered/changed textured PLY vertices; UV mapping is not verified')
    with Image.open(texture_path) as image:
        image_rgb = np.asarray(image.convert('RGB')).copy()
    return {'uv_per_vertex': uv, 'image_rgb': image_rgb, 'path': str(texture_path),
            'sha256': sha256(texture_path), 'uv_origin': 'bottom_left',
            'image_origin': 'top_left', 'vertex_order_verified': vertices_m is not None}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


class BOPDataset:
    def __init__(self, root, dataset, split=None):
        self.root = Path(root).resolve()
        self.dataset = dataset
        self.split = split or SPLITS.get(dataset, 'test')
        self.model_dir = self.root / dataset / ('models_cad' if dataset == 'tless' else 'models')
        self.models_info = json.loads((self.model_dir / 'models_info.json').read_text())

    def model(self, object_id):
        path = self.model_dir / f'obj_{object_id:06d}.ply'
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("CAD I/O needs existing Open3D environment; see runtime doctor") from exc
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices).copy() * 1e-3
        triangles = np.asarray(mesh.triangles).copy()
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
                or triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles)
                or not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError('Invalid BOP CAD mesh')
        colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else np.full_like(vertices, .7)
        if colors.shape != vertices.shape or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError('Invalid BOP vertex colors')
        texture = load_ply_texture(path, vertices)
        diameter_m = float(self.models_info[str(object_id)]['diameter']) * 1e-3
        if not np.isfinite(diameter_m) or diameter_m <= 0:
            raise ValueError('BOP CAD diameter must be finite and positive in native millimetres')
        return dict(vertices=vertices, triangles=triangles, colors=colors,
                    diameter_m=diameter_m, texture=texture,
                    appearance='texture' if texture is not None else ('vertex_colors' if mesh.has_vertex_colors() else 'assumed_gray_0.7'),
                    path=str(path), sha256=sha256(path), native_units='mm', internal_units='m')

    def frame(self, scene_id, image_id):
        scene = self.root / self.dataset / self.split / f'{scene_id:06d}'
        camera_path = scene / 'scene_camera.json'
        metadata = json.loads(camera_path.read_text())[str(image_id)]
        # Do not expose cam_R_w2c, cam_t_w2c or annotation fields to inference.
        K, depth_scale = validate_intrinsics(metadata['cam_K']), float(metadata['depth_scale'])
        image_paths = [scene / folder / f'{image_id:06d}{ext}' for folder in ('rgb', 'gray') for ext in ('.png', '.jpg', '.tif', '.jpeg', '.tiff')]
        rgb_path = next((p for p in image_paths if p.is_file()), None)
        depth_path = next((scene / 'depth' / f'{image_id:06d}{ext}' for ext in ('.png', '.tif', '.tiff') if (scene / 'depth' / f'{image_id:06d}{ext}').is_file()), None)
        if rgb_path is None or depth_path is None:
            raise FileNotFoundError(f'Missing existing BOP RGB/depth: {scene_id}/{image_id}')
        with Image.open(rgb_path) as image:
            image_mode = image.mode
            original_image = np.asarray(image)
            if original_image.dtype != np.uint8:
                raise ValueError('BOP RGB/gray source is not uint8; no implicit intensity retuning')
            rgb = np.asarray(image.convert('RGB')).copy()
        with Image.open(depth_path) as image:
            depth_mode, raw_depth = image.mode, np.asarray(image).copy()
        if raw_depth.ndim != 2 or raw_depth.shape != rgb.shape[:2] or not np.issubdtype(raw_depth.dtype, np.number):
            raise ValueError('BOP RGB/gray and numeric depth must have identical image dimensions')
        depth = depth_to_metres(raw_depth, depth_scale)
        return dict(rgb=rgb, depth_m=depth, K=K, depth_scale_mm=depth_scale,
                    source=dict(rgb=str(rgb_path), depth=str(depth_path), camera=str(camera_path),
                                rgb_sha256=sha256(rgb_path), depth_sha256=sha256(depth_path), camera_sha256=sha256(camera_path),
                                modality=rgb_path.parent.name, image_mode=image_mode, image_dtype=str(original_image.dtype),
                                depth_mode=depth_mode, depth_dtype=str(raw_depth.dtype), image_size_hw=list(raw_depth.shape)))


def input_inventory(root):
    root = Path(root)
    manifest = root / 'full_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Existing dataset inventory missing: {manifest}')
    d = json.loads(manifest.read_text())
    missing = []
    for row in d['images']:
        for key in ('rgb_path', 'depth_path', 'camera_path'):
            if not (root / row[key]).is_file():
                missing.append(row[key])
    return dict(images=len(d['images']), datasets=sorted(d['datasets']), missing=missing,
                manifest=str(manifest), manifest_sha256=sha256(manifest))
