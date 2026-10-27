"""Render the fixed CAD-only authoring pool; never reads observed crops."""
import hashlib
import json
import os
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
SHARED_INPUTS = REPO / 'outputs/datasets/nist_task_board/shared_inputs'
HERE = SHARED_INPUTS / 'cad_reference_views'
BASELINE = SHARED_INPUTS / 'direct_cad_captures'
SIZE = 448
# CAD-frame camera directions: axes, cube corners, then offset polar obliques.
CORNER_EL = float(np.degrees(np.arctan(1 / np.sqrt(2))))
DIRECTIONS = [(0, 0), (0, 90), (0, 180), (0, 270), (90, 0), (-90, 0)]
DIRECTIONS += [(el, az) for el in (CORNER_EL, -CORNER_EL)
               for az in (45, 135, 225, 315)]
DIRECTIONS += [(el, az) for el in (60, -60) for az in (0, 120, 240)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value):
    text = value.replace('\\', '/')
    if os.name != 'nt':
        text = text.replace('D:/', '/mnt/d/').replace('C:/', '/mnt/c/')
    return Path(text)


def vertices(model, data):
    mid = model.geom_dataid[0]
    points = model.mesh_vert[model.mesh_vertadr[mid]:
        model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    return points @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]


def make_model(mesh, color):
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(SIZE), offheight=str(SIZE))
    ET.SubElement(visual, 'headlight', ambient='.45 .45 .45',
                  diffuse='.55 .55 .55', specular='.1 .1 .1')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='part', file=str(mesh), scale='1 1 1')
    ET.SubElement(asset, 'texture', name='white', type='skybox', builtin='flat',
                  rgb1='1 1 1', rgb2='1 1 1', width='128', height='128')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', type='mesh', mesh='part',
                  rgba=' '.join(map(str, color + [1])), contype='0', conaffinity='0')
    ET.SubElement(world, 'camera', name='cad', pos='0 0 1', fovy='30')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def review_sheet(identity, record):
    """Labels appear only on review sheets, never on authoring input renders."""
    width, image_height, caption_height = 224, 224, 40
    sheet = Image.new('RGB', (5 * width, 4 * (image_height + caption_height) + 40), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), identity + ' | CAD-only review | 20 directions', fill='black', font=font)
    for index, view in enumerate(record['views']):
        x = index % 5 * width
        y = index // 5 * (image_height + caption_height) + 40
        with Image.open(HERE / view['image']) as image:
            sheet.paste(image.resize((width, image_height), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 6, y + image_height + 3),
                  f"V{view['view_index']:02d}  el={view['elevation_deg']:.2f}  az={view['azimuth_deg']:g}",
                  fill='black', font=font)
    path = HERE / 'cad_review' / (identity + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return dict(image=path.relative_to(HERE).as_posix(), image_sha256=digest(path),
                model_input=False)


def main():
    assert len(DIRECTIONS) == 20
    assert not (HERE / 'cad_inputs.json').exists(), 'CAD input pool already frozen; do not overwrite'
    metadata_path = BASELINE / 'cad_inputs.json'
    source_scene = BASELINE / 'raw/N00/scene.xml'
    protected = [metadata_path, source_scene, REPO / 'simulation/nist_capture.py',
                 REPO / 'docs/cad_view_prompt.txt']
    before = {str(path): digest(path) for path in protected}
    baseline = read(metadata_path)
    source = ET.parse(source_scene).getroot()
    meshes = {mesh.get('name'): mesh for mesh in source.findall('./asset/mesh')}
    records = {}
    review = {}
    for identity, item in baseline.items():
        mesh = local_path(item['mesh_file'])
        assert digest(mesh) == item['mesh_sha256'], (identity, 'native mesh changed')
        source_geom = next(geom for geom in source.findall('.//geom')
            if geom.get('name') == 'nist_part_' + identity + '_visual')
        source_mesh = meshes[source_geom.get('mesh')]
        assert local_path(source_mesh.get('file')) == mesh
        assert source_mesh.get('scale', '1 1 1') == '1 1 1'
        color = np.fromstring(source_geom.get('rgba'), sep=' ')[:3].tolist()
        assert color == item['color_rgb'], (identity, 'observed/source color mismatch')
        model = make_model(mesh, color)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        points = vertices(model, data)
        extent = np.ptp(points, axis=0)
        np.testing.assert_allclose(extent, item['native_extent_m'], rtol=0, atol=1e-12)
        center = (points.min(0) + points.max(0)) / 2
        distance = float(np.linalg.norm(points - center, axis=1).max()) / np.sin(np.deg2rad(15)) * 1.18
        camera = model.camera('cad').id
        views = []
        with mujoco.Renderer(model, height=SIZE, width=SIZE) as renderer:
            for index, (elevation, azimuth) in enumerate(DIRECTIONS, 1):
                az, el = np.deg2rad([azimuth, elevation])
                direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                hint = np.array([0., 1., 0.]) if abs(direction[2]) > .99 else np.array([0., 0., 1.])
                right = np.cross(hint, direction)
                right /= np.linalg.norm(right)
                up = np.cross(direction, right)
                model.cam_pos[camera] = center + distance * direction
                model.cam_quat[camera] = Rotation.from_matrix(np.column_stack([right, up, direction])).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='cad')
                rgb = renderer.render().copy()
                assert rgb.shape == (SIZE, SIZE, 3) and int(rgb.max()) > int(rgb.min())
                path = HERE / 'cad_views' / identity / f'view_{index:02d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(path)
                views.append(dict(view_index=index, image=path.relative_to(HERE).as_posix(),
                    image_sha256=digest(path), elevation_deg=elevation, azimuth_deg=azimuth,
                    camera_direction_cad=direction.tolist(), camera_position_cad_m=model.cam_pos[camera].tolist()))
        records[identity] = dict(mesh_file=item['mesh_file'], mesh_sha256=digest(mesh),
            color_rgb=color, native_extent_m=extent.tolist(), views=views,
            color_source_scene='../direct_cad_captures/raw/N00/scene.xml',
            color_source_scene_sha256=digest(source_scene), color_source_geom=source_geom.get('name'),
            color_matches_observation=True, render_width_height=[SIZE, SIZE], native_metric_scale=True,
            mesh_scale=[1, 1, 1], aspect_preserved=True, model_visible_labels=False,
            camera_frame='CAD metric frame; camera looks toward native bounding-box center',
            framing='Bounding-sphere camera distance; no anisotropic mesh scaling',
            direction_pool='six_axes_eight_cube_corners_six_offset_polar_obliques')
        review[identity] = review_sheet(identity, records[identity])
        print('CAD', identity, len(views), 'views', flush=True)
    assert len(records) == 12 and sum(len(item['views']) for item in records.values()) == 240
    after = {str(path): digest(path) for path in protected}
    assert before == after, 'Protected historical input changed'
    save(HERE / 'cad_inputs.json', records)
    save(HERE / 'render_audit.json', dict(status='passed', cad_count=12, render_count=240,
        common_direction_pool=[dict(view_index=index, elevation_deg=el, azimuth_deg=az)
            for index, (el, az) in enumerate(DIRECTIONS, 1)],
        native_mesh_hashes_equal_baseline=True, native_extents_equal_baseline=True,
        native_mesh_scale_equal_scene=True, color_equal_scene=True,
        protected_sha256=after, renderer_sha256=digest(Path(__file__)),
        runtime=dict(python=platform.python_version(), mujoco=mujoco.__version__,
            numpy=np.__version__, platform=platform.platform(), renderer='EGL'),
        contact_sheets=review, observed_crops_read=False, model_inference=False,
        views_expanded_beyond_twenty=False))


if __name__ == '__main__':
    main()
