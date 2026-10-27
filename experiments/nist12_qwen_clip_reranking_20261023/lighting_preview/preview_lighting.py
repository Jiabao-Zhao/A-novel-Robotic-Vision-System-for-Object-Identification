"""Render lighting proposals only; preserve all frozen scene and CAD inputs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCENE = REPO / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures/raw/N00/scene.xml'
CAD_SOURCE = REPO / 'outputs/datasets/nist_task_board/shared_inputs/cad_reference_views'
SIZE = 2560
CAD_SIZE = 448
GRAY = [.55, .55, .55]
CAMERA = 'robot0_eye_in_hand'
HEADLIGHT = dict(ambient='.6 .6 .6', diffuse='.25 .25 .25', specular='.02 .02 .02')
FILL_DIRECTIONS_CAMERA = [[.7, .35, -1], [-.7, .35, -1], [0, -.7, -1]]
FILL_DIFFUSE = '.12 .12 .12'
TARGETS = ('DSUB_Male', 'RGOCG16-50_16mm', 'KET16_Square_16mm', 'Gear_Large')
VIEWS = (1, 16)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compile_scene(root):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def rgb(model, data, size, camera):
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        return renderer.render().copy()


def directions(rotation):
    vectors = np.asarray(FILL_DIRECTIONS_CAMERA, dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors @ rotation.T


def balanced_scene(source, camera_rotation):
    root = copy.deepcopy(source)
    visual = root.find('visual')
    head = visual.find('headlight')
    if head is None:
        head = ET.SubElement(visual, 'headlight')
    head.attrib.update(HEADLIGHT)
    head.set('active', '1')
    world = root.find('worldbody')
    for light in list(world.findall('light')):
        world.remove(light)
    for index, direction in enumerate(directions(camera_rotation)):
        ET.SubElement(world, 'light', name=f'preview_balanced_fill_{index}',
            pos='0 0 2', dir=' '.join(map(str, direction)), type='directional',
            ambient='0 0 0', diffuse=FILL_DIFFUSE, specular='.02 .02 .02', castshadow='false')
    return root


def geometry_state(model, data):
    ids = [index for index in range(model.ngeom)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or '').startswith('nist_part_')]
    return dict(names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids],
        positions=data.geom_xpos[ids].copy(), rotations=data.geom_xmat[ids].copy(),
        camera_positions=data.cam_xpos.copy(), camera_rotations=data.cam_xmat.copy())


def compare_geometry(before, model, data):
    after = geometry_state(model, data)
    assert before['names'] == after['names']
    for key in ('positions', 'rotations', 'camera_positions', 'camera_rotations'):
        np.testing.assert_array_equal(before[key], after[key])


def scene_previews():
    source = ET.parse(SCENE).getroot()
    model, data = compile_scene(source)
    baseline_geometry = geometry_state(model, data)
    camera_id = model.camera(CAMERA).id
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    before = rgb(model, data, SIZE, CAMERA)
    Image.fromarray(before).save(HERE / 'scene_original.png')
    balanced = balanced_scene(source, camera_rotation)
    model_color, data_color = compile_scene(balanced)
    compare_geometry(baseline_geometry, model_color, data_color)
    Image.fromarray(rgb(model_color, data_color, SIZE, CAMERA)).save(HERE / 'scene_balanced_original_colors.png')
    neutral = copy.deepcopy(balanced)
    parts = []
    for geom in neutral.findall('.//geom'):
        name = geom.get('name', '')
        if name.startswith('nist_part_') and name.endswith('_visual'):
            geom.set('rgba', '.55 .55 .55 1')
            parts.append(name)
    assert len(parts) == 12
    model_neutral, data_neutral = compile_scene(neutral)
    compare_geometry(baseline_geometry, model_neutral, data_neutral)
    Image.fromarray(rgb(model_neutral, data_neutral, SIZE, CAMERA)).save(HERE / 'scene_balanced_neutral_gray.png')
    ET.ElementTree(balanced).write(HERE / 'scene_balanced_original_colors.xml')
    ET.ElementTree(neutral).write(HERE / 'scene_balanced_neutral_gray.xml')
    frozen = REPO / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures/scenes/N00/wrist_rgb.png'
    existing = np.asarray(Image.open(frozen).convert('RGB'))
    diff = np.abs(existing.astype(np.int16) - before.astype(np.int16))
    return dict(source_scene=SCENE.relative_to(REPO).as_posix(), source_scene_sha256=digest(SCENE),
        scene_label='Existing frozen twelve-product N00 scene; not the later eight-product J00 scene.',
        object_count=12, unchanged_camera=CAMERA, resolution=[SIZE, SIZE],
        original_lights=[light.attrib for light in source.findall('.//light')],
        proposal_lights=[light.attrib for light in balanced.findall('.//light')],
        proposal_headlight=HEADLIGHT, neutral_part_rgba=GRAY + [1.],
        source_capture_exact_pixel_replay=bool(np.array_equal(existing, before)),
        source_capture_replay_mean_absolute_difference=float(diff.mean()),
        source_capture_replay_max_absolute_difference=int(diff.max()),
        all_part_poses_and_camera_arrays_exactly_equal=True,
        variants=['original lighting and colors', 'balanced lighting; original colors', 'balanced lighting; neutral gray parts'])


def load_cad_renderer():
    file = REPO / 'simulation/nist_cad_views.py'
    spec = importlib.util.spec_from_file_location('preview_existing_cad_renderer', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cad_model(mesh, color, balanced):
    renderer = load_cad_renderer()
    model = renderer.make_model(mesh, color)
    if balanced:
        model.vis.headlight.ambient[:] = [.6] * 3
        model.vis.headlight.diffuse[:] = [.25] * 3
        model.vis.headlight.specular[:] = [.02] * 3
    return model


def cad_preview(identity, record, view, mode):
    renderer = load_cad_renderer()
    mesh = renderer.local_path(record['mesh_file'])
    assert digest(mesh) == record['mesh_sha256']
    color = GRAY if mode == 'balanced_neutral_gray' else record['color_rgb']
    # Add camera-relative directional fills to the same native mesh and fixed view.
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(CAD_SIZE), offheight=str(CAD_SIZE))
    attrs = HEADLIGHT if mode != 'original' else dict(ambient='.45 .45 .45', diffuse='.55 .55 .55', specular='.1 .1 .1')
    ET.SubElement(visual, 'headlight', **attrs)
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='part', file=str(mesh), scale='1 1 1')
    ET.SubElement(asset, 'texture', name='white', type='skybox', builtin='flat',
                  rgb1='1 1 1', rgb2='1 1 1', width='128', height='128')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', type='mesh', mesh='part', rgba=' '.join(map(str, color + [1])), contype='0', conaffinity='0', group='1')
    direction = np.asarray(view['camera_direction_cad'])
    hint = np.array([0., 1., 0.]) if abs(direction[2]) > .99 else np.array([0., 0., 1.])
    right = np.cross(hint, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    rotation = np.column_stack([right, up, direction])
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    ET.SubElement(world, 'camera', name='cad', pos=' '.join(map(str, view['camera_position_cad_m'])),
                  quat=' '.join(map(str, quaternion)), fovy='30')
    if mode != 'original':
        for index, light_direction in enumerate(directions(rotation)):
            ET.SubElement(world, 'light', name=f'preview_balanced_fill_{index}', pos='0 0 1',
                dir=' '.join(map(str, light_direction)), type='directional', castshadow='false',
                ambient='0 0 0', diffuse=FILL_DIFFUSE, specular='.02 .02 .02')
    model, data = compile_scene(root)
    points = renderer.vertices(model, data)
    np.testing.assert_allclose(np.ptp(points, axis=0), record['native_extent_m'], atol=1e-12, rtol=0)
    path = HERE / f'{identity}_V{view["view_index"]:02d}_{mode}.png'
    Image.fromarray(rgb(model, data, CAD_SIZE, 'cad')).save(path)
    return dict(file=path.name, mode=mode, target=identity, view_index=view['view_index'],
        source_render=(CAD_SOURCE / view['image']).relative_to(REPO).as_posix(),
        source_render_sha256=view['image_sha256'], mesh_sha256=record['mesh_sha256'],
        camera_direction_cad=view['camera_direction_cad'], camera_position_cad_m=view['camera_position_cad_m'],
        native_extent_m=record['native_extent_m'], mesh_scale=[1, 1, 1], color_rgb=color,
        headlight=attrs, fills_enabled=mode != 'original', shadow_casting_enabled=False,
        sha256=digest(path))


def main():
    assert not (HERE / 'settings.json').exists(), 'Preview already exists; preserve it.'
    catalog_path = CAD_SOURCE / 'cad_inputs.json'
    protected = {str(p): digest(p) for p in [SCENE, catalog_path, REPO / 'docs/cad_view_prompt.txt']}
    catalog = read(catalog_path)
    scene = scene_previews()
    cad = []
    for identity in TARGETS:
        for index in VIEWS:
            view, = [v for v in catalog[identity]['views'] if v['view_index'] == index]
            for mode in ('original', 'balanced_original_colors', 'balanced_neutral_gray'):
                cad.append(cad_preview(identity, catalog[identity], view, mode))
    assert protected == {str(p): digest(p) for p in [SCENE, catalog_path, REPO / 'docs/cad_view_prompt.txt']}
    save(HERE / 'settings.json', dict(preview_only=True, classification_inference=False,
        source_inputs_unchanged=True, source_hashes=protected, scene=scene, CAD=cad,
        proposed_headlight=HEADLIGHT, fill_directions_camera=FILL_DIRECTIONS_CAMERA,
        fill_diffuse=FILL_DIFFUSE, software=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        no_pixel_brightness_correction=True, source_sha256=digest(Path(__file__))))
    print('Lighting previews complete; no classification or old artifact changes.', flush=True)


if __name__ == '__main__':
    main()
