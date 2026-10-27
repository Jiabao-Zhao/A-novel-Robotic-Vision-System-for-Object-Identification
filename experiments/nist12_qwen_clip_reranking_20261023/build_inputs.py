"""Render the approved balanced-light/original-color N00--N07 input copies.

No model calls or identity labels enter this preparation. Historical depth
localizer rectangles are reused verbatim; simulator segmentation is never read.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'data'
SOURCE = REPO / 'outputs/datasets/nist_task_board/shared_inputs/direct_cad_captures'
CAD_SOURCE = REPO / 'outputs/datasets/nist_task_board/shared_inputs/cad_reference_views'
STATES = tuple(f'N{i:02d}' for i in range(8))
CAMERA = 'robot0_eye_in_hand'
SIZE = 2560
CAD_SIZE = 448


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preview = load_module('approved_lighting_helpers', HERE / 'lighting_preview/preview_lighting.py')
cad_renderer = load_module('source_cad_helpers', REPO / 'simulation/nist_cad_views.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(REPO).as_posix()


def record_source(path):
    return dict(file=relative(path), sha256=digest(path))


def render_options():
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    options.sitegroup[:] = 0
    return options


def scene_inputs(state, historical):
    raw_source = SOURCE / 'raw' / state
    scene_source = raw_source / 'scene.xml'
    source = ET.parse(scene_source).getroot()
    old_model, old_data = preview.compile_scene(source)
    before = preview.geometry_state(old_model, old_data)
    camera_id = old_model.camera(CAMERA).id
    camera_rotation = old_data.cam_xmat[camera_id].reshape(3, 3)
    balanced = preview.balanced_scene(source, camera_rotation)
    model, data = preview.compile_scene(balanced)
    preview.compare_geometry(before, model, data)
    # A lighting-only edit must leave visual colors and all non-light XML intact.
    old_geoms = [g.attrib for g in source.findall('.//geom')]
    new_geoms = [g.attrib for g in balanced.findall('.//geom')]
    assert old_geoms == new_geoms
    np.testing.assert_array_equal(old_model.geom_rgba, model.geom_rgba)
    np.testing.assert_array_equal(old_model.cam_fovy, model.cam_fovy)
    with mujoco.Renderer(model, height=SIZE, width=SIZE) as renderer:
        renderer.update_scene(data, camera=CAMERA, scene_option=render_options())
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=CAMERA, scene_option=render_options())
        rendered_depth = renderer.render().copy()
    reference_depth = np.load(raw_source / 'depth.npy')
    np.testing.assert_array_equal(rendered_depth, reference_depth)
    focal = SIZE / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
    intrinsics = np.array([[focal, 0, SIZE / 2], [0, focal, SIZE / 2], [0, 0, 1]])
    world_T_camera = np.eye(4)
    world_T_camera[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    world_T_camera[:3, 3] = data.cam_xpos[camera_id]
    np.testing.assert_array_equal(intrinsics, np.load(raw_source / 'intrinsics.npy'))
    np.testing.assert_array_equal(world_T_camera, np.load(raw_source / 'world_T_camera.npy'))
    raw = OUT / 'raw' / state
    scene = OUT / 'scenes' / state
    raw.mkdir(parents=True)
    (scene / 'crops').mkdir(parents=True)
    ET.ElementTree(balanced).write(raw / 'scene.xml')
    for name in ('depth.npy', 'intrinsics.npy', 'world_T_camera.npy'):
        shutil.copyfile(raw_source / name, raw / name)
        assert digest(raw_source / name) == digest(raw / name)
    rgb_path = scene / 'wrist_rgb.png'
    Image.fromarray(rgb).save(rgb_path)
    if state == 'N00':
        np.testing.assert_array_equal(rgb, np.asarray(Image.open(HERE / 'lighting_preview/scene_balanced_original_colors.png')))
    localization_source = SOURCE / 'scenes' / state / 'localization_audit.json'
    localization = read(localization_source)
    crops = []
    for item in historical['crops']:
        # Whitelist observed proposal information; no post-hoc identity is copied.
        proposal_id = int(item['proposal_id'])
        roi = {key: int(item['roi'][key]) for key in ('x1', 'y1', 'x2', 'y2')}
        x1, y1, x2, y2 = (roi[key] for key in ('x1', 'y1', 'x2', 'y2'))
        crop_path = scene / 'crops' / f'candidate_{proposal_id:03d}.png'
        Image.fromarray(rgb[y1:y2 + 1, x1:x2 + 1]).save(crop_path)
        assert [x2 - x1 + 1, y2 - y1 + 1] == item['native_crop_wh']
        crops.append(dict(kind='crop', state=state, proposal_id=proposal_id,
            file=relative(crop_path), sha256=digest(crop_path), roi=roi,
            native_crop_wh=[x2 - x1 + 1, y2 - y1 + 1],
            background_removed=False, localization_method=item['localization_method']))
    assert len(crops) == 12 and len({c['proposal_id'] for c in crops}) == 12
    metadata = dict(record_state=state, rgb_file=relative(rgb_path), rgb_sha256=digest(rgb_path),
        scene_file=relative(raw / 'scene.xml'), scene_sha256=digest(raw / 'scene.xml'),
        depth_file=relative(raw / 'depth.npy'), intrinsics_file=relative(raw / 'intrinsics.npy'),
        world_T_camera_file=relative(raw / 'world_T_camera.npy'),
        plane_model_camera=localization['plane_model_camera'],
        plane_distance_threshold_m=localization['config']['plane_distance_threshold_m'],
        crops=crops, source_scene=record_source(scene_source),
        source_localization=record_source(localization_source),
        source_metric_files=[record_source(raw_source / name) for name in ('depth.npy', 'intrinsics.npy', 'world_T_camera.npy')],
        source_rois_unchanged=True, camera_and_geometry_exactly_equal=True,
        original_colors_exactly_equal=True, rendered_depth_exactly_equal=True,
        depth_max_absolute_difference_m=float(np.abs(rendered_depth - reference_depth).max()),
        lighting_condition='balanced_original_colors')
    save(scene / 'localization_geometry.json', {key: metadata[key] for key in
        ('plane_model_camera', 'plane_distance_threshold_m', 'crops')})
    print(f'{state}: rendered balanced original colors; 12 unchanged depth proposals; depth/camera exact', flush=True)
    return metadata, crops


def camera_rotation(view):
    direction = np.asarray(view['camera_direction_cad'], dtype=float)
    hint = np.array([0., 1., 0.]) if abs(direction[2]) > .99 else np.array([0., 0., 1.])
    right = np.cross(hint, direction)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(direction, right), direction])


def cad_model(item):
    mesh = cad_renderer.local_path(item['mesh_file'])
    assert digest(mesh) == item['mesh_sha256']
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(CAD_SIZE), offheight=str(CAD_SIZE))
    ET.SubElement(visual, 'headlight', **preview.HEADLIGHT)
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='part', file=str(mesh), scale='1 1 1')
    ET.SubElement(asset, 'texture', name='white', type='skybox', builtin='flat', rgb1='1 1 1', rgb2='1 1 1', width='128', height='128')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='cad_surface', type='mesh', mesh='part',
        rgba=' '.join(map(str, item['color_rgb'] + [1])), contype='0', conaffinity='0', group='1')
    view = item['views'][0]
    rotation = camera_rotation(view)
    ET.SubElement(world, 'camera', name='cad', pos=' '.join(map(str, view['camera_position_cad_m'])),
        quat=' '.join(map(str, Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])), fovy='30')
    for index, direction in enumerate(preview.directions(rotation)):
        ET.SubElement(world, 'light', name=f'preview_balanced_fill_{index}', pos='0 0 1',
            dir=' '.join(map(str, direction)), type='directional', castshadow='false',
            ambient='0 0 0', diffuse=preview.FILL_DIFFUSE, specular='.02 .02 .02')
    return preview.compile_scene(root)


def cad_inputs(catalog):
    result, records = {}, []
    for target, item in catalog.items():
        model, data = cad_model(item)
        points = cad_renderer.vertices(model, data)
        np.testing.assert_allclose(np.ptp(points, axis=0), item['native_extent_m'], atol=1e-12, rtol=0)
        camera_id = model.camera('cad').id
        target_dir = OUT / 'cad_views' / target
        target_dir.mkdir(parents=True)
        views = []
        with mujoco.Renderer(model, height=CAD_SIZE, width=CAD_SIZE) as renderer:
            for view in item['views']:
                rotation = camera_rotation(view)
                model.cam_pos[camera_id] = view['camera_position_cad_m']
                model.cam_quat[camera_id] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
                model.light_dir[:] = preview.directions(rotation)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.cam_xmat[camera_id].reshape(3, 3), rotation, atol=1e-14, rtol=0)
                renderer.update_scene(data, camera='cad', scene_option=render_options())
                rgb = renderer.render().copy()
                index = int(view['view_index'])
                path = target_dir / f'view_{index:02d}.png'
                Image.fromarray(rgb).save(path)
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera='cad', scene_option=render_options())
                seg = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                mask = (seg[:, :, 0] == 0) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                assert mask.any()
                mask_path = target_dir / f'view_{index:02d}_mask.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                preview_path = HERE / f'lighting_preview/{target}_V{index:02d}_balanced_original_colors.png'
                if preview_path.exists():
                    np.testing.assert_array_equal(rgb, np.asarray(Image.open(preview_path)))
                source_image = CAD_SOURCE / view['image']
                assert digest(source_image) == view['image_sha256']
                new_view = copy.deepcopy(view)
                new_view.update(image=relative(path), image_sha256=digest(path),
                    mask_file=relative(mask_path), mask_sha256=digest(mask_path),
                    source_image=relative(source_image), source_image_sha256=view['image_sha256'])
                views.append(new_view)
                records.append(dict(kind='cad', target=target, view_index=index,
                    file=relative(path), sha256=digest(path), mask_file=relative(mask_path), mask_sha256=digest(mask_path)))
        new_item = copy.deepcopy(item)
        new_item.update(views=views, lighting_condition='balanced_original_colors',
            headlight=preview.HEADLIGHT, fill_directions_camera=preview.FILL_DIRECTIONS_CAMERA,
            fill_diffuse=preview.FILL_DIFFUSE, shadow_casting_enabled=False,
            color_source_scene=relative(SOURCE / 'raw/N00/scene.xml'))
        result[target] = new_item
        print(f'CAD {target}: 20 native-color balanced-light renders, unchanged metric mesh/cameras', flush=True)
    return result, records


def main():
    assert not OUT.exists(), 'Data already exists; never overwrite frozen inputs.'
    protected_paths = [SOURCE / 'capture_manifest.json', CAD_SOURCE / 'cad_inputs.json',
        REPO / 'docs/cad_view_prompt.txt', HERE / 'lighting_preview/preview_lighting.py',
        HERE / 'lighting_preview/settings.json']
    protected_paths += [SOURCE / 'raw' / state / 'scene.xml' for state in STATES]
    protected = {relative(path): digest(path) for path in protected_paths}
    assert protected['docs/cad_view_prompt.txt'] == '180dc9df66ab2cbd0b9b8057a054314bc1252d74b47db9d5dc98a15c391d2019'
    historical = read(SOURCE / 'capture_manifest.json')
    catalog = read(CAD_SOURCE / 'cad_inputs.json')
    assert len(catalog) == 12 and all(len(item['views']) == 20 for item in catalog.values())
    OUT.mkdir()
    states, records = {}, []
    for state in STATES:
        states[state], crops = scene_inputs(state, historical['states'][state])
        records.extend(crops)
        # Intermediate geometry manifest lets descriptor preparation follow scenes.
        save(OUT / 'capture_manifest.json', dict(states=states, complete=False))
    cad, cad_records = cad_inputs(catalog)
    records = cad_records + records
    assert len(records) == 336
    assert protected == {relative(path): digest(path) for path in protected_paths}
    save(OUT / 'cad_inputs.json', cad)
    save(OUT / 'input_records.json', records)
    save(OUT / 'capture_manifest.json', dict(states=states, complete=True,
        lighting_condition='balanced_original_colors', scenes=list(STATES),
        objects_per_scene=12, native_crops=96, native_cad_views=240,
        crop_localization='Exact historical depth-localizer rectangles; no RGB relocalization',
        simulator_segmentation_read=False, model_inference=False,
        original_part_colors_preserved=True, classifier_feedback_used=False))
    save(OUT / 'completion.json', dict(status='passed', source_sha256=digest(Path(__file__)),
        source_inputs_unchanged=True, protected_inputs=protected,
        all_scene_geometry_camera_depth_exactly_equal=True,
        scene_N00_and_selected_CAD_exact_preview_pixel_replay=True,
        cad_native_metric_scale_and_fixed_cameras_verified=True,
        cad_silhouettes='Isolated CAD render masks only; no scene simulator masks read',
        inputs=[record_source(OUT / name) for name in ('cad_inputs.json', 'input_records.json', 'capture_manifest.json')],
        scenes=8, objects_per_scene=12, crops=96, cad_views=240,
        software=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        qwen_calls=0, clip_calls=0, dino_calls=0))
    print('COMPLETE: 8 balanced original-color scenes, 96 native crops, 240 CAD references; no model inference.', flush=True)


if __name__ == '__main__':
    main()
