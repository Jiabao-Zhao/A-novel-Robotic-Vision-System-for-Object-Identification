"""Official CNOS Pyrender onboarding for new CADs; no scene pixels are read."""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import HERE as ROOT, SOURCES, setup, path, relative, read, save, digest


def main():
    manifest = setup()
    source = SOURCES/'cnos'
    sys.path.insert(0, str(source))
    import ctypes
    import os
    library_dir = HERE/'native_libs/usr/lib/x86_64-linux-gnu'
    os.environ['LD_LIBRARY_PATH'] = str(library_dir)+':'+os.environ.get('LD_LIBRARY_PATH','')
    opengl = HERE/'native_libs/usr/lib/x86_64-linux-gnu/libOpenGL.so.0'
    if opengl.exists():
        ctypes.CDLL(str(opengl), mode=ctypes.RTLD_GLOBAL)
    glu = HERE/'native_libs/usr/lib/x86_64-linux-gnu/libGLU.so.1'
    if glu.exists():
        ctypes.CDLL(str(glu), mode=ctypes.RTLD_GLOBAL)
    import pyglet
    pyglet.options['shadow_window'] = False
    import numpy as np
    import trimesh
    import pyrender
    from PIL import Image
    from src.poses.pyrender import render
    from src.utils.trimesh_utils import as_mesh
    directory = ROOT/'templates42'
    if (directory/'completion.json').exists():
        saved = read(directory/'completion.json')
        for name, expected in saved['sha256'].items():
            assert digest(path(name)) == expected, name
        print('Existing official42 template bank verified.', flush=True)
        return
    pose_root = source/'src/poses/predefined_poses'
    ids = np.load(pose_root/'idx_all_level0_in_level2.npy')
    all_poses = np.load(pose_root/'obj_poses_level2.npy')
    poses = all_poses[ids].copy()
    assert len(ids) == len(poses) == 42
    # Native render_template_with_pyrender: zoom translation by0.4, mm->m.
    poses[:, :3, 3] *= .4/1000
    # NOCTIS's own released level0 mapping additionally refers to canonical301.
    # Keep the CNOS/SAM6D42 bank unchanged while furnishing that native extra.
    extra_ids = [301]
    render_ids = np.r_[ids, extra_ids]
    render_poses = all_poses[render_ids].copy()
    render_poses[:, :3, 3] *= .4/1000
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1.]])
    directory.mkdir(exist_ok=True)
    np.save(directory/'template_poses.npy', poses)
    records, hashes = [], {}
    for item in manifest['objects']:
        obj_id = item['object_id']
        out = directory/f'obj_{obj_id:06d}'
        ordered = directory/'rendered_order'/f'obj_{obj_id:06d}'
        out.mkdir(exist_ok=True)
        ordered.mkdir(parents=True, exist_ok=True)
        mesh = as_mesh(trimesh.load(path(item['mesh']), process=False)).copy()
        np.testing.assert_allclose(mesh.extents, item['native_extent_m'], atol=1e-8)
        color = np.r_[np.asarray(item['color_rgb'])*255, 255].astype(np.uint8)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        color_file = HERE/'color_assets'/f'obj_{obj_id:06d}.ply'
        color_file.parent.mkdir(exist_ok=True)
        if not color_file.exists():
            mesh.export(color_file)
        center = np.asarray(mesh.bounding_box.centroid)
        recenter = np.eye(4)
        recenter[:3, 3] = -center
        render(mesh=pyrender.Mesh.from_trimesh(mesh), output_dir=str(ordered),
               obj_poses=render_poses.copy(), img_size=(480, 640), intrinsic=K,
               light_itensity=.6, re_center_transform=recenter)
        for order, canonical in enumerate(render_ids):
            original = ordered/f'{order:06d}.png'
            destination = out/f'{int(canonical):06d}.png'
            shutil.copyfile(original, destination)
            image = Image.open(destination)
            assert image.mode == 'RGBA' and image.size == (640, 480)
            assert np.count_nonzero(np.asarray(image.getchannel('A')))
            hashes[relative(destination)] = digest(destination)
        records.append(dict(object_id=obj_id, target=item['target'], mesh=item['mesh'],
            mesh_sha256=digest(path(item['mesh'])), color_ply=relative(color_file),
            color_ply_sha256=digest(color_file), native_bbox_center_m=center.tolist(),
            render_recenter_transform=recenter.tolist(), color_rgb=item['color_rgb']))
        print(f'OfficialPyrender42: {item["target"]}', flush=True)
    hashes[relative(directory/'template_poses.npy')] = digest(directory/'template_poses.npy')
    save(directory/'metadata.json', dict(renderer='Pinned official CNOS src.poses.pyrender.render',
        source_revision='298d1f3366171464ca271659f0e2f7a6eb8e39b4',
        renderer_source_sha256=digest(source/'src/poses/pyrender.py'),
        onboarding_variant='Released Pyrender/CAD-only reference route, not public BOP train_pbr crops',
        views=42, index_templates=ids.tolist(), object_poses_camera_m=poses.tolist(),
        additional_native_NOCTIS_indices=extra_ids,
        original_pose_sha256=digest(pose_root/'obj_poses_level2.npy'),
        level0_index_sha256=digest(pose_root/'idx_all_level0_in_level2.npy'),
        intrinsic=K.tolist(), image_wh=[640, 480], light_intensity=.6,
        geometry_units='native meters; re-centering only for reference rendering',
        geometry_verification_meshes='unchanged original native metric CADs',
        objects=records, observed_scene_images_read=False))
    hashes[relative(directory/'metadata.json')] = digest(directory/'metadata.json')
    save(directory/'completion.json', dict(passed=True, objects=12, views_per_object=42,
        canonical_images=504, additional_native_NOCTIS_images=12, images=516,
        new_CAD_only_views=True, old20_view_bank_reused=False, sha256=hashes))


if __name__ == '__main__':
    main()
