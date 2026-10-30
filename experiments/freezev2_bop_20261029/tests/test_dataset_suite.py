"""Seven-set readers and bounded diagnostics; fixtures never use GT poses."""
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from freezev2_repro.data import BOPDataset, DATASETS, SPLITS, load_ply_texture, read_ply_header, sha256
from freezev2_repro.dataset_checks import iter_json_array, run_dataset_checks, scan_texture_assets


def _dataset_fixture(root, dataset, *, mismatched=False, uint16_image=False):
    model_dir = root / dataset / ('models_cad' if dataset == 'tless' else 'models')
    model_dir.mkdir(parents=True)
    (model_dir / 'models_info.json').write_text('{"1":{"diameter":100}}')
    scene = root / dataset / SPLITS.get(dataset, 'test') / '000001'
    modality, suffix = ('gray', '.tif') if dataset == 'itodd' else ('rgb', '.png')
    (scene / modality).mkdir(parents=True)
    (scene / 'depth').mkdir()
    source = np.full((3, 4), 50, np.uint8) if dataset == 'itodd' else np.full((3, 4, 3), 50, np.uint8)
    if uint16_image:
        source = np.full((3, 4), 500, np.uint16)
    image = scene / modality / ('000001' + suffix)
    Image.fromarray(source).save(image)
    raw = np.array([[0, 1234, 2345, 5000]] * (2 if mismatched else 3), np.float32 if dataset == 'itodd' else np.uint16)
    depth = scene / 'depth' / ('000001' + suffix)
    Image.fromarray(raw).save(depth)
    scale = 0.1 if dataset in {'tless', 'ycbv'} else 1.0
    camera = scene / 'scene_camera.json'
    camera.write_text(json.dumps({'1': {'cam_K': [100, 0, 2, 0, 100, 1, 0, 0, 1], 'depth_scale': scale,
                                      'cam_R_w2c': 'forbidden', 'cam_t_w2c': 'forbidden'}}))
    (scene / 'scene_gt.json').write_text('MALFORMED GT MUST NEVER BE READ')
    return {'dataset': dataset, 'scene_id': 1, 'im_id': 1,
            'rgb_path': image.relative_to(root).as_posix(), 'depth_path': depth.relative_to(root).as_posix(),
            'camera_path': camera.relative_to(root).as_posix(), 'sha256': sha256(image), 'depth_sha256': sha256(depth)}, raw, scale


@pytest.mark.parametrize('dataset', DATASETS)
def test_each_bop_split_modality_dtype_and_depth_scale(tmp_path, dataset):
    _, raw, scale = _dataset_fixture(tmp_path, dataset)
    reader = BOPDataset(tmp_path, dataset)
    frame = reader.frame(1, 1)
    assert reader.split == SPLITS.get(dataset, 'test')
    assert frame['rgb'].shape == (3, 4, 3) and frame['rgb'].dtype == np.uint8
    assert np.all(frame['rgb'] == 50)
    np.testing.assert_allclose(frame['depth_m'], raw * scale * 1e-3, rtol=1e-7)
    assert frame['source']['depth_dtype'] == ('float32' if dataset == 'itodd' else 'uint16')
    assert frame['source']['modality'] == ('gray' if dataset == 'itodd' else 'rgb')
    assert frame['source']['image_size_hw'] == [3, 4]
    assert not any(key.startswith('cam_') for key in frame)
    assert 'cam_R_w2c' not in frame['source'] and 'cam_t_w2c' not in frame['source']


def test_reader_rejects_rgb_depth_dimension_mismatch_and_intensity_retuning(tmp_path):
    root = tmp_path / 'mismatch'
    _dataset_fixture(root, 'lmo', mismatched=True)
    with pytest.raises(ValueError, match='identical image dimensions'):
        BOPDataset(root, 'lmo').frame(1, 1)
    root = tmp_path / 'uint16'
    _dataset_fixture(root, 'itodd', uint16_image=True)
    with pytest.raises(ValueError, match='no implicit intensity retuning'):
        BOPDataset(root, 'itodd').frame(1, 1)


def _ply(path, *, textured=True, binary=False, with_uv=True):
    uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
    xyz = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.]])
    header = ['ply', 'format ' + ('binary_little_endian' if binary else 'ascii') + ' 1.0']
    if textured:
        header += ['comment TextureFile checker.png']
    header += ['element vertex 3', 'property float x', 'property float y', 'property float z']
    if with_uv:
        header += ['property float texture_u', 'property float texture_v']
    header += ['element face 1', 'property list uchar int vertex_indices', 'end_header']
    values = np.column_stack([xyz, uv]) if with_uv else xyz
    with path.open('wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode())
        if binary:
            stream.write(values.astype('<f4').tobytes())
            stream.write(bytes([3]) + np.array([0, 1, 2], '<i4').tobytes())
        else:
            np.savetxt(stream, values)
            stream.write(b'3 0 1 2\n')
    return xyz, uv


@pytest.mark.parametrize('binary', [False, True])
def test_textured_ply_preserves_uv_vertices_image_and_verified_origin(tmp_path, binary):
    model = tmp_path / 'model.ply'
    xyz, uv = _ply(model, binary=binary)
    image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], np.uint8)
    Image.fromarray(image).save(tmp_path / 'checker.png')
    texture = load_ply_texture(model, xyz * 1e-3)
    np.testing.assert_array_equal(texture['uv_per_vertex'], uv)
    np.testing.assert_array_equal(texture['image_rgb'], image)
    assert texture['uv_origin'] == 'bottom_left' and texture['image_origin'] == 'top_left'
    assert texture['vertex_order_verified']
    assert texture['sha256'] == sha256(tmp_path / 'checker.png')
    with pytest.raises(ValueError, match='reordered'):
        load_ply_texture(model, xyz[::-1] * 1e-3)


def test_texture_assets_do_not_silently_degrade_to_gray(tmp_path):
    model = tmp_path / 'missing.ply'
    _ply(model)
    with pytest.raises(FileNotFoundError, match='no gray fallback'):
        load_ply_texture(model)
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(tmp_path / 'checker.png')
    _ply(model, textured=True, with_uv=False)
    with pytest.raises(ValueError, match='explicit per-vertex'):
        load_ply_texture(model)
    _ply(model, textured=False, with_uv=True)
    with pytest.raises(ValueError, match='no TextureFile'):
        load_ply_texture(model)
    _ply(model, textured=False, with_uv=False)
    assert load_ply_texture(model) is None
    assert read_ply_header(model)['vertex_count'] == 3


def test_texture_path_cannot_escape_the_model_asset_directory(tmp_path):
    model = tmp_path / 'model.ply'
    _ply(model)
    model.write_text(model.read_text().replace('TextureFile checker.png', 'TextureFile ../outside.png'))
    with pytest.raises(ValueError, match='escapes'):
        load_ply_texture(model)


@pytest.mark.parametrize('chunk_chars', [1, 7, 1024])
def test_json_array_reader_handles_record_chunk_boundaries_without_full_load(tmp_path, chunk_chars):
    values = [{'scene_id': 1, 'segmentation': {'counts': 'abc\\\"def' * 5}}, 12345, -0.012, [1, 2, 3], 'string']
    path = tmp_path / 'public.json'
    path.write_text(json.dumps(values))
    assert list(iter_json_array(path, chunk_chars=chunk_chars, max_record_chars=1024)) == values
    compressed = tmp_path / 'public.json.gz'
    with gzip.open(compressed, 'wt') as stream:
        json.dump(values, stream)
    assert list(iter_json_array(compressed, chunk_chars=chunk_chars, max_record_chars=1024)) == values


@pytest.mark.parametrize('text', ['{}', '[{"a":1},]', '[{"a":1}', '[1] trailing', '[{"a":"unterminated}]'])
def test_json_array_reader_rejects_incomplete_or_non_array_data(tmp_path, text):
    path = tmp_path / 'invalid.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_chars=3, max_record_chars=128))


def test_json_array_reader_enforces_bounded_record_memory(tmp_path):
    path = tmp_path / 'huge.json'
    path.write_text(json.dumps([{'counts': 'x' * 200}]))
    with pytest.raises(ValueError, match='memory bound'):
        list(iter_json_array(path, chunk_chars=16, max_record_chars=64))


def test_bounded_seven_dataset_diagnostic_uses_actual_fixture_files_no_models(tmp_path):
    root = tmp_path / 'data'
    rows = []
    for dataset in DATASETS:
        row, _, _ = _dataset_fixture(root, dataset)
        rows.append(row)
        directory = root / dataset / ('models_cad' if dataset == 'tless' else 'models')
        _ply(directory / 'obj_000001.ply', textured=False, with_uv=False)
    (root / 'full_manifest.json').write_text(json.dumps({'images': rows}))
    output = tmp_path / 'diagnostic'
    report = run_dataset_checks(root, output, samples_per_dataset=1, render_cads=False)
    assert report['input_integrity_passed']
    assert not report['GT_pose_access'] and not report['learned_inference_run'] and not report['accuracy_metrics_computed']
    assert list(report['datasets']) == list(DATASETS)
    assert report['texture_assets']['model_count'] == 7
    assert all(len(value['frames']) == 1 for value in report['datasets'].values())
    assert (output / 'dataset_checks.json').is_file() and (output / 'input_mapping_contact_sheet.png').is_file()
    with pytest.raises(ValueError, match='preserve'):
        run_dataset_checks(root, output, render_cads=False)
    with pytest.raises(ValueError, match='bounded'):
        run_dataset_checks(root, tmp_path / 'other', samples_per_dataset=9, render_cads=False)
