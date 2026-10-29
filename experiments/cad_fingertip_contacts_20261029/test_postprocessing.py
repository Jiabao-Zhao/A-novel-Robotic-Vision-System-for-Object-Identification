"""Post-launch reporting tests; no grasp search, collision query or robot action."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import gallery, report


def selected_row():
    pose = np.eye(4)
    pose[2,3] = -.00749
    return dict(scene='N00',part='Synthetic',world_T_cad=np.eye(4).tolist(),
        table_plane_world=[0.,0.,1.,.1],selection=dict(status='selected',selected_grasp=dict(
            candidate_id=1,valid=True,validity={'status':'valid'},
            cad_T_grip_site=pose.tolist(),contacts_cad_m=[[-.01,0.,0.],[.01,0.,0.]],
            actual_contacts_hand_m=[[-.01,0.,.00749],[.01,0.,.00749]])))


def test_contact_height_uses_original_hand_frame_and_rejects_tampering():
    row = selected_row()
    result = report.selected_contact_measurements(row,.009249)
    assert result['contact_z_grip_site_m'] == pytest.approx([.00749,.00749])
    assert result['full_hand_distal_extent_beyond_contact_m'] == pytest.approx([.001759,.001759])
    assert result['contact_height_above_table_m'] == pytest.approx([.1,.1])
    bad = deepcopy(row)
    bad['selection']['selected_grasp']['actual_contacts_hand_m'][0][2] += .001
    with pytest.raises(AssertionError,match='disagrees'):
        report.selected_contact_measurements(bad,.009249)
    bad = deepcopy(row)
    bad['selection']['selected_grasp']['cad_T_grip_site'][2][3] += .001
    with pytest.raises(AssertionError,match='disagrees'):
        report.selected_contact_measurements(bad,.009249)


@pytest.mark.parametrize('tampered_field', ['file_hash','geometry_digest'])
def test_report_rejects_unbound_hand_calibration(tmp_path,tampered_field):
    run = tmp_path/'run'
    baseline = tmp_path/'baseline'
    run.mkdir()
    baseline.mkdir()
    rows = [dict(scene=f'N{s:02d}',part=f'Part{p:02d}',world_T_cad=np.eye(4).tolist(),
        table_plane_world=[0.,0.,1.,0.],mesh_path='synthetic.stl',selection=dict(
            status='no_valid_grasp_within_budget',selected_grasp=None,proposed_count=0,valid_count=0))
        for s in range(8) for p in range(12)]
    summary = dict(selected_placements=0,valid_poses=0,proposed_poses=0,elapsed_wall_seconds=0.)
    for folder in (run,baseline):
        (folder/'results.json').write_text(json.dumps(rows),encoding='utf-8')
        (folder/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    hand = dict(frame='synthetic grip site',opening_limits_m=[0.,.08],components=[dict(
        name='complete_synthetic_solid',is_pad=False,
        vertices_zero_m=[[-.01,-.01,-.01],[.01,.01,.01]],translation_per_opening=[0.,0.,0.])])
    hand_path = tmp_path/'hand.json'
    hand_path.write_text(json.dumps(hand),encoding='utf-8')
    calibration = dict(hand_geometry_sha256=report.hand_digest(hand),
        provenance=dict(source_hand_model_sha256=report.sha(hand_path)))
    if tampered_field == 'file_hash':
        calibration['provenance']['source_hand_model_sha256'] = '0'*64
        expected = 'unchanged original hand file'
    else:
        calibration['hand_geometry_sha256'] = '0'*64
        expected = 'component vertices'
    source = run/'contact_calibration.json'
    source.write_text(json.dumps(calibration),encoding='utf-8')
    with pytest.raises(AssertionError,match=expected):
        report.report(run,baseline_dir=baseline,calibration_path=source,hand_path=hand_path)


def test_gallery_label_override_is_restored_after_success_and_failure(tmp_path,monkeypatch):
    original = gallery.previous.NEW_LABEL
    output = tmp_path/'gallery'
    output.mkdir()
    (tmp_path/'results.json').write_text('[]',encoding='utf-8')

    def fake_render(*args,**kwargs):
        assert gallery.previous.NEW_LABEL == gallery.CONTACT_LABEL
        (output/'index.html').write_text(
            'two selected finger-pad contact points<div class="controls">',encoding='utf-8')
        return output

    monkeypatch.setattr(gallery.previous,'render',fake_render)
    gallery.render(tmp_path)
    assert gallery.previous.NEW_LABEL == original
    assert 'central groove' in (output/'index.html').read_text(encoding='utf-8')
    receipt = json.loads((output/'finger_contact_render_receipt.json').read_text())
    assert receipt['complete_original_hand_geometry'] and receipt['no_pad_extensions']

    def failed_render(*args,**kwargs):
        raise ValueError('Synthetic rendering failure')

    monkeypatch.setattr(gallery.previous,'render',failed_render)
    with pytest.raises(ValueError,match='Synthetic rendering failure'):
        gallery.render(tmp_path)
    assert gallery.previous.NEW_LABEL == original
