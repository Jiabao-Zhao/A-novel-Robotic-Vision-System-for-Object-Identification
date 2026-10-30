from copy import deepcopy
from pathlib import Path

from experiments.cad_grasp_selection_20261028.test_geometry import fixture,box
from .calibration import make_contact_calibration,hand_geometry_digest


def test_complete_supplied_hand_needs_no_local_historical_file(monkeypatch):
    kwargs,_=fixture();hand=kwargs['hand']
    for name,low,high,motion in [
        ('finger_left',[-.003,-.004,-.01],[0.,.004,.009],[-.5,0,0]),
        ('finger_right',[0.,-.004,-.01],[.003,.004,.009],[.5,0,0])]:
        hand['components'].append(dict(name=name,is_pad=False,
            vertices_zero_m=box(low,high)[0].tolist(),translation_per_opening=motion))
    before=deepcopy(hand)
    # Factory provenance is computed from the actual argument, not a cached
    # file belonging to one developer's installation.
    def unavailable(*args,**kwargs):raise FileNotFoundError('No historical local file')
    monkeypatch.setattr(Path,'read_bytes',unavailable)
    calibration=make_contact_calibration(hand)
    assert calibration['hand_geometry_sha256']==hand_geometry_digest(hand)
    assert calibration['provenance']['source_hand_model'] is None
    assert calibration['provenance']['source_kind'].startswith('caller-supplied')
    assert hand==before
