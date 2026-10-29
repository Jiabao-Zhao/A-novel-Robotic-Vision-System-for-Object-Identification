"""Offline equivalence and restoration checks for post-launch helpers."""
import copy
import json

import pytest

from . import report_streaming
from . import test_report as frozen_tests
from . import render_gallery


@pytest.fixture(autouse=True)
def use_streaming_audit(monkeypatch):
    monkeypatch.setattr(frozen_tests, 'module', report_streaming)


test_streaming_complete_hand_endpoint_audit = frozen_tests.test_table_audit_keeps_all_components_and_both_opening_endpoints
test_streaming_unavailable_is_not_success = frozen_tests.test_reporting_does_not_call_no_available_grasp_success
test_streaming_contacts_and_criteria = frozen_tests.test_nonempty_audit_recomputes_saved_contacts_criteria_and_world_frames


def test_title_override_is_documented_and_restored(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    source_results = tmp_path/'results.json'
    source_results.write_text('[]',encoding='utf-8')
    saved = source_results.read_bytes()
    original = Figure.suptitle
    output = tmp_path/'gallery'
    output.mkdir()
    def fake_render(*args, **kwargs):
        fig = Figure()
        fig.suptitle('Lines join the two selected pad-center contacts.')
        assert fig._suptitle.get_text() == 'Lines join the two selected finger-pad contacts.'
        return output
    monkeypatch.setattr(render_gallery.frozen, 'render', fake_render)
    render_gallery.render(tmp_path)
    assert Figure.suptitle is original
    assert source_results.read_bytes() == saved
    receipt = json.loads((output/'render_receipt.json').read_text())
    assert receipt['added_after_original_launch'] is True
    assert receipt['label_override']['changed_title_count'] == 1
    assert receipt['pose_or_geometry_changes'] is False
    def failing_render(*args, **kwargs):
        raise ValueError('Synthetic rendering failure')
    monkeypatch.setattr(render_gallery.frozen, 'render', failing_render)
    with pytest.raises(ValueError,match='Synthetic rendering failure'):
        render_gallery.render(tmp_path)
    assert Figure.suptitle is original
