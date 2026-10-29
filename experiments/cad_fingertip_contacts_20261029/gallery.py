"""Saved finger-contact selections using the preserved CAD/pickup renderer.

Only labels and explanatory HTML are adapted. Every hand collision component,
saved contact, CAD pose and selection is passed to the existing renderer intact.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.cad_grasp_refinement_20261029 import render_gallery as previous


ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
CONTACT_LABEL = 'two selected contacts on the original finger collision surfaces'
CAVEAT = ('The blue contacts are on the existing convex finger collision hulls. '
          'Gold shows the original pad solids; no pads were extended. The detailed '
          'visual finger has a central groove that this coarse collision hull bridges. '
          'These are modeled contacts, not verified physical contact surfaces.')


def render(run_dir, *, side_views=True, hand_path=None):
    old_label = previous.NEW_LABEL
    previous.NEW_LABEL = CONTACT_LABEL
    try:
        output = previous.render(run_dir, side_views=side_views, hand_path=hand_path or HAND)
    finally:
        previous.NEW_LABEL = old_label
    html = output/'index.html'
    original_html_sha = previous.sha(html)
    text = html.read_text(encoding='utf-8')
    text = text.replace('two selected finger-pad contact points',
                        'two selected contact points on the finger collision surfaces')
    marker = '<div class="controls">'
    if marker not in text:
        raise ValueError('Preserved gallery HTML has no expected controls marker')
    text = text.replace(marker, '<p class="note">'+CAVEAT+'</p>\n'+marker, 1)
    html.write_text(text, encoding='utf-8')
    receipt = dict(helper=str(Path(__file__).resolve()), helper_sha256=previous.sha(__file__),
        reused_wrapper=str(Path(previous.__file__).resolve()),
        reused_wrapper_sha256=previous.sha(previous.__file__),
        reused_renderer=str(Path(previous.frozen.__file__).resolve()),
        reused_renderer_sha256=previous.sha(previous.frozen.__file__),
        source_results_sha256=previous.sha(Path(output).parent/'results.json'),
        hand_source=str(hand_path or HAND), hand_source_sha256=previous.sha(hand_path or HAND),
        title_label=CONTACT_LABEL, label_configuration_restored=True,
        html_before_explanation_sha256=original_html_sha,
        html_after_explanation_sha256=previous.sha(html),
        explanatory_caveat=CAVEAT,
        contact_frame='Saved original grip-site frame via cad_T_grip_site',
        complete_original_hand_geometry=True, no_pad_extensions=True,
        saved_poses_and_selections_unchanged=True, no_new_candidate_generation=True,
        no_final_assembly_views=True, VLM_selection_excluded=True,
        pickup_and_assembly_execution=False, physical_contact_accuracy_validated=False)
    (output/'finger_contact_render_receipt.json').write_text(
        json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return output


main = render


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_fingertip_contacts_20261029.gallery RUN_DIRECTORY')
    render(sys.argv[1])
