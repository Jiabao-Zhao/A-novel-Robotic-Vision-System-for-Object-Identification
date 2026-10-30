"""Postprocessing of saved controlled selections with the preserved CAD renderer.

Added after benchmark launch. This wrapper changes labels and explanatory
metadata only; it never proposes, fits, checks or selects a grasp.
"""
from __future__ import annotations

from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from experiments.cad_grasp_refinement_20261029 import render_gallery as previous


CONTACT_LABEL='two selected hand contacts'
CAVEAT=('The combined policy assigns pad or finger contact before checking each pose. '
        'Each tile shows its selected mode. Pad means the original saved pad surfaces; '
        'finger means the original inward convex finger collision surfaces. The detailed '
        'visual finger has a groove bridged by that coarse collision hull. These are '
        'modeled contacts; physical contact accuracy and pickup or assembly success '
        'have not been verified.')


def render(run_dir,*,side_views=False,hand_path=None):
    """Render a 96-row primary branch, or another complete saved scene branch."""
    frozen=previous.frozen
    run_dir=frozen.source.local_path(run_dir)
    evidence=run_dir/'results.json'
    before=previous.sha(evidence)
    records=json.loads(evidence.read_text(encoding='utf-8'))
    branches={(r.get('sampler_strategy'),r.get('contact_mode'),r.get('seed_index')) for r in records}
    if len(branches)!=1:
        raise ValueError('Gallery requires one saved sampler/contact-policy/seed branch')
    if any(r.get('simulator_object_poses_read',False) for r in records):
        raise ValueError('Simulator labels/poses cannot supply the displayed observed alignment')
    original_outcome=frozen.outcome
    original_metadata=frozen.tile_metadata
    old_caption=frozen.CAPTION
    old_label=previous.NEW_LABEL

    def outcome(record):
        value=original_outcome(record)
        selected=frozen.selected_pose(record)
        if selected is not None:
            mode=selected.get('contact_mode')
            if mode not in ('pad','finger'):
                raise ValueError('Selected grasp must preserve its pad/finger contact mode')
            value+=' ('+mode+' contact)'
        return value

    def metadata(record,pair_world=None):
        result=original_metadata(record,pair_world)
        selected=frozen.selected_pose(record)
        result.update(contact_policy=record.get('contact_mode'),
            selected_contact_mode=None if selected is None else selected.get('contact_mode'),
            physical_contact_accuracy_validated=False)
        if selected is not None:
            result.update(actual_contacts_hand_m=selected.get('actual_contacts_hand_m'),
                actual_contact_component_names=selected.get('actual_contact_component_names'),
                contact_calibration_sha256=selected.get('contact_calibration_sha256'))
        return result

    frozen.outcome=outcome
    frozen.tile_metadata=metadata
    frozen.CAPTION='Geometry-selected grasps. No VLM, pickup or assembly execution. Modeled contacts; hardware contact unverified.'
    previous.NEW_LABEL=CONTACT_LABEL
    try:
        output=previous.render(run_dir,side_views=side_views,hand_path=hand_path)
    finally:
        frozen.outcome=original_outcome
        frozen.tile_metadata=original_metadata
        frozen.CAPTION=old_caption
        previous.NEW_LABEL=old_label

    manifest_path=output/'gallery_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    strategy,policy,seed_index=next(iter(branches))
    manifest.update(sampler_strategy=strategy,contact_policy=policy,seed_index=seed_index,
        contact_wording=CONTACT_LABEL,modeled_contact_caveat=CAVEAT,
        contact_mode_counts={mode:sum(tile['selected_contact_mode']==mode
            for sheet in manifest['overviews'] for tile in sheet['tiles']) for mode in ('pad','finger')},
        wrapper_added_after_benchmark_launch=True,
        wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=previous.sha(__file__),
        physical_contact_accuracy_validated=False)
    html=frozen.HTML.replace('two selected finger-pad contact points','two selected hand contact points')
    html=html.replace('A red label explains why no grasp was selected.',
        'Each selected tile identifies pad or finger contact. A red label explains why no grasp was selected.')
    html=html.replace('<div class="controls">','<p class="note">'+CAVEAT+'</p>\n<div class="controls">',1)
    packed=json.dumps(manifest,ensure_ascii=True).replace('<','\\u003c')
    (output/'index.html').write_text(html.replace('__DATA__',packed),encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if previous.sha(evidence)!=before:
        raise AssertionError('Saved numerical results changed during gallery rendering')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        helper_path=str(Path(__file__).resolve()),helper_sha256=previous.sha(__file__),
        postprocessing_source_added_after_benchmark_launch=True,
        preserved_renderer_path=str(Path(frozen.__file__).resolve()),
        preserved_renderer_sha256=previous.sha(frozen.__file__),
        preserved_wrapper_sha256=previous.sha(previous.__file__),
        source_results=str(evidence),source_results_sha256=before,
        numerical_results_unchanged=True,saved_poses_and_selections_unchanged=True,
        label_configuration_restored=True,selected_contacts_only=True,
        no_candidate_generation_or_selection=True,no_final_assembly_views=True,
        no_VLM_pickup_or_assembly_execution=True,physical_contact_accuracy_validated=False,
        output_sha256={p.name:previous.sha(p) for p in sorted(output.iterdir())
            if p.is_file() and p.name!='controlled_render_receipt.json'})
    (output/'controlled_render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return output


main=render


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_controlled_20261029.gallery PRIMARY_RESULTS_DIRECTORY')
    render(sys.argv[1])
