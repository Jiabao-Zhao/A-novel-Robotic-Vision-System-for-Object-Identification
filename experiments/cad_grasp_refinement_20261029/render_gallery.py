"""Post-launch wording wrapper around the unchanged frozen gallery renderer.

Only the figure title wording changes. Saved poses, selections, geometry and
the original renderer remain unchanged. The override and source hashes are
recorded separately because this helper was added after the original launch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from . import gallery as frozen


OLD_LABEL = 'two selected pad-center contacts'
NEW_LABEL = 'two selected finger-pad contacts'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render(run_dir, *, side_views=True, hand_path=None):
    from matplotlib.figure import Figure

    run_dir = frozen.source.local_path(run_dir)
    evidence = run_dir/'results.json'
    before = sha(evidence)
    original_source = Path(frozen.__file__).resolve()
    original_source_sha = sha(original_source)
    original_title = Figure.suptitle
    changed_titles = []

    def label_override(self, title, *args, **kwargs):
        new_title = title.replace(OLD_LABEL, NEW_LABEL) if isinstance(title, str) else title
        if new_title != title:
            changed_titles.append(dict(original=title, displayed=new_title))
        return original_title(self, new_title, *args, **kwargs)

    Figure.suptitle = label_override
    try:
        output = frozen.render(run_dir, side_views=side_views, hand_path=hand_path)
    finally:
        Figure.suptitle = original_title
    if sha(evidence) != before:
        raise AssertionError('Saved selections changed during gallery rendering')
    if sha(original_source) != original_source_sha:
        raise AssertionError('The frozen gallery source changed during rendering')
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        helper_path=str(Path(__file__).resolve()), helper_sha256=sha(__file__),
        added_after_original_launch=True,
        frozen_renderer_path=str(original_source), frozen_renderer_sha256=original_source_sha,
        source_results=str(evidence), source_results_sha256=before,
        label_override=dict(original=OLD_LABEL, displayed=NEW_LABEL,
                            changed_title_count=len(changed_titles), titles=changed_titles),
        override_scope='Matplotlib Figure.suptitle during this render call only; restored in finally',
        saved_results_unchanged=True, frozen_renderer_unchanged=True,
        pose_or_geometry_changes=False, candidate_generation_or_selection=False,
        VLM_selection_excluded=True, pickup_or_assembly_execution=False,
        no_final_assembly_views=True)
    (output/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n', encoding='utf-8')
    return output


main = render


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.cad_grasp_refinement_20261029.render_gallery RUN_DIRECTORY')
    render(sys.argv[1])
