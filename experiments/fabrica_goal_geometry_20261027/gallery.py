"""Render saved goal-filtered Fabrica candidates in the existing CAD grid.

Rendering never generates, filters, ranks or executes a grasp. Contact-pair
pagination is display only; the saved survivor bank remains complete.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from experiments.fabrica_original_20261027 import gallery as original


ROOT = original.ROOT
CAPTION = 'Nominal goal geometry only; insertion/release, VLM and execution not run.'


def status_note(record):
    notes = []
    region_status = str(record.get('mating_region_check_status', 'not_recorded'))
    if 'unavailable' in region_status or 'not_assessed' in region_status:
        notes.append('Endpoint only; mating-region check unavailable')
    elif region_status == 'not_recorded':
        notes.append('Mating-region status not recorded')
    outcome = original.outcome_label(record)
    if outcome:
        notes.append(outcome)
    return '; '.join(notes)


def verify_records(records):
    scenes, parts = original.validate_records(records)
    for record in records:
        pairs = original.pair_array(record)
        indices = list(record['retained_pair_indices'])
        if len(indices) != len(pairs) or len(indices) != len(set(indices)):
            raise ValueError('Repeated or missing surviving contact-pair index')
        supporting = {pose['pair_index'] for pose in record['retained_poses']}
        if supporting != set(indices):
            raise ValueError('Contact-pair sheet does not match surviving hand poses')
        lookup = dict(zip(indices, pairs))
        for pose in record['retained_poses']:
            if not np.allclose(lookup[pose['pair_index']], pose['contacts_world_m'],
                               rtol=0., atol=1e-12):
                raise ValueError('Saved contact pair differs from its surviving pose')
    return scenes, parts


def render_sheets(records, meshes, scenes, parts, output, preview_pairs=None):
    """Preserve the original grid/coordinate layout with task-specific captions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'savefig.bbox': None})
    lookup = {(r['scene'], r['part']): r for r in records}
    images = []
    for scene in scenes:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=original.IMAGE_DPI,
                                 constrained_layout=True)
        tiles, overlay_data, pair_artists = [], [], []
        for ax, part in zip(axes.flat, parts):
            record = lookup[(scene, part)]
            vertices, faces = meshes[part]
            transform = np.asarray(record['world_T_cad'], dtype=float)
            world = vertices @ transform[:3, :3].T + transform[:3, 3]
            center = (world.min(axis=0) + world.max(axis=0)) / 2
            points_mm = (world - center) * 1000
            ax.add_collection(PolyCollection(points_mm[faces][:, :, :2],
                facecolors=original.CAD_COLOR, edgecolors='none',
                antialiaseds=False, zorder=1))

            pairs = original.pair_array(record)
            hues = (.07 + np.arange(max(len(pairs), 1)) * .6180339887498949) % 1.
            colors = plt.get_cmap('turbo')(hues)
            shown = pairs if preview_pairs is None else pairs[:preview_pairs]
            tile_pairs = []
            for pair, color in zip(shown, colors):
                pair_mm = (pair - center) * 1000
                pair_artists.extend(ax.plot(pair_mm[:, 0], pair_mm[:, 1], color=color,
                                            linewidth=.8, zorder=4))
                pair_artists.append(ax.scatter(pair_mm[:, 0], pair_mm[:, 1], color=[color],
                                               s=8, edgecolors='none', zorder=5))
                tile_pairs.append((pair_mm[:, :2], color[:3]))
            radius = max(float(np.ptp(points_mm[:, :2], axis=0).max()) * .64, 2.)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect('equal')
            ax.set_axis_off()
            count = len(pairs)
            poses = original.pose_count(record)
            label = (part.replace('_', ' ') + '\n'
                     + f'{count} remaining contact ' + ('pair' if count == 1 else 'pairs')
                     + f'; {poses} gripper ' + ('pose' if poses == 1 else 'poses'))
            note = status_note(record)
            if note:
                label += '\n' + note
            ax.set_title(label, fontsize=9., pad=8,
                         color='#b33d32' if not count else '#263846')
            tiles.append(dict(part=part, contact_pairs=count, gripper_poses=poses,
                status=record.get('status'), note=note,
                mating_region_check_status=record.get('mating_region_check_status'),
                goal_geometry_note=record.get('goal_geometry_note', '')))
            overlay_data.append((ax, tile_pairs))
        explanation = ('Lines join remaining contact pairs; height and orientation are not shown.'
            if preview_pairs is None else
            f'First {preview_pairs} remaining contact pairs per part; full survivor counts are shown.')
        fig.suptitle(scene + '\n' + CAPTION + '\n' + explanation, fontsize=12)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        pixel_width, pixel_height = fig.canvas.get_width_height()
        for tile, (ax, pairs) in zip(tiles, overlay_data):
            tile['pair_overlays'] = []
            for index, (xy, rgb) in enumerate(pairs, 1):
                pixels = ax.transData.transform(xy)
                pixels[:, 1] = pixel_height - pixels[:, 1]
                tile['pair_overlays'].append(dict(pair_number=index,
                    endpoints_px=np.round(pixels, 4).tolist(),
                    color_rgb=np.rint(np.asarray(rgb) * 255).astype(int).tolist()))
        if preview_pairs is not None:
            path = output / f'{scene}_page1_preview.png'
            fig.savefig(path, dpi=original.IMAGE_DPI, facecolor='white')
            plt.close(fig)
            continue
        path = output / f'{scene}.png'
        fig.savefig(path, dpi=original.IMAGE_DPI, facecolor='white')
        for artist in pair_artists:
            artist.set_visible(False)
        base_path = output / f'{scene}_base.png'
        fig.savefig(base_path, dpi=original.IMAGE_DPI, facecolor='white')
        plt.close(fig)
        images.append(dict(scene=scene, image=path.name, base_image=base_path.name,
            pixel_width=pixel_width, pixel_height=pixel_height, tiles=tiles,
            contact_pairs=sum(t['contact_pairs'] for t in tiles),
            gripper_poses=sum(t['gripper_poses'] for t in tiles)))
        print('Goal-filtered CAD contact sheet: ' + scene, flush=True)
    return images


def gallery_html():
    html = original.HTML.replace('Fabrica CAD contact pairs',
                                 'Fabrica grasps after goal geometry checks')
    first = html.index('<p>Each colored line')
    stop = html.index('<div class="controls">', first)
    intro = '''<p>Each colored line joins one remaining contact pair on the CAD model.
These saved candidates passed the available mating-region and fully seated
gripper checks for nominal simulation goals. Connector mating regions are
unavailable: their remaining counts mean endpoint geometry only.</p>
<p>The page control shows twelve remaining contact pairs per part in saved order,
or all pairs together. A pair can support several gripper orientations; complete
pair and pose counts appear separately. Twelve is a display page size.</p>
<p class="note">Nominal goal geometry only; insertion/release, VLM and execution
not run. Views show the resting CAD from above; hand height and orientation are
not shown. This gallery performs no additional filtering or ranking.</p>
'''
    html = html[:first] + intro + html[stop:]
    return (html
        .replace('Open archived image with all contact pairs',
                 'Open archived image with all remaining contact pairs')
        .replace('All retained pairs shown; no ranking or candidate filtering.',
                 'All remaining pairs shown; no ranking or additional filtering.')
        .replace('This is pagination only; all candidates remain available.',
                 'This is pagination only; all saved survivors remain available.')
        .replace('Contact pairs</th>', 'Remaining contact pairs</th>')
        .replace("r.gripper_poses+' gripper poses in this scene. Counts do not measure pickup or assembly success.'",
                 "r.gripper_poses+' remaining gripper poses. Endpoint checks only; counts do not measure assembly success.'"))


def main(run_dir):
    run_dir = original.local_path(run_dir)
    source = run_dir / 'results.json'
    before = original.sha(source)
    records = json.loads(source.read_text(encoding='utf-8'))
    scenes, parts = verify_records(records)
    output = run_dir / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    meshes, mesh_sources = {}, {}
    for part in parts:
        record = next(r for r in records if r['part'] == part)
        path = original.local_path(record['mesh_path'])
        meshes[part] = original.read_stl(path)
        mesh_sources[part] = dict(path=str(path), sha256=original.sha(path))
    overviews = render_sheets(records, meshes, scenes, parts, output)
    render_sheets(records, meshes, [scenes[0]], parts, output,
                  preview_pairs=original.PAIRS_PER_PAGE)
    pairs = sum(len(original.pair_array(r)) for r in records)
    poses = sum(original.pose_count(r) for r in records)
    plotted_pairs = sum(len(t['pair_overlays']) for overview in overviews
                        for t in overview['tiles'])
    if plotted_pairs != pairs:
        raise ValueError('A remaining contact pair is missing from the gallery')
    for overview in overviews:
        for key in ('image', 'base_image'):
            if not (output / overview[key]).is_file():
                raise ValueError('Missing CAD sheet image')
    manifest = dict(placements=len(records), scenes=scenes, parts=parts,
        overviews=overviews, source_results=str(source), source_results_sha256=before,
        renderer_source_sha256=original.sha(__file__),
        original_renderer_source_sha256=original.sha(original.__file__),
        native_meshes=mesh_sources, caption=CAPTION,
        input_units='metres', plot_units='millimetres, centered per part for display only',
        contact_pairs=pairs, gripper_poses=poses,
        pairs_per_page=original.PAIRS_PER_PAGE, pagination_only=True,
        original_survivor_pair_order_preserved=True, every_remaining_pair_reachable=True,
        exact_pixel_overlay=True, gallery_removes_additional_candidates=False,
        no_repeated_line_per_hand_angle=True, no_CAD_simplification=True,
        no_candidate_generation_or_ranking=True, no_insertion_or_release_checks=True,
        no_VLM_calls=True, no_robot_execution=True, no_assembly_views=True,
        connector_region_check_unavailable=True,
        static_preview=f'{scenes[0]}_page1_preview.png')
    if original.sha(source) != before:
        raise ValueError('Saved filtered results changed during rendering')
    data = json.dumps(manifest, ensure_ascii=True).replace('<', '\\u003c')
    (output / 'index.html').write_text(gallery_html().replace('__DATA__', data),
                                     encoding='utf-8')
    (output / 'gallery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                                 encoding='utf-8')
    verification = dict(placements=len(records), scenes=len(scenes),
        plotted_remaining_pairs=plotted_pairs, remaining_poses=poses,
        pair_pose_support_matches=True, all_sheet_images_present=True,
        source_results_unchanged=True, browser_interaction_tested=False,
        static_preview=f'gallery/{scenes[0]}_page1_preview.png')
    (run_dir / 'gallery_verification.json').write_text(json.dumps(verification, indent=2)
                                                     + '\n', encoding='utf-8')
    print('GALLERY=' + str(output / 'index.html'), flush=True)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.fabrica_goal_geometry_20261027.gallery RUN_DIRECTORY')
    main(sys.argv[1])
