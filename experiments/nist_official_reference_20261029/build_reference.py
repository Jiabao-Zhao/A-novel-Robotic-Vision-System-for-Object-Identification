"""Publish read-only source-inspection evidence; no pose/grasp generation."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/nist_official_reference_20261029'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    web = json.loads((OUTPUT / 'web_findings.json').read_text(encoding='utf-8'))
    native = json.loads((OUTPUT / 'native_cad_audit.json').read_text(encoding='utf-8'))
    instructions = json.loads((OUTPUT / 'instructions_findings.json').read_text(encoding='utf-8'))
    assert sha(ROOT / instructions['source_path']) == instructions['source_sha256']
    assert sha(ROOT / native['archive']) == native['archive_sha256']
    frozen = ROOT / 'outputs/cad_grasp_selection_20261028/runs/20261028T190844Z'
    publication = json.loads((frozen / 'publication_manifest.json').read_text(encoding='utf-8'))
    for entry in publication['files']:
        assert sha(frozen / entry['path']) == entry['sha256'], entry['path']
    launch = json.loads((frozen / 'launch_manifest.json').read_text(encoding='utf-8'))
    pinned_count = 0
    for category in ['source_sha256', 'protected_sha256']:
        for name, expected in launch[category].items():
            assert sha(ROOT / name.replace('\\', '/')) == expected, name
            pinned_count += 1

    gallery = OUTPUT / 'gallery'
    gallery.mkdir(exist_ok=True)
    figures = [
        ('Official unassembled and assembled board', '../instructions/image1.png',
         'Original Figure 1 from NIST replication instructions. The completed board provides the intended arrangement.'),
        ('Waterproof: receiver faces up', '../instructions/image8.png',
         'Original Figure 8. The female connector is secured using four M3 × 35 mm bolts. On a horizontal board, the male mates vertically from above.'),
        ('DSUB: complete connector and housing', '../instructions/image7.png',
         'Original Figure 7. Use the printed receiver housing and full connector assembly; the contact mouth faces away from the board.'),
        ('Official fabrication roles', '../instructions/image2.png',
         'Original Figure 2. Red locations are drilled for round pins; blue locations are tapped for fixed bolts. Dimensions identify finishing tools, not every raw cut diameter.'),
        ('Mapped onto the unchanged board mesh', '../hole_roles.png',
         'Our labeled source inspection. Corrected USB/DSUB identities and M4/M12 assignments; no hole geometry was altered.'),
        ('Original saved SolidWorks preview', '../native_cad/GMC_Competition_Task_Board_Virtual.SLDASM/PreviewPNG.png',
         'Preview embedded in the official native assembly. Exact saved component matrices and parser checks are linked below; this is not a simulator render.'),
    ]
    cards = ''.join(
        f'<article><h2>{html.escape(title)}</h2><a href="{path}"><img src="{path}" alt="{html.escape(title)}"></a>'
        f'<p>{html.escape(caption)}</p></article>' for title, path, caption in figures)
    sources = ''.join(f'<li><a href="{html.escape(s["url"], quote=True)}">{html.escape(s["id"])}</a></li>'
                      for s in web['primary_sources'] if not s['id'].startswith('misumi'))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Official NIST Task Board 1 reference</title>
<style>body{margin:0;background:#f5f7f9;color:#203040;font:16px system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:24px}
p,li{line-height:1.6}.notice{padding:16px;background:#fff0d2;border-left:4px solid #b97900}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
article{background:white;border:1px solid #d6dce2;border-radius:8px;padding:18px}h2{font-size:21px}img{width:100%;height:auto;object-fit:contain;max-height:850px}a{color:#126394}
@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<main><h1>Official NIST Task Board 1 reference</h1>
<p>Source inspection, October 29, 2026. Original NIST photographs and native CAD evidence, with setup corrections.</p>
<p class="notice"><strong>Grasp generation and pickup/assembly execution remain paused. VLM selection is excluded.</strong>
The earlier horizontal waterproof reference was a mounting error. USB/DSUB locations were swapped, and M4/M12 fixtures occupied pin holes.
No simulation layout or historical numerical result was changed during this source audit.</p>
<p>Native board: <strong>384 × 384 × 8.9916 mm</strong>. Underside clearance: <strong>20 mm</strong>.
Finished round-hole drill sizes: <strong>4.1, 8.1, 12.2 and 16.5 mm</strong>. Keep the user-selected <strong>12 mm</strong> pin-tip depth as our task override.</p>
<p><a href="../../../docs/nist_task_board_1_reference.md">Detailed source report</a> ·
<a href="../native_cad/placement_summary.json">Saved native placements and discrepancies</a> ·
<a href="../native_cad_audit.json">Full CAD audit and matrices</a> ·
<a href="../native_cad/offline_checks.json">Offline parser checks</a> ·
<a href="../preservation_receipt.json">Preservation receipt</a></p>
<p>Saved placements are reference configurations, not physical fit certificates. Native working-face −Z differs from simulation +Z.
Waterproof/DSUB saved placements include about 0.25 mm Y differences against plate mounting centers; match mating datums before transfer.</p>
<div class="grid">''' + cards + '</div><h2>Primary sources</h2><ul>' + sources + '</ul></main></html>'
    target = gallery / 'index.html'
    target.write_text(page, encoding='utf-8')
    checked_links = 0
    for link in re.findall(r'(?:href|src)="([^"]+)"', page):
        if link.startswith(('https:', '#')) or link.endswith('preservation_receipt.json'):
            continue
        assert (gallery / link).resolve().is_file(), link
        checked_links += 1
    old = ROOT / 'outputs/nist_mating_goals_20261028'
    preserved = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                 for p in old.glob('*.json') if p.name != 'source_inventory.json'}
    receipt = dict(scope='Official document and static native-CAD inspection only',
        date='2026-10-29', source_docx_and_cad_archive_match_recorded_hashes=True,
        frozen_grasp_publication_files_verified=len(publication['files']),
        frozen_launched_source_and_protected_inputs_verified=pinned_count,
        local_gallery_links_checked=checked_links,
        earlier_mating_review_numerical_files_retained_sha256=preserved,
        current_scene_source_sha256=sha(ROOT / 'simulation/nist_task_board_1.py'),
        grasp_generation=False, VLM=False, pickup=False, assembly=False,
        physics_steps=False, simulation_layout_changed=False,
        native_matrix_checks='native_cad/offline_checks.json',
        note='Earlier review receives visible prose supersession notices only; no numerical outcome regeneration.')
    (OUTPUT / 'preservation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    inventory = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                 for p in sorted((ROOT / 'experiments/nist_official_reference_20261029').glob('*.py'))}
    (OUTPUT / 'source_inventory.json').write_text(json.dumps(dict(
        description='Post-inspection source inventory, not a pre-run freeze', sha256=inventory), indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in [
        'frozen_grasp_publication_files_verified', 'frozen_launched_source_and_protected_inputs_verified',
        'local_gallery_links_checked', 'grasp_generation', 'simulation_layout_changed']}))


if __name__ == '__main__':
    build()
