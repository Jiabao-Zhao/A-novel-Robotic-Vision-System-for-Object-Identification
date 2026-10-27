"""Post-hoc visual evidence for frozen pin failures; no model or score changes."""
import base64
import hashlib
import html
import io
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
REPO = EXP.parents[1]
FONT = Path('C:/Windows/Fonts/arial.ttf')
SELECTED = [('N00', 'RGOCG8-50_8mm'), ('N04', 'KET8_Square_8mm'),
            ('N06', 'RGOCG12-50_12mm')]


def read(name):
    return json.loads((EXP / name).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def label(name):
    if name.startswith('RGOCG'):
        return name.split('_')[-1].replace('mm', ' mm') + ' round pin'
    if name.startswith('KET'):
        return name.split('_')[-1].replace('mm', ' mm') + ' rectangular pin'
    return name.replace('_', ' ')


def font(size=20):
    return ImageFont.truetype(str(FONT), size=size)


def lines(draw, text, xy, width, size=20, fill='#25344a', spacing=5):
    x, y = xy
    if '\n' in str(text):
        for paragraph in str(text).split('\n'):
            y = lines(draw, paragraph, (x, y), width, size, fill, spacing)
        return y
    words = str(text).split()
    line = ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if draw.textlength(candidate, font=font(size)) > width and line:
            draw.text((x, y), line, font=font(size), fill=fill)
            y += size + spacing
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=font(size), fill=fill)
        y += size + spacing
    return y


def letterbox(image, size, background='#edf1f5'):
    image = image.copy().convert('RGB')
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, background)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def data_uri(image, size=(360, 300)):
    image = letterbox(image, size)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def main():
    records = read('data/input_records.json')
    descriptions = {(row['input_id'], row['mode']): row for row in read('descriptions.json')}
    decisions = read('analysis/decisions.json')
    failures = [row for row in decisions if row['method'] == 'clip_full_dino'
                and row['outcome'] == 'wrong' and row['same_family_wrong_size']]
    assert len(failures) == 28
    truth = {(row['state'], row['truth_proposal_id']): row['target']
             for row in decisions if row['method'] == 'dino'}
    candidates = {(row['state'], row['target'], row['proposal_id']): row
                  for row in read('clip/candidate_scores.json')}
    shortlists = {(row['state'], row['target']): row for row in read('clip/shortlist.json')
                  if row['method'] == 'clip_full'}
    crops = {(row['state'], row['proposal_id']): (index, row)
             for index, row in enumerate(records) if row['kind'] == 'crop'}
    cads = {(row['target'], row['view_index']): (index, row)
            for index, row in enumerate(records) if row['kind'] == 'cad'}
    prepared = np.load(EXP / 'dino/inputs_and_masks.npz')
    assert np.array_equal(prepared['encoded_record_indices'], np.arange(len(records)))
    source_paths = [EXP / name for name in ('analysis/decisions.json',
        'clip/candidate_scores.json', 'clip/shortlist.json', 'dino/scores.json',
        'descriptions.json', 'data/input_records.json', 'dino/input_records.json',
        'dino/inputs_and_masks.npz', 'protocol.json')]
    source_paths += [REPO / row['file'] for row in records]
    source_paths += [REPO / name for name in read('protocol.json')['protected']]
    source_paths = sorted(set(source_paths))
    frozen = {str(path.relative_to(REPO)).replace('\\', '/'): digest(path)
              for path in source_paths}
    for row in records:
        assert frozen[row['file']] == row['sha256']

    def crop_image(state, pid):
        return Image.open(REPO / crops[state, pid][1]['file']).convert('RGB')

    def cad_image(target, view):
        return Image.open(REPO / cads[target, view][1]['file']).convert('RGB')

    def shape_text(input_id):
        return '\n'.join(mode + ': ' + descriptions[input_id, mode]['fields']['SHAPE']
                         for mode in ('keywords', 'phrases'))

    def exact_text(input_id):
        return ''.join('<b>' + mode + '</b><pre>' +
            html.escape(descriptions[input_id, mode]['raw_output']) + '</pre>'
            for mode in ('keywords', 'phrases'))

    def evidence(row):
        state, target, true_pid = row['state'], row['target'], row['truth_proposal_id']
        selected = row['selected_ids'][0]
        shortlist = shortlists[state, target]
        ordered = shortlist['ranked_candidate_ids']
        items = []
        for pid in ordered[:5] + ([] if true_pid in ordered[:5] else [true_pid]):
            candidate = candidates[state, target, pid]
            items.append(dict(proposal_id=pid, product=truth[state, pid],
                role='TRUE' if pid == true_pid else ('SELECTED WRONG' if pid == selected else 'candidate'),
                clip_rank=ordered.index(pid) + 1,
                clip_score=candidate['clip_full']['score'],
                clip_components=candidate['clip_full']['components'],
                clip_view=candidate['clip_full']['selected_CLIP_CAD_view'],
                dino_view=candidate['dino_selected_CAD_view'],
                dino_global=candidate['dino_global_score'],
                dino_patch=candidate['dino_patch_score'], dino_final=candidate['dino_score']))
        return dict(state=state, target=target, truth_proposal_id=true_pid,
                    selected_proposal_id=selected, predicted_product=truth[state, selected],
                    truth_clip_rank=ordered.index(true_pid) + 1,
                    truth_in_top5=row['truth_in_top5'], rows=items)

    cases = [evidence(row) for row in failures]
    by_key = {(case['state'], case['target']): case for case in cases}
    # Compact review: same target CAD may select a different view for each crop.
    overview = Image.new('RGB', (1800, 7 * 410 + 110), 'white')
    draw = ImageDraw.Draw(overview)
    draw.text((25, 15), 'All 28 failed pin cases: true crop remains in top five in 27 cases', font=font(29), fill='#13263e')
    draw.text((25, 60), 'Each card: target CAD for true crop / true crop / target CAD for wrong crop / wrong crop. Identities are post-hoc audit labels.', font=font(18), fill='#44556a')
    for i, case in enumerate(cases):
        x, y = (i % 4) * 450 + 12, (i // 4) * 410 + 110
        draw.rounded_rectangle((x, y, x + 426, y + 395), radius=9, outline='#cbd4df', width=2)
        lines(draw, case['state'] + ' | ' + label(case['target']), (x + 10, y + 9), 410, 21)
        true_pid, wrong_pid = case['truth_proposal_id'], case['selected_proposal_id']
        true_row = candidates[case['state'], case['target'], true_pid]
        wrong_row = candidates[case['state'], case['target'], wrong_pid]
        views = [true_row['dino_selected_CAD_view'], wrong_row['dino_selected_CAD_view']]
        images = [cad_image(case['target'], views[0]), crop_image(case['state'], true_pid),
                  cad_image(case['target'], views[1]), crop_image(case['state'], wrong_pid)]
        titles = [f'CAD V{views[0]:02}', 'True crop', f'CAD V{views[1]:02}', 'Wrong crop']
        for j, (image, title) in enumerate(zip(images, titles)):
            ix, iy = x + 10 + (j % 2) * 208, y + 45 + (j // 2) * 125
            draw.text((ix, iy), title, font=font(16), fill='#33465f')
            overview.paste(letterbox(image, (198, 98)), (ix, iy + 22))
        text = f"Selected: {label(case['predicted_product'])}; true CLIP rank {case['truth_clip_rank']}"
        lines(draw, text, (x + 10, y + 300), 405, 17, '#9c2833')
        lines(draw, f"DINO true {true_row['dino_score']:.4f} | wrong {wrong_row['dino_score']:.4f}",
              (x + 10, y + 348), 405, 17)
    overview.save(OUT / 'all_pin_failures.png')

    css = '''body{font:16px Arial,sans-serif;max-width:1700px;margin:32px auto;padding:0 18px;color:#21334a;background:#f6f8fa}h1,h2{color:#102942}section{background:white;padding:22px;margin:28px 0;border:1px solid #dce2e9;border-radius:10px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #dce2e9;padding:8px;text-align:left}th{background:#eaf0f7}tr.true{background:#e8f6ef}tr.wrong{background:#fdecef}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.images img{width:100%;object-fit:contain;background:#edf1f5}.small{font-size:13px;color:#506379}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#f2f5f8;padding:10px}details{margin:12px 0;padding:8px;border:1px solid #dce2e9}a{color:#006bad}.score{font-variant-numeric:tabular-nums}'''
    chunks = ['<!doctype html><html><meta charset="utf-8"><title>Pin failure audit</title><style>' + css + '</style><body>',
        '<h1>28 pin-size failures: exact CAD, crop, text and score evidence</h1>',
        '<p>Post-hoc review of frozen eight-scene, twelve-object middle-lighting results. '
        'CLIP retrieves five crops; unchanged DINO selects among them. Both scores are uncalibrated. '
        '27/28 failed pin targets survive retrieval; N06 12 mm round pin is ranked eighth.</p>',
        '<p><b>Reading the views:</b> all reference images in a case are renders of the <em>supplied target CAD</em>, '
        'including the view matched to an incorrect-size crop. CLIP and DINO choose CAD views independently. '
        'DINO global is the mean of its top-five CAD-view CLS cosines; patch uses the displayed DINO view. '
        'Final DINO = (global + patch)/2. Four-decimal display rounding does not imply an exact tie.</p>',
        '<p>Qwen text is shown verbatim, including errors and unsupported material/shape claims. '
        'Product identities below come from completed-result audit labels; they were not score inputs.</p><ul>']
    for case in cases:
        anchor = case['state'] + '__' + case['target']
        chunks.append(f'<li><a href="#{anchor}">{case["state"]} — {html.escape(label(case["target"]))} → {html.escape(label(case["predicted_product"]))}</a></li>')
    chunks.append('</ul>')

    for case in cases:
        state, target = case['state'], case['target']
        chunks.append(f'<section id="{state}__{target}"><h2>{state} — target {html.escape(label(target))}; selected {html.escape(label(case["predicted_product"]))}</h2>')
        chunks.append(f'<p>True crop: p{case["truth_proposal_id"]:03}; CLIP rank {case["truth_clip_rank"]}. '
                      f'Top-five retention: {case["truth_in_top5"]}. Selected crop: p{case["selected_proposal_id"]:03}.</p>')
        chunks.append('<table><tr><th>CLIP rank</th><th>Crop / audit identity</th><th>Role</th><th>CLIP full</th><th>CLIP view</th><th>DINO global</th><th>DINO patch</th><th>DINO final</th><th>DINO view</th></tr>')
        for item in case['rows']:
            classname = 'true' if item['role'] == 'TRUE' else ('wrong' if item['role'] == 'SELECTED WRONG' else '')
            chunks.append(f'<tr class="{classname}"><td>{item["clip_rank"]}</td><td>p{item["proposal_id"]:03} · {html.escape(label(item["product"]))}</td><td>{item["role"]}</td>' +
                ''.join(f'<td>{item[key]:.6f}</td>' if key in ('clip_score','dino_global','dino_patch','dino_final') else f'<td>V{item[key]:02}</td>'
                        for key in ('clip_score','clip_view','dino_global','dino_patch','dino_final','dino_view')) + '</tr>')
        chunks.append('</table>')
        for item in case['rows']:
            pid, cview, dview = item['proposal_id'], item['clip_view'], item['dino_view']
            crop_id = f'crop__{state}__p{pid:03}'
            clip_id = f'cad__{target}__v{cview:02}'
            dino_id = f'cad__{target}__v{dview:02}'
            chunks.append(f'<details {"open" if item["role"] != "candidate" else ""}><summary>CLIP rank {item["clip_rank"]}: p{pid:03} · {html.escape(label(item["product"]))} · {item["role"]}</summary><div class="images">')
            for title, image, ident in ((f'Native crop p{pid:03}', crop_image(state,pid),crop_id),
                (f'Target CAD: CLIP view V{cview:02}',cad_image(target,cview),clip_id),
                (f'Target CAD: DINO view V{dview:02}',cad_image(target,dview),dino_id)):
                chunks.append('<div><h3>' + title + '</h3><img src="' + data_uri(image) + '">' + exact_text(ident) + '</div>')
            chunks.append('</div><details><summary>Actual DINO 224×224 inputs and CLIP component scores</summary><div class="images">')
            for title,index in ((f'Crop p{pid:03} DINO input',crops[state,pid][0]),
                                (f'CAD V{dview:02} DINO input',cads[target,dview][0])):
                chunks.append('<div><h3>' + title + '</h3><img src="' + data_uri(Image.fromarray(prepared['images'][index]),(224,224)) + '"></div>')
            chunks.append('<div><pre>' + html.escape(json.dumps(item['clip_components'],indent=2)) + '</pre></div></div></details></details>')
        chunks.append('</section>')
    chunks.append('</body></html>')
    (OUT / 'index.html').write_text('\n'.join(chunks),encoding='utf-8')

    for key in SELECTED:
        case = by_key[key]
        canvas = Image.new('RGB',(1800,1840),'white')
        draw = ImageDraw.Draw(canvas)
        state,target=key
        y=lines(draw,f'{state} | {label(target)} → {label(case["predicted_product"])}',(35,20),1730,38,'#142d48')
        lines(draw,'Frozen CLIP top-five retrieval → unchanged DINO selection; audit labels shown after scoring.',(35,y+4),1730,23)
        for row_index,pid in enumerate((case['truth_proposal_id'],case['selected_proposal_id'])):
            item=next(item for item in case['rows'] if item['proposal_id']==pid)
            y=140+row_index*545
            color='#1c7855' if row_index==0 else '#ab2e3b'
            lines(draw,f'{item["role"]}: {label(item["product"])} · p{pid:03} · CLIP rank {item["clip_rank"]}',(35,y),1730,29,color)
            panels=[(f'Target CAD — DINO V{item["dino_view"]:02}',cad_image(target,item['dino_view']),f'cad__{target}__v{item["dino_view"]:02}'),
                    (f'Native crop — {crops[state,pid][1]["native_crop_wh"]}',crop_image(state,pid),f'crop__{state}__p{pid:03}'),
                    (f'Target CAD — CLIP V{item["clip_view"]:02}',cad_image(target,item['clip_view']),f'cad__{target}__v{item["clip_view"]:02}')]
            for j,(title,image,ident) in enumerate(panels):
                x=35+j*580
                draw.text((x,y+47),title,font=font(23),fill='#263d57')
                canvas.paste(letterbox(image,(550,245)),(x,y+82))
                lines(draw,shape_text(ident),(x,y+337),550,20)
            score=(f'CLIP {item["clip_score"]:.6f} | DINO global {item["dino_global"]:.6f} + '
                   f'patch {item["dino_patch"]:.6f} → final {item["dino_final"]:.6f}')
            lines(draw,score,(35,y+465),1730,25,color)
        y=1255
        draw.text((35,y),'Full CLIP top five (true crop added below if excluded)',font=font(30),fill='#142d48')
        cols=[35,145,540,690,870,1060,1245,1435,1610]
        headers=['Rank','Crop / product','CLIP','CLIP view','DINO global','DINO patch','DINO final','DINO view','Role']
        for x,header in zip(cols,headers):
            draw.text((x,y+50),header,font=font(21),fill='#263d57')
        for i,item in enumerate(case['rows']):
            yy=y+95+i*46
            color='#1c7855' if item['role']=='TRUE' else '#ab2e3b' if item['role']=='SELECTED WRONG' else '#263d57'
            values=[str(item['clip_rank']),f'p{item["proposal_id"]:03} {label(item["product"])}',f'{item["clip_score"]:.4f}',f'V{item["clip_view"]:02}',f'{item["dino_global"]:.4f}',f'{item["dino_patch"]:.4f}',f'{item["dino_final"]:.4f}',f'V{item["dino_view"]:02}',item['role'].replace('SELECTED WRONG','WRONG')]
            for x,value in zip(cols,values):
                draw.text((x,yy),value,font=font(20),fill=color)
        lines(draw,'All CAD panels show the supplied target CAD. SHAPE fields are literal Qwen outputs; full COLOR / SHAPE / MATERIAL responses and every candidate are in the HTML.',(35,1685),1730,23)
        lines(draw,'DINO global averages five CAD-view CLS cosines; its patch term uses the displayed DINO view. Final=(global+patch)/2. CLIP and DINO views may differ. Scores are uncalibrated.',(35,1750),1730,22)
        canvas.save(OUT/f'{state}__{target}.png')
    checks={name:digest(REPO/name)==value for name,value in frozen.items()}
    assert all(checks.values())
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(pin_failures=len(cases),
        top5_retained=sum(case['truth_in_top5'] for case in cases),
        source_files_checked=len(checks),all_source_hashes_unchanged=True,
        new_model_calls=0,new_scores=0,selected_cases=SELECTED,
        self_contained_html=True,prepared_images_mapped_by_encoded_record_indices=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(pin_failures=len(cases),overview=str(OUT/'all_pin_failures.png'),
                         selected=SELECTED,source_files_checked=len(checks))))


if __name__=='__main__':
    main()
