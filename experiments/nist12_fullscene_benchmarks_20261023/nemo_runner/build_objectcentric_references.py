"""Prepare object-centric CAD references without observing any scene regions."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import digest, path, read, relative, save


def square_alpha_crop(source):
    rgba = Image.open(source).convert('RGBA')
    alpha = np.asarray(rgba.getchannel('A'))
    yy, xx = np.nonzero(alpha > 0)
    if len(xx) == 0:
        raise ValueError(f'CAD render has no foreground: {source}')
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    left = (bbox[0] + bbox[2] - side) // 2
    top = (bbox[1] + bbox[3] - side) // 2
    square = [left, top, left + side, top + side]
    cropped = rgba.crop(square)
    black = Image.new('RGBA', cropped.size, (0, 0, 0, 255))
    result = Image.alpha_composite(black, cropped).convert('RGB')
    return result, {'foreground_bbox_xyxy': bbox, 'square_crop_xyxy': square,
                    'crop_side_pixels': side, 'source_width_height': list(rgba.size),
                    'foreground_alpha_threshold': 0, 'units': 'render-image pixels',
                    'short_axis_square_padding': 'black; no extra long-axis margin'}


def contact_sheet(references):
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    heading = ImageFont.truetype(str(font_path), 28) if font_path.exists() else font
    object_ids = [1, 3, 6, 10, 11, 12]
    cell_width, cell_height = 340, 280
    canvas = Image.new('RGB', (4 * cell_width, len(object_ids) * cell_height + 100), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), 'NeMO CAD references: raw render -> object-centric square crop', fill='black', font=heading)
    draw.text((15, 50), 'Same CAD views and colors. Alpha only; no scene crops, labels or poses.', fill='black', font=font)
    raw = read(HERE / 'references.json')
    for row, object_id in enumerate(object_ids):
        item = references['objects'][str(object_id)]
        for pair, index in enumerate((0, 16)):
            for variant, filename in enumerate((raw['objects'][str(object_id)]['images'][index],
                                                 item['images'][index])):
                column = pair * 2 + variant
                x, y = column * cell_width, row * cell_height + 100
                label = ('Raw' if variant == 0 else 'Square alpha crop')
                draw.text((x + 8, y + 2), f'{item["target"]} | view {index}', fill='black', font=font)
                draw.text((x + 8, y + 27), label, fill='black', font=font)
                image = Image.open(path(filename)).convert('RGB')
                preview = ImageOps.contain(image, (cell_width - 16, cell_height - 65))
                canvas.paste(preview, (x + (cell_width - preview.width) // 2, y + 57))
    canvas.save(HERE / 'reference_QA.png')


def main():
    source = read(HERE / 'references.json')
    destination = HERE / 'cad_objectcentric'
    references = dict(source)
    references['route'] = 'Official NeMO encoder/detector with32object-centricCAD-renderedreferences; customCADPyrender, not exactBOPPBR.'
    references['raw_references_sha256'] = digest(HERE / 'references.json')
    references['preprocessing'] = 'Smallest square enclosing isolatedCADalpha>0, black short-axis padding; nativeNeMO224loader follows.'
    references['objects'] = {}
    crop_records = []
    for object_id, item in source['objects'].items():
        images = []
        for value in item['images']:
            source_path = path(value)
            output = destination / f'obj_{int(object_id):06d}' / source_path.name
            output.parent.mkdir(parents=True, exist_ok=True)
            image, crop = square_alpha_crop(source_path)
            image.save(output)
            images.append(relative(output))
            crop_records.append({'object_id': int(object_id), 'source': value,
                                 'source_sha256': digest(source_path), 'output': relative(output),
                                 'output_sha256': digest(output), **crop})
        references['objects'][object_id] = {
            'target': item['target'], 'images': images,
            'sha256': {value: digest(path(value)) for value in images},
        }
    save(HERE / 'primary_references.json', references)
    save(HERE / 'primary_crop_records.json', crop_records)
    save(HERE / 'pre_label_protocol_correction.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Raw full CAD render did not fulfill object-centric template reference contract.',
        'primary_evidence': {
            'paper': 'https://arxiv.org/html/2602.04343v1',
            'supplement7.3': 'Encoder template images depict object with small bbox variations.',
            'figure9': 'The32synthetic T-LESS reference templates are object-centric images.',
            'official_loader': 'src/nemolib/utils.py image_to_tensor usesResize((224,224)), requiring square input to preserve shape.',
        },
        'old_raw_arm': 'Preserve all existing outputs unchanged as nemo_raw_reference_diagnostic after active process exits.',
        'primary_arm': 'nemo with same32CADviews, samecolors, samecheckpoint andnativefullscene detector; isolatedCADalpha-squarecrop only.',
        'source_of_crops': 'Isolated rendered CAD alpha only; no scene information.',
        'ground_truth_read': False, 'evaluation_performed': False,
        'selection_based_on_outcomes': False,
        'primary_references_sha256': digest(HERE / 'primary_references.json'),
        'crop_records_sha256': digest(HERE / 'primary_crop_records.json'),
    })
    contact_sheet(references)
    print(f'Prepared {len(crop_records)} object-centric CAD references and reference_QA.png.')


if __name__ == '__main__':
    main()
