"""Arrange exact simulator outputs into labeled review figures; no color edits."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
INK = '#162b42'
MUTED = '#526274'
MODES = ('original', 'balanced_original_colors', 'balanced_neutral_gray')
LABELS = ('Original', 'Balanced light / same colors', 'Balanced light / uniform gray')


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def text(draw, x, y, value, size=22, color=INK):
    draw.text((x, y), value, font=font(size), fill=color)


def environment():
    panel_width, gap, margin = 560, 20, 24
    # Same workspace region in all three images. Full source renders are retained.
    crop = (180, 400, 2450, 2060)
    panel_height = round(panel_width * (crop[3]-crop[1]) / (crop[2]-crop[0]))
    width = 2*margin + 3*panel_width + 2*gap
    sheet = Image.new('RGB', (width, panel_height+156), 'white')
    draw = ImageDraw.Draw(sheet)
    text(draw, margin, 16, 'Simulation lighting preview | 12-product N00 layout', 30)
    for index, (mode, label) in enumerate(zip(MODES, LABELS)):
        x = margin + index*(panel_width+gap)
        text(draw, x, 65, label, 23)
        file = HERE / ('scene_original.png' if mode == 'original' else f'scene_{mode}.png')
        with Image.open(file) as source:
            panel = source.convert('RGB').crop(crop).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x, 100))
    text(draw, margin, panel_height+119,
         'Same camera and object poses. Enlarged workspace crop. Right panel also changes part colors.', 21, MUTED)
    sheet.save(HERE / 'environment_comparison.png')


def cad():
    margin, label_width, cell, row_height = 24, 170, 360, 304
    width = 2*margin+label_width+3*cell
    sheet = Image.new('RGB', (width, 110+4*row_height+62), 'white')
    draw = ImageDraw.Draw(sheet)
    text(draw, margin, 15, 'CAD reference preview | identical meshes and camera views', 28)
    for col, label in enumerate(LABELS):
        text(draw, margin+label_width+col*cell, 66, label, 21)
    examples = [('DSUB_Male', ['D-sub', 'connector']), ('RGOCG16-50_16mm', ['Round pin', '16 mm']),
                ('KET16_Square_16mm', ['Rectangular pin', '16 mm']), ('Gear_Large', ['Large gear'])]
    for row, (target, label) in enumerate(examples):
        y = 110+row*row_height
        if row % 2:
            draw.rectangle((margin, y, width-margin, y+row_height), fill='#f3f6f8')
        for line, word in enumerate(label):
            text(draw, margin+6, y+95+line*29, word, 21)
        for col, mode in enumerate(MODES):
            with Image.open(HERE / f'{target}_V16_{mode}.png') as source:
                panel = source.convert('RGB').resize((row_height, row_height), Image.Resampling.LANCZOS)
            sheet.paste(panel, (margin+label_width+col*cell+(cell-row_height)//2, y))
    text(draw, margin, 110+4*row_height+17,
         'Center: lighting only. Right: the same lighting plus neutral-gray material. No brightness postprocessing.', 20, MUTED)
    sheet.save(HERE / 'cad_comparison.png')


if __name__ == '__main__':
    environment()
    cad()
    print('Saved environment_comparison.png and cad_comparison.png')
