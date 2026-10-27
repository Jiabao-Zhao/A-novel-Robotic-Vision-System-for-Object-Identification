"""Contact sheets of actual, unannotated inference RGB inputs for input QA."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent


def sheet(ids,columns,title,destination):
    size=340
    rows=(len(ids)+columns-1)//columns
    result=Image.new('RGB',(columns*size,rows*(size+32)+45),'white')
    draw=ImageDraw.Draw(result)
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    draw.text((12,12),title,fill='black',font=font)
    for index,scene in enumerate(ids):
        image=Image.open(TRIAL/'inputs'/scene/'rgb.png').convert('RGB')
        image.thumbnail((size,size))
        x=(index%columns)*size;y=(index//columns)*(size+32)+45
        result.paste(image,(x,y))
        draw.text((x+8,y+size+7),scene,fill='black',font=font)
    result.save(destination)


def main():
    manifest=json.loads((TRIAL/'capture_manifest.json').read_text())
    assert manifest['complete']
    ids=[r['scene_id'] for r in manifest['scenes'] if r['condition']=='nominal']
    sheet(ids,5,'Twenty new layouts; actual inference RGB inputs',HERE/'nominal_preview.png')
    conditions=[r['scene_id'] for r in manifest['scenes'] if r['base_scene_id']=='R00']
    sheet(conditions,5,'R00 paired conditions; actual inference RGB inputs',HERE/'conditions_preview.png')


if __name__=='__main__':main()
