"""Display official reference-loader inputs, without loading an encoder."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import run_native as trial
manifest=trial.setup()
native=trial.official('cnos')
from omegaconf import OmegaConf
dataset=native.BOPTemplate(template_dir=str(trial.ROOT/'templates42'),
    obj_ids=[r['object_id'] for r in manifest['objects']],
    processing_config=OmegaConf.create({'image_size':224}),level_templates=0,pose_distribution='all')
sheet=Image.new('RGB',(4*180,12*205),'white')
draw=ImageDraw.Draw(sheet)
for index,obj in enumerate(manifest['objects']):
    templates=dataset[index]['templates'].numpy()
    for column,view in enumerate([0,16,31,41]):
        rgb=np.clip((templates[view].transpose(1,2,0)*[.229,.224,.225]+[.485,.456,.406])*255,0,255).astype(np.uint8)
        sheet.paste(Image.fromarray(rgb).resize((174,174)),(column*180+3,index*205+26))
        draw.text((column*180+3,index*205+3),obj['target'][:23],fill='black')
sheet.save(Path(__file__).resolve().parent/'reference_encoder_inputs_preview.png')
print('Native reference encoder-input contact sheet saved.')
