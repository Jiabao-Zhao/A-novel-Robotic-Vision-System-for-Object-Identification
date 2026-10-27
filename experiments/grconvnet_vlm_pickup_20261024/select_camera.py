"""Compare fixed virtual camera positions before freezing the new trial."""
from datetime import datetime, timezone
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from experiments.runtime_antipodal_assembly_20261023.robot_trial import RobotTrial
from experiments.runtime_antipodal_assembly_20261023.run_experiment import PARTS
from experiments.runtime_antipodal_assembly_20261023.clarification_20261023.run_8mm_runtime_followup import restore_other_part_visibility


root=Path.cwd()
out=root/'outputs/grconvnet_vlm_pickup_20261024/camera_selection'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
t=RobotTrial(root/'outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml',out,'Gear_Large')
restore_other_part_visibility(t)
camera=t.m.camera('birdview').id
t.m.cam_quat[camera]=[1,0,0,0]
t.m.cam_fovy[camera]=45
option=mujoco.MjvOption()
option.geomgroup[:]=[0,1,0,0,0,0]
option.sitegroup[:]=0
t.m.vis.global_.offwidth=max(1024,t.m.vis.global_.offwidth)
t.m.vis.global_.offheight=max(1024,t.m.vis.global_.offheight)
t.m.vis.quality.offsamples=0
rows=[]
with mujoco.Renderer(t.m,height=1024,width=1024) as renderer:
    for i,position in enumerate(((.10,0.,1.3),(.15,0.,1.3),(.20,0.,1.3))):
        t.m.cam_pos[camera]=position
        mujoco.mj_forward(t.m,t.d)
        renderer.update_scene(t.d,camera=camera,scene_option=option)
        Image.fromarray(renderer.render().copy()).save(out/f'camera_{i}.png')
        renderer.enable_segmentation_rendering()
        renderer.update_scene(t.d,camera=camera,scene_option=option)
        segmentation=renderer.render().copy()
        renderer.disable_segmentation_rendering()
        geom=segmentation[:,:,0]
        valid=(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(geom>=0)
        bodies=np.full(geom.shape,-1,dtype=int)
        bodies[valid]=t.m.geom_bodyid[geom[valid]]
        counts={part:int(np.count_nonzero(bodies==t.m.body('nist_part_'+part).id)) for part in PARTS}
        borders={part:bool(np.any((bodies==t.m.body('nist_part_'+part).id)[[0,-1],:]) or np.any((bodies==t.m.body('nist_part_'+part).id)[:,[0,-1]])) for part in PARTS}
        rows.append({'position_m':position,'fovy_deg':45,'target_mask_pixels':counts,'all14_nonempty':all(counts.values()),'target_touches_frame_border':borders})
        print(json.dumps(rows[-1]),flush=True)
(out/'selection.json').write_text(json.dumps(rows,indent=2)+'\n')
print('OUTPUT='+str(out),flush=True)
t.close()
