"""Export the existing Panda collision components; never step the simulator."""
from pathlib import Path
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml'


def load_hand():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    joints = [model.joint(f'gripper0_finger_joint{i}').id for i in (1, 2)]
    addresses = [int(model.jnt_qposadr[j]) for j in joints]
    site = model.site('gripper0_grip_site').id
    components = []
    for opening in (0., .08):
        data.qpos[addresses] = [opening / 2, -opening / 2]
        mujoco.mj_forward(model, data)
        origin = data.site_xpos[site].copy()
        rotation = data.site_xmat[site].reshape(3, 3).copy()
        for geom in range(model.ngeom):
            name = model.geom(geom).name or ''
            if not name.startswith('gripper0_'):
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[geom])
                first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                local = model.mesh_vert[first:first+count]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                                  for z in (-1, 1)]) * model.geom_size[geom]
            else:
                raise ValueError(f'Unsupported Panda hand geometry {name}: {kind}')
            vertices = ((local @ data.geom_xmat[geom].reshape(3, 3).T
                         + data.geom_xpos[geom]) - origin) @ rotation
            if opening == 0:
                components.append(dict(name=name, vertices_zero_m=vertices.tolist(),
                                       is_pad='pad' in name, body=model.body(int(model.geom_bodyid[geom])).name))
            else:
                record = next(c for c in components if c['name'] == name)
                deltas = (vertices - np.asarray(record['vertices_zero_m'])) / opening
                if np.max(np.ptp(deltas, axis=0)) > 1e-6:
                    raise ValueError('Hand components do not translate rigidly with opening')
                record['translation_per_opening'] = deltas.mean(axis=0).tolist()
    return dict(source_scene=str(SCENE), opening_limits_m=[0., .08],
                frame='Panda gripper0_grip_site; X jaw closing, +Z approach', components=components)


def vertices(component, opening):
    return (np.asarray(component['vertices_zero_m'])
            + opening * np.asarray(component['translation_per_opening']))


if __name__ == '__main__':
    hand = load_hand()
    output = ROOT / 'outputs/cad_grasp_geometry_20261026'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'hand_model.json').write_text(json.dumps(hand, indent=2))
    for component in hand['components']:
        v = vertices(component, 0.)
        print(component['name'], 'pad=', component['is_pad'],
              'min=', v.min(axis=0), 'max=', v.max(axis=0),
              'movement=', component['translation_per_opening'])
