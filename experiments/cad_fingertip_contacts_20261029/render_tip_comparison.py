"""Render the saved one-pose height audit; no new fit or selection."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from experiments.cad_grasp_geometry_20261026.gallery import read_stl
from experiments.cad_grasp_selection_20261028.geometry import component_vertices,transform_points
from .contact_audit import ROOT,sha


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    folder=ROOT/'outputs/cad_fingertip_contacts_20261029'
    source=folder/'declared_side_grip_audit.json'
    data=json.loads(source.read_text())
    hand_path=ROOT/'outputs/cad_grasp_geometry_20261026/hand_model.json'
    hand=json.loads(hand_path.read_text())
    row=data['observed_input'];pickup=np.asarray(row['world_T_cad'])
    plane=np.asarray(row['table_plane_world']);up=plane[:3]
    vertices,faces=read_stl(ROOT/row['mesh_path'])
    part_world=transform_points(vertices,pickup)
    old,new=data['declared_proposal'],data['actual_finger_fit']
    horizontal=(pickup@np.asarray(old['cad_T_grip_site']))[:3,0]
    horizontal-=up*(horizontal@up);horizontal/=np.linalg.norm(horizontal)
    center=part_world.mean(0)
    def project(world):return np.column_stack(((world-center)@horizontal,world@up+plane[3]))*1000
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,pose,label in zip(axes,(old,new),('Previous pad contact','New finger-end contact')):
        grip=np.asarray(pose['cad_T_grip_site']);world_grip=pickup@grip
        for component in hand['components']:
            world=transform_points(component_vertices(component,pose['holding_opening_m']),world_grip)
            xy=project(world);h=ConvexHull(xy)
            ax.fill(xy[h.vertices,0],xy[h.vertices,1],color='#e2ad50' if component['is_pad'] else '#80a7c4',alpha=.85)
        xy=project(part_world)
        ax.add_collection(PolyCollection(xy[np.asarray(faces)],facecolors='#cbd2d9',edgecolors='none',zorder=3))
        contacts=project(transform_points(pose['contacts_cad_m'],pickup))
        ax.scatter(contacts[:,0],contacts[:,1],color='#285bc2',s=30,zorder=5,label='Gripping contacts')
        tip=project((pickup@np.asarray(pose['cad_T_hand']))[:3,3][None])[0]
        ax.scatter(tip[0],tip[1],marker='+',color='#a43b2f',s=80,zorder=5,label='Fingertip pose origin')
        ax.axhline(0,color='#7b695a',linewidth=1.4)
        ax.text(-17,.25,'Table',color='#7b695a',fontsize=10)
        ax.set_xlim(-18,18);ax.set_ylim(-2.5,15);ax.set_aspect('equal')
        ax.set_xlabel('Across the gripper (mm)');ax.set_ylabel('Height above table (mm)')
        value=(data['old_near_pad_tip_static_height_audit']['declared_pose_complete_hand_minimum_clearance_m']
            if pose is old else data['actual_finger_validation']['checks']['table_minimum_clearance_m'])
        ax.set_title(f'{label}\nLowest hand point: {value*1000:+.3f} mm above table')
        ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('8 mm pin: saved straight-down finger-end detail\nNew pose clears the table; it still fails contact alignment and final-board checks.',fontsize=12)
    out=folder/'declared_side_grip_comparison.png'
    fig.savefig(out,dpi=170);plt.close(fig)
    receipt=dict(source=str(source),source_sha256=sha(source),hand_sha256=sha(hand_path),
        renderer_sha256=sha(__file__),no_new_contact_fit=True,no_new_selection=True,
        new_pose_is_not_valid=True,physical_execution=False)
    (folder/'declared_side_grip_comparison_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(out)
    return out


if __name__=='__main__':render()
