"""Map official Figure 2 processing roles onto unchanged native board loops.

This reads CAD boundaries and the documented fabrication roles. It does not
change the board mesh, mount parts, or run any grasp/robot/physics experiment.
Figure 2 is oriented with horizontal approximately 384-nativeY and vertical
nativeX; the output uses that viewing orientation for direct comparison.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.runtime_antipodal_assembly_20261023.extract_goal_audit import native_mesh, top_boundaries

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/nist_official_reference_20261029'
BOARD=ROOT/'outputs/simulation/nist_task_board_1/assets/meshes/GMC_Laser_Plate_Virtual_m.stl'
DOC=ROOT/'outputs/simulation/nist_task_board_1/assets/assembly_instructions.docx'

ROLES=[
    ('round_peg_4mm','drill',4.1,[191.45,347.176]),
    ('round_peg_8mm','drill',8.1,[116.45,347.176]),
    ('round_peg_12mm','drill',12.2,[41.45,122.176]),
    ('round_peg_16mm','drill',16.5,[266.45,122.176]),
    ('M4_fixed_bolt','tap',4,[191.45,122.176]),
    ('M8_fixed_bolt','tap',8,[41.45,47.176]),
    ('M12_fixed_bolt','tap',12,[266.45,347.176]),
    ('M16_fixed_bolt','tap',16,[341.45,47.176]),
]


def _display(points):
    points=np.asarray(points)
    return np.column_stack((384-points[:,1],points[:,0]))


def audit():
    vertices,faces=native_mesh(BOARD)
    circles,other=top_boundaries(vertices,faces)
    centers=np.array([x['center_board_xy_m'] for x in circles])*1000+192
    assigned=[]
    for name,process,size,target in ROLES:
        distances=np.linalg.norm(centers-target,axis=1)
        index=int(distances.argmin())
        assert distances[index]<.05
        hole=circles[index]
        assigned.append(dict(role=name,official_processing=process,
            documented_tool_diameter_or_thread_nominal_mm=size,
            audit_hole_id=hole['audit_hole_id'],
            measured_center_source_XY_mm=centers[index].tolist(),
            center_board_relative_XY_m=hole['center_board_xy_m'],
            nominal_layout_center_source_XY_mm=target,
            source_center_measurement_difference_mm=float(distances[index]),
            unchanged_STL_fitted_diameter_mm=hole['diameter_m']*1000,
            unchanged_STL_minimum_inscribed_diameter_mm=hole['minimum_inscribed_diameter_m']*1000,
            source_boundary_XY_mm=(np.asarray(hole['top_boundary_xy_m'])*1000+192).tolist(),
            evidence='Official fabrication instructions Figure 2 red/blue processing labels; paragraph indices 9,221,236,242',
            dimensions_note='STL loop dimensions and documented fabrication tool sizes are distinct evidence; no mesh resizing or finished-hole assumption.'))
    assert len({x['audit_hole_id'] for x in assigned})==8
    corrections=[
        dict(fixture='M4_Screw',previous_source_XY_mm=[191.45,347.176],
             previous_hole_role='round_peg_4mm',corrected_source_XY_mm=[191.45,122.176],
             corrected_hole_role='M4_fixed_bolt',reason='Previous mount occupied the official red 4.1-mm drill location, not the blue M4 tapping location.'),
        dict(fixture='M12_Screw',previous_source_XY_mm=[41.45,122.176],
             previous_hole_role='round_peg_12mm',corrected_source_XY_mm=[266.45,347.176],
             corrected_hole_role='M12_fixed_bolt',reason='Previous mount occupied the official red 12.2-mm drill location, not the blue M12 tapping location.'),
        dict(part='RGOCG12-50_12mm',previous_review_audit_hole='opening_27',
             previous_hole_role='M12_fixed_bolt',corrected_review_audit_hole='opening_26',
             corrected_hole_role='round_peg_12mm',
             reason='The tighter official round-12 hole was only occupied because our M12 fixture was assigned incorrectly; it is the intended round-12 destination.'),
        dict(fixture='USB_Female',previous_source_XY_mm=[266.45,197.176],
             corrected_source_XY_mm=[41.45,272.176],
             reason='The previous scene interchanged the USB and DSUB mounting groups. The native board assembly places U09-AF-AF-B_USB_Socket at this group; official Figure 1 agrees.'),
        dict(fixture='DSUB_Female',previous_source_XY_mm=[41.45,272.176],
             corrected_source_XY_mm=[266.45,197.176],
             reason='The native board assembly places DB_Holder and SETDSUB-B-ML-PS-25 at this group. Official Figures 1 and 7 corroborate the DSUB housing identity.'),
    ]
    all_circles=[]
    for index,hole in enumerate(circles):
        selected=next((x for x in assigned if x['audit_hole_id']==hole['audit_hole_id']),None)
        role=(selected['role'] if selected else
              'USB_M3_mount' if index in [0,1] else
              'RJ45_housing_M3_mount' if index in [2,3] else
              'Waterproof_M3x35_mount' if index in [4,5,6,7] else
              'DSUB_housing_M3_mount' if index in [8,9] else
              'BNC_M3_mount' if index in [10,11,12,13] else
              'gear_plate_M6_mount' if index in [16,17,18] else
              'corner_standoff_mount' if index in [19,20,21,22] else
              'BNC_central_clearance' if index==24 else 'unassigned')
        all_circles.append(dict(audit_hole_id=hole['audit_hole_id'],role=role,
            measured_center_source_XY_mm=centers[index].tolist(),
            unchanged_STL_fitted_diameter_mm=hole['diameter_m']*1000))
    rectangles=[]
    for hole in other:
        if max(hole['extent_xy_m'])>.1: continue
        extent=np.array(hole['extent_xy_m'])*1000
        key='KET4' if max(extent)<5 else 'KET8' if max(extent)<9 else 'KET12' if max(extent)<13 else 'KET16'
        rectangles.append(dict(role=key+'_rectangular_peg',
            measured_center_source_XY_mm=(np.array(hole['center_board_xy_m'])*1000+192).tolist(),
            unchanged_STL_extent_XY_mm=extent.tolist(),
            evidence='Figure 2 prismatic openings and official BOM cross-section sizes; paragraph 237 describes bottom-side-only filing if needed.'))
    connector_groups=[]
    for role,indices,component,bbox_center in [
        ('USB_M3_mount',[0,1],'U09-AF-AF-B_USB_Socket',[41.44961352432185,272.17577729518143]),
        ('DSUB_housing_M3_mount',[8,9],'SETDSUB-B-ML-PS-25',[266.449613524322,197.4257772951814]),
    ]:
        connector_groups.append(dict(role=role,
            audit_hole_ids=[circles[i]['audit_hole_id'] for i in indices],
            measured_mounting_group_midpoint_source_XY_mm=centers[indices].mean(axis=0).tolist(),
            corroborating_native_assembly_component=component,
            native_component_bbox_center_source_XY_mm=bbox_center,
            evidence='Original board assembly component transforms, official Figure 1 assembled layout and Figure 7 DSUB housing identity. Component bounding-box centre need not equal mounting-hole midpoint.'))
    result=dict(source_board_mesh=BOARD.relative_to(ROOT).as_posix(),
        source_board_sha256=hashlib.sha256(BOARD.read_bytes()).hexdigest(),
        source_instructions=DOC.relative_to(ROOT).as_posix(),
        source_instructions_sha256=hashlib.sha256(DOC.read_bytes()).hexdigest(),
        source_figure='instructions/image2.png',
        connector_identity_figures=['instructions/image1.png','instructions/image7.png'],
        connector_identity_native_assembly_audit='outputs/nist_official_reference_20261029/native_cad_audit.json',
        coordinate_frame='Original board STL XY in millimetres; native converted centred XY=(sourceXY-192)/1000 metres',
        figure_view_mapping='display horizontal = 384-sourceY; display vertical = sourceX; vertical display axis increases downward',
        identification_method='Unique red/blue Figure2 labels matched to native boundary centres using full layout orientation. This is not nearest-diameter assignment.',
        audit_hole_IDs_are_not_official_NIST_IDs=True,
        primary_processing_roles=assigned,circular_loop_roles=all_circles,
        rectangular_loop_roles=rectangles,connector_mounting_groups=connector_groups,
        required_corrections=corrections,
        source_mesh_modified=False,no_grasp_generation=True,no_VLM=True,no_robot_or_physics=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'hole_roles.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,10))
    for hole in [*circles,*other]:
        p=np.asarray(hole['top_boundary_xy_m'])*1000+192
        p=_display(np.vstack((p,p[0])))
        ax.plot(p[:,0],p[:,1],color='#aab1ba',lw=.8)
    colors={'drill':'#c83737','tap':'#255bba'}
    for item in assigned:
        p=_display(item['source_boundary_XY_mm']); p=np.vstack((p,p[0]))
        color=colors[item['official_processing']]
        ax.fill(p[:,0],p[:,1],color=color,alpha=.28)
        ax.plot(p[:,0],p[:,1],color=color,lw=1.8)
        x,y=_display([item['measured_center_source_XY_mm']])[0]
        suffix=(f"Drill {item['documented_tool_diameter_or_thread_nominal_mm']:g} mm\nround peg" if item['official_processing']=='drill'
                else f"Tap M{item['documented_tool_diameter_or_thread_nominal_mm']:g}\nfixed bolt")
        tx=x+(32 if x<192 else -32)
        ty=y+(-20 if y<220 else 20)
        ax.annotate(suffix,xy=(x,y),xytext=(tx,ty),ha='left' if x<192 else 'right',
                    va='center',fontsize=10,color=color,
                    arrowprops=dict(arrowstyle='-',color=color,lw=.9))
    groups=[('USB',[41.45,272.176],'#398747'),
            ('Waterproof\n4 × M3 × 35 mm',[116.45,197.176],'#5e6771'),
            ('BNC',[191.45,279.676],'#398747'),
            ('DSUB housing',[266.45,197.176],'#398747'),
            ('RJ45 housing',[341.45,347.176],'#398747'),
            ('Gear plate\n3 × M6',[171.45,47.176],'#c07a13')]
    for label,xy,color in groups:
        x,y=_display([xy])[0]
        ax.text(x,y,label,ha='center',va='center',fontsize=9,color=color,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    ax.set_xlim(-12,396);ax.set_ylim(396,-12);ax.set_aspect('equal')
    ax.set_xlabel('Figure 2 view horizontal = 384 − native Y (mm)')
    ax.set_ylabel('Figure 2 view vertical = native X (mm)')
    ax.set_title('Official Task Board 1 hole roles mapped onto unchanged CAD\nRed: drilling for round pegs · Blue: tapping for fixed bolts',fontsize=14,pad=15)
    ax.grid(alpha=.12)
    fig.text(.5,.018,'Positions measured from the STL. Tool sizes come from official Figure 2;\nraw CAD loop sizes differ and are not silently replaced with finished holes.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,1])
    fig.savefig(OUTPUT/'hole_roles.png',dpi=150,facecolor='white')
    plt.close(fig)
    return result


if __name__=='__main__':
    result=audit()
    print(json.dumps(dict(primary_roles=len(result['primary_processing_roles']),
                          circular_roles=len(result['circular_loop_roles']),
                          rectangular_roles=len(result['rectangular_loop_roles']),
                          unchanged_board=True)))
