"""Independent convex-program diagnosis of saved BNC pad-region overlap."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'outputs/cad_surface_grasps_20261026/runs/20261026T200202Z')


def chebyshev(first,second):
    # Halfspaces from independent convex hulls; maximize an inscribed disk.
    eq=np.vstack((ConvexHull(first).equations,ConvexHull(second).equations))
    normals=eq[:,:2]
    result=linprog([0.,0.,-1.],A_ub=np.c_[normals,np.linalg.norm(normals,axis=1)],b_ub=-eq[:,2],
                   bounds=[(None,None),(None,None),(0,None)],method='highs')
    return dict(feasible=result.success,inscribed_radius_m=float(result.x[2]) if result.success else None,
                center_yz_m=result.x[:2].tolist() if result.success else None)


def inspect(candidate):
    records=candidate['pad_surface_regions']
    pose=np.asarray(candidate['world_T_grip'])
    result=dict(id=candidate['id'],valid=candidate['valid'],rejection=candidate['rejection_reason'],
                direction_source=candidate['direction_source'],opening_m=candidate['opening_m'],
                actual_pad_normal_errors_deg=candidate['actual_pad_normal_errors_deg'],
                actual_witnesses_local_m=((np.asarray(candidate['contact_points_world_m'])-pose[:3,3])@pose[:3,:3]).tolist(),
                regions=[],pair_tests=[])
    for record in records:
        if not record['polygons']:
            result['regions'].append(dict(pad=record['pad'],polygons=0))
            continue
        points=np.concatenate([p['polygon_local_m'] for p in record['polygons']])
        result['regions'].append(dict(pad=record['pad'],polygons=len(record['polygons']),
            local_yz_min_m=points[:,1:].min(axis=0).tolist(),local_yz_max_m=points[:,1:].max(axis=0).tolist(),
            triangles=[p['triangle_index'] for p in record['polygons']]))
    for a in records[0]['polygons']:
        p=np.asarray(a['polygon_local_m'])[:,1:]
        for b in records[1]['polygons']:
            q=np.asarray(b['polygon_local_m'])[:,1:]
            gap=np.maximum(0,np.maximum(p.min(axis=0)-q.max(axis=0),q.min(axis=0)-p.max(axis=0)))
            opposition=float(np.asarray(a['normal_local'])@b['normal_local'])
            row=dict(first_triangle=a['triangle_index'],second_triangle=b['triangle_index'],
                     bounding_gap_yz_m=gap.tolist(),normals_dot=opposition)
            if np.max(gap)<=1e-10:
                row.update(chebyshev(p,q))
            else:
                row.update(feasible=False,inscribed_radius_m=None)
            result['pair_tests'].append(row)
    result['independent_positive_overlap_pairs']=sum(p['feasible'] and p['inscribed_radius_m']>1e-10 for p in result['pair_tests'])
    result['minimum_pair_box_gap_m']=min((float(np.linalg.norm(p['bounding_gap_yz_m'])) for p in result['pair_tests']),default=None)
    return result


def main():
    data=[]
    for scene in ('N00','N01','N02','N07'):
        folder=RUN/'cases'/scene/'BNC_Male'
        candidates=json.loads((folder/'candidate_checks.json').read_text())
        records=[inspect(c) for c in candidates if 'pad_surface_regions' in c]
        data.append(dict(scene=scene,candidates=records))
        print(scene,[(r['id'],r['rejection'],r['independent_positive_overlap_pairs'],r['minimum_pair_box_gap_m'],
              [(q['pad'],q['polygons'],q.get('local_yz_min_m'),q.get('local_yz_max_m')) for q in r['regions']]) for r in records])
    out=RUN/'bnc_region_diagnosis.json'
    out.write_text(json.dumps(dict(method='Independent convex-hull halfspaces and linear-program inscribed disks, not contacts.py clipping',scenes=data),indent=2)+'\n')


if __name__=='__main__':
    main()
