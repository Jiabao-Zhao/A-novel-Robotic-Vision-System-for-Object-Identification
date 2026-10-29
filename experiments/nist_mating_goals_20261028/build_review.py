"""Assemble explicit CAD goal references; no grasp or execution imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .gallery import OUTPUT, ROOT, ORDER


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid(value):
    value = np.asarray(value,float)
    assert value.shape == (4,4) and np.isfinite(value).all()
    assert np.allclose(value[3],[0,0,0,1],atol=1e-10,rtol=0)
    assert np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-8,rtol=0)
    assert abs(np.linalg.det(value[:3,:3])-1)<1e-8
    return value


def build():
    rows=[]
    for name in ['pins_review.json','connectors_review.json','gears_nut_review.json']:
        rows.extend(json.loads((OUTPUT/name).read_text())['parts'])
    rows.sort(key=lambda row:ORDER.index(row['part']))
    assert len(rows)==12 and len({row['part'] for row in rows})==12
    registered=[]
    for row in rows:
        transform=rigid(row['review_board_T_part'])
        old=rigid(row['old_board_T_part'])
        pf=rigid(row['part_mating_frame_cad'])
        rf=rigid(row['receiver_mating_frame_board'])
        d=np.asarray(row['axis_direction_cad'],float)
        assert abs(np.linalg.norm(d)-1)<1e-8
        assert np.allclose(transform[:3,:3]@d,row['axis_direction_board'],atol=1e-8)
        relationship=np.linalg.inv(rf)@transform@pf
        assert np.allclose(rf@relationship@np.linalg.inv(pf),transform,atol=1e-10)
        if row['proposed_board_T_part'] is not None:
            assert np.allclose(rigid(row['proposed_board_T_part']),transform,atol=1e-10)
        for evidence in row['evidence']:
            path=Path(evidence['path'])
            if not path.is_absolute(): path=ROOT/path
            assert sha(path)==evidence['sha256'],str(path)
        item=dict(part=row['part'],status=row['status'],
                  board_T_part=row['proposed_board_T_part'],
                  review_only_board_T_part=row['review_board_T_part'],
                  part_mesh=row['part_mesh'],receivers=row['receivers'],
                  axis_point_cad_m=pf[:3,3].tolist(),axis_direction_cad=d.tolist(),
                  part_T_part_mating=pf.tolist(),board_T_receiver_mating=rf.tolist(),
                  receiver_mating_T_part_mating_review=relationship.tolist(),
                  mating_frame_chain_is_representation_not_fit_proof=True,
                  full_3D_fit_certified=False,physical_assembly_validated=False,
                  grasp_generation_enabled=False)
        if row['part'].startswith(('KET','RGOCG')):
            m=row['measurements']
            assert abs(m['chosen_task_depth_m']-.012)<1e-12
            assert m['selected_minimum_polygon_side_clearance_m']>0
            board_top=float(rf[2,3])
            # Inserted CAD half-space is below the board's top plane at the new
            # final pose. Its sign reverses relative to the historical goals.
            n=-transform[:3,:3].T@np.array([0.,0.,1.])
            offset=board_top-transform[2,3]
            item.update(chosen_tip_depth_below_board_top_m=.012,
                        depth_source=m['depth_source'],
                        minimum_CAD_side_clearance_m=m['selected_minimum_polygon_side_clearance_m'],
                        protected_inserted_halfspace=dict(normal_cad=n.tolist(),offset_m=float(offset),
                            protected_when='normal_cad dot point_cad + offset_m >= 0',
                            note='Derived from corrected pose; historical negative-end contact masks must not be reused.'))
        registered.append(item)
    frozen=ROOT/'outputs/cad_grasp_selection_20261028/runs/20261028T190844Z'
    manifest=json.loads((frozen/'publication_manifest.json').read_text())
    for entry in manifest['files']:
        assert sha(frozen/entry['path'])==entry['sha256'],entry['path']
    launch=json.loads((frozen/'launch_manifest.json').read_text())
    for category in ['source_sha256','protected_sha256']:
        for name,expected in launch[category].items():
            assert sha(ROOT/Path(name.replace('\\','/')))==expected,name
    scope=dict(grasp_generation=False,VLM_selection=False,pickup_execution=False,
               assembly_execution=False,physics_steps=False,new_pose_estimation=False,
               simulator_object_poses_used=False)
    review=dict(scope=scope,units='metres, radians unless field states otherwise',
                frame='nominal board-centered XY with original scene world Z; not physical calibration',
                pin_depth_requirement='User explicitly chose 12 mm below board top in this review',
                original_results_preserved=True,parts=rows)
    registry=dict(scope=scope,frame=review['frame'],units='metres',
                  no_automatic_fallback_to_legacy_nominal_goals=True,
                  simulation_layout_modified=False,
                  note='This corrected reference registry is not connected to a grasp or execution runner while those stages are paused.',
                  nonnull_reference_goals=sum(r['board_T_part'] is not None for r in registered),
                  unresolved_goals=[r['part'] for r in registered if r['board_T_part'] is None],
                  parts=registered)
    (OUTPUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    (OUTPUT/'assembly_goals.json').write_text(json.dumps(registry,indent=2)+'\n')
    verification=dict(parts=12,proper_rotations_and_frame_chains_checked=True,
        directions_recomputed_from_final_rotation=True,pin_tip_depth_m=.012,
        pin_CAD_containment_and_positive_clearance_checked=True,
        source_CAD_hashes_unchanged=True,
        historical_run_manifest_files_verified=len(manifest['files']),
        historical_launch_source_and_protected_input_hashes_unchanged=True,
        no_grasp_generation_or_VLM_or_execution=True,
        full_3D_mating_fit_certified=False,
        available_reference_goals=registry['nonnull_reference_goals'],
        unresolved_goals=registry['unresolved_goals'])
    (OUTPUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    inventory={p.relative_to(ROOT).as_posix():sha(p)
               for p in sorted((ROOT/'experiments/nist_mating_goals_20261028').glob('*.py'))}
    (OUTPUT/'source_inventory.json').write_text(json.dumps(dict(
        description='Source inventory recorded after CAD review; not claimed as a pre-run source freeze',
        sha256=inventory),indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    return review,registry


if __name__=='__main__': build()
