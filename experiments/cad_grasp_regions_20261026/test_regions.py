"""Tests for contact-region reduction, not physical pickup quality."""
import numpy as np
import open3d as o3d

from experiments.cad_grasp_regions_20261026.regions import contact_regions,recast_pair


def box(low,high):
    mesh = o3d.geometry.TriangleMesh.create_box(*(np.asarray(high)-low))
    mesh.translate(low)
    return np.asarray(mesh.vertices),np.asarray(mesh.triangles)


def combine(parts):
    vertices,faces,offset = [],[],0
    for v,f in parts:
        vertices.append(v);faces.append(f+offset);offset += len(v)
    return np.concatenate(vertices),np.concatenate(faces)


def test_flat_box_is_one_region_not_hundreds_of_poses():
    v,f = box(np.array([-.008,-.025,0.]),np.array([.008,.025,.01]))
    out = contact_regions(v,f,[1,0,0],.067)
    assert out['metadata']['qualified_cells'] == 500
    assert len(out['regions']) == 1
    region = out['regions'][0]
    assert abs(region['center_y']) < 1e-12
    pair = recast_pair(out,0.,.005)
    assert pair is not None
    assert abs(pair['width_m']-.016) < 1e-7
    assert np.ptp(np.asarray(pair['contacts_world_m'])[:,2]) < 1e-12


def test_width_step_creates_distinct_regions():
    # Separate closed components touch at z=20mm; no target hull fills the step.
    v,f = combine([box(np.array([-.004,-.01,0.]),np.array([.004,.01,.02])),
                   box(np.array([-.008,-.01,.02]),np.array([.008,.01,.04]))])
    out = contact_regions(v,f,[1,0,0],.067)
    assert len(out['regions']) == 2
    assert np.allclose(sorted(r['median_width_m'] for r in out['regions']),[.008,.016],atol=1e-7)


def test_recasting_does_not_fill_native_opening():
    # Two bars separated along Y: a ray through their gap must miss both.
    v,f = combine([box(np.array([-.01,-.015,0.]),np.array([.01,-.005,.01])),
                   box(np.array([-.01,.005,0.]),np.array([.01,.015,.01]))])
    out = contact_regions(v,f,[1,0,0],.067)
    assert len(out['regions']) == 2
    assert recast_pair(out,0.,.005) is None
    assert recast_pair(out,.01,.005) is not None


def test_deterministic_regions_and_width_gate():
    v,f = box(np.array([-.008,-.025,0.]),np.array([.008,.025,.01]))
    a,b = contact_regions(v,f,[1,0,0],.067),contact_regions(v,f,[1,0,0],.067)
    assert a['metadata'] == b['metadata']
    assert np.array_equal(a['regions'][0]['grid_indices'],b['regions'][0]['grid_indices'])
    assert not contact_regions(v,f,[1,0,0],.01)['regions']


def test_smooth_round_body_can_remain_one_contact_region():
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=.006,height=.05,resolution=128)
    mesh.translate([0.,0.,.025])
    v,f = np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    # This graph tolerance joins the whole already-admitted +/-15deg band;
    # each ray still has to pass the original contact-normal qualification.
    out = contact_regions(v,f,[1,0,0],.067,neighbor_normal_angle_deg=30.)
    assert len(out['regions']) == 1
    assert abs(out['regions'][0]['center_y']) < 1e-9
    assert abs(out['regions'][0]['center_z']-.025) < 1e-9
