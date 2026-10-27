"""Check sanitized-model rejection, CAD bands and estimated-only state."""
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .planning import PlanningModel, band_vertices, native_triangles
from .sensors import robot_feedback
from .test_sensors import fixture


def box_stl(path):
    vertices = np.array([[x, y, z] for x in (-.015, .015) for y in (-.012, .012) for z in (-.0065, .0065)])
    faces = ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
             (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3))
    raw = bytearray(80) + struct.pack("<I", len(faces))
    for face in faces:
        raw.extend(struct.pack("<12fH", *([0., 0., 0.] + vertices[list(face)].ravel().tolist()), 0))
    path.write_bytes(raw)


class PlanningTests(unittest.TestCase):
    def test_four_CAD_bands_preserve_outer_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "native_m.stl"
            box_stl(path)
            triangles = native_triangles(path)
            bands = band_vertices(triangles)
            self.assertEqual(len(bands), 4)
            np.testing.assert_allclose(np.min(np.concatenate(bands), axis=0), triangles.reshape(-1, 3).min(0))
            np.testing.assert_allclose(np.max(np.concatenate(bands), axis=0), triangles.reshape(-1, 3).max(0))

    def test_estimated_surrogate_and_feedback_only(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model, data = fixture()
            template = folder / "robot_static.xml"
            mujoco.mj_saveLastXML(str(template), model)
            tree = ET.parse(template)
            for joint in tree.getroot().findall(".//joint"):
                if joint.get("name", "").startswith("robot0_"):
                    joint.set("range", "-2 2")
                    joint.set("limited", "true")
            tree.write(template)
            mesh = folder / "native_m.stl"
            box_stl(mesh)
            estimated = np.eye(4)
            estimated[:3, 3] = [.1, -.1, .04]
            feedback = robot_feedback(model, data)
            planner = PlanningModel(template, feedback, mesh, estimated)
            self.assertEqual(planner.m.body(planner.target_bid).name, "estimated_cad_surrogate")
            np.testing.assert_allclose(planner.d.qpos[planner.target_qadr:planner.target_qadr + 3], estimated[:3, 3])
            np.testing.assert_allclose(planner.pose(), feedback["world_T_grip"], atol=1e-10)
            qpos, time = planner.d.qpos.copy(), planner.d.time
            report = planner.collision_report(planner.d.qpos[planner.qa], .04)
            self.assertEqual(report["physics_steps"], 0)
            np.testing.assert_array_equal(planner.d.qpos, qpos)
            self.assertEqual(planner.d.time, time)
            tree = ET.parse(template)
            ET.SubElement(tree.getroot().find("worldbody"), "body", {"name": "nist_part_forbidden"})
            tree.write(template)
            with self.assertRaisesRegex(ValueError, "loose-product"):
                PlanningModel(template, feedback, mesh, estimated)


if __name__ == "__main__":
    unittest.main()
