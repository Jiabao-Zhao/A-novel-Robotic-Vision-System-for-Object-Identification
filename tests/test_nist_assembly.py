import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.nist_assembly import insertion_goal, peg_seating_metrics
from simulation.libero_planning import execute_simulation_plan, validate_simulation_plan


def fixture():
    return {'ready': True, 'cross_section': 'rectangle', 'cad_extent_m': [.016, .010, .050],
            'center_world_m': [0, 0, .03], 'entrance_z_m': .03, 'commanded_insertion_depth_m': .007,
            'polygon_world_m': [[-.0081, -.0051], [.0081, -.0051], [.0081, .0051], [-.0081, .0051]]}


class AssemblyTests(unittest.TestCase):
    def test_seating_requires_clearance_depth_and_alignment(self):
        job = fixture()
        self.assertTrue(peg_seating_metrics(np.array([0, 0, .048]), np.eye(3), job)['inserted'])
        for position, rotation in (([.001, 0, .048], np.eye(3)), ([0, 0, .08], np.eye(3)),
                                   ([0, 0, .048], Rotation.from_euler('z', 90, degrees=True).as_matrix()),
                                   ([0, 0, .048], Rotation.from_euler('x', 5, degrees=True).as_matrix())):
            self.assertFalse(peg_seating_metrics(np.array(position), rotation, job)['inserted'])

    def test_goal_preserves_estimated_grasp_offset_and_aligns_rectangle(self):
        cad = np.eye(4)
        cad[:3, :3] = Rotation.from_euler('z', 40, degrees=True).as_matrix()
        cad[:3, 3] = [-.2, -.1, .025]
        offset = np.eye(4)
        offset[:3, 3] = [.002, 0, .005]
        item = {'world_T_cad': cad, 'world_T_grasp': cad @ offset, 'cad_extent_m': [.016, .010, .050]}
        destination = {'insertion_supported': True, 'assembly_binding': fixture()}
        goal = insertion_goal(item, destination)
        np.testing.assert_allclose(goal[:3, 3], [.002, 0, .053])
        np.testing.assert_allclose(goal[:3, :3], np.eye(3), atol=1e-10)

    def test_fallen_peg_has_no_projected_clearance(self):
        result = peg_seating_metrics(np.array([0., 0., .035]),
            Rotation.from_euler('x', 90, degrees=True).as_matrix(), fixture())
        self.assertFalse(result['inserted'])
        self.assertIsNone(result['minimum_clearance_mm'])

    def test_tip_below_board_is_not_constrained_by_the_hole(self):
        job = fixture()
        rotation = Rotation.from_euler('y', .4, degrees=True).as_matrix()
        position = np.array([0., 0., .025])
        self.assertFalse(peg_seating_metrics(position, rotation, job)['inserted'])
        job.update(scoring_revision=2, board_bottom_z_m=.020,
                   cross_section_polygon_cad_m=[[-.008,-.005],[-.008,.005],[.008,.005],[.008,-.005]],
                   contact_numerical_tolerance_m=.000001)
        self.assertTrue(peg_seating_metrics(position, rotation, job)['inserted'])
        self.assertFalse(peg_seating_metrics(position + [.0005, 0, 0], rotation, job)['inserted'])

    def test_unsupported_destination_rejected_before_pick(self):
        item = {'object_id': 'part', 'world_T_grasp': np.eye(4).tolist(),
                'world_T_cad': np.eye(4).tolist(), 'cad_extent_m': [.016, .010, .050]}
        destination = {'object_id': 'hole', 'insertion_supported': True, 'assembly_binding': fixture()}
        context = {'objects': [item, destination], 'completion_condition': 'assembled_and_released'}
        plan = {'status': 'ready', 'reason': '', 'actions': [{'action': 'pick', 'object_id': 'part'},
            {'action': 'insert', 'object_id': 'part', 'destination_id': 'hole'}]}
        validate_simulation_plan(plan, context)
        invalid = copy.deepcopy(context)
        invalid['objects'][1]['insertion_supported'] = False
        with patch('simulation.libero_planning.pick_object') as pick:
            with self.assertRaises(ValueError):
                execute_simulation_plan(None, {}, plan, invalid)
            pick.assert_not_called()
        with patch('simulation.libero_planning.pick_object', return_value={'picked': True}) as pick, \
             patch('simulation.nist_assembly.insert_object', return_value={'inserted': True}) as insert:
            self.assertEqual(execute_simulation_plan(None, {}, plan, context), {'inserted': True})
            self.assertEqual(insert.call_args.args[2], item)
            self.assertEqual(insert.call_args.args[3], destination)


if __name__ == '__main__':
    unittest.main()
