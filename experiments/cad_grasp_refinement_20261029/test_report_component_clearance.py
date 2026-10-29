"""Post-processing regression: global and component minima have distinct scopes."""
from copy import deepcopy

import numpy as np
import pytest

from .report_streaming import candidate_table_clearance, compare_saved_table_clearances
from experiments.cad_grasp_selection_20261028.test_geometry import fixture


def test_distinct_component_heights_compare_against_their_own_minima():
    kwargs, _ = fixture()
    hand = kwargs['hand']
    candidate = dict(cad_T_grip_site=np.eye(4).tolist(),
                     holding_opening_m=.01, pre_opening_m=.02)
    minimum, _, components = candidate_table_clearance(candidate, np.eye(4),
        np.array([0.,0.,1.,.1]), hand, return_component_minima=True)
    assert minimum == pytest.approx(.075)
    assert components['left_pad'] == pytest.approx(.096)
    assert components['palm'] == pytest.approx(.075)
    saved = dict(checks=dict(table_minimum_clearance_m=minimum,
        component_geometry=[dict(component=name,table_minimum_clearance_m=value)
                            for name,value in components.items()]))
    assert compare_saved_table_clearances(saved, minimum, components) == (1,3)
    bad_component = deepcopy(saved)
    bad_component['checks']['component_geometry'][0]['table_minimum_clearance_m'] = minimum
    with pytest.raises(AssertionError,match='component left_pad'):
        compare_saved_table_clearances(bad_component, minimum, components)
    bad_global = deepcopy(saved)
    bad_global['checks']['table_minimum_clearance_m'] = components['left_pad']
    with pytest.raises(AssertionError,match='complete-hand'):
        compare_saved_table_clearances(bad_global, minimum, components)
