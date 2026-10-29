"""The experiment boundary must reject simulator-pose or ambiguous inputs."""
from copy import deepcopy

import pytest

from .run_nist import validate_observed_rows


def rows():
    return [dict(scene=f'N{s:02d}',part=f'part_{p}',simulator_object_poses_read=False)
            for s in range(8) for p in range(12)]


@pytest.mark.parametrize('flag',[True,None])
def test_simulator_or_unspecified_pose_provenance_is_rejected(flag):
    data=rows()
    validate_observed_rows(data)
    changed=deepcopy(data)
    changed[0]['simulator_object_poses_read']=flag
    with pytest.raises(ValueError,match='explicitly exclude simulator'):
        validate_observed_rows(changed)


def test_missing_placement_cannot_be_hidden_by_a_duplicate():
    data=rows()
    data[-1]=data[0]
    with pytest.raises(ValueError,match='twelve CAD products'):
        validate_observed_rows(data)
