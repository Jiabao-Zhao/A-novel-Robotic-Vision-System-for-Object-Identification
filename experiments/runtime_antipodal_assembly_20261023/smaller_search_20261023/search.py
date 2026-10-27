"""Small, fixed-budget CAD grasp search; units are metres in the world frame."""
from dataclasses import replace

import numpy as np

from ..grasp_planning import candidate_frames


FINGER_DEPTHS_M = (.004, 0., -.004, -.008, .006, .008)
SMALL = {"surface_rays": 512, "contact_pairs": 24, "shortlist": 12}
LARGE = {"surface_rays": 2048, "contact_pairs": 320, "shortlist": 48}


def frames_with_full_finger_depths(pairs, rotation, center, pair_budget):
    """Keep the original four depths and the two successful follow-up depths."""
    original = candidate_frames(pairs, rotation, center, pair_budget)
    extra = []
    for candidate in original:
        if not np.isclose(candidate.pad_depth_m, .004):
            continue
        for depth in (.006, .008):
            pose = candidate.world_T_grip.copy()
            pose[:3, 3] -= pose[:3, 2] * (depth - candidate.pad_depth_m)
            extra.append(replace(candidate, world_T_grip=pose, pad_depth_m=depth))
    return sorted(original + extra, key=lambda item: item.score)


def same_contact_region(first, second):
    """Nearby touches and approach, treating exchanged jaws as equivalent."""
    if np.linalg.norm(first.contacts_world_m.mean(0) - second.contacts_world_m.mean(0)) > .005:
        return False
    a, b = first.world_T_grip[:3, :3], second.world_T_grip[:3, :3]
    return (abs(np.dot(a[:, 0], b[:, 0])) > np.cos(np.deg2rad(15))
            and np.dot(a[:, 2], b[:, 2]) > np.cos(np.deg2rad(15)))


def small_shortlist(candidates, limit=12):
    """Prefer different touch regions before adding a second depth per region.

    The original geometric score/order is retained within each pass. A second
    pass permits another depth or exchanged-jaw orientation, preserving those
    useful alternatives without filling the list with neighbouring contacts.
    """
    kept, groups = [], []
    for quota in (1, 2):
        for candidate in candidates:
            if any(candidate is old for old in kept):
                continue
            match = next((group for group in groups if same_contact_region(candidate, group[0])), None)
            if match is None:
                groups.append([candidate])
                kept.append(candidate)
            elif len(match) < quota:
                # A near duplicate at the same depth and jaw direction adds no
                # useful alternative; a new depth or reversed jaws can.
                if any(abs(old.pad_depth_m-candidate.pad_depth_m) < .001
                       and np.dot(old.world_T_grip[:3, 0], candidate.world_T_grip[:3, 0]) > .97
                       for old in match):
                    continue
                match.append(candidate)
                kept.append(candidate)
            if len(kept) == limit:
                return kept
    return kept
