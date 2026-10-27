"""Bounded NumPy seed batching for the unchanged table-constrained depth search."""
from time import perf_counter
import numpy as np
from native_loader import load_native

native = load_native()
MAX_SEEDS_PER_BATCH = 72
MAX_BATCH_OBSERVATIONS = 131072


def refine_batch(cad, observed, normal, basis, initial, iterations=native.ITERATIONS):
    """Run native per-seed stopping rules with one batched KD query per iteration."""
    pose = np.asarray(initial, dtype=float).copy()
    observed = np.asarray(observed, dtype=float)
    count, size = len(pose), len(observed)
    local = np.matmul(observed[None] - pose[:, None, :3, 3], pose[:, :3, :3])
    distances, indices = cad.tree.query(local.reshape(-1, 3), workers=1)
    distances, indices = distances.reshape(count, size), indices.reshape(count, size)
    rmse = np.sqrt(np.mean(distances ** 2, axis=1))
    best_rmse, best_pose = rmse.copy(), pose.copy()
    histories = [[float(value)] for value in rmse]
    stopped = ['iteration_limit'] * count
    active = np.ones(count, dtype=bool)
    observed_center = observed.mean(0)
    b = (observed - observed_center) @ basis
    skew = np.array([[0., -normal[2], normal[1]], [normal[2], 0., -normal[0]],
                     [-normal[1], normal[0], 0.]])
    skew_squared = skew @ skew
    for _ in range(int(iterations)):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        current = pose[ids]
        matched = np.matmul(cad.points[indices[ids]], current[:, :3, :3].transpose(0, 2, 1)) + current[:, None, :3, 3]
        cad_center = matched.mean(1)
        a = np.matmul(matched - cad_center[:, None], basis)
        cross = np.sum(a[:, :, 0] * b[None, :, 1] - a[:, :, 1] * b[None, :, 0], axis=1)
        dot = np.sum(a * b[None], axis=(1, 2))
        yaw = np.arctan2(cross, dot)
        yaw[(cross == 0) & (dot == 0)] = 0.
        delta = (np.eye(3)[None] + np.sin(yaw)[:, None, None] * skew[None]
                 + (1 - np.cos(yaw))[:, None, None] * skew_squared[None])
        shift = (cad_center - np.matmul(delta, cad_center[:, :, None])[:, :, 0]
                 + np.matmul(basis, np.matmul(basis.T, (observed_center[None] - cad_center)[:, :, None]))[:, :, 0])
        candidate = current.copy()
        candidate[:, :3, :3] = np.matmul(delta, current[:, :3, :3])
        candidate[:, :3, 3] = np.matmul(delta, current[:, :3, 3, None])[:, :, 0] + shift
        local = np.matmul(observed[None] - candidate[:, None, :3, 3], candidate[:, :3, :3])
        distances, new_indices = cad.tree.query(local.reshape(-1, 3), workers=1)
        distances, new_indices = distances.reshape(len(ids), size), new_indices.reshape(len(ids), size)
        updated = np.sqrt(np.mean(distances ** 2, axis=1))
        reject = updated > rmse[ids] + 1e-12
        for index in ids[reject]:
            stopped[index] = 'no_icp_improvement'
        active[ids[reject]] = False
        accepted_ids, accepted_local = ids[~reject], np.flatnonzero(~reject)
        change = rmse[accepted_ids] - updated[accepted_local]
        pose[accepted_ids] = candidate[accepted_local]
        indices[accepted_ids] = new_indices[accepted_local]
        rmse[accepted_ids] = updated[accepted_local]
        for index in accepted_ids:
            histories[index].append(float(rmse[index]))
        improve = rmse[accepted_ids] < best_rmse[accepted_ids]
        improved_ids = accepted_ids[improve]
        best_rmse[improved_ids], best_pose[improved_ids] = rmse[improved_ids], pose[improved_ids]
        finished_ids = accepted_ids[change <= 1e-8]
        active[finished_ids] = False
        for index in finished_ids:
            stopped[index] = 'rmse_tolerance'
    records = [dict(rmse_m=float(best_rmse[i]), initial_rmse_m=histories[i][0],
                    rmse_history_m=histories[i], iterations=len(histories[i])-1,
                    stop_reason=stopped[i]) for i in range(count)]
    return best_pose, records


def fit_pair(cad, observed_points, plane, K, depth_roi, mask_roi, roi_xy=(0, 0), clearances=(0.,)):
    """Same candidates, objective, projection and winner rule as frozen fit_pair."""
    started = perf_counter()
    depth, mask = np.asarray(depth_roi), np.asarray(mask_roi, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError('Depth and observed mask must share a camera image region.')
    if not mask.any():
        return dict(status='unavailable_empty_observed_mask', arms=[], seconds=perf_counter()-started)
    observed = np.asarray(observed_points, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 3 or not len(observed):
        return dict(status='unavailable_empty_observed_cloud', arms=[], seconds=perf_counter()-started)
    observed = native.voxel_means(observed)
    normal, distance, basis = native.table_frame(plane, observed)
    yy, xx = np.nonzero(mask)
    observed_bbox = (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1)
    observed_total = len(xx)
    declarations = [(face_index, yaw) for face_index in range(len(cad.supports)) for yaw in native.YAW_DEGREES]
    batch_size = min(MAX_SEEDS_PER_BATCH, max(1, MAX_BATCH_OBSERVATIONS // len(observed)))
    arms = []
    for clearance in clearances:
        begin, bank = perf_counter(), []
        for first in range(0, len(declarations), batch_size):
            group = declarations[first:first+batch_size]
            initial = np.stack([native.contact_pose(cad.vertices, observed, normal, distance, basis,
                      cad.supports[index]['outward_normal'], yaw, float(clearance)) for index, yaw in group])
            poses, refinements = refine_batch(cad, observed, normal, basis, initial)
            for (index, yaw), pose, refinement in zip(group, poses, refinements):
                verification = native.projection_score(cad, pose, K, depth, mask, roi_xy, observed_bbox, observed_total)
                heights = (cad.vertices @ pose[:3, :3].T + pose[:3, 3]) @ normal + distance
                if not np.isclose(heights.min(), clearance, atol=1e-10):
                    raise AssertionError('Table contact changed during batched refinement.')
                bank.append(dict(support_index=index, initial_yaw_deg=yaw, clearance_m=float(clearance),
                    T_camera_from_cad=pose.tolist(), minimum_table_height_m=float(heights.min()),
                    fitting=refinement, verification=verification))
        available = [i for i, row in enumerate(bank) if row['verification']['geometry_score'] is not None]
        best = max(available, key=lambda i: bank[i]['verification']['geometry_score']) if available else None
        best_rmse = min(range(len(bank)), key=lambda i: bank[i]['fitting']['rmse_m'])
        arm_name = 'contact_0mm' if clearance == 0 else f'clearance_{clearance*1000:g}mm'
        arms.append(dict(arm_name=arm_name, clearance_m=float(clearance), status='available' if available else 'unavailable_all_projections',
            best_geometry_index=best, best_geometry=None if best is None else bank[best],
            best_rmse_index=best_rmse, best_rmse=bank[best_rmse], pose_bank=bank,
            hypothesis_count=len(bank), available_projections=len(available), seconds=perf_counter()-begin))
    return dict(status='complete', arms=arms, observed_voxel_points=len(observed), cad_voxel_points=len(cad.points),
        support_faces=cad.supports, cad_native_extent_m=np.ptp(cad.vertices, axis=0).tolist(), seconds=perf_counter()-started)
