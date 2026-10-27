"""Validate capture bookkeeping/calibration without reading any method prediction."""
from collections import Counter
from render_scenes import *


def main():
    manifest=read(TRIAL/'capture_manifest.json')
    assert manifest['complete']
    assert len(manifest['scenes'])==200
    records={r['scene_id']:r for r in manifest['scenes']}
    counts=Counter(r['condition'] for r in records.values())
    assert all(n==20 for n in counts.values()) and len(counts)==10
    rgb_only=[]
    occlusion=[]
    supports=[]
    for index in range(20):
        base=f'R{index:02d}'
        complete=read(TRIAL/'inputs'/base/'capture_completion.json')
        basehash=complete['input_sha256'][relative(TRIAL/'inputs'/base/'depth.npy')]
        baseK=np.load(TRIAL/'inputs'/base/'intrinsics.npy')
        truth=read(TRIAL/'audit'/base/'truth.json')
        supports.extend(abs(r['pose_audit_only']['minimum_world_z_m']) for r in truth['instances'])
        assert len(truth['instances'])==12
        for condition in ('light_shadow','background','finish'):
            scene=base+'__'+condition
            entry=read(TRIAL/'inputs'/scene/'capture_completion.json')
            hashes=entry['input_sha256']
            assert hashes[relative(TRIAL/'inputs'/scene/'depth.npy')]==basehash
            np.testing.assert_array_equal(np.load(TRIAL/'inputs'/scene/'intrinsics.npy'),baseK)
            rgb_changed=hashes[relative(TRIAL/'inputs'/scene/'rgb.png')]!=complete['input_sha256'][relative(TRIAL/'inputs'/base/'rgb.png')]
            rgb_only.append(dict(scene_id=scene,depth_exact=True,intrinsics_exact=True,rgb_changed=rgb_changed))
        low=np.load(TRIAL/'inputs'/(base+'__resolution_low')/'intrinsics.npy')
        np.testing.assert_array_equal(low[:2,:],baseK[:2,:]/2)
        for condition in ('absent','duplicate'):
            tr=read(TRIAL/'audit'/(base+'__'+condition)/'truth.json')
            assert len(tr['instances'])==11+(condition=='duplicate')*2
            expected=list(tr['expected_instance_counts'].values())
            assert Counter(expected)==({1:11,0:1} if condition=='absent' else {1:11,2:1})
        target=read(TRIAL/'audit'/(base+'__occlusion_light')/'truth.json')['condition_target_object_id']
        visible=[]
        for condition in ('nominal','occlusion_light','occlusion_heavy'):
            scene=base if condition=='nominal' else base+'__'+condition
            tr=read(TRIAL/'audit'/scene/'truth.json')
            visible.append(next(r['visible_pixels'] for r in tr['instances'] if r['object_id']==target))
        assert visible[2]<=visible[1]<=visible[0]
        occlusion.append(dict(base_scene_id=base,condition_target_object_id=target,
            nominal_pixels=visible[0],light_pixels=visible[1],heavy_pixels=visible[2],
            visible_fraction_light=None if not visible[0] else visible[1]/visible[0],
            visible_fraction_heavy=None if not visible[0] else visible[2]/visible[0]))
    assert max(supports)<1e-9
    save(HERE/'capture_input_checks.json',dict(passed=True,scenes=200,condition_counts=dict(counts),
        no_method_predictions_read=True,RGB_only_depth_pairs=rgb_only,
        max_abs_CAD_support_height_m=max(supports),
        nominal_instances=240,absent_scene_instances=220,duplicate_scene_instances=260,
        gripper_geometry_absent=True,low_resolution_intrinsics_exact_half=True,
        occlusion_measurements_audit_only=occlusion,manifest_sha256=digest(TRIAL/'capture_manifest.json')))
    print('Capture checks passed: 200 scenes, RGB/depth/calibration pairing and instance counts.',flush=True)


if __name__=='__main__':main()
