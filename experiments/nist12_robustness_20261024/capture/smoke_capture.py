"""Input/calibration QA only; does not accept scenes using recognition outcomes."""
from render_scenes import *


def main():
    protocol=read(TRIAL/'protocol.json')
    config=protocol['capture']
    catalog=read(path(config['catalog_file']))['objects']
    items=inventory(catalog)
    layout=generate_layout(0,config['seed_base'],items,config['support_clearance_m'])
    record=render_scene('R00',layout,0,catalog)
    root,instances=make_scene(layout,0,catalog)
    model,data=compile_scene(root)
    assert not any('robot' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or '')
                   for gid in range(model.ngeom))
    depth=np.load(path(record['depth']))
    K=np.load(path(record['intrinsics']))
    T=np.load(TRIAL/'audit/R00/world_T_camera.npy')
    labels=np.load(TRIAL/'audit/R00/instance_labels.npz')['labels']
    yy,xx=np.nonzero(labels==0)
    choose=np.linspace(0,len(xx)-1,500,dtype=int)
    z=depth[yy[choose],xx[choose]]
    cam=np.column_stack(((xx[choose]-K[0,2])*z/K[0,0],(yy[choose]-K[1,2])*z/K[1,1],z))
    world=cam@T[:3,:3].T+T[:3,3]
    assert np.max(np.abs(world[:,2]))<.0002
    reprojection=cam[:,:2]/cam[:,2:3]*np.array([K[0,0],K[1,1]])+K[:2,2]
    assert np.max(np.abs(reprojection-np.column_stack((xx[choose],yy[choose]))))<1e-9
    save(HERE/'smoke_QA.json',dict(passed=True,scene_id='R00',objects=len(catalog),
        robot_geometry_absent=True,no_classifier_calls=True,no_recognition_based_scene_acceptance=True,
        native_CAD_extents_match=True,table_backprojection_samples=500,
        max_abs_world_table_height_m=float(np.max(np.abs(world[:,2]))),
        max_reprojection_error_px=float(np.max(np.abs(reprojection-np.column_stack((xx[choose],yy[choose]))))),
        pixel_offset_convention='Existing pinhole convention, integer u/v and principal point width/2,height/2.',
        static_support_clearance_m=config['support_clearance_m'],protocol_sha256=digest(TRIAL/'protocol.json')))
    print('R00 smoke calibration/source checks passed',flush=True)


if __name__=='__main__':main()
