import os
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from gendp.real_world.aloha_bimanual_puppet import AlohaBimanualPuppet
from gendp.real_world.aloha_bimanual_master import AlohaBimanualMaster
from gendp.real_world.multi_realsense import MultiRealsense
from d3fields.utils.draw_utils import aggr_point_cloud_from_data
from gendp.common.aloha_utils import MASTER2PUPPET_JOINT_FN
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import np2o3d, o3dVisualizer

def test_aloha_calibration(sides=['right', 'left']):
    view_ctrl_info = {
        "front": [ -0.78451141301854455, -0.30765995570025595, 0.53841173325083047 ],
        "lookat": [ -0.25162616981933222, -0.12788398312192117, 0.23992266979173191 ],
        "up": [ 0.55335827616805278, 0.044557205847718349, 0.8317507280449864 ],
        "zoom": 0.23999999999999957
    }
    o3d_visualizer = o3dVisualizer(view_ctrl_info=view_ctrl_info)
    o3d_visualizer.start()
    o3d_visualizer.add_triangle_mesh('origin', 'origin', size=0.1)

    # boundaries for visualizer
    boundaries = {
        'x_lower': -1.0,
        'x_upper': 1.0,
        'y_lower': -1.0,
        'y_upper': 1.0,
        'z_lower': -1.0,
        'z_upper': 1.0,
    }

    # pose manual guess
    robot_base_in_world = np.array([[[0,-1,0,-0.13],
                                    [1,0,0,-0.51],
                                    [0,0,1,0.02],
                                    [0,0,0,1]],
                                    [[0,1,0,-0.13],
                                    [-1,0,0,0.27],
                                    [0,0,1,0.02],
                                    [0,0,0,1]],])
    
    # save current robot base pose in world
    curr_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(curr_path, 'aloha_extrinsics')
    os.system(f'mkdir -p {save_dir}')
    for rob_i, side in enumerate(sides):
        np.save(os.path.join(save_dir, f'{side}_base_pose_in_world.npy'), robot_base_in_world[rob_i])

    # start robot
    kin_helper = KinHelper(robot_name='trossen_vx300s_v3')
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    init_qpos = np.array([0.0, -0.865, 0.6, 0.0, 1.35, 0.0, -0.5, 0.0, -0.94, 0.72, 0.0, 1.31, 0.0, -0.5])
    puppet_robot = AlohaBimanualPuppet(shm_manager=shm_manager, verbose=False, frequency=50, robot_sides=['right', 'left'], init_qpos=init_qpos)
    puppet_robot.start()
    master_robot = AlohaBimanualMaster(shm_manager=shm_manager, verbose=False, frequency=50, robot_sides=['right', 'left'])
    master_robot.start()

    # visualize realsense point cloud
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(exposure=200, gain=64)
        realsense.set_white_balance(white_balance=2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)

        while True:
            out = realsense.get()

            # set robot joint pos
            master_state = master_robot.get_motion_state()
            target_state = master_state['joint_pos'].copy()
            for rob_i in range(2):
                target_state[7*rob_i + 6] = MASTER2PUPPET_JOINT_FN(master_state['joint_pos'][7*rob_i + 6])
            target_time = time.time() + 0.1
            puppet_robot.set_target_joint_pos(target_state, target_time=target_time)

            # compute robot point cloud
            curr_robot_joint_pos = puppet_robot.get_state()['curr_full_joint_pos']
            base_pcd_ls = []
            for rob_i, side in enumerate(sides):
                base_pcd = kin_helper.compute_robot_pcd(curr_robot_joint_pos[rob_i*8:(rob_i+1)*8], pcd_name='rob_pcd')
                base_pcd_i = base_pcd.copy()
                base_pcd_i = robot_base_in_world[rob_i] @ np.concatenate([base_pcd_i, np.ones((base_pcd_i.shape[0], 1))], axis=-1).T
                base_pcd_i = base_pcd_i[:3].T
                base_pcd_ls.append(base_pcd_i)
            base_pcd = np.concatenate(base_pcd_ls, axis=0)

            # visualize camera point cloud
            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())
            extrinsics = np.stack(value['extrinsics'] for value in out.values())
            pcd = aggr_point_cloud_from_data(colors=colors,
                                            depths=depths,
                                            Ks=intrinsics,
                                            poses=extrinsics,
                                            downsample=False,
                                            boundaries=boundaries,)
            
            o3d_visualizer.update_pcd(pcd, 'pcd')
            o3d_visualizer.update_pcd(np2o3d(base_pcd), 'rob_pcd')
            
            # render
            o3d_visualizer.render()

if __name__ == '__main__':
    test_aloha_calibration()
