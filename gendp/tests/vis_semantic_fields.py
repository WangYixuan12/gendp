#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import colormaps
from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding
from d3fields.fusion import Fusion
import scipy.spatial.transform as st

### hyper param
epi_range = [0]
vis_robot = True
vis_action = True
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{curr_dir}/../../data/sapien_demo/hang_mug_demo'
robot_name = 'panda'
cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']

### set up shape_meta
shape_meta = {
    'shape': [6, 4000],
    'type': 'spatial',
    'info': {
        'reference_frame': 'world',
        'distill_dino': True,
        'distill_obj': 'mug',
        'view_keys': ['left_bottom_view', 'right_bottom_view', 'left_top_view', 'right_top_view'],
        'N_gripper': 400,
        'boundaries': {
            'x_lower': -0.35,
            'x_upper': 0.35,
            'y_lower': -0.3,
            'y_upper': 0.5,
            'z_lower': 0.01,
            'z_upper': 0.5
        },
        'resize_ratio': 0.5
    }
}

### create visualizer
visualizer = o3dVisualizer()
visualizer.start()

### create kinematics helper
kin_helper = KinHelper(robot_name='panda')

### create fusion
fusion = Fusion(num_cam=len(cam_keys), dtype=torch.float16)

for i in tqdm(epi_range):
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    # add meshes to visualize actions
    if vis_action:
        init_cart = data_dict['cartesian_action'][0] # (horizon, 7)
        action_horizon = init_cart.shape[0]
        action_cm = colormaps.get_cmap('plasma')
        action_colors = action_cm(np.linspace(0, 1, init_cart.shape[0], endpoint=True))[:, :3] # (horizon, 3)
        for a_i in range(action_horizon):
            visualizer.add_triangle_mesh('sphere', f'action_{a_i}', action_colors[a_i], radius=0.01)
    
    T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
    robot_base_in_world_seq = data_dict['observations']['robot_base_pose_in_world'][()]
    
    for t in tqdm(range(T)):
        # visualize point cloud
        robot_base_in_world = robot_base_in_world_seq[t]
        colors = np.stack([data_dict['observations']['images'][f'{cam_key}_color'][t:t+1] for cam_key in cam_keys], axis=1) # (N, H, W, 3)
        depths = np.stack([data_dict['observations']['images'][f'{cam_key}_depth'][t:t+1] for cam_key in cam_keys], axis=1) / 1000. # (N, H, W)
        intrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_intrinsic'][t:t+1] for cam_key in cam_keys], axis=1)
        extrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_extrinsic'][t:t+1] for cam_key in cam_keys], axis=1)
        pcd, pcd_feats = d3fields_proc(
            fusion=fusion,
            shape_meta=shape_meta,
            color_seq=colors,
            depth_seq=depths,
            extri_seq=extrinsics,
            intri_seq=intrinsics,
            robot_base_pose_in_world_seq=robot_base_in_world_seq,
            teleop_robot=kin_helper,
            qpos_seq=data_dict['observations']['full_joint_pos'][t:t+1],
        )
        pcd = pcd[0]
        pcd_feats = pcd_feats[0]
        pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
        pcd = pcd.T[:, :3]
        feats_cmap = colormaps.get_cmap('viridis')
        pcd_colors = feats_cmap(pcd_feats[:, 0])[:, :3]
        pcd = np2o3d(pcd, pcd_colors)
        visualizer.update_pcd(pcd, 'pcd')
        
        # visualize robot
        if vis_robot:
            robot_meshes = kin_helper.gen_robot_meshes(qpos = data_dict['observations']['full_joint_pos'][t])
            for m_i, mesh in enumerate(robot_meshes):
                visualizer.update_custom_mesh(mesh, f'mesh_{m_i}')
            
        if vis_action:
            # update action box
            t_start = t
            t_end = min(t_start + action_horizon, T)
            ee_target_pose = data_dict['cartesian_action'][t_start:t_end] # (horizon, 7)
            ee_target_pose_mat = np.tile(np.eye(4)[None], (t_end - t_start, 1, 1))
            ee_target_pose_mat[:, :3, 3] = ee_target_pose[:, :3]
            ee_target_pose_mat[:, :3, :3] = st.Rotation.from_euler('xyz', ee_target_pose[:, 3:6]).as_matrix()
            ee_target_pose_mat = np.linalg.inv(robot_base_in_world) @ ee_target_pose_mat
            for a_i in range(t_end - t_start):
                visualizer.update_triangle_mesh(f'action_{a_i}', tf=ee_target_pose_mat[a_i])
        
        visualizer.render()
