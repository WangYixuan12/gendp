#!/usr/bin/env python
# coding: utf-8

import os
import copy
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d
from gild.common.data_utils import load_dict_from_hdf5
from gild.common.rob_mesh_utils import load_mesh, mesh_poses_to_pc
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d
import transforms3d
import scipy.spatial.transform as st

num_episodes = 1
epi_s = 0
epi_range = list(range(epi_s, epi_s+num_episodes))
vis_robot = False
vis_action = True

data_dir = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag_v2'

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

action_box_ls = []
action_box_pts_ls = []
action_horizon = 8
action_vis_size_ls = np.linspace(0.05, 0.01, action_horizon)
if vis_action:
    # add action box
    # action_box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
    # action_box.compute_vertex_normals()
    # action_box.paint_uniform_color([0.1, 0.7, 0.1])
    # box_pts = np.asarray(action_box.vertices).copy() - np.array([0.025, 0.025, 0.025])
    # visualizer.add_geometry(action_box)
    
    # add a list of boxes
    for i in range(action_horizon):
        action_box = o3d.geometry.TriangleMesh.create_coordinate_frame(size=action_vis_size_ls[i])
        action_box.compute_vertex_normals()
        box_pts = np.asarray(action_box.vertices).copy()
        action_box_ls.append(action_box)
        action_box_pts_ls.append(box_pts)
        visualizer.add_geometry(action_box_ls[-1])
    
    # add last box specifically for current state
    action_box = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    action_box.compute_vertex_normals()
    box_pts = np.asarray(action_box.vertices).copy()
    action_box_ls.append(action_box)
    action_box_pts_ls.append(box_pts)
    visualizer.add_geometry(action_box_ls[-1])

for i in tqdm(epi_range):
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    if vis_robot:
        finger_names = data_dict['observations']['finger_pos'].keys()
        finger_meshes, mesh_offsets = load_mesh('trossen_vx300s', finger_names)

    cam_keys = ['camera_0', 'camera_1', 'camera_2']
    
    T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
    robot_base_in_world_seq = data_dict['observations']['robot_base_pose_in_world'][()]
    
    timestamp = data_dict['timestamp'][()]
    print(f'timestamp: {timestamp - timestamp[0]}')
    timediff = np.diff(timestamp)
    avg_dt = np.mean(timediff)
    print(f'avg_dt: {avg_dt}')
    
    for t in tqdm(range(T)):
        robot_base_in_world = robot_base_in_world_seq[t, 0]
        print()
        print('curr freq: ', 1/timediff[t] if t < T-1 else 1/timediff[-1])
        colors = np.stack([data_dict['observations']['images'][f'{cam_key}_color'][t] for cam_key in cam_keys]) # (N, H, W, 3)
        depths = np.stack([data_dict['observations']['images'][f'{cam_key}_depth'][t] for cam_key in cam_keys]) / 1000. # (N, H, W)
        intrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_intrinsics'][t] for cam_key in cam_keys])
        extrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_extrinsics'][t] for cam_key in cam_keys])
        pcd, pcd_colors = aggr_point_cloud_from_data(colors, depths, intrinsics, extrinsics, downsample=False, out_o3d=False)
        pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
        pcd = pcd.T[:, :3]
        pcd = np2o3d(pcd, pcd_colors)
        
        if vis_robot:
            finger_poses = np.stack([data_dict['observations']['finger_pos'][finger_name][t] for finger_name in finger_names])
            finger_pcd_np = mesh_poses_to_pc(finger_poses, finger_meshes, mesh_offsets, [1000 for _ in range(len(finger_names))], scale=0.001)
            finger_colors = np.zeros_like(finger_pcd_np)
            finger_colors[:, 0] = 1.0
            figner_pcd_o3d = np2o3d(finger_pcd_np, finger_colors)
            pcd = pcd + figner_pcd_o3d
        
        if t == 0 and i == epi_range[0]:
            curr_pcd = copy.deepcopy(pcd)
            visualizer.add_geometry(curr_pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visualizer.add_geometry(origin)
            visualizer.update_geometry(curr_pcd)
            visualizer.poll_events()
            visualizer.update_renderer()
            visualizer.run()
        else:
            curr_pcd.points = pcd.points
            curr_pcd.colors = pcd.colors
            visualizer.update_geometry(curr_pcd)
            
            if vis_action:
                # update action box
                t_start = t
                t_end = min(t_start + action_horizon, T)
                ee_target_pose = data_dict['cartesian_action'][t_start:t_end] # (horizon, 7)
                ee_target_pose_mat = np.tile(np.eye(4)[None], (t_end - t_start, 1, 1))
                ee_target_pose_mat[:, :3, 3] = ee_target_pose[:, :3]
                ee_target_pose_mat[:, :3, :3] = st.Rotation.from_euler('xyz', ee_target_pose[:, 3:6]).as_matrix()
                for act_i in range(t_end - t_start):
                    box_pts_in_world = (ee_target_pose_mat[act_i] @ np.concatenate([action_box_pts_ls[act_i], np.ones((action_box_pts_ls[act_i].shape[0], 1))], axis=-1).T).T[:, :3]
                    action_box_ls[act_i].vertices = o3d.utility.Vector3dVector(box_pts_in_world)
                    visualizer.update_geometry(action_box_ls[act_i])
                
                # update last box
                curr_ee_pose = data_dict['observations']['ee_pos'][t]
                curr_ee_pose_mat = np.eye(4)
                curr_ee_pose_mat[:3, 3] = curr_ee_pose[:3]
                curr_ee_pose_mat[:3, :3] = st.Rotation.from_euler('xyz', curr_ee_pose[3:6]).as_matrix()
                box_pts_in_world = (curr_ee_pose_mat @ np.concatenate([action_box_pts_ls[-1], np.ones((action_box_pts_ls[-1].shape[0], 1))], axis=-1).T).T[:, :3]
                action_box_ls[-1].vertices = o3d.utility.Vector3dVector(box_pts_in_world)
                visualizer.update_geometry(action_box_ls[-1])
            
            visualizer.poll_events()
            visualizer.update_renderer()
        if t == 0 and i == epi_range[0]:
            visualizer.run()
        print('curr gripper pos (from ee_pos): ', data_dict['observations']['ee_pos'][t][-1])
        print('curr gripper pos (from joint_pos): ', data_dict['observations']['joint_pos'][t][-1])
        print('target gripper pos (from cartesian): ', data_dict['cartesian_action'][t][-1])
        print('target gripper pos (from joint action): ', data_dict['joint_action'][t][-1])
