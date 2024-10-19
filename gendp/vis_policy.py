# import packages
import time
import os

import dill
import hydra
import torch
from omegaconf import open_dict
import diffusers
import cv2
import matplotlib
from matplotlib import colormaps as cm
import numpy as np
import pickle
import transforms3d
from pytorch3d.transforms import rotation_6d_to_matrix
import matplotlib.pyplot as plt

from gendp.dataset.real_dataset import RealDataset
from gendp.workspace.base_workspace import BaseWorkspace
from gendp.policy.base_image_policy import BaseImagePolicy
from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc, d3fields_proc_for_vis
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import np2o3d, aggr_point_cloud_from_data, draw_keypoints, poseInterp, viewCtrlInterp, vis_pcd_plotly, voxel_downsample
from d3fields.fusion import Fusion

# hacks to work with old models
import gendp
import sys
sys.modules["diffusion_policy"] = gendp

def extract_pcd_from_data_dict(data_dict, t, fusion, shape_meta, kin_helper):
    # vis raw pcd
    view_keys = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
    color_seq = np.stack([data_dict['observations']['images'][f'{k}_color'][t] for k in view_keys], axis=0) # (V, H ,W, C)
    # color_seq = np.stack([white_balance_loops(color_seq[i]) for i in range(color_seq.shape[0])], axis=0)
    depth_seq = np.stack([data_dict['observations']['images'][f'{k}_depth'][t] for k in view_keys], axis=0) / 1000. # (V, H ,W)
    extri_seq = np.stack([data_dict['observations']['images'][f'{k}_extrinsics'][t] for k in view_keys], axis=0) # (V, 4, 4)
    intri_seq = np.stack([data_dict['observations']['images'][f'{k}_intrinsics'][t] for k in view_keys], axis=0) # (V, 3, 3)
    qpos_seq = data_dict['observations']['full_joint_pos'][t] # (8*num_bots)

    if 'd3fields' in shape_meta['obs']:
        boundaries = shape_meta['obs']['d3fields']['info']['boundaries']
        d3fields_shape_meta = shape_meta['obs']['d3fields']
    else:
        boundaries = {
            'x_lower': -0.3,
            'x_upper': 0.1,
            'y_lower': -0.28,
            'y_upper': 0.05,
            'z_lower': 0.01,
            'z_upper': 0.2,
        }
        d3fields_shape_meta = {
            'shape': [5, 1000],
            'type': 'spatial',
            'info': {
                'feat_type': 'no_feats',
                'reference_frame': 'robot',
                'use_seg': False,
                'use_dino': False,
                'distill_dino': False,
                'distill_obj': 'knife',
                'rob_pcd': True,
                'view_keys': ['camera_0', 'camera_1', 'camera_2', 'camera_3'],
                'query_texts': ['cup', 'pad'],
                'query_thresholds': [0.3],
                'N_per_inst': 100,
                'boundaries': boundaries,
                'resize_ratio': 0.5,
            }
        }

    # compute robot pcd
    num_bots = qpos_seq.shape[0] // 8
    curr_qpos = qpos_seq
    qpos_dim = curr_qpos.shape[0] // num_bots

    # compute robot pcd
    robot_pcd_ls = []
    for rob_i in range(num_bots):
        robot_pcd = kin_helper.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(kin_helper.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
        robot_base_pose_in_world = data_dict['observations']['robot_base_pose_in_world'][t, rob_i]
        robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
        robot_pcd_ls.append(robot_pcd)
    robot_pcd = np.concatenate(robot_pcd_ls, axis=0)

    pcd, pcd_colors = aggr_point_cloud_from_data(color_seq, depth_seq, intri_seq, extri_seq, downsample=False, boundaries=boundaries, out_o3d=False, excluded_pts=robot_pcd, exclude_threshold=0.02) # (N, 3), (N, 3)
    robot_base_pose_in_world = data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
    pcd = np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).transpose() # (4, N)
    pcd = pcd[:3].transpose() # (N, 3)

    # vis d3fields
    aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feat_ls, rob_mesh_ls = d3fields_proc_for_vis(
        fusion=fusion,
        shape_meta=d3fields_shape_meta,
        color_seq=color_seq[None],
        depth_seq=depth_seq[None],
        extri_seq=extri_seq[None],
        intri_seq=intri_seq[None],
        robot_base_pose_in_world_seq=data_dict['observations']['robot_base_pose_in_world'][t:t+1],
        teleop_robot=kin_helper,
        qpos_seq=qpos_seq[None],
        exclude_threshold=0.02,
        return_raw_feats=True,
    )

    if len(aggr_feats_ls) == 0:
        return pcd, pcd_colors, aggr_src_pts_ls[0], None, None, rob_mesh_ls[0]
    else:
        return pcd, pcd_colors, aggr_src_pts_ls[0], aggr_feats_ls[0], aggr_raw_feat_ls[0], rob_mesh_ls[0]


### hyperparam
data_root = "/home/yixuan/gendp"

# scene_name = "knife_1"
# ckpt_path = f"{data_root}/data/outputs/knife/checkpoints/latest.ckpt"
# dataset_dir = f"{data_root}/data/outputs/knife/eval_0"
# epi_path = f"{dataset_dir}/episode_0.hdf5"
# vis_time = 21 # the frame to visualize

# scene_name = "knife_2"
# ckpt_path = f"{data_root}/data/outputs/knife/checkpoints/latest.ckpt"
# dataset_dir = f"{data_root}/data/outputs/knife/eval_1"
# epi_path = f"{dataset_dir}/episode_0.hdf5"
# vis_time = 13 # the frame to visualize

# scene_name = "can_1"
# ckpt_path = f"{data_root}/data/outputs/can/checkpoints/epoch=400.ckpt"
# dataset_dir = f"{data_root}/data/outputs/can/eval_0"
# epi_path = f"{dataset_dir}/episode_0.hdf5"
# vis_time = 11 # the frame to visualize

scene_name = "can_2"
ckpt_path = f"{data_root}/data/outputs/can/checkpoints/epoch=400.ckpt"
dataset_dir = f"{data_root}/data/outputs/can/eval_1"
epi_path = f"{dataset_dir}/episode_0.hdf5"
vis_time = 30 # the frame to visualize

### load checkpoint
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)

# [Optional] compatiable with old model due to renaming
with open_dict(cfg):
    cfg.policy.shape_meta.obs.d3fields.info.N_gripper = cfg.policy.shape_meta.obs.d3fields.info.N_per_inst
    cfg.shape_meta.obs.d3fields.info.N_gripper = cfg.shape_meta.obs.d3fields.info.N_per_inst
    cfg.task.dataset.shape_meta.obs.d3fields.info.N_gripper = cfg.task.dataset.shape_meta.obs.d3fields.info.N_per_inst
    cfg.task.shape_meta.obs.d3fields.info.N_gripper = cfg.task.shape_meta.obs.d3fields.info.N_per_inst

# load workspace
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# load policy
policy: BaseImagePolicy
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

device = torch.device('cuda')
policy.eval().to(device)

# set inference params
policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
policy.num_inference_steps = 16 # DDIM inference iterations
noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    set_alpha_to_one=True,
    steps_offset=0,
    prediction_type='epsilon'
)
if 'd3fields' in cfg.task.dataset.shape_meta['obs']:
    # cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.022 # knife
    cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.025 # can
policy.noise_scheduler = noise_scheduler

# create dataset class
cfg.data_root = data_root
cfg.task.dataset.dataset_dir = dataset_dir
cfg.task.dataset.use_cache = True
with open_dict(cfg.task.dataset):
    cfg.task.dataset.vis_input = False
dataset : RealDataset = hydra.utils.instantiate(cfg.task.dataset)

raw_data_dict, _ = load_dict_from_hdf5(epi_path)

fusion = Fusion(num_cam=4, dtype=torch.float16)
kin_helper = KinHelper(robot_name='trossen_vx300s_v3')

i = vis_time
data = dataset[i]
rotate_mat = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]]) # rotate point cloud for better viewpoint

# visualize raw observation
start_time = time.time()
vis_shape_meta = cfg.task.dataset.shape_meta.copy()
if 'd3fields' in vis_shape_meta['obs']:
    vis_shape_meta['obs']['d3fields']['info']['boundaries']['z_lower'] = 0.01
    vis_shape_meta['obs']['d3fields']['info']['boundaries']['z_upper'] = 0.2
    vis_shape_meta['obs']['d3fields']['info']['boundaries']['x_lower'] = -0.3
    vis_shape_meta['obs']['d3fields']['info']['boundaries']['x_upper'] = 0.1
    vis_shape_meta['obs']['d3fields']['shape'][1] = 40000
pcd, pcd_colors, feats_pcd, distilled_feats, raw_feats, rob_mesh = extract_pcd_from_data_dict(raw_data_dict, i, fusion, vis_shape_meta, kin_helper)

pcd_down, pcd_colors_down = voxel_downsample(pcd, pcd_color=pcd_colors, voxel_size=0.005)
pcd_down = (rotate_mat @ pcd_down.T).T
raw_pcd = np2o3d(pcd_down, color=pcd_colors_down)

vis_pcd_plotly([raw_pcd], size_ls=[4], output_name=f'{scene_name}_raw_obs')

# infer actions
obs_dict = data['obs']
gt_action = data['action'][None].to(device)
for key in obs_dict.keys():
    obs_dict[key] = obs_dict[key].unsqueeze(0).to(device)
with torch.no_grad():
    result = policy.predict_action(obs_dict)
    pred_action = result['action_pred'] # (N, T, 10 * num_bots)
    num_bots = pred_action.shape[-1] // 10

    # Offset prediction action for more clear visualization.
    # Otherwise, the action represents pose at gripper bar, which is unclear to see
    N, T = pred_action.shape[:2]
    pred_action = pred_action.reshape(N*T, num_bots, 10)
    pred_action_rot = pred_action[:, :, 3:9] # (N, T, 6)
    pred_action_mat = rotation_6d_to_matrix(pred_action_rot).cpu().numpy() # (N, T, 3, 3)
    theta = -0.2
    ee_rot_adj = np.array([[np.cos(theta), 0, -np.sin(theta)],
                            [0, 1, 0],
                            [np.sin(theta), 0, np.cos(theta)]]) # (3, 3)
    ee_rot_adj = np.tile(ee_rot_adj[None, None], (pred_action_mat.shape[0], pred_action_mat.shape[1], 1, 1)) # (N, T, 3, 3)
    pred_action_mat = pred_action_mat @ ee_rot_adj # (N, T, 3, 3)
    ee_axis = pred_action_mat[:, :, :, 0] # (N, T, 3)

    pred_action_pos = pred_action[:, :, :3].cpu().numpy() # (N, T, 3)
    offset_val = 0.10
    pred_action_pos = pred_action_pos + offset_val * ee_axis # (N, T, 3)
    pred_action_pos = pred_action_pos.reshape(N, T, num_bots*3) # (N, T, 3*num_bots)

# visualize inputs to the policy
cmap = matplotlib.colormaps.get_cmap('viridis')
d3fields = obs_dict['d3fields'][0, -1].cpu().numpy().transpose() # (N, 3+D)
pred_act_pos = pred_action[0, :, :3].cpu().numpy()
d3fields_geo = d3fields[:, :3]
d3fields_sim = d3fields[:, 3]
d3fields_sim_color = cmap(d3fields_sim)[:,:3]
d3fields_geo = (rotate_mat @ d3fields_geo.T).T
input_d3fields_o3d = np2o3d(d3fields_geo, color=d3fields_sim_color)
vis_pcd_plotly([input_d3fields_o3d], size_ls=[4], output_name=f'{scene_name}_input')

# visualize 3D semantic fields
d3fields_sim = distilled_feats[:, 0]
vis_boundaries = cfg.task.dataset.shape_meta['obs']['d3fields']['info']['boundaries'].copy()
vis_boundaries['z_lower'] = 0.004
robot_base_pose_in_world = raw_data_dict['observations']['robot_base_pose_in_world'][0, 0] # (4, 4)
d3fields_bound_mask = (feats_pcd[:, 2] > vis_boundaries['z_lower'])
d3fields_sim[~d3fields_bound_mask] = 0
d3fields_sim_color = cmap(d3fields_sim)[:,:3]
high_sim_mask = d3fields_sim > 0.6

# transform feats_pcd to robot base frame
feats_pcd = (rotate_mat @ feats_pcd.T).T # transform to better viewpoint in plotly
d3fields_o3d = np2o3d(feats_pcd, color=d3fields_sim_color)
vis_pcd_plotly([d3fields_o3d], size_ls=[4], output_name=f'{scene_name}_d3fields')

# visualize action
pred_act_pos = pred_action_pos[0] # (T, 3*num_bots)
pred_act_pos = pred_act_pos.reshape(T * num_bots, 3) # (T*num_bots, 3)
action_cmap = cm.get_cmap('plasma')
act_len = pred_act_pos.shape[0] // num_bots
pred_act_pos_color = action_cmap(np.arange(act_len)/(act_len-1))[:,:3]
pred_act_pos_color = np.repeat(pred_act_pos_color, num_bots, axis=0) # (T, 3)
pred_act_pos_color = np.tile(pred_act_pos_color, (num_bots, 1)) # (T*num_bots, 3)
pred_act_pos = (rotate_mat @ pred_act_pos.T).T
vis_pcd_plotly([np2o3d(pred_act_pos, color=pred_act_pos_color), d3fields_o3d], size_ls=[6, 4], output_name=f'{scene_name}_action')
