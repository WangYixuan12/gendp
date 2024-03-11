#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import cv2
import matplotlib.cm as cm
from gild.common.data_utils import load_dict_from_hdf5

# num_episodes = 1
# epi_s = 50
# epi_range = range(epi_s, epi_s+num_episodes)

# epi_range = [1, 18, 23, 24, 32, 36, 46, 54, 65, 69, 91, 94, 95]
# epi_range = [222, 272, 322, 372]
epi_range = [0]

# data_dir = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag'
# data_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_demo/mixed_mug_demo_400_v6'
# data_dir = '/media/yixuan_2T/diffusion_policy/data/real_aloha_demo/Guang_pen_v3'
# data_dir = '/media/yixuan_2T/diffusion_policy/data/outputs/knife/2024.01.23_knife_Guang_knife_v4_no_seg_distill_dino_N_1000_pn2.1_r_0.04/eval_subset'
data_dir = '/media/yixuan_2T/diffusion_policy/data/real_aloha_demo/heatmap_vis_shoe'
save_dir = f'{data_dir}/data_vis'
os.system(f'mkdir -p {save_dir}')

def vis_format(img, format):
    # img: (H, W, C) or (H, W) numpy array
    # format: 'bgr', 'rgb' or 'uint16depth' or 'float32depth'
    if format == 'bgr':
        return img
    elif format == 'rgb':
        return img[..., ::-1]
    elif format == 'uint16depth':
        cmap = cm.get_cmap('plasma')
        depth_norm = img / 1000.0
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = cmap(depth_norm)[:, :, :3]
        return (depth_vis * 255).astype(np.uint8)[..., ::-1]
    elif format == 'float32depth':
        cmap = cm.get_cmap('plasma')
        depth_norm = img / 1000.0
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = cmap(depth_norm)[:, :, :3]
        return (depth_vis * 255).astype(np.uint8)[..., ::-1]

for i in epi_range:
    print(f'episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    # obs_keys = ['left_bottom_view_color',
    #             'right_bottom_view_color',
    #             'left_top_view_color',
    #             'right_top_view_color',]
    # obs_format = ['bgr', 'bgr', 'bgr', 'bgr']
    obs_keys = [
                'camera_0_color',
                'camera_1_color',
                'camera_2_color',
                'camera_3_color',
                # 'camera_0_depth',
                # 'camera_1_depth',
                # 'camera_2_depth',
                ]
    obs_format = [
                'rgb',
                'rgb',
                'rgb',
                'rgb',
                # 'uint16depth',
                # 'uint16depth',
                # 'uint16depth'
                ]

    obs_ls = []
    for obs_key in obs_keys:
        obs_ls.append(data_dict['observations']['images'][obs_key])
    
    if False:
        save_keys = ['left_bottom_view_color',
                    'right_bottom_view_color',
                    'left_top_view_color',
                    'right_top_view_color',]
        for key in save_keys:
            img = data_dict['observations']['images'][key][0]
            cv2.imwrite(f'{key}_{i}.png', img)

    T = obs_ls[0].shape[0]

    vid_ls = []
    for obs_i, obs in enumerate(obs_ls):
        for t in range(T):
            img = vis_format(obs[t], obs_format[obs_i])
            cv2.imshow('img', img)
            cv2.waitKey(1)
            if t == 0:
                vid_ls.append(cv2.VideoWriter(f'{save_dir}/episode_{i}_{obs_keys[obs_i]}.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (img.shape[1], img.shape[0])))
            vid_ls[obs_i].write(img)
            cv2.imwrite(f'{save_dir}/episode_{i}_{obs_keys[obs_i]}_{t}.png', img)
        vid_ls[obs_i].release()
