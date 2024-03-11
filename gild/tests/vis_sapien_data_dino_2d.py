#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sys
sys.path.append('/home/yixuan/general_dp')
sys.path.append('/home/yixuan/general_dp/diffusion_policy/d3fields_dev')

import h5py
import numpy as np
import cv2
import torch

from gild.common.data_utils import load_dict_from_hdf5
from d3fields.fusion import Fusion

num_episodes = 1
epi_s = 0

data_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/tmp'
save_dir = f'{data_dir}/vis'
os.system(f'mkdir -p {save_dir}')

model_type = 'dinov2'
device = 'cuda'
fusion = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float16)
pca_name = 'can'
pca_path = f'pca_model/{pca_name}.pkl'
pca_model = pickle.load(open(pca_path, 'rb'))

for i in range(epi_s, epi_s+num_episodes):
    print(f'episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    obs_key = 'left_bottom_view_color'

    obs = data_dict['observations']['images'][obs_key]

    T = obs.shape[0]

    for t in range(T):
        img = obs[t]
        H, W = img.shape[:2]
        params = {
            'patch_h': H // 10,
            'patch_w': W // 10,
        }
        feat_map = fusion.extract_features(img[None], params=params)
        feat_map = feat_map[0].cpu().numpy()
        if t == 0:
            os.system(f'mkdir -p temp')
            np.save(f'temp/feat_map_0_pepsi.npy', feat_map)
        feat_map_pca = pca_model.transform(feat_map.reshape(-1, feat_map.shape[-1])).reshape(feat_map.shape[:2] + (3,))
        if t == 0:
            min_val = feat_map_pca.min()
            max_val = feat_map_pca.max()
        for ch_i in range(3):
            feat_map_pca[:,:,ch_i] = (feat_map_pca[:,:,ch_i] - min_val) * 255 / (max_val - min_val)
        
        feat_map_pca_resize = cv2.resize(feat_map_pca, (W, H), interpolation=cv2.INTER_NEAREST)
        
        concat_img = np.concatenate([img, feat_map_pca_resize], axis=1).astype(np.uint8)
        
        cv2.imshow('img', concat_img)
        cv2.waitKey(10)
        # if t == 0:
        #     vid = cv2.VideoWriter(f'{save_dir}/episode_{i}.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (img.shape[1], img.shape[0]))
        # vid.write(img)
