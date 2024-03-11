#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sys

import h5py
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from gild.common.data_utils import load_dict_from_hdf5

### measure how jittery the action is

num_episodes = 99
epi_s = 1

data_dir = '/home/yixuan/general_dp/data/real_aloha_demo/pick_place_cup_v2'

all_action_diff = list()

for i in range(epi_s, epi_s+num_episodes):
    print(f'Loading data for episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'
    data_dict, _ = load_dict_from_hdf5(data_path)

    action_key = 'cartesian_action'
    
    # measure action diff
    action = data_dict[action_key]
    action_diff = np.diff(action[:,:3], axis=0)
    all_action_diff.append(action_diff)

all_action_diff = np.concatenate(all_action_diff, axis=0)
all_action_diff_norm = np.linalg.norm(all_action_diff, axis=-1)

num_abnormal = all_action_diff_norm > 0.03

print(num_abnormal.sum())

print(all_action_diff_norm.shape)

plt.boxplot(all_action_diff_norm[:,None])
plt.show()

