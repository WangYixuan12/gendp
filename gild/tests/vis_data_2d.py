#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np

import cv2


# In[2]:


data_path = '/media/yixuan_2T/diffusion_policy/data/robomimic/datasets/can/mh/image_rgb_84_with_ref.hdf5'


# In[3]:


f = h5py.File(data_path, 'r')


# In[4]:


f


# In[5]:


f.keys()


# In[6]:


print(len(list(f['data'].keys())))
print(list(f['data'].keys()))


# In[7]:


f['data']['demo_0']['obs'].keys()


# In[8]:


np.array(f['data']['demo_0']['obs']['agentview_extrinsic']).shape


# In[9]:


camera_names = ['agentview', 'robot0_eye_in_hand']

T, H, W = np.array(f['data']['demo_0']['obs']['agentview_image']).shape[:3]

vid_ls = [cv2.VideoWriter('out_{}.avi'.format(camera_name),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (W,H)) for camera_name in camera_names]

for t in range(T):
    for cam_idx, camera_name in enumerate(camera_names):
        color = np.array(f['data']['demo_0']['obs'][camera_name + '_image'])[t,:,:,:][..., ::-1]
        vid_ls[cam_idx].write(color)

for vid in vid_ls:
    vid.release()

