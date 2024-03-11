#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import sys
sys.path.append('/home/yixuan/general_dp/')
from utils.draw_utils import aggr_point_cloud_from_data
import numpy as np
import open3d as o3d
import copy

import cv2


# In[2]:


data_path = '/media/yixuan_2T/diffusion_policy/data/robomimic/datasets/can/mh/image_rgbd.hdf5'


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

T = np.array(f['data']['demo_0']['obs']['agentview_depth']).shape[0]

for t in range(T):
    depths = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_depth'])[t,:,:,0] for camera_name in camera_names], axis=0)
    colors = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_image'])[t,:,:,:] for camera_name in camera_names], axis=0)
    Ks = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_intrinsic'])[t,:,:] for camera_name in camera_names], axis=0)
    Rs = np.stack([np.linalg.inv(np.array(f['data']['demo_0']['obs'][camera_name + '_extrinsic'])[t,:,:]) for camera_name in camera_names], axis=0)
    curr_pcd = aggr_point_cloud_from_data(colors, depths, Ks, Rs, downsample=False)
    
    # o3d.visualization.draw_geometries([curr_pcd])
    
    if t == 0:
        pcd = copy.deepcopy(curr_pcd)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        visualizer.add_geometry(pcd)
        
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.run()
        
        img = visualizer.capture_screen_float_buffer()
        img = np.asarray(img)[..., ::-1]
        
        vid = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (img.shape[1],img.shape[0]))
        vid.write((img*255).astype(np.uint8))
    else:
        pcd.points = curr_pcd.points
        pcd.colors = curr_pcd.colors
        
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()

        img = visualizer.capture_screen_float_buffer()
        img = np.asarray(img)[..., ::-1]
        img = np.asarray(img)
        vid.write((img*255).astype(np.uint8))
        
vid.release()


# In[10]:


colors = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_image'])[t,:,:,:] for camera_name in camera_names], axis=0)
colors.shape


# In[ ]:





# In[ ]:




