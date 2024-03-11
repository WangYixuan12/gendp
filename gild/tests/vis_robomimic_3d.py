import h5py
import numpy as np
import open3d as o3d
import copy

import cv2

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def aggr_point_cloud_from_data(colors, depths, Ks, poses, downsample=True, masks=None):
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    colors = colors / 255.
    start = 0
    end = N
    # end = 2
    step = 1
    # visualize scaled COLMAP poses
    pcds = []
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 1.5)
        else:
            mask = masks[i]
        # mask = np.ones_like(depth, dtype=bool)
        pcd = depth2fgpcd(depth, mask, cam_param)
        
        pose = poses[i]
        pose = np.linalg.inv(pose)
        
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        
        # plt.subplot(1, 4, 1)
        # plt.imshow(trans_pcd[:, 0].reshape(H, W))
        # plt.subplot(1, 4, 2)
        # plt.imshow(trans_pcd[:, 1].reshape(H, W))
        # plt.subplot(1, 4, 3)
        # plt.imshow(trans_pcd[:, 2].reshape(H, W))
        # plt.subplot(1, 4, 4)
        # plt.imshow(color)
        # plt.show()
        
        pcd_o3d = np2o3d(trans_pcd, color[mask])
        # downsample
        if downsample:
            radius = 0.01
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
        pcds.append(pcd_o3d)
    aggr_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        aggr_pcd += pcd
    return aggr_pcd


# In[2]:


data_path = '/DIR_TO_HDF5/image_rgbd.hdf5'


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
