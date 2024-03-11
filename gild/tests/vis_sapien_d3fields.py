import h5py
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
import numpy as np
import open3d as o3d
import copy

import cv2
import zarr
from gild.common.replay_buffer import ReplayBuffer

def text_3d(text, pos, direction=None, degree=0.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (1., 0., 0.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getbbox(text)[2:]

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

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

# cache_zarr_path = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/middle_view_demo_100/cache_ori.zarr.zip'
# cache_zarr_path = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/hang_mug/nescafe_mug_demo_10/cache.zarr.zip'
# cache_zarr_path = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/hang_mug/nescafe_mug_demo_1_closer_views_central_obj/cache.zarr.zip'
# cache_zarr_path = '/home/yixuan/general_dp/data/real_aloha_demo/pick_place_cup/cache.zarr.zip'
cache_zarr_path = '/media/yixuan_2T/diffusion_policy/data/real_aloha_demo/Guang_align_v4/cache_seg_distill_dino_eef.zarr.zip'

# 12, 17, 18, 19, 82 - 84

use_pca = False
if use_pca:
    pca_name = 'can'
    pca_path = f'pca_model/{pca_name}.pkl'
    import pickle
    pca_model = pickle.load(open(pca_path, 'rb'))

with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
    replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore())

d3fields = replay_buffer['/data/d3fields']
T = d3fields.shape[0]

colors = [np.array([1.0, 0.0, 0.0]),
          np.array([0.0, 1.0, 0.0]),
          np.array([0.0, 0.0, 1.0]),]

# # compute the number of missing objetcs
# missing_obj = 0
# for t in range(T // 150):
#     pts = d3fields[t * 150,:,:3]
#     if np.any(pts == 0):
#         missing_obj += 1
#         print(f'episode {t} has missing objects')

# print(f'number of missing objects: {missing_obj}')

for t in range(T):
    # depths = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_depth'])[t,:,:,0] for camera_name in camera_names], axis=0)
    # colors = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_image'])[t,:,:,:] for camera_name in camera_names], axis=0)
    # Ks = np.stack([np.array(f['data']['demo_0']['obs'][camera_name + '_intrinsic'])[t,:,:] for camera_name in camera_names], axis=0)
    # Rs = np.stack([np.linalg.inv(np.array(f['data']['demo_0']['obs'][camera_name + '_extrinsic'])[t,:,:]) for camera_name in camera_names], axis=0)
    # curr_pcd = aggr_point_cloud_from_data(colors, depths, Ks, Rs, downsample=False)
    # n_pts = d3fields.shape[1]
    # n_obj = n_pts // 100
    # n_sample = 100
    # if not use_pca:
    #     curr_pcd_color = np.zeros((n_pts, 3))
    #     for i in range(n_obj):
    #         curr_pcd_color[i*100:(i+1)*100,:] = colors[i]
    # else:
    #     curr_feats = d3fields[t,:,3:]
    #     curr_feats_after_pca = pca_model.transform(curr_feats)
    #     curr_pcd_color = np.zeros((n_pts, 3))
    #     for i in range(n_obj):
    #         for ch_i in range(3):
    #             curr_pcd_color[i*100:(i+1)*100, ch_i] = (curr_feats_after_pca[i*100:(i+1)*100, ch_i] - curr_feats_after_pca[i*100:(i+1)*100, ch_i].min()) / (curr_feats_after_pca[i*100:(i+1)*100, ch_i].max() - curr_feats_after_pca[i*100:(i+1)*100, ch_i].min())
    curr_pcd_color = None
    curr_pts = d3fields[t,:,:3]
    # curr_pts = curr_pts.reshape(n_obj, 100, 3)[:, :n_sample, :].reshape(n_obj * n_sample, 3)
    # curr_pcd_color = curr_pcd_color.reshape(n_obj, 100, 3)[:, :n_sample, :].reshape(n_obj * n_sample, 3)
    curr_pcd = np2o3d(curr_pts, curr_pcd_color)
    
    # o3d.visualization.draw_geometries([curr_pcd])
    # epi = t // 300
    # curr_text_pcd = text_3d(f'episode {epi}', [0, 0, 0.5], font_size=36, density=2, degree=90.0)
    
    if t == 0:
        pcd = copy.deepcopy(curr_pcd)
        # text_pcd = copy.deepcopy(curr_text_pcd)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        visualizer.add_geometry(pcd)
        # visualizer.add_geometry(text_pcd)
        
        visualizer.update_geometry(pcd)
        # visualizer.update_geometry(text_pcd)
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
        # text_pcd.points = curr_text_pcd.points
        # text_pcd.colors = curr_text_pcd.colors
        
        visualizer.update_geometry(pcd)
        # visualizer.update_geometry(text_pcd)
        visualizer.poll_events()
        visualizer.update_renderer()

        img = visualizer.capture_screen_float_buffer()
        img = np.asarray(img)[..., ::-1]
        img = np.asarray(img)
        vid.write((img*255).astype(np.uint8))
        
vid.release()
