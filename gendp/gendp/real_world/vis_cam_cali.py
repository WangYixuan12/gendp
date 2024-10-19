import time
import argparse

import numpy as np
import open3d as o3d
from d3fields.utils.draw_utils import aggr_point_cloud_from_data
from gendp.real_world.multi_realsense import MultiRealsense

def visualize_calibration_result(iterative=False):
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(200, 64)
        realsense.set_white_balance(2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

            # # visualize depth
            # import matplotlib.cm as cm
            # import cv2
            # depth_1 = out[1]['depth'] / 1000.
            # cmap = cm.get_cmap('jet')
            # depth_min = 0.2
            # depth_max = 0.8
            # depth_1 = (depth_1 - depth_min) / (depth_max - depth_min)
            # depth_1 = np.clip(depth_1, 0, 1)
            # depth_1_vis = cmap(depth_1.reshape(-1))[..., :3].reshape(depth_1.shape + (3,))
            # depth_1_vis = depth_1_vis[..., ::-1]
            # cv2.imshow('depth_1', depth_1_vis)
            # cv2.waitKey(1)

        colors = np.stack(value['color'] for value in out.values())[..., ::-1]
        depths = np.stack(value['depth'] for value in out.values()) / 1000.
        intrinsics = np.stack(value['intrinsics'] for value in out.values())
        extrinsics = np.stack(value['extrinsics'] for value in out.values())
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        if iterative:
            for i in range(colors.shape[0]):
                pcd = aggr_point_cloud_from_data(colors=colors[i:i+1], depths=depths[i:i+1], Ks=intrinsics[i:i+1], poses=extrinsics[i:i+1], downsample=False)
                o3d.visualization.draw_geometries([pcd, origin])
        pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False)
        o3d.visualization.draw_geometries([pcd, origin])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterative', action='store_true')
    args = parser.parse_args()
    visualize_calibration_result(args.iterative)
