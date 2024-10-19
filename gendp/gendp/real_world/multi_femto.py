import os
from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
from pyorbbecsdk import Context, DeviceList, \
    OBPropertyID, OBSensorType, OBFormat, OBAlignMode, \
        Pipeline, Config, StreamProfileList, VideoFrame
import numpy as np
import open3d as o3d
from gendp.real_world.single_femto import SingleFemto
from gendp.real_world.aloha_bimanual_puppet import AlohaBimanualPuppet
from gendp.real_world.video_recorder import VideoRecorder
from d3fields.utils.draw_utils import aggr_point_cloud_from_data

class MultiFemto:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,960),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleFemto.get_connected_devices_serial()
            print(f'Found {len(serial_numbers)} connected devices: {serial_numbers}')
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)

        video_recorder = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)

        cameras = dict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleFemto(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )
        
        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])

    def set_depth_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value[i])
    
    def set_depth_preset(self, preset):
        n_camera = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 1.0)
        else:
            # manual exposure
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 0.0)
            if exposure is not None:
                self.set_color_option(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure)
            if gain is not None:
                self.set_color_option(OBPropertyID.OB_PROP_COLOR_GAIN_INT, gain)
    
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, 1.0)
        else:
            # manual exposure
            self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, 0.0)
            if exposure is not None:
                self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, exposure)
            if gain is not None:
                self.set_depth_option(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, 1.0)
        else:
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, 0.0)
            self.set_color_option(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, white_balance)
    
    def get_intrinsics(self):
        return np.array([c.get_intrinsics() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def calibrate_extrinsics(self, visualize=True, board_size=(6, 9), squareLength=0.03, markerLength=0.022):
        for camera in self.cameras.values():
            camera.calibrate_extrinsics(visualize=visualize, board_size=board_size, squareLength=squareLength, markerLength=markerLength)


def visualize_calibration_result():
    # import matplotlib.pyplot as plt
    with MultiFemto(
        resolution=(1280, 960),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        for _ in range(200):
            out = realsense.get()

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
        for i in range(colors.shape[0]):
            # plt.subplot(1, 2, 1)
            # plt.imshow(colors[i])
            # plt.subplot(1, 2, 2)
            # plt.imshow(depths[i])
            # plt.show()
            pcd_i = aggr_point_cloud_from_data(colors=colors[i:i+1],
                                                depths=depths[i:i+1],
                                                Ks=intrinsics[i:i+1],
                                                poses=extrinsics[i:i+1],
                                                downsample=False)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries([pcd_i, origin])
        pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries([pcd, origin])

def test_aloha_calibration(sides=['right', 'left']):
    from gendp.common.kinematics_utils import KinHelper
    from d3fields.utils.draw_utils import np2o3d

    # pose manual guess
    robot_base_in_world = np.array([[[0,-1,0,-0.13],
                                    [1,0,0,-0.51],
                                    [0,0,1,0.02],
                                    [0,0,0,1]],
                                    [[0,1,0,-0.13],
                                    [-1,0,0,0.27],
                                    [0,0,1,0.02],
                                    [0,0,0,1]],])
    # robot_base_in_world = np.array([[1.0, 0, 0.0, -0.52],
    #                                 [0, 1.0, 0.0, -0.06],
    #                                 [0.0, 0.0, 1.0, 0.03],
    #                                 [0.0, 0.0, 0.0, 1.0]])
    
    # save current robot base pose in world
    curr_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(curr_path, 'aloha_extrinsics')
    os.system(f'mkdir -p {save_dir}')
    for rob_i, side in enumerate(sides):
        np.save(os.path.join(save_dir, f'{side}_base_pose_in_world.npy'), robot_base_in_world[rob_i])

    # visualize robot base point cloud
    kin_helper = KinHelper(robot_name='trossen_vx300s_v3')
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    init_qpos = np.array([0.0, -0.865, 0.6, 0.0, 1.35, 0.0, -0.5, 0.0, -0.94, 0.72, 0.0, 1.31, 0.0, -0.5])
    tool_names = ['pusher', 'dustpan']
    puppet_robot = AlohaBimanualPuppet(shm_manager=shm_manager, verbose=False, frequency=50, robot_sides=['right', 'left'], init_qpos=init_qpos)
    puppet_robot.start()
    time.sleep(10)
    curr_robot_joint_pos = puppet_robot.get_state()['curr_full_joint_pos']
    # base_pcd = kin_helper.compute_robot_pcd(np.zeros(8), link_names=['vx300s/base_link'], num_pts=[10000])
    base_pcd_o3d_ls = []
    for rob_i, side in enumerate(sides):
        base_pcd = kin_helper.compute_robot_pcd(curr_robot_joint_pos[rob_i*8:(rob_i+1)*8])
        tool_pcd = kin_helper.compute_tool_pcd(curr_robot_joint_pos[rob_i*8:(rob_i+1)*8], tool_name=tool_names[rob_i])
        base_pcd = np.concatenate([base_pcd, tool_pcd], axis=0)
        base_pcd_i = base_pcd.copy()
        base_pcd_i = robot_base_in_world[rob_i] @ np.concatenate([base_pcd_i, np.ones((base_pcd_i.shape[0], 1))], axis=-1).T
        base_pcd_i = base_pcd_i[:3].T
        base_pcd_o3d = np2o3d(base_pcd_i)
        base_pcd_o3d_ls.append(base_pcd_o3d)

    # visualize realsense point cloud
    with MultiFemto(
        resolution=(640, 360),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(exposure=200, gain=64)
        realsense.set_white_balance(white_balance=2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(2000, 16)
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

        colors = np.stack(value['color'] for value in out.values())[..., ::-1]
        depths = np.stack(value['depth'] for value in out.values()) / 1000.
        intrinsics = np.stack(value['intrinsics'] for value in out.values())
        extrinsics = np.stack(value['extrinsics'] for value in out.values())
        for i in range(colors.shape[0]):
            pcd_i = aggr_point_cloud_from_data(colors=colors[i:i+1],
                                               depths=depths[i:i+1],
                                               Ks=intrinsics[i:i+1],
                                               poses=extrinsics[i:i+1],
                                               downsample=False)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
            # visualize pcd_i
            o3d.visualization.draw_geometries([pcd_i, origin] + base_pcd_o3d_ls)
        
        # visualize all
        pcd = aggr_point_cloud_from_data(colors=colors,
                                         depths=depths,
                                         Ks=intrinsics,
                                         poses=extrinsics,
                                         downsample=False)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin] + base_pcd_o3d_ls)

def calibrate_all():
    with MultiFemto(
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        while True:
            # realsense.calibrate_extrinsics(visualize=True, board_size=(6,9), squareLength=0.03, markerLength=0.022)
            realsense.set_exposure(exposure=200, gain=16)
            realsense.calibrate_extrinsics(visualize=True, board_size=(6,5), squareLength=0.04, markerLength=0.03)
            time.sleep(0.1)

def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

if __name__ == '__main__':
    # calibrate_all()
    visualize_calibration_result()
    # test_aloha_calibration()
