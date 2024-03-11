import os
from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from gild.common.aloha_utils import (
    torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions, prep_puppet_robot,
    START_ARM_POSE, START_EE_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, DT, MASTER2PUPPET_JOINT_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_JOINT2POS, PUPPET_POS2JOINT
)
from gild.real_world.single_realsense import SingleRealsense
from gild.real_world.aloha_bimanual_puppet import AlohaBimanualPuppet
from gild.real_world.aloha_bimanual_master import AlohaBimanualMaster
from gild.real_world.video_recorder import VideoRecorder
from d3fields.utils.draw_utils import aggr_point_cloud_from_data

class MultiRealsense:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,720),
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
            serial_numbers = SingleRealsense.get_connected_devices_serial()
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
            cameras[serial] = SingleRealsense(
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
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)
    
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
        pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

def test_aloha_calibration(sides=['right', 'left']):
    from gild.common.kinematics_utils import KinHelper
    from d3fields.utils.draw_utils import np2o3d

    vis_rob = True

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
    tool_names = ['pusher_v2', 'dustpan_v2']
    puppet_robot = AlohaBimanualPuppet(shm_manager=shm_manager, verbose=False, frequency=50, robot_sides=['right', 'left'], init_qpos=init_qpos)
    puppet_robot.start()
    master_robot = AlohaBimanualMaster(shm_manager=shm_manager, verbose=False, frequency=50, robot_sides=['right', 'left'])
    master_robot.start()
    time.sleep(10)

    # visualize realsense point cloud
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(exposure=200, gain=64)
        realsense.set_white_balance(white_balance=2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        curr_pcd = o3d.geometry.PointCloud()
        curr_base_pcd = o3d.geometry.PointCloud()
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        iter_idx = 0
        while True:
            out = realsense.get()

            # set robot joint pos
            master_state = master_robot.get_motion_state()
            target_state = master_state['joint_pos'].copy()
            for rob_i in range(2):
                target_state[7*rob_i + 6] = MASTER2PUPPET_JOINT_FN(master_state['joint_pos'][7*rob_i + 6])
            target_time = time.time() + 0.1
            puppet_robot.set_target_joint_pos(target_state, target_time=target_time)

            if vis_rob:
                # compute robot base point cloud
                curr_robot_joint_pos = puppet_robot.get_state()['curr_full_joint_pos']
                # base_pcd = kin_helper.compute_robot_pcd(np.zeros(8), link_names=['vx300s/base_link'], num_pts=[10000])
                base_pcd_ls = []
                for rob_i, side in enumerate(sides):
                    base_pcd = kin_helper.compute_robot_pcd(curr_robot_joint_pos[rob_i*8:(rob_i+1)*8], pcd_name='rob_pcd')
                    tool_pcd = kin_helper.compute_tool_pcd(curr_robot_joint_pos[rob_i*8:(rob_i+1)*8], tool_name=tool_names[rob_i], pcd_name='tool_pcd')
                    base_pcd = np.concatenate([base_pcd, tool_pcd], axis=0)
                    base_pcd_i = base_pcd.copy()
                    base_pcd_i = robot_base_in_world[rob_i] @ np.concatenate([base_pcd_i, np.ones((base_pcd_i.shape[0], 1))], axis=-1).T
                    base_pcd_i = base_pcd_i[:3].T
                    base_pcd_ls.append(base_pcd_i)
                base_pcd = np.concatenate(base_pcd_ls, axis=0)
                base_pcd_o3d = np2o3d(base_pcd)

            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())
            extrinsics = np.stack(value['extrinsics'] for value in out.values())
            # for i in range(colors.shape[0]):
            #     pcd_i = aggr_point_cloud_from_data(colors=colors[i:i+1],
            #                                     depths=depths[i:i+1],
            #                                     Ks=intrinsics[i:i+1],
            #                                     poses=extrinsics[i:i+1],
            #                                     downsample=False)
            #     origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
            #     # visualize pcd_i
            #     o3d.visualization.draw_geometries([pcd_i, origin] + base_pcd_o3d_ls)
            
            # visualize all
            pcd = aggr_point_cloud_from_data(colors=colors,
                                            depths=depths,
                                            Ks=intrinsics,
                                            poses=extrinsics,
                                            downsample=False)
            curr_pcd.points = pcd.points
            curr_pcd.colors = pcd.colors
            curr_base_pcd.points = base_pcd_o3d.points
            curr_base_pcd.colors = base_pcd_o3d.colors
            if iter_idx == 0:
                visualizer.add_geometry(curr_pcd)
                if vis_rob:
                    visualizer.add_geometry(curr_base_pcd)
                visualizer.add_geometry(origin)
            
            visualizer.update_geometry(curr_pcd)
            if vis_rob:
                visualizer.update_geometry(curr_base_pcd)

            view_control = visualizer.get_view_control()
            view_control.set_front([ -0.78451141301854455, -0.30765995570025595, 0.53841173325083047 ])
            view_control.set_lookat([ -0.25162616981933222, -0.12788398312192117, 0.23992266979173191 ])
            view_control.set_up([ 0.55335827616805278, 0.044557205847718349, 0.8317507280449864 ])
            view_control.set_zoom(0.23999999999999957)

            visualizer.poll_events()
            visualizer.update_renderer()

            if iter_idx == 0:
                visualizer.run()
            iter_idx += 1

def calibrate_all():
    with MultiRealsense(
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
