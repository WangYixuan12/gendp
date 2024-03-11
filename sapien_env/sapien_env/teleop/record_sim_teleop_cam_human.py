import numpy as np
import os
import h5py
import cv2
import sys
import torch
sys.path.append(os.environ['SAPIEN_ROOT'])
from sapien_env.rl_env.relocate_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


def plot_contact_color_map(contact_data, sensor_number_dim):
    contact_data_left= contact_data[:sensor_number_dim]
    contact_data_right= contact_data[sensor_number_dim:]
    contact_data_left= contact_data_left.reshape((4,4))
    contact_data_right= contact_data_right.reshape((4,4))
    
    contact_data_left_norm = (contact_data_left - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
    contact_data_right_norm = (contact_data_right -  MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

    contact_data_left_norm_scaled = (contact_data_left_norm * 255).astype(np.uint8)
    contact_data_right_norm_scaled = (contact_data_right_norm * 255).astype(np.uint8)

    colormap_left = cv2.applyColorMap(contact_data_left_norm_scaled, cv2.COLORMAP_VIRIDIS)
    colormap_right = cv2.applyColorMap(contact_data_right_norm_scaled, cv2.COLORMAP_VIRIDIS)
    
    separator = np.ones((colormap_left.shape[0],1,3), dtype=np.uint8) * 255

    combined_colormap = np.concatenate((colormap_left, separator,colormap_right), axis=1)
    cv2.imshow("Left finger and Right finger Contact Data", combined_colormap )
    cv2.waitKey(1)



def main_env():
    env = RelocateRLEnv(use_gui=True, robot_name="trossen_vx300s_tactile_thin",
                        object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    # viewer = Viewer(base_env.renderer)
    # viewer.set_scene(base_env.scene)
    # base_env.viewer = viewer

    
    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.viewer.set_camera_rpy(0, -0.7, 0.01)
    gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
    scene = env.scene
    steps = 0
    scene.step()
    
    for link in env.robot.get_links():
        print(link.get_name(),link.get_pose())
    for joint in env.robot.get_active_joints():
        print(joint.get_name())

    plot_contact = True

    if plot_contact:
        # Create a named window with the WINDOW_NORMAL flag
        cv2.namedWindow("Left finger and Right finger Contact Data", cv2.WINDOW_NORMAL)

        # Set the window size
        cv2.resizeWindow("Left finger and Right finger Contact Data",WINDOW_WIDTH, WINDOW_HEIGHT)

        sensor_number_dim = int(env.sensors.shape[0]/2)
    teleop = TeleopRobot()
    cartisen_action =teleop.init_cartisen_action(env.robot.get_qpos()[:])
    action = np.zeros(7)
    arm_dof = env.arm_dof

    # Set data dir
    dataset_dir = "data/teleop_data"
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    
    cams = gui.cams
    for cam in cams:
        data_dict[f'/observations/images/{cam.name}'] = []
    
    max_timesteps = 1000
    
    for i in range(max_timesteps):
        cartisen_action = teleop.keyboard_control(gui.viewer, cartisen_action)
        action[:arm_dof] = teleop.ik_vx300s(env.robot.get_qpos()[:],cartisen_action)
        action[arm_dof:] = cartisen_action[6]
        obs, reward, done, _ = env.step(action[:7])
        data_dict['/observations/qpos'].append(env.robot.get_qpos()[:-1])
        data_dict['/observations/qvel'].append(env.robot.get_qvel()[:-1])
        data_dict['/action'].append(action.copy())
        for cam in gui.cams:
            cam.take_picture()
            rgb_tensor = torch.clamp(torch.utils.dlpack.from_dlpack(cam.get_dl_tensor("Color"))[..., :3], 0, 1) * 255
            rgb = rgb_tensor.type(torch.uint8).cpu().numpy()
            data_dict[f'/observations/images/{cam.name}'].append(rgb)
        
        if plot_contact:
            contact_data = env.sensors
            plot_contact_color_map(contact_data, sensor_number_dim)
        gui.render()
    
    import time
    t=time.localtime()

    episode_index = 1
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_index}'+time.strftime("%Y-%m-%d-%H-%M-%S", t))

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam in gui.cams:
            _ = image.create_dataset(cam.name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )


        qpos = obs.create_dataset('qpos', (max_timesteps, 7))
        qvel = obs.create_dataset('qvel', (max_timesteps, 7))
        action = root.create_dataset('action', (max_timesteps, 7))
        for name, array in data_dict.items():
            root[name][...] = array

if __name__ == '__main__':
    main_env()