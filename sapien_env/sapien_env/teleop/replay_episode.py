import numpy as np
import os
import h5py
import cv2
import sys
sys.path.append(os.environ['SAPIEN_ROOT'])
import matplotlib.pyplot as plt
from sapien_env.rl_env.relocate_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        #TODO image
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
    return qpos, qvel, action, image_dict

def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

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

def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')

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
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
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
    
    # Set data dir
    dataset_dir = 'data/scripted/scripted_random005'
    dataset_name = "episode_0"
    qpos, qvel,action,image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, 0.02, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    print(action.shape)

    re_simulate = False
    if re_simulate:
        max_timesteps = 1000
        for i in range(max_timesteps):
            single_action = action[i]
            obs, reward, done, _ = env.step(single_action[:action.shape[0]])
            if plot_contact:
                contact_data = env.sensors
                plot_contact_color_map(contact_data, sensor_number_dim)
            gui.render()


if __name__ == '__main__':
    main_env()