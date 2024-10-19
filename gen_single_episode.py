import os

import cv2
import numpy as np
import transforms3d
import sapien.core as sapien
from omegaconf import OmegaConf
import hydra

from sapien_env.rl_env.mug_collect_env import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS, YX_TABLE_TOP_CAMERAS
from gendp.common.data_utils import save_dict_to_hdf5
from gendp.common.kinematics_utils import KinHelper

def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic

def transform_action_from_world_to_robot(action : np.ndarray, pose : sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4,4))
    action_mat[:3,:3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3,3] = action[:3]
    action_mat[3,3] = 1
    action_mat_in_robot = np.matmul(np.linalg.inv(pose.to_transformation_matrix()),action_mat)
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3,3]
    action_robot[3:6] = transforms3d.euler.mat2euler(action_mat_in_robot[:3,:3],axes='sxyz')
    action_robot[6] = action[6]
    return action_robot

def task_to_cfg(task, manip_obj=None):
    if task == 'hang_mug':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.hang_mug_env.HangMugRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'nescafe_mug' if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.hang_mug_scripted_policy.SingleArmPolicy',
            }
        )
    elif task == 'mug_collect':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.mug_collect_env.MugCollectRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'pepsi' if manip_obj is None else manip_obj,
                'randomness_level': 'half'
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.mug_collect_scripted_policy.SingleArmPolicy',
            }
        )
    elif task == 'pen_insertion':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.pen_insertion_env.PenInsertionRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'pencil' if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.pen_insertion_scripted_policy.SingleArmPolicy',
            }
        )
    else:
        raise ValueError(f'Unknown task {task}')
    return cfg, policy_cfg

def main_env(episode_idx, dataset_dir, headless, mode, task_name, manip_obj=None):
    # initialize env
    os.system(f'mkdir -p {dataset_dir}')
    kin_helper = KinHelper(robot_name="panda")

    cfg, policy_cfg = task_to_cfg(task_name, manip_obj=manip_obj)
    
    with open(os.path.join(dataset_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f.name)

    # collect data
    env : BaseRLEnv = hydra.utils.instantiate(cfg)
    env.seed(episode_idx)
    env.reset()
    arm_dof = env.arm_dof
    
    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer,headless=headless)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)
    if not gui.headless:
        gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
        gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    scene = env.scene
    scene.step()
    
    timesteps = 0
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    scripted_policy = hydra.utils.instantiate(policy_cfg)
    
    # set up data saving hyperparameters
    init_poses = env.get_init_poses()
    data_dict = {
        'observations': 
            {'joint_pos': [],
             'joint_vel': [],
             'full_joint_pos': [], # this is to compute FK
             'robot_base_pose_in_world': [],
             'ee_pos': [],
             'ee_vel': [],
            #  'finger_pos': {},
             'images': {},},
        'joint_action': [],
        'cartesian_action': [],
        'info':
            {'init_poses': init_poses}
    }
    # finger_names = ['left_finger_link','right_finger_link']
    # for finger in finger_names:
    #     data_dict['observations']['finger_pos'][finger] = []
    cams = gui.cams
    for cam in cams:
        data_dict['observations']['images'][f'{cam.name}_color'] = []
        data_dict['observations']['images'][f'{cam.name}_depth'] = []
        data_dict['observations']['images'][f'{cam.name}_intrinsic'] = []
        data_dict['observations']['images'][f'{cam.name}_extrinsic'] = []
    attr_dict = {
        'sim': True,
    }
    config_dict = {
        'observations':
            {
                'images': {}
            }
    }
    for cam_idx, cam in enumerate(gui.cams):
        color_save_kwargs = {
            'chunks': (1, cam.height, cam.width, 3), # (1, 480, 640, 3)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint8',
        }
        depth_save_kwargs = {
            'chunks': (1, cam.height, cam.width), # (1, 480, 640)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint16',
        }
        config_dict['observations']['images'][f'{cam.name}_color'] = color_save_kwargs
        config_dict['observations']['images'][f'{cam.name}_depth'] = depth_save_kwargs

    
    while True:
        action = np.zeros(arm_dof+1)
        cartisen_action, quit = scripted_policy.single_trajectory(env,env.palm_link.get_pose(),mode=mode)
        # transform cartisen_action from robot to world frame
        if quit:
            break
        cartisen_action_in_rob = transform_action_from_world_to_robot(cartisen_action,env.robot.get_pose())
        action[:arm_dof] = kin_helper.compute_ik_sapien(env.robot.get_qpos()[:],cartisen_action_in_rob)[:arm_dof]
        action[arm_dof:] = cartisen_action_in_rob[6]
        # print(action)
        obs, reward, done, _ = env.step(action[:arm_dof+1])
        rgbs, depths = gui.render(depth=True)
        
        data_dict['observations']['joint_pos'].append(env.robot.get_qpos()[:-1])
        data_dict['observations']['joint_vel'].append(env.robot.get_qvel()[:-1])
        data_dict['observations']['full_joint_pos'].append(env.robot.get_qpos())
        data_dict['observations']['robot_base_pose_in_world'].append(env.robot.get_pose().to_transformation_matrix())
        ee_translation = env.palm_link.get_pose().p
        ee_rotation = transforms3d.euler.quat2euler(env.palm_link.get_pose().q,axes='sxyz')
        ee_gripper = env.robot.get_qpos()[arm_dof]
        ee_pos = np.concatenate([ee_translation,ee_rotation,[ee_gripper]])
        ee_vel = np.concatenate([env.palm_link.get_velocity(),env.palm_link.get_angular_velocity(),env.robot.get_qvel()[arm_dof:arm_dof+1]])
        data_dict['observations']['ee_pos'].append(ee_pos)
        data_dict['observations']['ee_vel'].append(ee_vel)
        data_dict['joint_action'].append(action.copy())
        data_dict['cartesian_action'].append(cartisen_action.copy())
        # for finger_idx, finger in enumerate(finger_names):
        #     pose = env.finger_tip_links[finger_idx].get_pose()
        #     data_dict['observations']['finger_pos'][finger].append(pose.to_transformation_matrix())
        for cam_idx, cam in enumerate(gui.cams):
            data_dict['observations']['images'][f'{cam.name}_color'].append(rgbs[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_depth'].append(depths[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_intrinsic'].append(cam.get_intrinsic_matrix())
            data_dict['observations']['images'][f'{cam.name}_extrinsic'].append(cam.get_extrinsic_matrix())
        timesteps += 1
    if reward < 1:
        print("Failed at episode {}".format(episode_idx))
    data_dict = stack_dict(data_dict)
    save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
    if not gui.headless:
        gui.viewer.close()
        cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('episode_idx', help='random seed for the episode')
    parser.add_argument('dataset_dir', help='directory to save the dataset')
    parser.add_argument('task_name', help='task name, including hang_mug, mug_collect, pen_insertion')
    parser.add_argument('--headless', action='store_true', help='whether to run in headless mode')
    parser.add_argument('--obj_name', default=None, help='manipulated object name. The full list is shown in YX_DEFAULT_SCALE at sapien_env/sapien_env/utils/yx_object_utils.py')
    parser.add_argument('--mode', default='straight', help='mode for scripted policy. Examples are shown in generate_trajectory() at sapien_env/sapien_env/teleop/mug_collect_scripted_policy.py')
    args = parser.parse_args()

    main_env(episode_idx=int(args.episode_idx),
             dataset_dir=args.dataset_dir,
             headless=args.headless,
             mode=args.mode,
             manip_obj=args.obj_name,
             task_name=args.task_name)
