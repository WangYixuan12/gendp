import numpy as np
import cv2
import os
import sys
sys.path.append(os.environ['SAPIEN_ROOT'])
from sapien_env.rl_env.franka_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.rl_env.para import ARM_INIT
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300

def main_env():
    env = RelocateRLEnv(use_gui=True, robot_name="panda",frame_skip=10, use_visual_obs=False, use_ray_tracing=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
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

    plot_contact =True
    teleop = TeleopRobot(robot_name="panda")
    print(env.robot.get_qpos()[:])
    cartisen_action =teleop.init_cartisen_action_franka(env.robot.get_qpos()[:9])
    action = np.zeros(8)
    arm_dof = env.arm_dof
    # gui.viewer.toggle_pause(True)
    print(arm_dof)
    # env.reset()
    for i in range(10000):
        # print(cartisen_action)
        cartisen_action = teleop.keyboard_control(gui.viewer, cartisen_action)
        action[:arm_dof] = teleop.ik_panda(env.robot.get_qpos()[:],cartisen_action)
        action[arm_dof:] = cartisen_action[6]
        # print(action)
        obs, reward, done, _ = env.step(action[:arm_dof+1])
        gui.render()


if __name__ == '__main__':
    main_env()