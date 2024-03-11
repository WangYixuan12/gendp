
from pynput import keyboard
from functools import cached_property
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer
import sys
sys.path.append('/home/bing4090/sapien_teleop')
from sapien_env.rl_env.relocate_env import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS

OBJECT_LIFT_LOWER_LIMIT = -0.03
ARM_INIT = np.array([-0.4, 0., 0])

class TeleopRobot():
    def __init__(self):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.vx300 = loader.load("assets/robot/trossen_description/vx300.urdf")
        self.trossen_model = self.vx300.create_pinocchio_model()
        self.joint_names = [joint.get_name() for joint in self.vx300.get_active_joints()]
        self.link_name2id = {self.vx300.get_links()[i].get_name(): i for i in range(len(self.vx300.get_links()))}

    def init_action(self,ee_link_pose):
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        action = np.zeros(6)
        p = ee_link_pose.p-ARM_INIT
        q=ee_link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        cartisen_action[0:3] = p
        cartisen_action[3:6] = eluer
        return action, cartisen_action
    
    def teleop(self,initial_qpos,cartesian):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        # print(pose)
        active_qmask= np.array([True,True,True,True,True,False,False])
        # end_effector_link_name = "vx300/gripper_bar_link"
        qpos = self.trossen_model.compute_inverse_kinematics(link_index=7, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask)  
        return qpos[0][0:5]
                
    def keyboard_control(self, viewer, cartesian):
        delta_position = 0.001
        constant_x = 1.2
        constant_y = 2
        delta_orientation = 0.005
        # x
        if viewer.window.key_down("i"):
            cartesian[0]+=delta_position*constant_x
        if viewer.window.key_down("k"):
            cartesian[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("j"):
            cartesian[1]+=delta_position*constant_y
        if viewer.window.key_down("l"):
            cartesian[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("u"):
            cartesian[2]+=delta_position
        if viewer.window.key_down("o"):
            cartesian[2]-=delta_position
        # roll
        if viewer.window.key_down("r"):
            cartesian[3]+=delta_orientation
        if viewer.window.key_down("f"):
            cartesian[3]-=delta_orientation
        # pitch
        if viewer.window.key_down("t"):
            cartesian[4]+=delta_orientation
        if viewer.window.key_down("g"):
            cartesian[4]-=delta_orientation
        # yaw
        if viewer.window.key_down("y"):
            cartesian[5]+=delta_orientation
        if viewer.window.key_down("h"):
            cartesian[5]-=delta_orientation

        # gripper open or close
        if viewer.window.key_down("z"):
            cartesian[6]+=0.001
        if viewer.window.key_down("x"):
            cartesian[6]-=0.001
        return cartesian


def main_env():
    env = RelocateRLEnv(use_gui=True, robot_name="trossen_vx300_tactile_map_4x4",
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
    y_list = np.zeros((1,env.sensors.shape[0]))

    if plot_contact:
        colormap = plt.cm.get_cmap('viridis') 
        x = []               
        y=[]
        left = plt.subplot(1,2,1)
        right = plt.subplot(1,2,2)
        norm = colors.Normalize(vmin=0, vmax=0.5)

    sensor_number_dim = int(env.sensors.shape[0]/2)
    teleop = TeleopRobot()
    ee_link_name = "vx300/ee_gripper_link"
    ee_link = [link for link in env.robot.get_links() if link.get_name() == ee_link_name][0]
    ee_link_pose = ee_link.get_pose()
    action, cartisen_action =teleop.init_action(ee_link_pose)
    
    for i in range(10000):
        cartisen_action = teleop.keyboard_control(gui.viewer, cartisen_action)
        action[:5] = teleop.teleop(env.robot.get_qpos()[:],cartisen_action)
        action[5:] = cartisen_action[6]
        obs, reward, done, _ = env.step(action[:6])

        if plot_contact and i%10==0:
            # x.append(i)
            contact_data = env.sensors
            contact_data_left= contact_data[:sensor_number_dim]
            contact_data_right= contact_data[sensor_number_dim:]
            contact_data_left= contact_data_left.reshape((4,4))
            contact_data_right= contact_data_right.reshape((4,4))
            left.imshow(contact_data_left, cmap=colormap, norm=norm)
            right.imshow(contact_data_right, cmap=colormap, norm=norm)
            plt.pause(0.001)
        gui.render()
        # env.render()




if __name__ == '__main__':
    main_env()
    plt.show()