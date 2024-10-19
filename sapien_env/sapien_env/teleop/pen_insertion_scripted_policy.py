import numpy as np
from pyquaternion import Quaternion
import transforms3d
import sapien.core as sapien

from sapien_env.rl_env.pen_insertion_env import PenInsertionRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS, YX_TABLE_TOP_CAMERAS
from gendp.common.kinematics_utils import KinHelper


class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        # quat = curr_quat + (next_quat - curr_quat) * t_frac
        # interpolate quaternion using slerp
        curr_quat_obj = Quaternion(curr_quat)
        next_quat_obj = Quaternion(next_quat)
        quat = Quaternion.slerp(curr_quat_obj, next_quat_obj, t_frac).elements
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self,env, ee_link_pose, mode='straight'):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose, mode)


        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        
        if len(self.trajectory) == 0:
            quit = True
            return None, quit
        else:
            quit = False

        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)


        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        eluer = transforms3d.euler.quat2euler(quat,axes='sxyz')
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper
        return cartisen_action, quit
    
    def generate_trajectory(self, env , ee_link_pose, mode='straight'):

        # mug_pose_mat = env.manipulated_object.get_pose().to_transformation_matrix()
        # grasp_pose_in_mug_mat = grasp_pose.to_transformation_matrix()
        # grasp_pose_in_world_mat = mug_pose_mat @ grasp_pose_in_mug_mat
        # grasp_pose_in_world = sapien.Pose.from_transformation_matrix(grasp_pose_in_world_mat)
        
        # pre_grasp_pose_in_mug_mat = pre_grasp_pose.to_transformation_matrix()
        # pre_grasp_pose_in_world_mat = mug_pose_mat @ pre_grasp_pose_in_mug_mat
        # pre_grasp_pose_in_world = sapien.Pose.from_transformation_matrix(pre_grasp_pose_in_world_mat)
        
        
        # place pose setting
        # mug_tree_mat = env.mug_tree.get_pose().to_transformation_matrix()
        
        # # place_q = np.array(transforms3d.euler.euler2quat(0, 0.0, 0.0, axes='sxyz').tolist())
        # place_pose = sapien.Pose(place_p,place_q)
        # place_pose_in_mug_tree_mat = place_pose.to_transformation_matrix()
        # place_pose_in_world_mat = mug_tree_mat @ place_pose_in_mug_tree_mat
        # place_pose_in_world = sapien.Pose.from_transformation_matrix(place_pose_in_world_mat)
        
        # post_place_pose = sapien.Pose(post_place_p,post_place_q)
        # post_place_pose_in_mug_tree_mat = post_place_pose.to_transformation_matrix()
        # post_place_pose_in_world_mat = mug_tree_mat @ post_place_pose_in_mug_tree_mat
        # post_place_pose_in_world = sapien.Pose.from_transformation_matrix(post_place_pose_in_world_mat)
        
        # leave_pose = sapien.Pose(leave_p,leave_q)
        # leave_pose_in_mug_tree_mat = leave_pose.to_transformation_matrix()
        # leave_pose_in_world_mat = mug_tree_mat @ leave_pose_in_mug_tree_mat
        # leave_pose_in_world = sapien.Pose.from_transformation_matrix(leave_pose_in_world_mat)
        pen_pose = env.manipulated_object.get_pose()
        pen_pose_mat = pen_pose.to_transformation_matrix()
        if env.manip_obj_name == 'pencil':
            pen_postion = pen_pose.p+np.array([0.,0.,0.105])
        elif env.manip_obj_name == 'pencil_2':
            pen_postion = pen_pose.p+np.array([0.,0.,0.105])
        elif env.manip_obj_name == 'pencil_4':
            pen_postion = pen_pose.p+np.array([0.,0.,0.105])
        else:
            pen_postion = pen_pose.p+np.array([0.,0,0.105])
        eef_pose_rot = pen_pose_mat[0:3,0:3] @ np.array([[0,-1,0],
                                                         [0,0,-1],
                                                         [1,0,0]])
        # adjust eef_pose_rot
        eef_pose_rot[:3, 2] = np.array([0, 0, -1])
        eef_pose_rot[:3, 1] = np.cross(eef_pose_rot[:3, 2], eef_pose_rot[:3, 0])
        
        pen_pose_quat = transforms3d.quaternions.mat2quat(eef_pose_rot)

        pencil_sharpener_pose = env.pencil_sharpener.get_pose()
        if env.manip_obj_name == 'pencil':
            pencil_sharpener_position = pencil_sharpener_pose.p+np.array([0.01,-0.1,0.24])
        elif env.manip_obj_name == 'pencil_2':
            pencil_sharpener_position = pencil_sharpener_pose.p+np.array([0.01,-0.1,0.24])
        elif env.manip_obj_name == 'pencil_4':
            pencil_sharpener_position = pencil_sharpener_pose.p+np.array([0.01,-0.1,0.23])
        else:
            pencil_sharpener_position = pencil_sharpener_pose.p+np.array([0.01,-0.1,0.24])
        if mode == 'straight':
            self.trajectory = [
                {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.09},
                {"t": 30, "xyz": pen_postion, "quat":pen_pose_quat, "gripper": 0.09},
                {"t": 40, "xyz": pen_postion, "quat":pen_pose_quat, "gripper": 0.09},
                {"t": 60, "xyz": pen_postion, "quat":pen_pose_quat, "gripper": 0.0},
                {"t": 80, "xyz": pen_postion, "quat":pen_pose_quat, "gripper": 0.0},
                {"t": 140, "xyz": pencil_sharpener_position+np.array([0,-0.2,0]), "quat":ee_link_pose.q , "gripper": 0.0},
                {"t": 180, "xyz": pencil_sharpener_position, "quat": ee_link_pose.q, "gripper": 0.0},
                {"t": 200, "xyz": pencil_sharpener_position, "quat": ee_link_pose.q, "gripper": 0.0},
            ]
        else:
            raise RuntimeError('mode not implemented')

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

def main_env():
    kin_helper = KinHelper(robot_name="panda")
    max_timesteps = 400
    num_episodes = 50
    onscreen = True
    success = 0
    success_rate =0
    env = PenInsertionRLEnv(manip_obj='pencil',use_gui=True, robot_name="panda", frame_skip=10, use_visual_obs=False)
    env.seed(0)
    env.reset()
    arm_dof = env.arm_dof
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer,headless=False)
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
    viewer = gui.viewer
    viewer.toggle_pause(True)
    scripted_policy = SingleArmPolicy()
    while True:
        action = np.zeros(arm_dof+1)
        cartisen_action, quit = scripted_policy.single_trajectory(env,env.palm_link.get_pose(),mode='straight')
        # transform cartisen_action from robot to world frame
        if quit:
            break
        cartisen_action_in_rob = transform_action_from_world_to_robot(cartisen_action,env.robot.get_pose())
        action[:arm_dof] = kin_helper.compute_ik_sapien(env.robot.get_qpos()[:],cartisen_action_in_rob)[:arm_dof]
        action[arm_dof:] = cartisen_action_in_rob[6]
        # print(action)
        obs, reward, done, _ = env.step(action[:arm_dof+1])
        rgbs, depths = gui.render(depth=True)
        
        ee_translation = env.palm_link.get_pose().p
        ee_rotation = transforms3d.euler.quat2euler(env.palm_link.get_pose().q,axes='sxyz')
        ee_gripper = env.robot.get_qpos()[arm_dof]
        ee_pos = np.concatenate([ee_translation,ee_rotation,[ee_gripper]])
        ee_vel = np.concatenate([env.palm_link.get_velocity(),env.palm_link.get_angular_velocity(),env.robot.get_qvel()[arm_dof:arm_dof+1]])
        # timesteps += 1
    if not gui.headless:
        gui.viewer.close()
    env.close()


if __name__ == '__main__':
    main_env()