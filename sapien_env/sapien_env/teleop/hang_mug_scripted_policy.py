import numpy as np
from pyquaternion import Quaternion
import transforms3d
import sapien.core as sapien

from sapien_env.rl_env.hang_mug_env import HangMugRLEnv

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
    
    def generate_trajectory(self, env : HangMugRLEnv, ee_link_pose, mode='straight'):
        if env.manip_obj_name == 'nescafe_mug':
            pre_grasp_p = np.array([0.0, 0.2, 0.03])
            grasp_p = np.array([0.0, 0.12, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,0.0,0.0,axes='sxyz').tolist())
            
            # place_p = np.array([-0.2, 0.33, 0.14])
            # place_q = np.array(transforms3d.euler.euler2quat(0.0, -np.pi, -np.pi/2.0, axes='sxyz').tolist())
            
            # post_place_p = np.array([0.02, 0.33, 0.14])
            # post_place_q = np.array(transforms3d.euler.euler2quat(0.0, -np.pi, -np.pi/2.0, axes='sxyz').tolist())
            
            # leave_p = np.array([0.01, 0.29, 0.3])
            # leave_q = np.array(transforms3d.euler.euler2quat(0.0, -np.pi, -np.pi/2.0, axes='sxyz').tolist())
            
            # place_q_in_world = np.array(transforms3d.euler.euler2quat(1.15*np.pi, -0.5*np.pi, np.pi/2.0, axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.05, -0.1, 0.42])
            
            post_place_p_in_world = np.array([0.05, 0.1, 0.42])
            
            leave_p_in_world = np.array([0.05, 0.1, 0.45])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            # place_q_in_world_mat = np.array([[1.0, 0.0, 0.0],
            #                                  [0.0, -1.0, 0.0],
            #                                  [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
            # place_q_in_world = np.array(transforms3d.euler.euler2quat(1.15*np.pi, -0.5*np.pi, np.pi/2.0, axes='sxyz').tolist())
        elif env.manip_obj_name == 'blue_mug':
            pre_grasp_p = np.array([0.0, 0.3, 0.03])
            grasp_p = np.array([0.0, 0.18, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,0.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.05, -0.1, 0.42])
            
            post_place_p_in_world = np.array([0.05, 0.1, 0.42])
            
            leave_p_in_world = np.array([0.05, 0.1, 0.45])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
        elif env.manip_obj_name == 'kor_mug':
            pre_grasp_p = np.array([0.0, 0.3, 0.03])
            grasp_p = np.array([0.0, 0.15, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,0.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.06, -0.1, 0.40])
            
            post_place_p_in_world = np.array([0.06, 0.13, 0.40])
            
            leave_p_in_world = np.array([0.06, 0.13, 0.45])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
        elif env.manip_obj_name == 'low_poly_mug':
            pre_grasp_p = np.array([0.0, 0.3, 0.03])
            grasp_p = np.array([0.0, 0.13, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,0.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.06, -0.1, 0.45])
            
            post_place_p_in_world = np.array([0.06, 0.1, 0.45])
            
            leave_p_in_world = np.array([0.06, 0.1, 0.50])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
        elif env.manip_obj_name == 'aluminum_mug':
            pre_grasp_p = np.array([0.03, 0.3, 0.03])
            grasp_p = np.array([0.03, 0.1, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,np.pi/4.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.06, -0.1, 0.38])
            
            post_place_p_in_world = np.array([0.06, 0.1, 0.38])
            
            leave_p_in_world = np.array([0.06, 0.1, 0.43])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
        elif env.manip_obj_name == 'black_mug':
            pre_grasp_p = np.array([0.03, 0.4, 0.0])
            grasp_p = np.array([0.03, 0.2, 0.0])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,np.pi/2.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.05, -0.1, 0.43])
            
            post_place_p_in_world = np.array([0.05, 0.12, 0.43])
            
            leave_p_in_world = np.array([0.05, 0.12, 0.48])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
        elif env.manip_obj_name == 'white_mug':
            pre_grasp_p = np.array([0.0, 0.3, 0.03])
            grasp_p = np.array([0.0, 0.14, 0.03])
            grasp_q = np.array(transforms3d.euler.euler2quat(np.pi/2.0,0.0,0.0,axes='sxyz').tolist())
            
            place_p_in_world = np.array([0.06, -0.1, 0.45])
            
            post_place_p_in_world = np.array([0.06, 0.1, 0.45])
            
            leave_p_in_world = np.array([0.06, 0.1, 0.50])
            
            place_q_in_world_mat = np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]])
            place_q_in_world = -np.array(transforms3d.quaternions.mat2quat(place_q_in_world_mat).tolist())
            
        pre_grasp_pose = sapien.Pose(pre_grasp_p,grasp_q)
        grasp_pose = sapien.Pose(grasp_p,grasp_q)
        
        mug_pose_mat = env.manipulated_object.get_pose().to_transformation_matrix()
        grasp_pose_in_mug_mat = grasp_pose.to_transformation_matrix()
        grasp_pose_in_world_mat = mug_pose_mat @ grasp_pose_in_mug_mat
        grasp_pose_in_world = sapien.Pose.from_transformation_matrix(grasp_pose_in_world_mat)
        
        pre_grasp_pose_in_mug_mat = pre_grasp_pose.to_transformation_matrix()
        pre_grasp_pose_in_world_mat = mug_pose_mat @ pre_grasp_pose_in_mug_mat
        pre_grasp_pose_in_world = sapien.Pose.from_transformation_matrix(pre_grasp_pose_in_world_mat)
        
        
        # place pose setting
        mug_tree_mat = env.mug_tree.get_pose().to_transformation_matrix()
        
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
        
        
        if mode == 'straight':
            self.trajectory = [
                {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.09},
                {"t": 20, "xyz": pre_grasp_pose_in_world.p, "quat": pre_grasp_pose_in_world.q, "gripper": 0.09},
                {"t": 70, "xyz": grasp_pose_in_world.p, "quat": grasp_pose_in_world.q, "gripper": 0.09},
                {"t": 90, "xyz": grasp_pose_in_world.p, "quat": grasp_pose_in_world.q, "gripper": 0.00},
                {"t": 120, "xyz": grasp_pose_in_world.p, "quat": grasp_pose_in_world.q, "gripper": 0.00},
                {"t": 140, "xyz": pre_grasp_pose_in_world.p, "quat": pre_grasp_pose_in_world.q, "gripper": 0.00},
                {"t": 190, "xyz": place_p_in_world, "quat": place_q_in_world, "gripper": 0.00},
                {"t": 200, "xyz": place_p_in_world, "quat": place_q_in_world, "gripper": 0.00},
                {"t": 250, "xyz": post_place_p_in_world, "quat": place_q_in_world, "gripper": 0.00},
                {"t": 280, "xyz": post_place_p_in_world, "quat": place_q_in_world, "gripper": 0.00},
                {"t": 290, "xyz": post_place_p_in_world, "quat": place_q_in_world, "gripper": 0.02},
                {"t": 300, "xyz": leave_p_in_world, "quat": place_q_in_world, "gripper": 0.02},
            ]
        else:
            raise RuntimeError('mode not implemented')