import copy
import time
import numpy as np
import sapien.core as sapien
import transforms3d
from pathlib import Path
from urdfpy import URDF
import pybullet as p
# from gild.common.data_utils import load_dict_from_hdf5

class TeleopRobot():
    def __init__(self, robot_name="trossen_vx300s_tactile_thin"):
        clid = p.connect(p.SHARED_MEMORY)
        if (clid < 0):
            p.connect(p.GUI)
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        current_dir = Path(__file__).parent
        package_dir = (current_dir.parent / "assets").resolve()
        if "trossen" in robot_name:
            trossen_urdf_prefix = "_".join(robot_name.split("_")[1:])
            urdf_path = f"{package_dir}/robot/trossen_description/{trossen_urdf_prefix}.urdf"
            self.eef_name = 'vx300s/ee_arm_link'
        elif "panda" in robot_name:
            urdf_path = f"{package_dir}/robot/panda/panda.urdf"
            self.eef_name = 'panda_hand'
        self.sapien_robot = loader.load(urdf_path)
        # for link_i, link in enumerate(self.sapien_robot.get_links()):
        #     if link.name == self.eef_name:
        #         self.eef_link_idx = link_i
        #         break
        self.urdf_robot = URDF.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.robot_name = robot_name
        self.bullet_robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.bullet_active_joints = []
        for i in range(p.getNumJoints(self.bullet_robot)):
            jointInfo = p.getJointInfo(self.bullet_robot, i)
            if jointInfo[2]== p.JOINT_REVOLUTE or jointInfo[2] == p.JOINT_PRISMATIC:
                self.bullet_active_joints.append(i)
            if jointInfo[12].decode("utf-8") == self.eef_name:
                self.eef_link_idx = i
        # self.eef_link_idx = 8
        self.bullet_ll = []
        self.bullet_ul = []
        for joint_idx, joint in enumerate(self.urdf_robot.joints):
            if joint.limit is not None and joint.limit.lower is not None:
                self.bullet_ll.append(joint.limit.lower)
                self.bullet_ul.append(joint.limit.upper)
            else:
                self.bullet_ll.append(-0.01)
                self.bullet_ul.append(0.01)
        self.bullet_jr = list(np.array(self.bullet_ul) - np.array(self.bullet_ll))

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        for link in self.urdf_robot.links:
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if len(collision.geometry.mesh.meshes) > 0:
                    mesh = collision.geometry.mesh.meshes[0]
                    self.meshes[link.name] = mesh.as_open3d
                    self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
                    self.offsets[link.name] = collision.origin
        self.pcd_dict = {}

    # copute fk in sapien in robot frame with qpos
    def compute_fk(self, qpos): 
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose = self.robot_model.get_link_pose(8)
        # print("fk",link_pose)
        return link_pose

    def mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):
        # poses: (N, 4, 4) numpy array
        # offsets: (N, ) list of offsets
        # meshes: (N, ) list of meshes
        # num_pts: (N, ) list of int
        # scales: (N, ) list of float
        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if pcd_name is None or pcd_name not in self.pcd_dict or len(self.pcd_dict[pcd_name]) <= index:
                mesh = copy.deepcopy(meshes[index])  # .copy()
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud= mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]
            
            # tf_obj_to_link = np.eye(4)
            # tf_obj_to_link[:3, 3] = offsets[index][3:]
            # tf_obj_to_link[:3, :3] = transforms3d.euler.euler2mat(offsets[index][0], offsets[index][1], offsets[index][2], axes='sxyz')
            tf_obj_to_link = offsets[index]
            
            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc =np.concatenate(all_pc,axis=0)
        return all_pc

    # compute robot pcd given qpos
    def compute_robot_pcd(self, qpos, link_names = None, num_pts = None, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        if num_pts is None:
            num_pts = [500] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([self.robot_model.get_link_pose(link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self.mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls, offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd

    def compute_fk_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i))
        return link_pose_ls

    def compute_fk_pb_links(self, qpos, link_idx):
        for i in range(len(qpos)):
            p.resetJointState(self.bullet_robot, self.bullet_active_joints[i], qpos[i])
        link_pose_ls = []
        for i in link_idx:
            link_state = p.getLinkState(self.bullet_robot, i)
            link_pose_ls.append([link_state[0], link_state[1]])
        return link_pose_ls

    def init_cartisen_action(self,qpos):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        self.ee_link_pose = self.robot_model.get_link_pose(8)
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        p = self.ee_link_pose.p
        q=self.ee_link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        cartisen_action[0:3] = p
        cartisen_action[3:6] = eluer
        cartisen_action[6] = 0.021
        return cartisen_action

    def init_cartisen_action_franka(self,qpos):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        self.ee_link_pose = self.robot_model.get_link_pose(11)
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        p = self.ee_link_pose.p
        q=self.ee_link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        cartisen_action[0:3] = p
        cartisen_action[3:6] = eluer
        cartisen_action[6] = 0.021
        return cartisen_action

    def ik_vx300(self,initial_qpos,cartesian):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        active_qmask= np.array([True,True,True,True,True,False,False])
        qpos = self.robot_model.compute_inverse_kinematics(link_index=7, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask)  
        return qpos[0][0:5]

    def ik_vx300s(self,initial_qpos,cartesian):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        active_qmask= np.array([True,True,True,True,True,True,False,False])
        qpos = self.robot_model.compute_inverse_kinematics(link_index=9, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask) 
        # print(qpos) 
        return qpos[0][0:6]
    
    def ik_pb_vx300s(self,initial_qpos,cartesian):
        pos = cartesian[0:3]
        orn = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        # transforms3d: wxyz
        # pybullet: xyzw
        orn = orn[1:4] + orn[0:1]
        jointPoses = p.calculateInverseKinematics(self.bullet_robot, self.eef_link_idx, pos, orn, self.bullet_ll, self.bullet_ul,
                                                  self.bullet_jr, initial_qpos)[:6]
        for i in range(6):
            p.resetJointState(self.bullet_robot, i, jointPoses[i])
        return jointPoses
    
    def ik_vx300s_sapien_pose(self,initial_qpos,pose):
        active_qmask= np.array([True,True,True,True,True,True,False,False])
        qpos = self.robot_model.compute_inverse_kinematics(link_index=9, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask) 
        # print(qpos) 
        return qpos[0][0:6]

    def ik_panda(self,initial_qpos,cartesian):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        active_qmask= np.array([True,True,True,True,True,True,True,False,False])
        qpos = self.robot_model.compute_inverse_kinematics(link_index=9, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask) 
        # print(qpos) 
        return qpos[0][0:7]

    def keyboard_control(self, viewer, cartesian):
        delta_position = 0.001
        constant_x = 1
        constant_y = 1
        constant_z = 1
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
            cartesian[2]+=delta_position*constant_z
        if viewer.window.key_down("o"):
            cartesian[2]-=delta_position*constant_z
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
            cartesian[6]+=0.0005
        if viewer.window.key_down("x"):
            cartesian[6]-=0.0005
        cartesian[6] = np.clip(cartesian[6],0.01,0.09)
        return cartesian

    def yx_keyboard_control(self, viewer, cartesian):
        delta_position = 0.001
        constant_x = 1
        constant_y = 1
        constant_z = 1
        delta_orientation = 0.005
        prev_cartesian = cartesian.copy()
        # x
        if viewer.window.key_down("k"):
            cartesian[0]+=delta_position*constant_x
        if viewer.window.key_down("i"):
            cartesian[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("l"):
            cartesian[1]+=delta_position*constant_y
        if viewer.window.key_down("j"):
            cartesian[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("up"):
            cartesian[2]+=delta_position*constant_z
        if viewer.window.key_down("down"):
            cartesian[2]-=delta_position*constant_z
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
            cartesian[6]+=0.0005
        if viewer.window.key_down("x"):
            cartesian[6]-=0.0005
        cartesian[6] = np.clip(cartesian[6],0.01,0.09)
        
        if viewer.window.key_down("n"):
            quit = True
        else:
            quit = False
        moved = np.linalg.norm(cartesian-prev_cartesian) > 0.01 * min(delta_position, delta_orientation) * min(constant_x, constant_y, constant_z)
        return cartesian, quit, moved

    def keyboard_control_2arm(self, viewer, cartesian,cartesian_2):
        delta_position = 0.0005
        constant_x = 1
        constant_y = 1
        constant_z = 1
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
            cartesian[2]+=delta_position*constant_z
        if viewer.window.key_down("o"):
            cartesian[2]-=delta_position*constant_z
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
            cartesian[6]+=0.0005
        if viewer.window.key_down("x"):
            cartesian[6]-=0.0005
        cartesian[6] = np.clip(cartesian[6],0.01,0.09)

        # x
        if viewer.window.key_down("up"):
            cartesian_2[0]+=delta_position*constant_x
        if viewer.window.key_down("down"):
            cartesian_2[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("left"):
            cartesian_2[1]+=delta_position*constant_y
        if viewer.window.key_down("right"):
            cartesian_2[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("0"):
            cartesian_2[2]+=delta_position*constant_z
        if viewer.window.key_down("p"):
            cartesian_2[2]-=delta_position*constant_z
        # # roll
        # if viewer.window.key_down("r"):
        #     cartesian_2[3]+=delta_orientation
        # if viewer.window.key_down("f"):
        #     cartesian_2[3]-=delta_orientation
        # # pitch
        # if viewer.window.key_down("t"):
        #     cartesian_2[4]+=delta_orientation
        # if viewer.window.key_down("g"):
        #     cartesian_2[4]-=delta_orientation
        # # yaw
        # if viewer.window.key_down("y"):
        #     cartesian_2[5]+=delta_orientation
        # if viewer.window.key_down("h"):
        #     cartesian_2[5]-=delta_orientation

        # gripper open or close
        if viewer.window.key_down("n"):
            cartesian_2[6]+=0.0005
        if viewer.window.key_down("m"):
            cartesian_2[6]-=0.0005
        cartesian_2[6] = np.clip(cartesian_2[6],0.01,0.09)
        return cartesian, cartesian_2

def test_teleop_robot():
    # robot_name = 'trossen_vx300s'
    # finger_names = ['vx300s/left_finger_link','vx300s/right_finger_link']
    # num_pts = [1000, 1000]
    # init_qpos = np.array([0,0,0,0,0,0,0,0])
    # end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.057, -0.057])
    # end_qpos = np.array([-0.10277671366930008, -0.5092816352844238, 0.5445631742477417, -0.26077672839164734, 0.7685244083404541, 0.27458256483078003, 0.0, -0.0])
    # init_qpos = np.array([-0.10277671366930008, -0.5092816352844238, 0.5445631742477417, -0.26077672839164734, 0.7685244083404541, 0.27458256483078003, 0.057, -0.057])
    
    robot_name = 'panda'
    finger_names = ['panda_leftfinger','panda_rightfinger', 'panda_hand']
    num_pts = [1000, 1000, 1000]
    # finger_names = None
    # num_pts = None
    init_qpos = np.array([0,0,0,0,0,0,0,0,0])
    end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0, 0.09, 0.09])
    
    teleop_robot = TeleopRobot(robot_name=robot_name)
    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        pcd = teleop_robot.compute_robot_pcd(curr_qpos, link_names=finger_names, num_pts=num_pts)
        from d3fields.utils.draw_utils import np2o3d
        import open3d as o3d
        pcd_o3d = np2o3d(pcd)
        if i == 0:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            curr_pcd = copy.deepcopy(pcd_o3d)
            visualizer.add_geometry(curr_pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visualizer.add_geometry(origin)
        curr_pcd.points = pcd_o3d.points
        curr_pcd.colors = pcd_o3d.colors
        visualizer.update_geometry(curr_pcd)
        visualizer.update_geometry(origin)
        visualizer.poll_events()
        visualizer.update_renderer()
        if i == 0:
            visualizer.run()

def test_fk():
    robot_name = 'trossen_vx300s'
    # init_qpos = np.array([0,0,0,0,0,0,0,0])
    # end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.09, -0.09])
    fn = '/media/yixuan_2T/diffusion_policy/data/real_aloha_demo/pick_place_cup_v2/episode_1.hdf5'
    dict, _ = load_dict_from_hdf5(fn)
    qpos = dict['joint_action'][()]
    
    teleop_robot = TeleopRobot(robot_name=robot_name)
    for i in range(qpos.shape[0]):
        curr_qpos = qpos[i]
        curr_qpos = np.concatenate([curr_qpos[:6], np.zeros(2)])
        fk = teleop_robot.compute_fk_links(curr_qpos, [8])[0]
        fk_euler = transforms3d.euler.quat2euler(fk.q, axes='sxyz')
        pb_fk = teleop_robot.compute_fk_pb_links(curr_qpos, [6])[0] # pybullet: xyzw
        pb_fk_euler = transforms3d.euler.quat2euler(pb_fk[1][3:4] + pb_fk[1][0:3], axes='sxyz') # sapien: wxyz
        print('sapien fk:', list(fk.p) +  list(fk_euler))
        print('pb fk:', list(pb_fk[0]) + list(pb_fk_euler))
        
        ik_qpos = teleop_robot.ik_pb_vx300s(curr_qpos, list(fk.p) + list(fk_euler))
        print('gt qpos:', curr_qpos)
        print('ik qpos:', ik_qpos)
        
        for i in range(len(ik_qpos)):
            p.resetJointState(teleop_robot.bullet_robot, teleop_robot.bullet_active_joints[i], ik_qpos[i])
        
        print()
        time.sleep(0.1)

if __name__ == "__main__":
    # test_teleop_robot()
    test_fk()
