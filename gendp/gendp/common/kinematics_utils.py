import os
import copy
import time
import numpy as np
import transforms3d
from pathlib import Path
import open3d as o3d
import urchin
import warnings
warnings.filterwarnings("always", category=RuntimeWarning)

# test sapien
import sapien.core as sapien

class KinHelper():
    def __init__(self, robot_name="trossen_vx300s_tactile_thin", headless=True):
        # clid = p.connect(p.SHARED_MEMORY)
        # if (clid < 0):
        #     if headless:
        #         p.connect(p.DIRECT)
        #     else:
        #         p.connect(p.GUI)
        
        # load robot
        current_dir = Path(__file__).parent
        for _ in range(3):
            current_dir = current_dir.parent
        package_dir = (current_dir / "sapien_env" / "sapien_env" / "assets").resolve()
        if "trossen" in robot_name:
            trossen_urdf_prefix = "_".join(robot_name.split("_")[1:])
            urdf_path = f"{package_dir}/robot/trossen_description/{trossen_urdf_prefix}.urdf"
            self.eef_name = 'vx300s/ee_arm_link'
        elif "panda" in robot_name:
            urdf_path = f"{package_dir}/robot/panda/panda.urdf"
            self.eef_name = 'panda_hand'
        self.robot_name = robot_name
        # with suppress_stdout(): # suppress pybullet annoying print
        #     self.bullet_robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.urdf_robot = urchin.URDF.load(urdf_path)
        
        # load sapien robot
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == self.eef_name:
                self.sapien_eef_idx = link_idx
                break
        # link_names = [link.get_name() for link in self.sapien_robot.get_links()]
        # joint_names = [joint.get_name() for joint in self.sapien_robot.get_active_joints()]
        # self.planner = mplib.Planner(
        #     urdf=urdf_path,
        #     user_link_names=link_names,
        #     user_joint_names=joint_names,
        #     move_group=self.eef_name,
        #     joint_vel_limits=np.ones(len(joint_names)),
        #     joint_acc_limits=np.ones(len(joint_names)),)
        
        # # read joint info
        # self.bullet_active_joints = []
        # self.bullet_ll = []
        # self.bullet_ul = []
        # for i in range(p.getNumJoints(self.bullet_robot)):
        #     jointInfo = p.getJointInfo(self.bullet_robot, i)
        #     if jointInfo[2]== p.JOINT_REVOLUTE or jointInfo[2] == p.JOINT_PRISMATIC:
        #         self.bullet_active_joints.append(i)
        #         self.bullet_ll.append(jointInfo[8])
        #         self.bullet_ul.append(jointInfo[9])
        #     if jointInfo[12].decode("utf-8") == self.eef_name:
        #         self.eef_link_idx = i
        # self.bullet_jr = list(np.array(self.bullet_ul) - np.array(self.bullet_ll))

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
                    self.meshes[link.name].compute_vertex_normals()
                    self.meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
                    self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
                    self.offsets[link.name] = collision.origin
        self.pcd_dict = {}
        self.tool_meshes = {}

        # copy variables
        self.package_dir = package_dir
        
        # # link name to index map
        # self.link_name_to_idx_dict = {}
    
    # def _link_name_to_idx(self, link_name):
    #     if link_name not in self.link_name_to_idx_dict:
    #         for joint_idx in range(p.getNumJoints(self.bullet_robot)):
    #             jointInfo = p.getJointInfo(self.bullet_robot, joint_idx)
    #             if jointInfo[12].decode("utf-8") == link_name:
    #                 self.link_name_to_idx_dict[link_name] = joint_idx
    #                 break
    #     if link_name not in self.link_name_to_idx_dict:
    #         self.link_name_to_idx_dict[link_name] = -1
    #     return self.link_name_to_idx_dict[link_name]

    def _mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):
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
        pcd = self._mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls, offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd

    def gen_robot_meshes(self, qpos, link_names = None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([self.robot_model.get_link_pose(link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        meshes_ls = []
        for link_idx, link_name in enumerate(link_names):
            import copy
            mesh = copy.deepcopy(self.meshes[link_name])
            mesh.scale(0.001, center=np.array([0, 0, 0]))
            tf = link_pose_ls[link_idx] @ offsets_ls[link_idx]
            mesh.transform(tf)
            meshes_ls.append(mesh)
        return meshes_ls

    def compute_tool_pcd(self, qpos, tool_name, num_pts = None, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if num_pts is None:
            num_pts = 500
        
        # a hack to get the tool pose in eef frame
        if tool_name == 'dustpan':
            tool_pose_in_eef = np.array([[np.cos(np.pi/4.0), np.sin(np.pi/4.0), 0., 0.14],
                                         [0., 0., -1., 0.075],
                                         [-np.sin(np.pi/4.0), np.cos(np.pi/4.0), 0., -0.005],
                                         [0., 0., 0., 1.]])
        elif tool_name == 'pusher':
            tool_pose_in_eef = np.array([[0., -1., 0., 0.27],
                                         [0., 0., 1., -0.05],
                                         [1., 0., 0., -0.02],
                                         [0., 0., 0., 1.]])
        elif tool_name == 'spoon':
            theta = 0.1
            tool_pose_in_eef = np.array([[0., -np.cos(theta), np.sin(theta), 0.28],
                                         [1., 0., 0., -0.03],
                                         [0., np.sin(theta), np.cos(theta), -0.07],
                                         [0., 0., 0., 1.]])
        elif tool_name == 'pusher_v2':
            tool_pose_in_eef = np.array([[0., -1., 0., 0.28],
                                         [0., 0., 1., -0.16],
                                         [1., 0., 0., -0.04],
                                         [0., 0., 0., 1.]])
        elif tool_name == 'dustpan_v2':
            theta = 0.3
            tool_pose_in_eef = np.array([[np.cos(theta), np.sin(theta), 0., 0.1],
                                         [0., 0., -1., 0.14],
                                         [-np.sin(theta), np.cos(theta), 0., -0.07],
                                         [0., 0., 0., 1.]])
        else:
            raise RuntimeError('tool name not supported')
        
        if tool_name not in self.tool_meshes:
            tool_mesh_path = os.path.join(self.package_dir, 'tool', tool_name + '.stl')
            tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
            self.tool_meshes[tool_name] = tool_mesh
        
        eef_link_name = 'vx300s/ee_arm_link'
        eef_link_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == eef_link_name:
                eef_link_idx = link_idx
                break
        
        eef_pose = self.robot_model.get_link_pose(eef_link_idx).to_transformation_matrix()
        tool_pose = eef_pose @ tool_pose_in_eef

        return self._mesh_poses_to_pc(poses=tool_pose[None], meshes=[self.tool_meshes[tool_name]], offsets=[np.eye(4)], num_pts=[num_pts], scales=[0.001], pcd_name=pcd_name)

    # def compute_fk_links(self, qpos, link_idx):
    #     for i in range(len(qpos)):
    #         p.resetJointState(self.bullet_robot, self.bullet_active_joints[i], qpos[i])
    #     link_pose_ls = []
    #     for i in link_idx:
    #         if i == -1:
    #             # base link will have index -1
    #             link_pose_ls.append(np.eye(4))
    #             continue
    #         link_state = p.getLinkState(self.bullet_robot, i)
    #         link_state_mat = np.eye(4)
    #         link_state_mat[:3, 3] = link_state[4]
    #         link_state_mat[:3, :3] = np.array(p.getMatrixFromQuaternion(link_state[5])).reshape(3, 3)
    #         link_pose_ls.append(link_state_mat)
    #     link_pose_ls = np.stack(link_pose_ls)
    #     return link_pose_ls
    
    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls
    
    # def compute_ik(self,initial_qpos,cartesian):
    #     # cartesian: (8, ) numpy array
    #     pos = cartesian[0:3]
    #     orn = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
    #     orn = orn[3:4] + orn[0:3]
    #     # orn = cartesian[3:7]
    #     # orn = p.getQuaternionFromEuler(cartesian[3:6])
    #     for i in range(initial_qpos.shape[0]):
    #         p.resetJointState(self.bullet_robot, self.bullet_active_joints[i], initial_qpos[i])
    #     jointPoses = p.calculateInverseKinematics(self.bullet_robot, self.eef_link_idx, pos, orn,
    #                                               lowerLimits=self.bullet_ll, upperLimits=self.bullet_ul,
    #                                               jointRanges=self.bullet_jr, restPoses=initial_qpos.tolist())
    #     return np.array(jointPoses)

    def compute_ik_sapien(self,initial_qpos,cartesian,pose_fmt='euler'):
        # p = cartesian[0:3]
        # q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        # pose = sapien.Pose(p=p, q=q)
        if pose_fmt == 'euler':
            tf_mat = np.eye(4)
            tf_mat[:3, :3] = transforms3d.euler.euler2mat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
            tf_mat[:3, 3] = cartesian[0:3]
        elif pose_fmt == 'mat':
            tf_mat = cartesian
        pose = sapien.Pose.from_transformation_matrix(tf_mat)
        if 'trossen' in self.robot_name:
            active_qmask= np.array([True,True,True,True,True,True,False,False])
        elif 'panda' in self.robot_name:
            active_qmask= np.array([True,True,True,True,True,True,True,True,True])
        qpos = self.robot_model.compute_inverse_kinematics(link_index=self.sapien_eef_idx, pose=pose, initial_qpos=initial_qpos,active_qmask=active_qmask, eps=1e-3, damp=1e-1)
        # verify ik
        fk_pose = self.compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        # print('target pose for IK:', tf_mat)
        # print('fk pose for IK:', fk_pose)
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf_mat[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf_mat[:3, :3])
        if pose_diff > 0.01 or rot_diff > 0.01:
            print('ik pose diff:', pose_diff)
            print('ik rot diff:', rot_diff)
            warnings.warn('ik pose diff or rot diff too large. Return initial qpos.', RuntimeWarning, stacklevel=2, )
            return initial_qpos
        # qpos = self.robot_model.compute_inverse_kinematics(link_index=7, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask) 
        # print(qpos) 
        return qpos[0]

def test_kin_helper():
    robot_name = 'trossen_vx300s_v3'
    # finger_names = ['vx300s/left_finger_link','vx300s/right_finger_link']
    # num_pts = [1000, 1000]
    total_steps = 100
    finger_names = None
    num_pts = None
    init_qpos = np.array([0.851939865243963, -0.229601035617388, 0.563932102437065, -0.098902024821519, 1.148033168114365, 1.016116677288259, 0.0, -0.0])
    # init_qpos = np.array([0.788165775078753, -0.243655597686374, 0.573832680057706, -0.075632950397682, 1.260574309772582, 2.000622093036658, 0.0, -0.0])
    end_qpos = np.array([0.788165775078753, -0.243655597686374, 0.573832680057706, -0.075632950397682, 1.260574309772582, 2.000622093036658, 0.0, -0.0])

    # end_qpos = np.array([-0.10277671366930008, -0.5092816352844238, 0.5445631742477417, -0.26077672839164734, 0.7685244083404541, 0.27458256483078003, 0.0, -0.0])
    # init_qpos = np.array([-0.10277671366930008, -0.5092816352844238, 0.5445631742477417, -0.26077672839164734, 0.7685244083404541, 0.27458256483078003, 0.057, -0.057])
    
    # robot_name = 'panda'
    # # finger_names = ['panda_leftfinger','panda_rightfinger', 'panda_hand']
    # # num_pts = [1000, 1000, 1000]
    # finger_names = None
    # num_pts = None
    # init_qpos = np.array([0,0,0,0,0,0,0,0,0])
    # end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0, 0.09, 0.09])
    
    kin_helper = KinHelper(robot_name=robot_name)
    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk_pose = kin_helper.compute_fk_sapien_links(curr_qpos, [kin_helper.sapien_eef_idx])[0]
        print('fk pose:', fk_pose)
        start_time = time.time()
        pcd = kin_helper.compute_robot_pcd(curr_qpos, link_names=finger_names, num_pts=num_pts, pcd_name='finger')
        # tool_pcd = kin_helper.compute_tool_pcd(curr_qpos, 'pusher', num_pts=1000, pcd_name='tool')
        # pcd = np.concatenate([pcd, tool_pcd], axis=0)
        print('compute_robot_pcd time:', time.time() - start_time)
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
    robot_name = 'trossen_vx300s_v3'
    init_qpos = np.array([0,0,0,0,0,0,0,0])
    end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.09, -0.09])
    # fn = '/media/yixuan_2T/diffusion_policy/data/real_aloha_demo/pick_place_cup_v2/episode_1.hdf5'
    # # fn = '/home/yixuan/general_dp/data/real_aloha_demo/pick_place_cup_v2/episode_2.hdf5'
    # dict, _ = load_dict_from_hdf5(fn)
    # qpos = dict['observations']['full_joint_pos'][()]
    
    kin_helper = KinHelper(robot_name=robot_name, headless=False)
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    # for i in range(qpos.shape[0]):
    #     curr_qpos = qpos[i]
    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk = kin_helper.compute_fk_sapien_links(curr_qpos, [kin_helper.sapien_eef_idx])[0]
        # fk[:3, 3] +=  np.random.normal(0, 0.005, 3)
        fk_euler = transforms3d.euler.mat2euler(fk[:3, :3], axes='sxyz')
        
        # init_ik_qpos = qpos[0].copy()
        # init_ik_qpos[4] = -np.pi/4.0
        if i == 0:
            init_ik_qpos = np.array(START_ARM_POSE)
        # ik_qpos = kin_helper.compute_ik_sapien_post_process(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        ik_qpos = kin_helper.compute_ik_sapien(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        re_fk_pos_mat = kin_helper.compute_fk_sapien_links(ik_qpos, [kin_helper.sapien_eef_idx])[0]
        re_fk_euler = transforms3d.euler.mat2euler(re_fk_pos_mat[:3, :3], axes='sxyz')
        re_fk_pos = re_fk_pos_mat[:3, 3]
        print('re_fk_pos diff:', np.linalg.norm(re_fk_pos - fk[:3, 3]))
        print('re_fk_euler diff:', np.linalg.norm(np.array(re_fk_euler) - np.array(fk_euler)))
        
        # ik_qpos = kin_helper.compute_ik_mplib(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        # ik_qpos = np.clip(ik_qpos, kin_helper.bullet_ll, kin_helper.bullet_ul)
        # ik_qpos = kin_helper.compute_ik(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        # ik_qpos = np.clip(ik_qpos, kin_helper.bullet_ll, kin_helper.bullet_ul)
        # if (ik_qpos[:6] - kin_helper.bullet_ll[:6]).min() < 0.0 or \
        #     (ik_qpos[:6] - kin_helper.bullet_ul[:6]).max() > 0.0:
        #     raise RuntimeError('ik qpos out of bound')
        init_ik_qpos = ik_qpos.copy()
        print('fk_euler:', fk_euler)
        print('gt qpos:', curr_qpos)
        print('ik qpos:', ik_qpos)
        print('qpos diff:', np.linalg.norm(ik_qpos[:6] - curr_qpos[:6]))
        qpos_diff = np.linalg.norm(ik_qpos[:6] - curr_qpos[:6])
        if qpos_diff > 0.01:
            warnings.warn('qpos diff too large', RuntimeWarning, stacklevel=2, )
        
        # for i in range(len(ik_qpos)):
        #     p.resetJointState(kin_helper.bullet_robot, kin_helper.bullet_active_joints[i], ik_qpos[i])
        
        print()
                
        time.sleep(0.1)

if __name__ == "__main__":
    test_kin_helper()
    # test_fk()
