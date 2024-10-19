import os
import copy

import numpy as np
import open3d as o3d
import transforms3d


def load_mesh(robot_name, finger_names=None):
    repo_path = os.path.dirname(os.path.abspath(__file__))
    for i in range(3):
        repo_path = os.path.dirname(repo_path)
    if 'panda' in robot_name:
        STL_FILE_DICT = {'left_finger_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/panda/franka_description/meshes/collision/finger.stl",
                        'right_finger_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/panda/franka_description/meshes/collision/finger.stl",
                        'hand_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/panda/franka_description/meshes/collision/hand.stl",}

        STL_OFFSET_DICT = {'left_finger_link': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           'right_finger_link': np.array([0.0, 0.0, 3.14159265359, 0.0, 0.0, 0.0]),
                           'hand_link': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

        finger_names = ['left_finger_link','right_finger_link', 'hand_link']
    elif 'trossen' in robot_name:
        # STL_FILE_DICT = {,
        #                  'base_link_right': f"{repo_path}/sapien_env/sapien_env/assets/robot/trossen_description/meshes/vx300s_meshes/vx300s_1_base.stl",}
        # STL_OFFSET_DICT = {'base_link_left': np.array([0, 0 ,1.5707963267948966,0 ,0, 0]),
        #                    'base_link_right': np.array([0, 0 ,1.5707963267948966,0 ,0, 0]),}
        # finger_names = ['base_link_left','base_link_right',]
        STL_FILE_DICT = {'base_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/trossen_description/meshes/vx300s_meshes/vx300s_1_base.stl",
                         'left_finger_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/trossen_description/meshes/vx300s_meshes/vx300s_10_gripper_finger.stl",
                         'right_finger_link': f"{repo_path}/sapien_env/sapien_env/assets/robot/trossen_description/meshes/vx300s_meshes/vx300s_10_gripper_finger.stl",}
        STL_OFFSET_DICT = {'base_link': np.array([0, 0 ,1.5707963267948966,0 ,0, 0]),
                           'left_finger_link': np.array([1.5707963267948966 ,-3.141592653589793 ,1.5707963267948966,-0.0404 ,-0.0575, 0]),
                           'right_finger_link': np.array([-1.5707963267948966, 3.141592653589793, -1.5707963267948966,-0.0404, 0.0575, 0]),}
        if finger_names is None:
            finger_names = STL_FILE_DICT.keys()
    else:
        raise RuntimeError('unsupported robot name')

    meshes = []
    offsets = []
    for name in finger_names:
        try:
            mesh = o3d.io.read_triangle_mesh(STL_FILE_DICT[name])
            meshes.append(mesh)
            offsets.append(STL_OFFSET_DICT[name])
        except:
            continue
    return meshes, offsets

def mesh_poses_to_pc(poses, meshes, offsets, num_pts, scale=1.0):
    # poses: (N, 4, 4) numpy array
    # offsets: (N, ) list of offsets
    # meshes: (N, ) list of meshes
    # num_pts: (N, ) list of int
    try:
        assert poses.shape[0] == len(meshes)
        assert poses.shape[0] == len(offsets)
    except:
        raise RuntimeError('poses and meshes must have the same length')

    N = poses.shape[0]
    all_pc = []
    for index in range(N):
        mat = poses[index]
        mesh = copy.deepcopy(meshes[index])  # .copy()
        mesh.scale(scale, center=np.array([0, 0, 0]))
        sampled_cloud= mesh.sample_points_uniformly(number_of_points=num_pts[index])
        cloud_points = np.asarray(sampled_cloud.points)
        
        tf_obj_to_link = np.eye(4)
        tf_obj_to_link[:3, 3] = offsets[index][3:]
        tf_obj_to_link[:3, :3] = transforms3d.euler.euler2mat(offsets[index][0], offsets[index][1], offsets[index][2], axes='sxyz')
        
        mat = mat @ tf_obj_to_link
        transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
        all_pc.append(transformed_points)
    all_pc =np.concatenate(all_pc,axis=0)
    return all_pc
