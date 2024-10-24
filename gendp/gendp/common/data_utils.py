from typing import Optional
import time

import numpy as np
import h5py
import cv2
import torch
from tqdm import tqdm

from d3fields.utils.draw_utils import np2o3d

def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
    y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
    z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape

### ALOHA fixed constants
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, '/', dic, config_dict)

def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(path + key, shape=item.shape, **config_dict[key])
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, config_dict[key])
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, 'r')
    return recursively_load_dict_contents_from_group(h5file, '/'), h5file

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, 'r+') as h5file:
        recursively_modify_hdf5_from_dict(h5file, '/', dic)

def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot modify %s type'%type(item))

def vis_distill_feats(pts, feats):
    """visualize distilled features

    Args:
        pts (np.ndarray): (N, 3)
        feats (np.ndarray): (N, f) ranging in [0, 1]
    """
    import open3d as o3d
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    for i in range(pts.shape[1]):
        feats_i = feats[:, i]
        colors = cmap(feats_i)[:, :3]
        pts_o3d = np2o3d(pts, color=colors)
        o3d.visualization.draw_geometries([pts_o3d])

def d3fields_proc(fusion, shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                  robot_base_pose_in_world_seq = None, teleop_robot = None, qpos_seq=None, expected_labels=None,
                  tool_names=[None], exclude_threshold=0.01, exclude_colors=[]):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    query_texts = []
    query_thresholds = []
    boundaries = shape_meta['info']['boundaries']
    use_seg = False
    use_dino = False
    distill_dino = shape_meta['info']['distill_dino'] if 'distill_dino' in shape_meta['info'] else False
    distill_obj = shape_meta['info']['distill_obj'] if 'distill_obj' in shape_meta['info'] else False
    if "N_gripper" in shape_meta['info']:
        N_gripper = shape_meta['info']['N_gripper']
    elif "N_per_inst" in shape_meta['info']:
        N_gripper = shape_meta['info']['N_per_inst'] # legacy name
    else:
        N_gripper = 100
    N_total = shape_meta['shape'][1]
    max_pts_num = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = robot_base_pose_in_world_seq.shape[1] if len(robot_base_pose_in_world_seq.shape) == 4 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):
        # print()
        obs = {
            'color': color_seq[t],
            'depth': depth_seq[t],
            'pose': extri_seq[t][:,:3,:],
            'K': intri_seq[t],
        }
        
        fusion.update(obs, update_dino=(use_dino or distill_dino))
        
        # compute robot pcd
        if 'panda' in teleop_robot.robot_name:
            finger_names = ['panda_leftfinger', 'panda_rightfinger', 'panda_hand']
            dense_num_pts = [50, 50, 400]
            sparse_num_pts = [int(0.2 * N_gripper), int(0.2 * N_gripper), int(0.6 * N_gripper)]
        elif 'trossen_vx300s' in teleop_robot.robot_name:
            finger_names = ['vx300s/left_finger_link', 'vx300s/right_finger_link']
            dense_num_pts = [250, 250]
            sparse_num_pts = [int(0.5 * N_gripper), int(0.5 * N_gripper)]
        else:
            raise RuntimeError('unsupported')

        # DEPRECATED: use qpos_seq instead
        # if qpos_seq[t].shape[0] == 8 and 'panda' in teleop_robot.robot_name:
        #     # NOTE: tentative fix for panda
        #     curr_qpos = np.concatenate([qpos_seq[t], qpos_seq[t][7:8]])
        # elif qpos_seq[t].shape[0] == 7 and 'trossen_vx300s' in teleop_robot.robot_name:
        #     # NOTE: tentative fix for vx300s
        #     gripper_joint_norm = PUPPET_GRIPPER_JOINT_NORMALIZE_FN(qpos_seq[t][6])
        #     gripper_pos_unnorm = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(gripper_joint_norm)
        #     curr_qpos = np.concatenate([qpos_seq[t][:6], np.array([gripper_pos_unnorm, -gripper_pos_unnorm])])
        # else:
        #     curr_qpos = qpos_seq[t]

        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots
        
        dense_ee_pcd_ls = []
        ee_pcd_ls = []
        robot_pcd_ls = []
        tool_pcd_ls = []
        for rob_i in range(num_bots):
            tool_name = tool_names[rob_i]
            # compute robot pcd
            dense_ee_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], finger_names, dense_num_pts, pcd_name=f'dense_ee_pcd_{rob_i}') # (N, 3)
            ee_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], finger_names, sparse_num_pts, pcd_name=f'ee_pcd_{rob_i}')
            robot_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
            if tool_name is not None:
                tool_pcd = teleop_robot.compute_tool_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], tool_name, N_gripper, pcd_name=f'tool_pcd_{rob_i}')
        
            # transform robot pcd to world frame    
            robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i] if robot_base_pose_in_world_seq is not None else None
            dense_ee_pcd = (robot_base_pose_in_world @ np.concatenate([dense_ee_pcd, np.ones((dense_ee_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            ee_pcd = (robot_base_pose_in_world @ np.concatenate([ee_pcd, np.ones((ee_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            if tool_name is not None:
                tool_pcd = (robot_base_pose_in_world @ np.concatenate([tool_pcd, np.ones((tool_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            
            # save to list
            dense_ee_pcd_ls.append(dense_ee_pcd)
            ee_pcd_ls.append(ee_pcd)
            robot_pcd_ls.append(robot_pcd)
            if tool_name is not None:
                tool_pcd_ls.append(tool_pcd)
        # convert to numpy array
        dense_ee_pcd = np.concatenate(dense_ee_pcd_ls + tool_pcd_ls, axis=0)
        ee_pcd = np.concatenate(ee_pcd_ls + tool_pcd_ls, axis=0)
        robot_pcd = np.concatenate(robot_pcd_ls + tool_pcd_ls, axis=0)
        
        
        # post process robot pcd
        ee_pcd_tensor = torch.from_numpy(ee_pcd).to(device=fusion.device, dtype=fusion.dtype)
        
        if use_dino or distill_dino:
            ee_eval_res = fusion.eval(ee_pcd_tensor, return_names=['dino_feats'])
            ee_feats = ee_eval_res['dino_feats']
        
        if use_seg:
            fusion.text_queries_for_inst_mask(query_texts, query_thresholds, boundaries, expected_labels=expected_labels, robot_pcd=dense_ee_pcd, voxel_size=0.03, merge_iou=0.15)
        
        if use_seg:
            obj_pcd = fusion.extract_masked_pcd(list(range(1, fusion.get_inst_num())), boundaries=boundaries)
            src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(obj_pcd, N_gripper, per_instance=True, use_seg=use_seg, use_dino=(use_dino or distill_dino))
        else:
            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.002, excluded_pts=robot_pcd, exclude_threshold=exclude_threshold, exclude_colors=exclude_colors)
            src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(obj_pcd, N_total - ee_pcd.shape[0], per_instance=True, use_seg=use_seg, use_dino=(use_dino or distill_dino))
        
        aggr_src_pts = np.concatenate(src_pts_list, axis=0) # (N, 3)
        aggr_feats = torch.concat(src_feat_list, axis=0).detach().cpu().numpy() if (use_dino or distill_dino) else None # (N, 1024)
        
        # only to adjust point number when using segmentation
        if use_seg:
            max_obj_pts_num = max_pts_num - ee_pcd.shape[0]
            if aggr_src_pts.shape[0] > max_obj_pts_num:
                aggr_src_pts = aggr_src_pts[:max_obj_pts_num]
                aggr_feats = aggr_feats[:max_obj_pts_num] if (use_dino or distill_dino) else None
            elif aggr_src_pts.shape[0] < max_obj_pts_num:
                aggr_src_pts = np.pad(aggr_src_pts, ((0,max_obj_pts_num-aggr_src_pts.shape[0]),(0,0)), mode='constant')
                aggr_feats = np.pad(aggr_feats, ((0,max_obj_pts_num-aggr_feats.shape[0]),(0,0)), mode='constant') if use_dino else None
        
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0)
        aggr_feats = np.concatenate([aggr_feats, ee_feats.detach().cpu().numpy()], axis=0) if use_dino else None
        
        if distill_dino:
            aggr_feats = fusion.eval_dist_to_sel_feats(torch.concat(src_feat_list + [ee_feats], axis=0),
                                                       obj_name=distill_obj,).detach().cpu().numpy()
        
        try:
            assert aggr_src_pts.shape[0] == N_total
            assert aggr_feats.shape[0] == N_total if use_dino else True
        except:
            raise RuntimeError('aggr_src_pts.shape[0] != N_total')
        
        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            # aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
        
        # decomp as part
        new_aggr_src_pts = []
        N_per_part = N_gripper // aggr_src_pts.shape[1]
        for i in range(aggr_src_pts.shape[0] // N_per_part):
            # only decomp the first instance
            if i == 0:
                aggr_src_pts_part = aggr_src_pts[i*N_per_part:(i+1)*N_per_part]
        
        
        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        aggr_feats_ls.append(aggr_feats.astype(np.float32) if (use_dino or distill_dino) else None)
    
    return aggr_src_pts_ls, aggr_feats_ls

# basically the same as d3fields_proc, but to keep the original code clean, we create a new function
def d3fields_proc_for_vis(fusion, shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                          robot_base_pose_in_world_seq = None, teleop_robot = None, qpos_seq=None, exclude_threshold=0.01,
                          exclude_colors=[], return_raw_feats=True):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    boundaries = shape_meta['info']['boundaries']
    use_dino = False
    distill_dino = shape_meta['info']['distill_dino'] if 'distill_dino' in shape_meta['info'] else False
    distill_obj = shape_meta['info']['distill_obj'] if 'distill_obj' in shape_meta['info'] else False
    N_total = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = robot_base_pose_in_world_seq.shape[1] if len(robot_base_pose_in_world_seq.shape) == 4 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_raw_feats_ls = []
    rob_mesh_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):
        # print()
        obs = {
            'color': color_seq[t],
            'depth': depth_seq[t],
            'pose': extri_seq[t][:,:3,:],
            'K': intri_seq[t],
        }
        
        fusion.update(obs, update_dino=(use_dino or distill_dino))
        
        if teleop_robot is not None:
            # compute robot pcd
            curr_qpos = qpos_seq[t]
            qpos_dim = curr_qpos.shape[0] // num_bots
            
            robot_pcd_ls = []
            robot_meshes_ls = []
            for rob_i in range(num_bots):
                # compute robot pcd
                robot_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
                robot_meshes = teleop_robot.gen_robot_meshes(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], link_names=['vx300s/left_finger_link', 'vx300s/right_finger_link', 'vx300s/gripper_bar_link', 'vx300s/gripper_prop_link', 'vx300s/gripper_link'])
            
                # transform robot pcd to world frame    
                robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i]
                robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
                # transform mesh to the frame of first robot
                for mesh in robot_meshes:
                    first_robot_base_pose_in_world = robot_base_pose_in_world_seq[t, 0] # (4, 4)
                    mesh.transform(np.linalg.inv(first_robot_base_pose_in_world) @ robot_base_pose_in_world)
                robot_meshes_ls = robot_meshes_ls + robot_meshes
                
                # save to list
                robot_pcd_ls.append(robot_pcd)
            # convert to numpy array
            robot_pcd = np.concatenate(robot_pcd_ls, axis=0)
            
            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.002, excluded_pts=robot_pcd, exclude_threshold=exclude_threshold, exclude_colors=exclude_colors)
        else:
            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.002, exclude_colors=exclude_colors)
        # res = fusion.eval(obj_pcd)
        # src_pts_list = [obj_pcd]
        # src_feat_list = [res['dino_feats']]
        src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(obj_pcd, N_total, per_instance=True, use_seg=False, use_dino=(use_dino or distill_dino))
        
        aggr_src_pts = np.concatenate(src_pts_list, axis=0) # (N, 3)
        aggr_feats = torch.concat(src_feat_list, axis=0).detach().cpu().numpy() if (use_dino or distill_dino) else None # (N, 1024)
        
        if return_raw_feats:
            if aggr_feats is not None:
                aggr_raw_feats = aggr_feats.copy()
                aggr_raw_feats_ls.append(aggr_raw_feats)
        
        if distill_dino:
            aggr_feats = fusion.eval_dist_to_sel_feats(torch.concat(src_feat_list, axis=0),
                                                       obj_name=distill_obj,).detach().cpu().numpy()
        
        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            # aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
        
        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        if use_dino or distill_dino:
            aggr_feats_ls.append(aggr_feats.astype(np.float32))
        if teleop_robot is not None:
            rob_mesh_ls.append(robot_meshes_ls)
    
    if return_raw_feats:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feats_ls, rob_mesh_ls
    return aggr_src_pts_ls, aggr_feats_ls, rob_mesh_ls

# basically the same as d3fields_proc, but to keep the original code clean, we create a new function
def d3fields_proc_for_tsne(fusion, shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                          robot_base_pose_in_world_seq = None, teleop_robot = None, qpos_seq=None, exclude_threshold=0.01,
                          exclude_colors=[], return_raw_feats=True):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    boundaries = shape_meta['info']['boundaries']
    use_dino = False
    distill_dino = shape_meta['info']['distill_dino'] if 'distill_dino' in shape_meta['info'] else False
    distill_obj = shape_meta['info']['distill_obj'] if 'distill_obj' in shape_meta['info'] else False
    N_total = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = robot_base_pose_in_world_seq.shape[1] if len(robot_base_pose_in_world_seq.shape) == 4 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_raw_feats_ls = []
    rob_mesh_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):
        # print()
        obs = {
            'color': color_seq[t],
            'depth': depth_seq[t],
            'pose': extri_seq[t][:,:3,:],
            'K': intri_seq[t],
        }
        
        fusion.update(obs, update_dino=(use_dino or distill_dino))
        
        grid, _ = create_init_grid(boundaries, step_size=0.005) # (N, 3)
        grid = grid.to(device=fusion.device, dtype=fusion.dtype)
        # res = fusion.eval(obj_pcd)
        # src_pts_list = [obj_pcd]
        # src_feat_list = [res['dino_feats']]
        # with torch.no_grad():
        #     grid_eval_res = fusion.batch_eval(grid, return_names=['dino_feats'])
        
        # grid_feats = grid_eval_res['dino_feats']
        
        src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(grid.detach().cpu().numpy(), -1, per_instance=False, use_seg=False, use_dino=(use_dino or distill_dino))
        aggr_src_pts = np.concatenate(src_pts_list, axis=0) # (N, 3)
        aggr_feats = torch.concat(src_feat_list, axis=0).detach().cpu().numpy() if (use_dino or distill_dino) else None # (N, 1024)
        
        if return_raw_feats:
            if aggr_feats is not None:
                aggr_raw_feats = aggr_feats.copy()
                aggr_raw_feats_ls.append(aggr_raw_feats)
        
        if distill_dino:
            aggr_feats = fusion.eval_dist_to_sel_feats(torch.concat(src_feat_list, axis=0),
                                                       obj_name=distill_obj,).detach().cpu().numpy()
        
        
        # # transform to reference frame
        # if reference_frame == 'world':
        #     pass
        # elif reference_frame == 'robot':
        #     # aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
        #     aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
        
        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        if use_dino or distill_dino:
            aggr_feats_ls.append(aggr_feats.astype(np.float32))
    
    if return_raw_feats:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feats_ls, rob_mesh_ls
    return aggr_src_pts_ls, aggr_feats_ls, rob_mesh_ls

def vis_actions(raw_actions):
    # (T, 7)
    import open3d as o3d
    import transforms3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # add a random mesh
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
    action_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    action_mesh_pts = np.asarray(action_mesh.vertices).copy()
    visualizer.add_geometry(action_mesh)
    for t in range(raw_actions.shape[0]):
        pos = raw_actions[t, :3]
        rot_euler = raw_actions[t, 3:6]
        rot_mat = transforms3d.euler.euler2mat(rot_euler[0], rot_euler[1], rot_euler[2])
        if t == 0:
            print('rot_mat: ', rot_mat)
        action_mesh_pts_new = rot_mat @ action_mesh_pts.T + pos.reshape(-1, 1)
        action_mesh_pts_new = action_mesh_pts_new.T
        action_mesh.vertices = o3d.utility.Vector3dVector(action_mesh_pts_new)
        visualizer.update_geometry(action_mesh)
        visualizer.poll_events()
        visualizer.update_renderer()
        if t == 0:
            visualizer.run()
        time.sleep(0.03)
    visualizer.destroy_window()

def vis_post_actions(actions):
    # (T, 10)
    import open3d as o3d
    import torch
    import pytorch3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # add a random mesh
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=0.05))
    action_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    action_mesh_pts = np.asarray(action_mesh.vertices).copy()
    visualizer.add_geometry(action_mesh)
    for t in range(actions.shape[0]):
        pos = actions[t, :3]
        rot_6d = actions[t, 3:9]
        rot_mat = pytorch3d.transforms.rotation_6d_to_matrix(torch.from_numpy(rot_6d).unsqueeze(0)).squeeze(0).numpy()
        if t == 0:
            print('rot_6d: ', rot_6d)
            print('rot_mat: ', rot_mat)
        action_mesh_pts_new = rot_mat @ action_mesh_pts.T + pos.reshape(-1, 1)
        action_mesh_pts_new = action_mesh_pts_new.T
        action_mesh.vertices = o3d.utility.Vector3dVector(action_mesh_pts_new)
        visualizer.update_geometry(action_mesh)
        visualizer.poll_events()
        visualizer.update_renderer()
        if t == 0:
            visualizer.run()
        time.sleep(0.03)
    visualizer.destroy_window()

def _convert_actions(raw_actions, rotation_transformer, action_key):
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 14)
    # vis_actions(raw_actions[:,:7])
    # vis_actions(raw_actions[:,7:])
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    if action_key == 'cartesian_action':
        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    elif action_key == 'joint_action':
        pass
    else:
        raise RuntimeError('unsupported action_key')
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    # vis_post_actions(actions[:,10:])
    return actions
