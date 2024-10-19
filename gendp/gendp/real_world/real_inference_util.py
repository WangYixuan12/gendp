from typing import Dict, Callable, Tuple, Optional
import numpy as np
import cv2
from gendp.common.cv2_util import get_image_transform
from gendp.common.data_utils import d3fields_proc

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        fusion = None,
        expected_labels = None,
        teleop = None,
        exclude_colors = [],
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr['type']
        shape = attr['shape']
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'depth':
            this_depths_in = env_obs[key]
            t,hi,wi = this_depths_in.shape
            this_depths_in = this_depths_in.reshape(t,hi,wi,1)
            this_depths_in = np.clip(this_depths_in, 0, 1000).astype(np.uint16)
            co,ho,wo = shape
            assert co == 1
            out_depths = this_depths_in
            if (ho != hi) or (wo != wi):
                out_depths = np.stack([cv2.resize(x, (wo,ho)) for x in this_depths_in])
                if this_depths_in.dtype == np.uint16:
                    out_depths = out_depths.astype(np.float32) / 1000.
            # THWC to TCHW
            out_depths = out_depths[..., None]
            obs_dict_np[key] = np.moveaxis(out_depths,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
        elif type == 'spatial':
            try:
                assert key == 'd3fields'
            except AssertionError:
                raise RuntimeError('Only support d3fields as spatial type.')
            try:
                assert fusion is not None
            except AssertionError:
                raise RuntimeError('fusion is None, but d3fields is requested.')

            # construct inputs for d3fields processing
            view_keys = attr['info']['view_keys']
            use_dino = False
            distill_dino = attr['info']['distill_dino'] if 'distill_dino' in attr['info'] else False
            tool_names = [None, None]
            if 'right_tool' in attr['info']:
                tool_names[0] = attr['info']['right_tool']
            if 'left_tool' in attr['info']:
                tool_names[1] = attr['info']['left_tool']
            color_seq = np.stack([env_obs[f'{k}_color'] for k in view_keys], axis=1) # (T, V, H ,W, C)
            depth_seq = np.stack([env_obs[f'{k}_depth'] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
            extri_seq = np.stack([env_obs[f'{k}_extrinsics'] for k in view_keys], axis=1) # (T, V, 4, 4)
            intri_seq = np.stack([env_obs[f'{k}_intrinsics'] for k in view_keys], axis=1) # (T, V, 3, 3)
            qpos_seq = env_obs['full_joint_pos'] if 'full_joint_pos' in env_obs else env_obs['joint_pos'] # (T, -1)
            if 'robot_base_pose_in_world' in env_obs:
                robot_base_pose_in_world_seq = env_obs['robot_base_pose_in_world'] # (T, 4, 4)

            aggr_src_pts_ls, aggr_feats_ls = \
                d3fields_proc(fusion=fusion,
                            shape_meta=attr,
                            color_seq=color_seq,
                            depth_seq=depth_seq,
                            extri_seq=extri_seq,
                            intri_seq=intri_seq,
                            robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                            qpos_seq=qpos_seq,
                            teleop_robot=teleop,
                            expected_labels=expected_labels,
                            tool_names=tool_names,
                            exclude_colors=exclude_colors,)
            aggr_src_pts = np.stack(aggr_src_pts_ls)
            aggr_feats = np.stack(aggr_feats_ls)
            if use_dino or distill_dino:
                aggr_pts_feats = np.concatenate([aggr_src_pts, aggr_feats], axis=-1)
            else:
                aggr_pts_feats = aggr_src_pts
            obs_dict_np[key] = aggr_pts_feats.transpose(0,2,1)

    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
