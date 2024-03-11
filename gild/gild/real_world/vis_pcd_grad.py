import time
import os

import dill
import hydra
import torch
from omegaconf import open_dict
import diffusers
import open3d as o3d
import copy

from gild.workspace.base_workspace import BaseWorkspace
from gild.policy.base_image_policy import BaseImagePolicy
from d3fields.utils.my_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

def vis_pcd_grad(d3fields, grad = None):
    # d3fields: (B, n_hist, 3+D, N)
    # grad: (B, n_hist, 3+D, N)
    from d3fields.utils.draw_utils import np2o3d
    import open3d as o3d
    import numpy as np
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    
    if grad is not None:
        pts_grad_ls = []
        for t_idx in range(d3fields.shape[1]):
            pts = d3fields[0, t_idx, :3].detach().cpu().numpy().transpose()
            grad_t = np.abs(grad[0, t_idx, :3].detach().cpu().numpy().transpose())
            grad_t = np.linalg.norm(grad_t, axis=1)
            grad_t = grad_t * 10.0 / grad_t.max()
            # grad_t = grad_t * 10000
            grad_t = np.clip(grad_t, 0, 1)
            grad_t_color = cmap(grad_t)[:,:3]
            pts_grad_i = np2o3d(pts, grad_t_color)
            pts_grad_ls.append(pts_grad_i)
    else:
        pts_grad_ls = None
        
    pts_feats_ls = []
    feats = d3fields[0, 0, 3:].detach().cpu().numpy().transpose()
    pts = d3fields[0, 0, :3].detach().cpu().numpy().transpose()
    for i in range(feats.shape[1]):
        feat_i = feats[:, i]
        feat_i_color = cmap(feat_i)[:,:3]
        pts_feat_i = np2o3d(pts, feat_i_color)
        pts_feats_ls.append(pts_feat_i)
    
    return pts_grad_ls, pts_feats_ls

# input_dir = "/home/yixuan/general_dp/data/outputs/aloha_pick_place_cup/2023.12.06_aloha_d3fields_no_feats_with_rob_pcd_pick_place_cup/checkpoints/epoch=7200.ckpt"
# input_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/hang_mug_sim/2024.01.15_hang_mug_sim_mixed_mug_demo_200_v5_no_seg_no_dino_N_4000_pn2.1_r_0.04/checkpoints/epoch=200.ckpt"
# input_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/flip/2024.01.17_flip_Guang_flip_v2_no_seg_distill_dino_N_4000_pn2.1_r_0.04/checkpoint/latest.ckpt"

vis_name = 'flip'

if vis_name == 'hang_mug':
    is_demo = True
    data_root = "/media/yixuan_2T/diffusion_policy"
    dataset_dir = "/media/yixuan_2T/diffusion_policy/data/sapien_demo/mixed_mug_demo_200_v5"
    input_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/hang_mug_sim/2024.01.15_hang_mug_sim_mixed_mug_demo_200_v5_no_seg_distill_dino_N_4000_pn2.1_r_0.04/checkpoints/epoch=200.ckpt"
elif vis_name == 'flip':
    is_demo = False
    data_root = "/media/yixuan_2T/diffusion_policy"
    dataset_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/flip/2024.01.17_flip_Guang_flip_v2_no_seg_distill_dino_N_4000_pn2.1_r_0.04/eval"
    input_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/flip/2024.01.17_flip_Guang_flip_v2_no_seg_distill_dino_N_4000_pn2.1_r_0.04/checkpoints/latest.ckpt"
    

output_dir = input_dir.split('/')
output_dir = '/'.join(output_dir[:-2])
output_dir = output_dir + '/policy_vis'
os.system(f"mkdir -p {output_dir}")

# load checkpoint
ckpt_path = input_dir
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# hacks for method-specific setup.
action_offset = 0
delta_action = False
if 'diffusion' in cfg.name:
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    # policy.eval().to(device)
    policy.to(device)

    # set inference params
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    policy.num_inference_steps = 16 # DDIM inference iterations
    noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )
    policy.noise_scheduler = noise_scheduler

else:
    raise RuntimeError("Unsupported policy type: ", cfg.name)

cfg.data_root = data_root
cfg.task.dataset.dataset_dir = dataset_dir
cfg.task.dataset.use_cache = True
cfg.task.dataset.val_ratio = 0.0
cfg.task.dataset.manual_val_mask = False
cfg.task.dataset.manual_val_start = 1
with open_dict(cfg.task.dataset):
    cfg.task.dataset.vis_input = False
dataset = hydra.utils.instantiate(cfg.task.dataset)

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
pcd = o3d.geometry.PointCloud()

for i in range(len(dataset)):
    start_time = time.time()
    data = copy.copy(dataset[i])
    obs_dict = data['obs']
    gt_action = data['action'][None].to(device) if is_demo else None
    for key in obs_dict.keys():
        obs_dict[key] = obs_dict[key].unsqueeze(0).to(device)
        obs_dict[key].requires_grad = True
    # with torch.no_grad():
    result = policy.predict_action(obs_dict)
    pred_action = result['action_pred']
    if gt_action is not None:
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        print(f"mse: {mse}")
        mse.backward()
        # for key in obs_dict.keys():
        #     print(f"grad: {obs_dict[key].grad}")
        obs_dict['d3fields'].grad[0, :, :, -cfg.shape_meta.obs.d3fields.info.N_per_inst:] = 0
        pts_grad_ls, pts_feats_ls = vis_pcd_grad(obs_dict['d3fields'], obs_dict['d3fields'].grad)
    else:
        pts_grad_ls, pts_feats_ls = vis_pcd_grad(obs_dict['d3fields'], None)
    
    for feat_idx, pts_feat_i in enumerate(pts_feats_ls):
        pcd.points = pts_feat_i.points
        pcd.colors = pts_feat_i.colors
            
        if i == 0 and feat_idx == 0:
            visualizer.add_geometry(pcd)
            
            view_control = visualizer.get_view_control()
            view_control.set_front([-0.5041976199714987, -0.792452798282268, 0.34322488620389874])
            view_control.set_lookat([-0.0249892015906369, 0.017080835760080269, 0.22190024528036578])
            view_control.set_up([0.25269929663900503, 0.24466133314996144, 0.93610036723603296])
            view_control.set_zoom(1.0200000000000002)
            
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        if not os.path.exists(f"{output_dir}/feat/{feat_idx}"):
            os.system(f"mkdir -p {output_dir}/feat/{feat_idx}")
        visualizer.capture_screen_image(f"{output_dir}/feat/{feat_idx}/feat_{i}.png")
    
    if is_demo:
        for t_idx, pts_grad_i in enumerate(pts_grad_ls):
            pcd.points = pts_grad_i.points
            pcd.colors = pts_grad_i.colors
                
            visualizer.update_geometry(pcd)
            visualizer.poll_events()
            visualizer.update_renderer()
            if not os.path.exists(f"{output_dir}/grad/{t_idx}"):
                os.system(f"mkdir -p {output_dir}/grad/{t_idx}")
            visualizer.capture_screen_image(f"{output_dir}/grad/{t_idx}/grad_{i}.png")
    
    # measure frequency
    end_time = time.time()
    # print(f"freq: {1/(end_time-start_time)}")

# generate video
if is_demo:
    len_t = len(os.listdir(f"{output_dir}/grad"))
    for t_idx in range(len_t):
        os.system(f"ffmpeg -framerate 10 -i {output_dir}/grad/{t_idx}/grad_%d.png {output_dir}/grad_{t_idx}.mp4")

len_feat = len(os.listdir(f"{output_dir}/feat"))
for feat_idx in range(len_feat):
    os.system(f"ffmpeg -framerate 10 -i {output_dir}/feat/{feat_idx}/feat_%d.png {output_dir}/feat_{feat_idx}.mp4")

