import time

import dill
import hydra
import torch
from omegaconf import open_dict
import diffusers

from gild.dataset.real_dataset import RealDataset
from gild.workspace.base_workspace import BaseWorkspace
from gild.policy.base_image_policy import BaseImagePolicy

# input_dir = "/home/yixuan/general_dp/data/outputs/aloha_pick_place_cup/2023.12.06_aloha_d3fields_no_feats_with_rob_pcd_pick_place_cup/checkpoints/epoch=7200.ckpt"
input_dir = "/media/yixuan_2T/diffusion_policy/data/outputs/hang_mug_sim/2024.01.25_hang_mug_sim_mixed_mug_demo_200_v5_original_rgbd/checkpoints/latest.ckpt"

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
    policy.eval().to(device)

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

cfg.data_root = "/media/yixuan_2T/diffusion_policy"
cfg.task.dataset.dataset_dir = "/media/yixuan_2T/diffusion_policy/data/sapien_demo/mixed_mug_demo_200_v5"
with open_dict(cfg.task.dataset):
    cfg.task.dataset.vis_input = False
dataset : RealDataset = hydra.utils.instantiate(cfg.task.dataset)

for i in range(len(dataset)):
    start_time = time.time()
    data = dataset[i]
    obs_dict = data['obs']
    gt_action = data['action'][None].to(device)
    for key in obs_dict.keys():
        obs_dict[key] = obs_dict[key].unsqueeze(0).to(device)
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred']
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        print(f"mse: {mse}")
    # measure frequency
    end_time = time.time()
    print(f"freq: {1/(end_time-start_time)}")
