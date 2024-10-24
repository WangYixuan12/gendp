import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import open_dict
import gendp
sys.modules['diffusion_policy'] = gendp
from gendp.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--max_steps', default=-1, type=int)
@click.option('--n_test', default=-1, type=int)
@click.option('--n_train', default=-1, type=int)
@click.option('--n_test_vis', default=-1, type=int)
@click.option('--n_train_vis', default=-1, type=int)
@click.option('--train_start_idx', default=-1, type=int)
@click.option('--test_start_idx', default=-1, type=int)
@click.option('--dataset_dir', default=None, type=str)
@click.option('--repetitive', default=False, type=bool)
@click.option('--vis_3d', default=False, type=bool)
@click.option('--test_obj_ls', default=None, type=str, multiple=True)
@click.option('--train_obj_ls', default=None, type=str, multiple=True)
@click.option('--data_root', default="", type=str)
def main(checkpoint, output_dir, device,
         max_steps=-1, n_test=-1, n_train=-1, n_test_vis=-1, n_train_vis=-1, train_start_idx=-1, test_start_idx=-1, dataset_dir=None, vis_3d=False, repetitive=False,
         test_obj_ls=[], train_obj_ls=[], data_root=""):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    cfg.task.env_runner._target_ = 'gendp.env_runner.sapien_series_image_runner.SapienSeriesImageRunner'
    cfg.task.env_runner.max_steps = max_steps if max_steps > 0 else cfg.task.env_runner.max_steps
    cfg.task.env_runner.n_test = n_test if n_test >= 0 else cfg.task.env_runner.n_test
    cfg.task.env_runner.n_train = n_train if n_train >= 0 else cfg.task.env_runner.n_train
    cfg.task.env_runner.n_test_vis = n_test_vis if n_test_vis >= 0 else cfg.task.env_runner.n_test_vis
    cfg.task.env_runner.n_train_vis = n_train_vis if n_train_vis >= 0 else cfg.task.env_runner.n_train_vis
    cfg.task.env_runner.train_start_idx = train_start_idx if train_start_idx >= 0 else cfg.task.env_runner.train_start_idx
    cfg.task.env_runner.dataset_dir = dataset_dir if dataset_dir is not None else cfg.task.env_runner.dataset_dir
    with open_dict(cfg.task.env_runner):
        cfg.task.env_runner.test_start_seed = test_start_idx if test_start_idx >= 0 else cfg.task.env_runner.test_start_seed
        cfg.task.env_runner.repetitive = repetitive
        # cfg.task.env_runner.train_obj_ls = ['sprite', 'pepsi']
        # cfg.task.env_runner.test_obj_ls = ['sprite', 'pepsi']
        # cfg.task.env_runner.train_obj_ls = ['pepsi', 'sprite', 'obamna', 'mtndew', 'cola']
        # cfg.task.env_runner.test_obj_ls = ['pepsi', 'sprite', 'obamna', 'mtndew', 'cola']
        # cfg.task.env_runner.train_obj_ls = ['nescafe_mug']
        # cfg.task.env_runner.test_obj_ls = []
        cfg.task.env_runner.train_obj_ls = train_obj_ls
        cfg.task.env_runner.test_obj_ls = test_obj_ls
    # add sapien_env from dataset_dir
    import sys
    sys.path.append(cfg.task.env_runner.dataset_dir)
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy, vis_3d)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
