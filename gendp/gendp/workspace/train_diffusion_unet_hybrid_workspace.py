if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import time
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from gendp.workspace.base_workspace import BaseWorkspace
from gendp.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.env_runner.base_image_runner import BaseImageRunner
from gendp.common.checkpoint_util import TopKCheckpointManager
from gendp.common.json_logger import JsonLogger
from gendp.common.pytorch_util import dict_apply, optimizer_to
from gendp.common.time_utils import get_time_now
from gendp.model.diffusion.ema_model import EMAModel
from gendp.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        if hasattr(cfg, 'is_real'):
            self.is_real = cfg.is_real
        else:
            self.is_real = False

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        start_time = get_time_now()
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        if not self.is_real:
            env_runner: BaseImageRunner
            is_env_rollout = ((cfg.task.env_runner.n_test + cfg.task.env_runner.n_train + cfg.task.env_runner.n_test_vis + cfg.task.env_runner.n_train_vis) > 0)
            if is_env_rollout:
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir)
                assert isinstance(env_runner, BaseImageRunner)
        else:
            is_env_rollout = False

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # iter_start = time.time()
                        # device transfer
                        # device_start = time.time()
                        
                        # remove d3fields entry with empty points
                        # if 'd3fields' in batch['obs']:
                        #     d3fields = batch['obs']['d3fields']
                        #     d3fields_mask = torch.any(d3fields[:, :, :3, :] == 0, dim=-1) # (B, T, 3)
                        #     d3fields_mask = torch.any(d3fields_mask, dim=-1) # (B, T)
                        #     d3fields_mask = torch.any(d3fields_mask, dim=-1) # (B,)
                        #     if (~d3fields_mask).max() == False:
                        #         continue
                        #     for key in batch['obs']:
                        #         batch['obs'][key] = batch['obs'][key][~d3fields_mask]
                        #     batch['action'] = batch['action'][~d3fields_mask]
                        
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # print(f'device transfer time: {time.time()-device_start}')

                        # compute loss
                        # forward_start = time.time()
                        raw_loss = self.model.compute_loss(batch)
                        # print(f'forward time: {time.time()-forward_start}')
                        # backward_start = time.time()
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        # print(f'backward time: {time.time()-backward_start}')
                        
                        # update ema
                        ema_start = time.time()
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        # print(f'ema time: {time.time()-ema_start}')

                        # logging
                        # logging_start = time.time()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'time_hrs': (get_time_now() - start_time).total_seconds() / (60*60),
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        # print(f'logging time: {time.time()-logging_start}')
                        # print(f'iter time: {time.time()-iter_start}')
                        # print()

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        time.sleep(1) # wait for the checkpoint to be written
                        self.save_checkpoint(use_thread=False)
                        time.sleep(1) # wait for the checkpoint to be written
                    if cfg.checkpoint.save_last_snapshot:
                        time.sleep(1) # wait for the checkpoint to be written
                        self.save_snapshot(use_thread=False)
                        time.sleep(1) # wait for the checkpoint to be written

                # run rollout
                if is_env_rollout:
                    if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch > 0:
                        # try:
                        runner_log = env_runner.run(policy)
                        # log all
                        step_log.update(runner_log)
                        # except Exception as e:
                        #     print(f"Error running rollout: {e}")
                        #     print("Skipping rollout")

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_action_mse_errors = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                # if 'd3fields' in batch['obs']:
                                #     d3fields = batch['obs']['d3fields']
                                #     d3fields_mask = torch.any(d3fields[:, :, :3, :] == 0, dim=-1) # (B, T, 3)
                                #     d3fields_mask = torch.any(d3fields_mask, dim=-1) # (B, T)
                                #     d3fields_mask = torch.any(d3fields_mask, dim=-1) # (B,)
                                #     if (~d3fields_mask).max() == False:
                                #         continue
                                #     for key in batch['obs']:
                                #         batch['obs'][key] = batch['obs'][key][~d3fields_mask]
                                #     batch['action'] = batch['action'][~d3fields_mask]
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                                
                                result = policy.predict_action(batch['obs'])
                                pred_action = result['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, batch['action'])
                                val_action_mse_errors.append(mse.item())
                                
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            step_log['val_action_mse_error'] = np.mean(val_action_mse_errors)

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0: # and self.epoch > 0:
                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # strategy 1: save topk checkpoints
                    # # We can't copy the last checkpoint here
                    # # since save_checkpoint uses threads.
                    # # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     time.sleep(1) # wait for the checkpoint to be written
                    #     self.save_checkpoint(path=topk_ckpt_path)
                    #     time.sleep(1) # wait for the checkpoint to be written
                    
                    # strategy 2: save checkpoint at this epoch
                    time.sleep(1)
                    self.save_checkpoint(tag=f'epoch={self.epoch:03d}', use_thread=False)
                    time.sleep(1)
                    
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                # print('epoch done')

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
