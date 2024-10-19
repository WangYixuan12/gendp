from typing import Dict, OrderedDict
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from gendp.model.common.normalizer import LinearNormalizer
from gendp.policy.base_image_policy import BaseImagePolicy
from gendp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from gendp.model.diffusion.mask_generator import LowdimMaskGenerator
from gendp.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmoc
from robomimic.models.obs_nets import ObservationGroupEncoder
import gendp.model.vision.crop_randomizer as dmvc
from gendp.common.pytorch_util import dict_apply, replace_submodules
from gendp.model.diffusion.build_act import build_ACT_model_and_optimizer

class ACTPolicy(BaseImagePolicy):
    def __init__(self,
                 enc_layers,
                 dec_layers,
                 nheads,
                 lr_backbone,
                 lr,
                 backbone,
                 kl_weight,
                 hidden_dim,
                 dim_feedforward,
                 chunk_size,
                 shape_meta,
                 n_action_steps,):
        super().__init__()
        camera_names = [k for k in shape_meta['obs'].keys() if shape_meta['obs'][k]['type'] == 'rgb']
        policy_config = {'lr': lr,
                         'num_queries': chunk_size,
                         'kl_weight': kl_weight,
                         'hidden_dim': hidden_dim,
                         'dim_feedforward': dim_feedforward,
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'action_dim': shape_meta['action']['shape'][0],
                         'camera_names': camera_names,
                         'shape_meta': shape_meta,}
        model, optimizer = build_ACT_model_and_optimizer(policy_config)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = policy_config['kl_weight']
        
        self.normalizer = LinearNormalizer()
        self.camera_names = camera_names
        self.horizon = chunk_size
        self.n_action_steps = n_action_steps
        self.n_obs_steps = 1
        
        self.shape_meta = shape_meta
        
        print("ACT params: ", sum(p.numel() for p in self.model.parameters()))

    def configure_optimizers(self):
        return self.optimizer
    
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        To = self.n_obs_steps

        is_spatial = False
        spatial_key = ""
        for k in self.shape_meta['obs'].keys():
            if self.shape_meta['obs'][k]['type'] == 'spatial':
                spatial_key = k
                is_spatial = True
                break
        qpos = nobs['joint_pos'][:,0]
        if not is_spatial:
            image = [nobs[k][:,0:1] for k in self.camera_names]
            image = torch.concat(image, dim=1)
            action_pred, _, (_, _) = self.model(qpos, image, None) # no action, sample from prior
        else:
            spatial_in = nobs[spatial_key][:,0:1]
            action_pred, _, (_, _) = self.model(qpos, spatial_in, None) # no action, sample from prior

        action_pred = self.normalizer['action'].unnormalize(action_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        act_dim = nactions.shape[2]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        is_spatial = False
        spatial_key = ""
        for k in self.shape_meta['obs'].keys():
            if self.shape_meta['obs'][k]['type'] == 'spatial':
                spatial_key = k
                is_spatial = True
                break
        qpos = nobs['joint_pos'][:,0]
        is_pad = batch['is_pad']
        if not is_spatial:
            image = [nobs[k][:,0:1] for k in self.camera_names]
            image = torch.concat(image, dim=1)
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, None, nactions, is_pad)
        else:
            spatial_in = nobs[spatial_key][:,0:1]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, spatial_in, None, nactions, is_pad)
        
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss_dict = dict()
        all_l1 = F.l1_loss(nactions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict['loss']

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


