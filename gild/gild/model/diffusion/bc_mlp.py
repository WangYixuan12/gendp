from typing import Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class BCMLP(nn.Module):
    def __init__(self, act_dim, obs_dim):
        super().__init__()
        
        self.nets = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, obs, **kwargs):
        """
        obs: (B,obs_dim)
        """
        return self.nets(obs)
