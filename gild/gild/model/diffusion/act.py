from typing import Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PointToken(nn.Module):
    def __init__(self, pos_feat_dim = 128, dino_feat_dim = 128, has_feat = True):
        super().__init__()
        self.pos_feat_dim = pos_feat_dim
        self.dino_feat_dim = dino_feat_dim
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_feat_dim),
        )
        if has_feat:
            self.dino_feat_mlp = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.dino_feat_dim),
            )
            self.n_emb = self.pos_feat_dim + self.dino_feat_dim
        else:
            self.n_emb = self.pos_feat_dim
    
    def get_emb_dim(self):
        return self.n_emb
    
    def forward(self, pcd):
        # pcd: (B, D, N)
        B, D, N = pcd.shape
        pcd = pcd.transpose(1, 2).reshape(B*N, D)
        pos_feat = self.pos_mlp(pcd[:, :3])
        if self.has_feat:
            dino_feat = self.dino_feat_mlp(pcd[:, 3:]) 
            token = torch.cat([pos_feat, dino_feat], dim=1) # (B*N, n_emb)
        else:
            token = pos_feat # (B*N, n_emb)
        token = token.reshape(B, N, self.n_emb).transpose(1, 2) # (B, n_emb, N)
        return token

class PointNetToken(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pcd):
        pass

class ACT(nn.Module):
    def __init__(self, n_head=8, n_layer=6, p_drop_attn=0.1):
        super().__init__()
        
        self.tokenizer = PointToken(pos_feat_dim=128,
                                    dino_feat_dim=128,
                                    has_feat=False)
        
        n_emb = self.tokenizer.get_emb_dim()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer
        )
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, obs, **kwargs):
        """
        obs: (B,obs_dim)
        """
        pass
