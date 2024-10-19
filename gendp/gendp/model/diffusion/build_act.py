import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from gendp.model.detr.models import build_ACT_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    # parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
    #                     help="Name of the convolutional backbone to use")
    # use image backbone
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")
    
    # use point cloud backbone
    parser.add_argument('--backbone', default='pointnet', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--local_channels', default=(64, 128, 256), type=tuple, # will be overridden
                        help="Number of channels in each local block")
    parser.add_argument('--global_channels', default=(512,), type=tuple, # will be overridden
                        help="Number of channels in each global block")
    parser.add_argument('--observation_space', default=(512,3), type=tuple, 
                        help="observation space")
    
    # parser.add_argument('--num_points', default=512, type=int,
    #                     help="Number of points in each point cloud")
    # parser.add_argument('--pointnet_layers', default=3, type=int,
    #                     help="Number of layers in the PointNet backbone")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser


def build_ACT_model_and_optimizer(args_override):
    # default dictionary
    default_dict = {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'batch_size': 2,
        'weight_decay': 1e-4,
        'epochs': 300,
        'lr_drop': 200,
        'clip_max_norm': 0.1,
        'dilation': False,
        'position_embedding': 'sine',
        'camera_names': [],
        'backbone': 'pointnet',
        'enc_layers': 4,
        'dec_layers': 6,
        'dim_feedforward': 2048,
        'hidden_dim': 256,
        'dropout': 0.1,
        'nheads': 8,
        'num_queries': 400,
        'pre_norm': False,
        'masks': False,
        'eval': False,
        'onscreen_render': False,
        'ckpt_dir': '',
        'policy_class': '',
        'task_name': '',
        'seed': 0,
        'num_epochs': 0,
        'kl_weight': 0,
        'chunk_size': 0,
        'temporal_agg': False,
    }
    
    for k, v in args_override.items():
        default_dict[k] = v
    
    args = OmegaConf.create(default_dict)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer
