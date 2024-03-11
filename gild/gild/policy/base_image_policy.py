from typing import Dict
import torch
import torch.nn as nn
from gild.model.common.module_attr_mixin import ModuleAttrMixin
from gild.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **args) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def compute_loss(self, batch):
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

    def forward(self, obs_dict: Dict[str, torch.Tensor], **args) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        return self.predict_action(obs_dict, **args)
    
    # def forward(self, batch):
    #     """
    #     obs_dict:
    #         str: B,To,*
    #     return: B,Ta,Da
    #     """
    #     return self.compute_loss(batch)
