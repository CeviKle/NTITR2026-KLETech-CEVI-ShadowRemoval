import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os

from model_convnext import fusion_net

from Restormer.restormer_arch import Restormer


class final_net(nn.Module):
    def __init__(self):
        super(final_net, self).__init__()
        self.remove_model = fusion_net()
        self.enhancement_model =  Restormer()
        self.use_grad_checkpointing = False

    def set_gradient_checkpointing(self, enabled=True):
        self.use_grad_checkpointing = enabled

    def forward(self, input, scale=0.05):
        if self.training and self.use_grad_checkpointing:
            x = checkpoint(self.remove_model, input, use_reentrant=False)
            enhanced = checkpoint(self.enhancement_model, x, use_reentrant=False)
        else:
            x = self.remove_model(input)
            enhanced = self.enhancement_model(x)
        x_ = (enhanced * scale + x ) / (1 + scale)
        return x_


