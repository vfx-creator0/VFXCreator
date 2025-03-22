import torch
from diffusers import ModelMixin, ConfigMixin
from torch import nn 
from torch.nn import functional as F
from diffusers.configuration_utils import register_to_config
import os
from typing import Any, Dict, Optional, Tuple, Union

class TimeSettingEmbedding(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, embedding_dim=4096, hidden_dim=4096,
                 num_layers=1, activation=None, layer_norm=True,
                 zero_init=True, logize_input=True):
        '''
        Maping the camera setting from EXIF to same dimension as the token
        embedding in CLIP. 
        '''
        super().__init__()

        self.zero_init = zero_init
        self.logize_input = logize_input
        self.activation = activation

        self.embed_focal_length = []
        self.embed_aperture = []
        for i in range(num_layers):
            if num_layers == 1 and i == 0:
                self.embed_focal_length.append(nn.Linear(1, embedding_dim))
                self.embed_aperture.append(nn.Linear(1, embedding_dim))
            elif i == 0:
                self.embed_focal_length.append(nn.Linear(1, hidden_dim))
                self.embed_aperture.append(nn.Linear(1, hidden_dim))
            elif i == num_layers - 1:
                self.embed_focal_length.append(nn.Linear(hidden_dim, embedding_dim))
                self.embed_aperture.append(nn.Linear(hidden_dim, embedding_dim))
            else:
                self.embed_focal_length.append(nn.Linear(hidden_dim, hidden_dim))
                self.embed_aperture.append(nn.Linear(hidden_dim, hidden_dim))
               
            
            if self.zero_init:
                nn.init.zeros_(self.embed_focal_length[-1].weight)
                nn.init.zeros_(self.embed_aperture[-1].weight)

                nn.init.zeros_(self.embed_focal_length[-1].bias)
                nn.init.zeros_(self.embed_aperture[-1].bias)

            if layer_norm and i != num_layers - 1:
                self.embed_focal_length.append(nn.LayerNorm(hidden_dim))
                self.embed_aperture.append(nn.LayerNorm(hidden_dim))
            elif layer_norm and i == num_layers - 1:
                self.embed_focal_length.append(nn.LayerNorm(embedding_dim))
                self.embed_aperture.append(nn.LayerNorm(embedding_dim))
            if i != num_layers - 1 and self.activation is not None:

                if self.activation == 'silu':
                    activation_layer = nn.SiLU()
                elif activation == 'relu':
                    activation_layer = nn.ReLU()
                elif activation == 'gelu':
                    activation_layer = nn.GELU()

                self.embed_focal_length.append(activation_layer)
                self.embed_aperture.append(activation_layer)
        self.embed_focal_length = nn.Sequential(*self.embed_focal_length)
        self.embed_aperture = nn.Sequential(*self.embed_aperture)

    
    def forward(self, x_focal_length,x_aperture):
        if self.logize_input:
            x_focal_length = torch.log(x_focal_length + 1e-6)
            x_aperture = torch.log(x_aperture + 1e-6)
        y_focal_length = self.embed_focal_length(x_focal_length.view(-1,1)).unsqueeze(1)#([1, 1, 1024])    
        y_aperture = self.embed_aperture(x_aperture.view(-1,1)).unsqueeze(1)
        y = torch.cat([y_focal_length, y_aperture], dim=1)
        return y#([1, 4, 1024])

class MaskSettingEmbedding(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, embedding_dim=4096, hidden_dim=1024,
                 num_layers=1, activation=None, layer_norm=True,
                 zero_init=True, logize_input=True):
        '''
        Maping the camera setting from EXIF to same dimension as the token
        embedding in CLIP. 
        '''
        super().__init__()

        self.zero_init = zero_init
        self.logize_input = logize_input
        self.activation = activation

        self.embed_focal_length = []
        self.embed_aperture = []

        self.embed_focal_length.append(nn.Linear(hidden_dim, embedding_dim))

               
            
        if self.zero_init:
            nn.init.zeros_(self.embed_focal_length[-1].weight)

            nn.init.zeros_(self.embed_focal_length[-1].bias)


        self.embed_focal_length.append(nn.LayerNorm(embedding_dim))

        self.embed_focal_length = nn.Sequential(*self.embed_focal_length)
    
    def forward(self, x_focal_length):
        y_focal_length = self.embed_focal_length(x_focal_length).unsqueeze(1)#([1, 1, 1024])    
        return y_focal_length#([1, 4, 1024])
    
ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    
# class TimestepEmbedding(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         time_embed_dim: int,
#         act_fn: str = "silu",
#         out_dim: int = None,
#         post_act_fn: Optional[str] = None,
#         cond_proj_dim=None,
#         sample_proj_bias=True,
#     ):
#         super().__init__()

#         self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

#         if cond_proj_dim is not None:
#             self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
#         else:
#             self.cond_proj = None

#         self.act = get_activation(act_fn)

#         if out_dim is not None:
#             time_embed_dim_out = out_dim
#         else:
#             time_embed_dim_out = time_embed_dim
#         self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

#         if post_act_fn is None:
#             self.post_act = None
#         else:
#             self.post_act = get_activation(post_act_fn)

#     def forward(self, sample, condition=None):
#         if condition is not None:
#             sample = sample + self.cond_proj(condition)
#         sample = self.linear_1(sample)

#         if self.act is not None:
#             sample = self.act(sample)

#         sample = self.linear_2(sample)

#         if self.post_act is not None:
#             sample = self.post_act(sample)
#         return sample

class TimestepEmbedding(ModelMixin, ConfigMixin):
    def __init__(
        self,
        in_channels=49,
        time_embed_dim=512,
        act_fn = "silu",
        out_dim = None,
        post_act_fn= None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample