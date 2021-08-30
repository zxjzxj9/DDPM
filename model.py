#! /usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate Positional Embedding
class PosEmbedding(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

    def forward(self, bs, sz):
        timesteps = torch.arange(sz, dtype=torch.float32).repeat(bs, 1)
        half_dim = self.ndim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.unsqueeze(-1) * emb.view(1, 1, -1)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        print(emb.shape, torch.zeros(bs, sz, 1, dtype=torch.float32).shape)
        if self.ndim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros(bs, sz, 1, dtype=torch.float32)], dim=-1)
        return emb

class ResNetBlock(nn.Module):

    def __init__(self, in_chan, out_chan, nembed, act=F.relu):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 2, 1)
        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, 2, 1)
        self.res = nn.Conv2d(in_chan, out_chan, 1)
        self.fc = nn.Conv2d(nembed, out_chan, 1)
        self.act = act

    def forward(self, x, t_embed):

        # Embedding
        t_embed = t_embed.view(t_embed.size(0), t_embed.size(1), 1, 1)
        t_embed = self.fc(t_embed)

        # Downsample -> Upsample
        h = self.act(x)
        h = self.conv1(h)
        h = self.act(h + t_embed)
        h = self.conv2(h)
        h = self.act(h)

        # Residue
        x = self.res(x)
        return x + h

# see https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py

class UNet(nn.Module):

    def __init__(self, nchan_scale = [1, 2, 4, 8]):
        super().__init__()

        self.nchan_scale = nchan_scale

    def forward(selfs):
        pass


class Diffusion(nn.Module):

    def __init__(self, alpha):
        super().__init__()

        if alpha > 1 or alpha < 0:
            raise Exception(f"Invalid alpha value: {alpha:12.6f}")
        self.beta1 = math.sqrt(alpha)
        self.beta2 = math.sqrt(1.0-alpha)

    def _forward(self, x):
        x = self.beta1*x + self.beta2*torch.randn_like(x)
        return x

    def _backward(self, x):
        pass

    def forward(self, x):
        pass