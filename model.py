#! /usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):

    def __init__(self, nchan, act=F.relu):
        super().__init__()

        self.down_sample = nn.Conv2d(nchan, nchan, 3, 2, 1)
        self.up_sample = nn.ConvTranspose2d(nchan, nchan, 3, 2, 1)
        self.act = act

    def forward(self, x):

        h = self.down_sample(x)
        h = self.act(h)
        h = self.up_sample(x)
        h = self.act(h)

        return x + h

class PixelCNN(nn.Module):
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