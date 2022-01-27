#! /usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# Residue Conv Block
class ResNetBlock(nn.Module):

    def __init__(self, in_chan, out_chan, nembed, act=F.relu):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_chan, out_chan, 3, 1, 1)
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

class LinearProj(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.w = nn.Parameter(torch.zeros(out_chan, in_chan))
        self.b = nn.Parameter(torch.zeros(1, out_chan, 1, 1))
        nn.init.xavier_normal_(self.w.data)

    def forward(self, x):
        x = torch.einsum('nchw,kc->nkhw', x, self.w)
        x = x + self.b
        return x

class SelfAttn(nn.Module):
    def __init__(self, nchan):
        super().__init__()

        self.nchan = nchan
        self.scale = 1.0/math.sqrt(nchan)
        self.kproj = LinearProj(nchan, nchan)
        self.qproj = LinearProj(nchan, nchan)
        self.vproj = LinearProj(nchan, nchan)
        self.hproj = LinearProj(nchan, nchan)

    def forward(self, x):

        b, c, h, w = x.shape
        k = self.kproj(x)
        q = self.qproj(x)
        v = self.vproj(x)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k)*self.scale
        w = w.view(b, h, w, h*w).softmax(dim=-1).view(b, h, w, h, w)
        v = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.hproj(v)

        return h + x

# see https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
class UNet(nn.Module):

    def __init__(self, nchan, nembed, nchan_scale = [1, 2, 4, 8]):
        super().__init__()
        self.nchan = nchan
        self.nchan_scale = nchan_scale
        # Downsample
        self.res1 = nn.ModuleList([ResNetBlock(nchan*s, nchan*s, nembed) for s in nchan_scale])
        self.vit1 = nn.ModuleList([SelfAttn(nchan*s) for s in nchan_scale])
        self.down_sample = nn.ModuleList([
            nn.Conv2d(nchan*s1, nchan*s2, 3, 2, 1) for s1, s2 in zip(nchan_scale[:-1], nchan_scale[1:])
        ])
        # Middle model
        self.res_m1 = nn.Conv2d(nchan*nchan_scale[-1], nchan*nchan_scale[-1], 3, 1, 1)
        self.attn_m1 = SelfAttn(nchan*nchan_scale[-1])
        self.res_m2 = nn.Conv2d(nchan*nchan_scale[-1], nchan*nchan_scale[-1], 3, 1, 1)
        # Upsample
        self.up_sample = nn.ModuleList([
            nn.ConvTranspose2d(nchan*s1, nchan*s2, 4, 2) for s2, s1 in reversed(
                list(zip(nchan_scale[:-1], nchan_scale[1:])))
        ])
        self.vit2 = nn.ModuleList([SelfAttn(2*nchan*s) for s in nchan_scale])
        self.res2 = nn.ModuleList([ResNetBlock(2*nchan * s, nchan*s, nembed) for s in reversed(nchan_scale)])

        self.conv_out = nn.Conv2d(nchan, nchan, 3, 1)

    def forward(self, x, t_embed):
        hs = []

        h = x
        for mres, vit, mds in zip(self.res1, self.vit1,  self.down_sample):
            h = mres(h, t_embed)
            h = vit(h)
            hs.append(h)
            h = mds(h)

        h = self.res1(h, t_embed)
        h = self.attn_m1(h)
        h = self.res2(h, t_embed)

        # TODO: fix hidden input size
        for mus, mres, vit, prev_h in zip(self.up_sample, self.res2, self.vit2, hs):
            h = mus(torch.cat([h, prev_h], dim=1))
            h = F.relu(h)
            h = mres(h, t_embed)
            h = vit(h)

        h = F.relu(h)
        h = self.conv_out(h)
        return h


class GaussDiffuse(nn.Module):

    def __init__(self, nchan, nembed, nchan_scale, h, w, tstep, mu, sigma):
        super().__init__()

        self.nchan = nchan
        self.nembed = nembed
        self.nchan_scale = nchan_scale
        self.tstep = tstep
        self.embed = PosEmbedding(self.nembed)
        self.unet = UNet(nchan, nembed, nchan_scale)
        self.mu = mu
        self.sigma = sigma
        self.h = h
        self.w = w

        # This alpha should be multiplication of alphas, alpha = alpha_1 * alpha_2 * ... * alpha_t
        # self.alpha = alpha
        # self.a1 = math.sqrt(self.alpha)
        # self.a2 = math.sqrt(1-self.alpha)

    def _diffuse(self, bs, x, alpha):
        # From 0 to T
        t = torch.randint(0, self.tstep, bs)
        eps = torch.randn(bs, 3, self.h, self.w)
        t_embed = self.embed(bs, t)
        a1 = math.sqrt(alpha)
        a2 = math.sqrt(1-alpha)
        z_t = self.unet(a1*x + a2*eps, t_embed)
        return (eps - z_t).square().mean()

    def _denoise(self, bs, x, t, alpha, z=None):
        if z is None:
            z = 0.0
        eps = torch.randn(bs, 3, self.h, self.w)
        t_embed = self.embed(bs, t)
        z_t = self.unet(x, t_embed)
        a2 = math.sqrt(1 - alpha)
        x = 1.0/alpha * (x - (1 - alpha)*z_t/a2) + eps*z
        return x

    def forward(self, x):
        if self.training:
            # In training mode, diffusing the input dataset
            return self._diffuse(x)
        else:
            return self._denoise(x)


if __name__ == "__main__":
    model1 = ResNetBlock(32, 64, 128)
    model2 = UNet(3, 12)
    model3 = GaussDiffuse(32, 64, 12, 128, 128, 16, 0.0, 1.0)