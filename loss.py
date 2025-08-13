import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, model, cfg=False, cfg_w=1.0, cfg_dropout=0.9, num_labels=0):
        super(DiffusionLoss, self).__init__()
        self.model = model
        self.cfg = cfg
        self.cfg_w = cfg_w
        self.cfg_dropout = cfg_dropout
        self.num_labels = num_labels

        self.lerp = lambda e, x, t: ((1 - t) * e + t * x)
        self.v = lambda x1, t1, t2, c: self.model(x1, t1, t2, c) - x1

    @staticmethod
    def sample_t(x):
        B, _ = x.shape
        samples = torch.rand(B, 3, device=x.device)
        t0 = samples[:, 2:]
        t1 = torch.minimum(samples[:, 0:1], samples[:, 1:2])
        t2 = torch.maximum(samples[:, 0:1], samples[:, 1:2])
        tm = (t1 + t2) / 2
        return t0, t1, t2, tm

    def forward(self, x, c):
        t0, t1, t2, tm = self.sample_t(x)
        e = torch.randn_like(x)
        z0 = self.lerp(e, x, t0)
        z1 = self.lerp(e, x, t1)
        v_tgt = x - e

        if self.cfg:
            uncond = torch.ones_like(c) * self.num_labels
            cfg_mask = torch.rand(*c.shape, device=c.device) < self.cfg_dropout
            c = torch.where(cfg_mask, c, uncond)
            v_uncond = self.v(z0, t0, t0, uncond)
            v_tgt = torch.where(cfg_mask.unsqueeze(-1), v_tgt * self.cfg_w + v_uncond * (1 - self.cfg_w), v_tgt)

        v_t0 = self.v(z0, t0, t0, c)
        v_t1t2 = self.v(z1, t1, t2, c)
        v_t1tm = self.v(z1, t1, tm, c)
        v_tmt2 = self.v(z1 + v_t1tm * (tm - t1), tm, t2, c)
        v_t1tmt2 = v_t1tm + v_tmt2

        loss1 = F.mse_loss(v_t0, v_tgt.detach())
        loss2 = F.mse_loss(v_t1t2 * 2, v_t1tmt2.detach())

        loss = loss1 + 0.25 * loss2
        return loss