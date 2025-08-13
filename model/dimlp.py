import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, std=1.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([dim // 2]) * std)

    def forward(self, x: torch.Tensor):
        f = 2 * math.pi * x * self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class DiMLP(nn.Module):
    def __init__(self, num_labels, dim=256, output_size=2):
        super(DiMLP, self).__init__()

        self.t1_embedder = TimestepEmbedder(dim)
        self.t2_embedder = TimestepEmbedder(dim)
        self.e_embedder = nn.Linear(2, dim, bias=False)
        self.c_embedder = nn.Embedding(num_labels + 1, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, output_size)
        )
        
    def forward(self, e, t1=None, t2=None, c=None):
        if t1 is None and t2 is None:
            t1 = torch.zeros(e.shape[0], 1, device=e.device)
            t2 = torch.ones(e.shape[0], 1, device=e.device)
        x = self.e_embedder(e) + self.t1_embedder(t1) + self.t2_embedder(t2) + self.c_embedder(c)
        return self.mlp(x)
