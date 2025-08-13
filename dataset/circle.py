import math
import torch
from torch.utils.data import Dataset


class Circle(Dataset):
    def __init__(
        self,
        num_samples: int,
        radius_range: tuple = (3.0, 3.0),
        center: tuple = ((2, 2), (-2, 2), (-2, -2), (2, -2)),
    ):
        self.num_samples = num_samples
        self.radius_range = radius_range
        self.center = center

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r = torch.empty(1).uniform_(*self.radius_range).item()
        theta0 = torch.empty(1).uniform_(to=2 * math.pi).item()
        c = torch.randint(0, 4, size=(1,)).item()
        cx, cy = self.center[c]
        x = cx + r * math.cos(theta0)
        y = cy + r * math.sin(theta0)
        point = torch.tensor([x, y])
        return point, c
