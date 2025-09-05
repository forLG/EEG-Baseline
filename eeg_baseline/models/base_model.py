import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")