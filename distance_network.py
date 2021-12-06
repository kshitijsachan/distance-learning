import torch
from torch import nn

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(34, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
