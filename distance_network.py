import torch
from torch import nn

class DistanceNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DistanceNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, output_dim),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # probs = logits.softmax(dim=1)
        return logits 
