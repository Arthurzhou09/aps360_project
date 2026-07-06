import torch
import torch.nn as nn   



class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) model for regression tasks.
    """
    def __init__(self, input_dim,bottleneck_hidden_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_hidden_dim = bottleneck_hidden_dim

        self.fcl = nn.Sequential(
            nn.Linear(input_dim, bottleneck_hidden_dim*2),
            nn.LayerNorm(bottleneck_hidden_dim*2),
            nn.SiLU(),
            nn.Linear(bottleneck_hidden_dim*2, bottleneck_hidden_dim),
            nn.LayerNorm(bottleneck_hidden_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_hidden_dim, bottleneck_hidden_dim),
            nn.LayerNorm(bottleneck_hidden_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_hidden_dim, 1)
        )
       
    def forward(self, x):
        return self.fcl(x)
    