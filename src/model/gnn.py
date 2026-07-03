import torch.nn as nn
import torch
from model.blocks import EncoderLayer, DecoderLayer

class Tem1BetaGNN(nn.Module):
    """
    A simple GCN model with a single variate regression head for TEM-1 beta-lactamase fitness prediction.
    """
    def __init__(self, in_channels, hidden_channels, reg_hidden_channels, mp_layers, head_layers):
        super().__init__()
        self.encoder = EncoderLayer(in_channels, hidden_channels, mp_layers)
        self.decoder = DecoderLayer(in_channels, hidden_channels, reg_hidden_channels, mp_layers, head_layers)

    def forward(self, x, edge_index, edge_attr):
        # Simple GCN layer implementation
        x = self.encoder(x, edge_index, edge_attr)
        x = self.decoder(x, edge_index, edge_attr)
        return x