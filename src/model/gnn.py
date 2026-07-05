import torch.nn as nn
import torch
from model.blocks import EncoderLayer, DecoderLayer

class Tem1BetaGNN(nn.Module):
    """
    A simple GCN model with a single variate regression head for TEM-1 beta-lactamase fitness prediction.
    """
    def __init__(self, node_in_channels, edge_features_dim, hidden_channels, reg_hidden_channels, mp_layers, head_layers):
        super().__init__()

        self.node_in_channels = node_in_channels
        self.edge_features_dim = edge_features_dim
        self.hidden_channels = hidden_channels
        self.reg_hidden_channels = reg_hidden_channels
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        
        self.encoder = EncoderLayer(node_in_channels, edge_features_dim, hidden_channels, mp_layers)
        self.decoder = DecoderLayer(hidden_channels, reg_hidden_channels, mp_layers, head_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        # Simple GCN layer implementation
        x, edge_attribute = self.encoder(x, edge_index, edge_attr)
        x = self.decoder(x, edge_index, edge_attribute, batch)
        return x