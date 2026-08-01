import torch.nn as nn
import torch
from model.blocks import EncoderLayer, DecoderLayer, MaskedResidueDecoderLayer

class Tem1BetaGNN(nn.Module):
    """
    A simple GCN model with a single variate regression head for TEM-1 beta-lactamase fitness prediction.
    """
    def __init__(self, node_in_channels, edge_features_dim, hidden_channels, reg_hidden_channels, mp_layers, head_layers, dropout=0.0):
        super().__init__()

        self.node_in_channels = node_in_channels
        self.edge_features_dim = edge_features_dim
        self.hidden_channels = hidden_channels
        self.reg_hidden_channels = reg_hidden_channels
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.dropout = dropout

        self.encoder = EncoderLayer(node_in_channels, edge_features_dim, hidden_channels, mp_layers, dropout)
        self.decoder = DecoderLayer(hidden_channels, reg_hidden_channels, mp_layers, head_layers, dropout)

    def forward(self, x, edge_index, edge_attr, batch, mutation_idx):
        # Simple GCN layer implementation
        x, edge_attribute = self.encoder(x, edge_index, edge_attr)
        x = self.decoder(x, edge_index, edge_attribute, batch, mutation_idx)
        return x


class Tem1MaskedResidueGNN(nn.Module):
    """
    Structure-conditioned residue predictor for self-supervised pretraining.
    Shares the same EncoderLayer as Tem1BetaGNN. The decoder outputs 20
    AA logits instead of a pooled fitness scalar, and there is no mutation_idx/batch
    pooling step since every node gets its own prediction. Used to score a mutation
    zero-shot [log P(mutant|structure) - log P(WT|structure) (mutation score)] at a masked position, without ever training on DMS fitness labels.
    """
    def __init__(self, node_in_channels, edge_features_dim, hidden_channels, head_hidden_channels, mp_layers, head_layers, dropout=0.0):
        super().__init__()

        self.node_in_channels = node_in_channels
        self.edge_features_dim = edge_features_dim
        self.hidden_channels = hidden_channels
        self.head_hidden_channels = head_hidden_channels
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.dropout = dropout

        self.encoder = EncoderLayer(node_in_channels, edge_features_dim, hidden_channels, mp_layers, dropout)
        self.decoder = MaskedResidueDecoderLayer(hidden_channels, head_hidden_channels, mp_layers, head_layers, dropout)

    def forward(self, x, edge_index, edge_attr):
        x, edge_attribute = self.encoder(x, edge_index, edge_attr)
        logits = self.decoder(x, edge_index, edge_attribute)
        return logits