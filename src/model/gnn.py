



class Tem1BetaGCN(nn.Module):
    """
    A simple GCN model for TEM-1 beta-lactamase fitness prediction.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # Simple GCN layer implementation
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x