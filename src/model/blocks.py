import torch.nn as nn 
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

class EncoderLayer(MessagePassing):
    """
    Implment node and edge updates using message passing encoder.

    args:
        in_dim: dimension of input node+edge features (i.e: we could do in_dim/2 by matching rbf kernel size to node feature size.)
        hidden_units: dimension of hidden units in message passing
        message_passing_layers: number of layers in the message passing MLP
    """
    def __init__(self, node_in_dim: int, edge_features_dim: int,hidden_units: int, message_passing_layers=3 ):
        super().__init__(aggr="mean")

        self.node_in_dim = node_in_dim
        self.hidden_units = hidden_units
        self.edge_features_dim = edge_features_dim


        fcl_edge_message =[]
        cur_in_dim = edge_features_dim
        for _ in range(message_passing_layers):
            fcl_edge_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_edge_message.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_edge = nn.Sequential(*fcl_edge_message)

        fcl_message = []
        cur_in_dim = 2*hidden_units # add with edge projection features
        for _ in range(message_passing_layers):
            fcl_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_message.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_message = nn.Sequential(*fcl_message)

        self.fcl_middle = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units*2),
            nn.SiLU(),
            nn.Linear(self.hidden_units*2, self.hidden_units)
        )

        self.input_proj = nn.Linear(self.node_in_dim, self.hidden_units)


        self.norm = nn.LayerNorm(self.hidden_units)

    def forward(self, x, edge_index: torch.long, edge_attr):
        """
        args:
            x: node features, shape [N, F]
            edge_index: edge indices, shape [2, E]
            edge_attr: edge features, shape [E, D]
        returns:
            x: updated node features, shape [N, hidden_units]
            edge_attr: updated edge features, shape [E, hidden_units]
        """
        edge_attr = self.fcl_edge(edge_attr) # update edge features first, (E,D) -> (E,hidden_units)
        x = self.input_proj(x)
        x_propogate = self.propagate(edge_index, x=x, edge_attr=edge_attr) #aggregation and updates, message creation is defined using message method. (update)

        #res blocks: there should be no isolated nodes (is this just for robutness confirm...)
        x = self.norm(x + x_propogate)
        x =self.norm(x + self.fcl_middle(x))

        # update edge features
        #edge_attr_propogate = self.edge_updater(edge_index, x=x, edge_attr=edge_attr) # same but for edge features. aggregation brings (E,hidden_units) -> (N,hidden_units)
        #edge_attr = self.norm(edge_attr + edge_attr_propogate) # a

        return x, edge_attr

    
    def message(self, x_j, edge_attr):
        message = self.fcl_message(torch.cat([x_j, edge_attr], dim=-1))
        return message

class DecoderLayer(MessagePassing):
    """
    Message passing with an mlp regression head for decoding fitness.

    args:
        in_dim: dimension of input node features
        hidden_units: dimension of hidden units in message passing 
        reg_hidden_units: dimension of hidden units in regression MLP
        message_passing_layers: number of layers in the message passing MLP
        head_layers: number of layers in the regression MLP head
    """
    def __init__(self,  hidden_units: int, reg_hidden_units: int, message_passing_layers=3, head_layers=2):
        super().__init__(aggr="mean")

        self.hidden_units = hidden_units
        self.reg_hidden_units = reg_hidden_units
        self.message_passing_layers = message_passing_layers
        self.head_layers = head_layers
        
        message_fcl = []
        cur_in_dim = 2*self.hidden_units # add with edge projection features
        for _ in range(self.message_passing_layers):
            message_fcl.append(nn.Linear(cur_in_dim, self.hidden_units))
            message_fcl.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_message = nn.Sequential(*message_fcl)

        output =[]
        cur_in_dim = self.hidden_units
        for _ in range(self.head_layers):
            output.append(nn.Linear(cur_in_dim, self.reg_hidden_units)) # 164 - > 64, 32
            output.append(nn.SiLU())
            cur_in_dim = self.reg_hidden_units
        output.append(nn.Linear(cur_in_dim, 1))
        self.output = nn.Sequential(*output)

        self.norm = nn.LayerNorm(self.hidden_units)

        # pooling
        self.gate = nn.Linear(hidden_units, hidden_units) #feature
        self.score = nn.Linear(hidden_units, 1)

        
    def forward(self,x, edge_index, edge_attr, batch):
        x_propogate = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        x = self.norm(x + x_propogate)

        # attention pool
        gate = torch.sigmoid(self.gate(x))
        scores = self.score(x)
        weights = softmax(scores, batch)
        x = weights*gate*x
        x = global_mean_pool(x, batch) # (B, hidden_units)
  
        x = self.output(x)
        return x

    def message(self, x_j, edge_attr):
        message = self.fcl_message(torch.cat([x_j, edge_attr], dim=-1))
        return message

