import torch.nn as nn 
from torch_geometric.nn import MessagePassing
import torch

class EncoderLayer(MessagePassing):
    """
    Implment node and edge updates using message passing encoder.

    args:
        in_dim: dimension of input node+edge features (i.e: we could do in_dim/2 by matching rbf kernel size to node feature size.)
        hidden_units: dimension of hidden units in message passing
        message_passing_layers: number of layers in the message passing MLP
    """
    def __init__(self, in_dim: int, hidden_units: int, message_passing_layers=3 ):
        super().__init__(aggr="mean")

        self.in_dim = in_dim
        self.hidden_units = hidden_units

        fcl_message = []
        cur_in_dim = in_dim
        for _ in range(message_passing_layers):
            fcl_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_message.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_message = nn.Sequential(*fcl_message)

        fcl_edge_message =[]
        cur_in_dim = in_dim
        for _ in range(message_passing_layers):
            fcl_edge_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_edge_message.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_edge_message = nn.Sequential(*fcl_edge_message)

        self.fcl_middle = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units*2),
            nn.SiLU(),
            nn.Linear(self.hidden_units*2, self.hidden_units)
        )

        self.norm = nn.LayerNorm(self.hidden_units)

    def forward(self, x, edge_index, edge_attr):
        """
        args:
            x: node features, shape [N, F]
            edge_index: edge indices, shape [2, E]
            edge_attr: edge features, shape [E, D]
        returns:
            x: updated node features, shape [N, hidden_units]
            edge_attr: updated edge features, shape [E, hidden_units]
        """
        x_propogate = self.propagate(edge_index, x=x, edge_attr=edge_attr) #aggregation and updates, message creation is defined using message method. (update)

        #res blocks: there should be no isolated nodes (is this just for robutness confirm...)
        x = self.norm(x + x_propogate)
        x =self.norm(x + self.fcl_middle(x))

        # update edge features
        edge_attr_propogate = self.edge_updater(edge_attr, x=x, edge_attr=edge_attr) # same but for edge features. aggregation brings (E,hidden_units) -> (N,hidden_units)
        edge_attr = self.norm(edge_attr + edge_attr_propogate) # a

        return x, edge_attr

    def edge_update(self, x_j, edge_attr):
        """ x_j is indexed by edge indexs (connections), therfore (N,F) -> (E, F) during this step"""
        message = self.fcl_edge_message(torch.cat([x_j, edge_attr], dim=-1)) # (E,F) + (E,D) -> (E, E+D = hidden_units)
        return message
    
    def message(self, x_j, edge_attr):
        message = self.fcl_message(torch.cat([x_j, edge_attr], dim=-1))
        return message

class ClassifierDecoder(MessagePassing):
    """
    Message passing with a simple mlp classifier head for decoding fitness.

    args:
        in_dim: dimension of input node features
        hidden_units: dimension of hidden units in message passing and classifier
        message_passing_layers: number of layers in the message passing MLP
        classifier_layers: number of layers in the classifier MLP
        message_passing_layers: number of layers in the message passing MLP
        classifier_layers: number of layers in the classifier MLP
    """
    def __inint__(self, in_dim: int, hidden_units: int, classifier_hidden_units: int, message_passing_layers=3, classifier_layers=2):
        super().__init__(aggr="mean")
        self.in_dim = in_dim
        self.hidden_units = hidden_units
        self.classifier_hidden_units = classifier_hidden_units
        self.message_passing_layers = message_passing_layers
        self.classifier_layers = classifier_layers

        message_fcl = []
        cur_in_dim = in_dim
        for _ in range(self.message_passing_layers):
            message_fcl.append(nn.Linear(cur_in_dim, self.hidden_units))
            message_fcl.append(nn.SiLU())
            cur_in_dim = self.hidden_units
        self.fcl_message = nn.Sequential(*message_fcl)

        output =[]
        for _ in range(self.classifier_layers):
            output.append(nn.Linear(self.hidden_units, self.classifier_hidden_units))
            output.append(nn.SiLU())
            cur_in_dim = self.classifier_hidden_units

        output.append(nn.Linear(cur_in_dim, 1))
        self.output = nn.Sequential(*output)

        self.norm = nn.LayerNorm(self.hidden_units)

    def forward(self,x, edge_index, edge_attr):
        x_propogate = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        x = self.norm(x + x_propogate)
        x = self.output(x)
        return x

    def message(self, x_j, edge_attr):
        message = self.fcl_message(torch.cat([x_j, edge_attr], dim=-1))
        return message

