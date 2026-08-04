import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax

class MessagePassingRound(MessagePassing):
    """
    One round of attention-gated message passing followed by a gated residual and a
    feed-forward block. Pulled out of EncoderLayer so several rounds can be stacked with
    independent weights.

    Note the distinction from message_passing_layers, which is the DEPTH OF THE MLP that
    builds a single message. Stacking rounds is what grows the receptive field: one round
    lets a node see its neighbours, two rounds lets it see neighbours-of-neighbours, and
    so on. ProteinMPNN  uses 3 encoder and 3 decoder rounds.

    args:
        hidden_units: dimension of hidden units in message passing
        message_passing_layers: number of layers in the message passing MLP
    """
    def __init__(self, hidden_units: int, message_passing_layers=3, dropout=0.0):
        super().__init__(aggr="add") # messages are pre-weighted by softmax attention (see message()), so "add" sums a weighted average instead of "mean"'s uniform 1/degree dilution

        self.hidden_units = hidden_units

        fcl_message = []
        cur_in_dim = 2*hidden_units # add with edge projection features
        for _ in range(message_passing_layers):
            fcl_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_message.append(nn.SiLU())
            fcl_message.append(nn.Dropout(dropout))
            cur_in_dim = self.hidden_units
        self.fcl_message = nn.Sequential(*fcl_message)

        # attention-weighted aggregation: same score/gate pattern DecoderLayer uses for its mutation-site pooling, applied per-edge instead of per-graph. attn_score gives each neighbor a softmax weight over the receiving node's incoming edges (replacing mean's uniform 1/degree weighting), attn_gate lets a message be suppressed independently of its attention weight.
        self.attn_score = nn.Linear(hidden_units, 1)
        self.attn_gate = nn.Linear(hidden_units, hidden_units)
        nn.init.constant_(self.attn_gate.bias, 2.0) # start ~sigmoid(2)=0.88 open (not 0.5)

        self.fcl_middle = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units*2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_units*2, self.hidden_units)
        )

        # gates how much of the aggregated neighbor update reaches the residual stream, so a
        # node can learn to mostly keep its own identity when neighbors aren't informative
        # instead of always taking the aggregate at full strength.
        self.residual_gate = nn.Linear(self.hidden_units, self.hidden_units)
        nn.init.constant_(self.residual_gate.bias, 2.0) # same open-start reasoning as attn_gate

        self.norm = nn.LayerNorm(self.hidden_units)
        self.output_nomr = nn.LayerNorm(self.hidden_units)

    def forward(self, x, edge_index, edge_attr):
        x_propogate = self.propagate(edge_index, x=x, edge_attr=edge_attr) #aggregation and updates, message creation is defined using message method. (update)

        #res blocks: there should be no isolated nodes (is this just for robutness confirm...)
        gate = torch.sigmoid(self.residual_gate(x))
        x = self.norm(x + gate * x_propogate)
        x = self.output_nomr(x + self.fcl_middle(x))
        return x

    def message(self, x_j, edge_attr, index, size_i):
        message = self.fcl_message(torch.cat([x_j, edge_attr], dim=-1))
        weight = softmax(self.attn_score(message), index, num_nodes=size_i) # normalize over each node's incoming edges
        gate = torch.sigmoid(self.attn_gate(message))
        return weight * gate * message


class EncoderLayer(nn.Module):
    """
    Implment node and edge updates using message passing encoder.

    args:
        in_dim: dimension of input node+edge features (i.e: we could do in_dim/2 by matching rbf kernel size to node feature size.)
        hidden_units: dimension of hidden units in message passing
        message_passing_layers: number of layers in the message passing MLP
        mp_rounds: number of message passing rounds (independent weights each). 1 reproduces
            the original single-propagate behaviour; higher values widen the receptive field.
    """
    def __init__(self, node_in_dim: int, edge_features_dim: int,hidden_units: int, message_passing_layers=3, dropout=0.0, mp_rounds=1):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.hidden_units = hidden_units
        self.edge_features_dim = edge_features_dim
        self.mp_rounds = mp_rounds

        fcl_edge_message =[]
        cur_in_dim = edge_features_dim
        for _ in range(message_passing_layers):
            fcl_edge_message.append(nn.Linear(cur_in_dim, self.hidden_units))
            fcl_edge_message.append(nn.SiLU())
            fcl_edge_message.append(nn.Dropout(dropout))
            cur_in_dim = self.hidden_units
        self.fcl_edge = nn.Sequential(*fcl_edge_message)

        self.input_proj = nn.Linear(self.node_in_dim, self.hidden_units)

        self.rounds = nn.ModuleList([
            MessagePassingRound(hidden_units, message_passing_layers, dropout)
            for _ in range(mp_rounds)
        ])

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
        edge_attr = self.fcl_edge(edge_attr) # update edge features first, (E,D) -> (E,hidden_units)
        x = self.input_proj(x)
        for mp_round in self.rounds:
            x = mp_round(x, edge_index, edge_attr)
        return x, edge_attr



class DecoderLayer(nn.Module):
    """
    Message passing with an mlp regression head for decoding fitness.

    args:
        in_dim: dimension of input node features
        hidden_units: dimension of hidden units in message passing
        reg_hidden_units: dimension of hidden units in regression MLP
        message_passing_layers: number of layers in the message passing MLP
        head_layers: number of layers in the regression MLP head
        mp_rounds: number of message passing rounds (independent weights each)
    """
    def __init__(self,  hidden_units: int, reg_hidden_units: int, message_passing_layers=3, head_layers=2, dropout=0.0, mp_rounds=1):
        super().__init__()

        self.hidden_units = hidden_units
        self.reg_hidden_units = reg_hidden_units
        self.message_passing_layers = message_passing_layers
        self.head_layers = head_layers
        self.mp_rounds = mp_rounds

        self.rounds = nn.ModuleList([
            MessagePassingRound(hidden_units, message_passing_layers, dropout)
            for _ in range(mp_rounds)
        ])

        output =[]
        cur_in_dim = self.hidden_units
        for _ in range(self.head_layers):
            output.append(nn.Linear(cur_in_dim, self.reg_hidden_units)) # 164 - > 64, 32
            output.append(nn.SiLU())
            output.append(nn.Dropout(dropout))
            cur_in_dim = self.reg_hidden_units
        output.append(nn.Linear(cur_in_dim, 1))
        self.output = nn.Sequential(*output)

        # pooling
        self.gate = nn.Linear(hidden_units, hidden_units) #feature
        self.score = nn.Linear(hidden_units, 1)


    def forward(self, x, edge_index, edge_attr, batch, mutation_idx):
        for mp_round in self.rounds:
            x = mp_round(x, edge_index, edge_attr)

        gate = torch.sigmoid(self.gate(x))

        # attention pool over the mutated residue(s) only: non-mutated nodes get -inf
        scores = self.score(x)
        scores = scores.masked_fill(~mutation_idx.unsqueeze(-1), float('-inf'))
        weights = softmax(scores, batch)
        x = weights*gate*x
        x = global_add_pool(x, batch) # (B, hidden_units)

        x = self.output(x)
        return x


class MaskedResidueDecoderLayer(nn.Module):
    """
    Message passing with a per-node classification head for masked-residue self-supervised
    pretraining. Unlike DecoderLayer, there is no pooling since every node gets 20
    amino acid logits, and the loss is computed only at masked positions via
    nn.CrossEntropyLoss(ignore_index=-100) against the label tensor. All else is the same.

    args:
        hidden_units: dimension of hidden units in message passing
        head_hidden_units: dimension of hidden units in the classification MLP
        message_passing_layers: number of layers in the message passing MLP
        head_layers: number of layers in the classification MLP head
    """
    def __init__(self, hidden_units: int, head_hidden_units: int, message_passing_layers=3, head_layers=2, dropout=0.0, mp_rounds=1):
        super().__init__()

        self.hidden_units = hidden_units
        self.head_hidden_units = head_hidden_units
        self.message_passing_layers = message_passing_layers
        self.head_layers = head_layers
        self.mp_rounds = mp_rounds

        self.rounds = nn.ModuleList([
            MessagePassingRound(hidden_units, message_passing_layers, dropout)
            for _ in range(mp_rounds)
        ])

        output = []
        cur_in_dim = self.hidden_units
        for _ in range(self.head_layers):
            output.append(nn.Linear(cur_in_dim, self.head_hidden_units)) # 64 -> 32
            output.append(nn.SiLU())
            output.append(nn.Dropout(dropout))
            cur_in_dim = self.head_hidden_units
        output.append(nn.Linear(cur_in_dim, 20)) # 20 canonical amino
        self.output = nn.Sequential(*output)

    def forward(self, x, edge_index, edge_attr):
        for mp_round in self.rounds:
            x = mp_round(x, edge_index, edge_attr)

        logits = self.output(x) # [N, 20], logits not normalized
        return logits
