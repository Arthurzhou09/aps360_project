
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np

class DataClass(Dataset):
    """
    Parent class for dataset(s).
    """
    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_num_nodes(self) -> np.ndarray:
        """Returns the number of nodes for each point in the dataset"""
        raise NotImplementedError
    

class ProteinData(Data):
    def __init__(
        self,
        distance_features=None,
        node_features=None,
        sequence=None,
        res_idx=None,
        edge_index=None,
        seq_mask=None,
        mutation_idx=None,
    ):
        """
        A class representing single or pairwise mutation data.

        Args:
            distance_features: distances between residues (pairwise distances between N, Ca, C, O),
            node_features: additional numeric features for each residue, expected shape [N, F], where F is the number of features.
            sequence: Amino acid sequence of the protein, expected shape [N], where N is the number of residues.
            res_idx: Indices indicating the residue position (Ambler) in a chain, expected shape [N_c], where N_c is the number of residues.
            edge_index: Edge indices representing the connectivity of residues, expected shape [2, E]
            seq_mask: Mask indicating valid residues in the sequence, expected shape [N,]
            mutation_idx: Indice(s) of the mutated residue(s), expected shape [N]
        """
        super(ProteinData, self).__init__()
        if distance_features is not None:
            self.distance_features = distance_features
        if node_features is not None:
            self.node_features = node_features
        if edge_index is not None:
            self.edge_index = edge_index
        if sequence is not None:
            self.sequence = sequence
        if res_idx is not None:
            self.res_idx = res_idx
            self.num_nodes = len(res_idx)
        if seq_mask is not None:
            self.seq_mask = seq_mask
        if mutation_idx is not None:
            self.mutation_idx = mutation_idx