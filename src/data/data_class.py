from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np

RESIDUE_LETTERS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

LETTER_TO_TOKEN = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

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
    

class ProteinGraphData(Data):
    def __init__(
        self,
        distance_features=None,
        node_features=None,
        sequence=None,
        edge_index=None,
        mutation_idx=None,
        fitness=None,
        is_single=None,
    ):
        """
        A class representing single or pairwise mutation data.

        Args:
            distance_features: distances between residues (pairwise distances between N, Ca, C, O),
            node_features: additional numeric features for each residue, expected shape [N, F], where F is the number of features.
            sequence: Amino acid sequence of the protein, expected shape [N], where N is the number of residues.
            edge_index: Edge indices representing the connectivity of residues, expected shape [2, E]
            mutation_index: Mask indicating mutation positions in the sequence, expected shape [N,]
            fitness: Fitness value for the mutation(s).
            is_single: 1 for a single mutant, 0 for a double. Carried per graph so training
                and inference can report metrics WITHIN each group. A combined Spearman is
                not a clean measure of residue-level understanding: double mutants are
                systematically more damaging (mean standardized fitness -0.26 vs +0.34 on
                the test split), so simply counting the mutations scores ~0.32 on its own.
        """
        super(ProteinGraphData, self).__init__()
        if distance_features is not None:
            self.distance_features = distance_features
        if node_features is not None:
            self.node_features = node_features
        if edge_index is not None:
            self.edge_index = edge_index
        if sequence is not None:
            self.sequence = sequence
        if mutation_idx is not None:
            self.mutation_idx = mutation_idx
        if fitness is not None:
            self.fitness = fitness
        if is_single is not None:
            self.is_single = is_single


class HomologGraphData(Data):
    def __init__(
        self,
        distance_features=None,
        node_features=None,
        sequence=None,
        edge_index=None,
        mask_idx=None,
        labels=None,
        mut_aa_idx=None,
        fitness=None,
    ):
        """
        A class representing a masked-residue sample: a sequence threaded onto the WT
        structure graph with one or more residues masked out. Used both for
        self-supervised pretraining (a random homolog position is masked, labels holds
        its true identity) and for zero-shot DMS scoring (the known mutated position(s)
        are masked instead, labels holds the WT identity there, and mut_aa_idx/fitness
        carry what's needed to score the mutation: log P(mutant) - log P(WT) vs measured
        fitness).

        Args:
            distance_features: distances between residues (pairwise distances between N, Ca, C, O),
            node_features: additional numeric features for each residue, expected shape [N, F], where F is the number of features. Masked position(s) get zeroed identity/property features.
            sequence: WT amino acid sequence of the protein (reference only, not used by the model), expected shape [N].
            edge_index: Edge indices representing the connectivity of residues, expected shape [2, E]
            mask_idx: Mask indicating the masked position(s) in the sequence, expected shape [N,]
            labels: True (WT, for DMS scoring) amino acid index (0-19) at masked position(s), -100 (CrossEntropyLoss ignore_index) elsewhere, expected shape [N,]
            mut_aa_idx: DMS scoring only - mutant amino acid index (0-19) at masked position(s), -1 elsewhere, expected shape [N,]
            fitness: DMS scoring only - measured fitness for this mutation, to correlate against the zero-shot score
        """
        super(HomologGraphData, self).__init__()
        if distance_features is not None:
            self.distance_features = distance_features
        if node_features is not None:
            self.node_features = node_features
        if edge_index is not None:
            self.edge_index = edge_index
        if sequence is not None:
            self.sequence = sequence
        if mask_idx is not None:
            self.mask_idx = mask_idx
        if labels is not None:
            self.labels = labels
        if mut_aa_idx is not None:
            self.mut_aa_idx = mut_aa_idx
        if fitness is not None:
            self.fitness = fitness