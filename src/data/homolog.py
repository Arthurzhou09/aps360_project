from sys import platform
import torch
from data.data_class import DataClass, HomologGraphData
from data.feature_utils import *
from data.data_utils import *
import pandas as pd
import numpy as np
import platform

if platform.system() != "Windows":
    PROCESSED_DATA_DIR = "/Users/arthurzhou/github/aps360_project/src/data/processed"
else:
    PROCESSED_DATA_DIR = 'C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed'


class HomologMaskedDataset(DataClass):
    """
    Self-supervised masked-residue dataset for structure-conditioned pretraining. Homolog sequences (same protein family as the WT structure) are threaded onto the WT structure graph, one aligned residue is masked per sample, and the label is the true AA identity at that position. No DMS fitness labels are used here - this is pretraining only, to be scored zero-shot against fitness afterward.

    args:
        homolog_data: DataFrame with columns ['homolog_ID', 'percent_identity', 'sequence'] (from homolog_process.py / split_by_cluster).
        pdb_id: PDB ID of the WT protein structure to load.
        directed: Whether to create directed edges in the graph.
        max_neighbours: Maximum number of nearest neighbors to consider for edge construction.
        seed: seed for the per-sample mask position RNG.
    returns:
        HomologGraphData
    """

    def __init__(self, homolog_data: pd.DataFrame, pdb_id: str, directed=True, max_neighbours=16, seed=1012):
        self.wt_sequence, self.atomic_pos = parse_structure(load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))
        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")
        self.rng = np.random.default_rng(seed)

        # align every homolog onto the WT structure once, same as Tem1BetaLactamaseDataset
        # does for the DMS experimental sequences.
        self.homolog_data = homolog_data.reset_index(drop=True)
        self.encoded_sequences = []
        self.valid_positions = []
        for sequence in self.homolog_data['sequence']:
            mapping, _ = align_sequence(sequence, self.wt_sequence)
            encoded = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)
            for homolog_idx, wt_idx in mapping.items():
                encoded[wt_idx] = RESIDUE_LETTERS.index(sequence[homolog_idx])
            self.encoded_sequences.append(encoded)
            self.valid_positions.append(np.flatnonzero(encoded != -1))

        # static features shared across all homologs: same WT structure graph.
        self.distance_features = build_distance_features(self.atomic_pos, k=max_neighbours, directed=directed)
        self.edge_index = build_backbone_edge_index(self.atomic_pos, k=max_neighbours, directed=directed)

    def __len__(self):
        return len(self.homolog_data)

    def __getitem__(self, idx):
        """
        Returns a single sample with one randomly masked (aligned) residue. The mask
        position is resampled on every call, so repeated epochs see different positions for the same homolog.
        """
        encoded_sequence = self.encoded_sequences[idx]
        mask_pos = self.rng.choice(self.valid_positions[idx])

        true_aa_idx = encoded_sequence[mask_pos]
        masked_encoded_sequence = encoded_sequence.copy()
        masked_encoded_sequence[mask_pos] = -1 # hide the residue identity, same convention as unaligned/gap positions

        mask_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mask_idx[mask_pos] = True

        # -100 is the ignore_index for nn.CrossEntropyLoss
        labels = np.full(len(self.wt_sequence), fill_value=-100, dtype=int)
        labels[mask_pos] = true_aa_idx

        # build node features: masked position gets zeroed one-hot/aaindex features (aa_idx == -1)
        node_features = build_node_features(masked_encoded_sequence, self.aa_index)
        node_features = np.concatenate([mask_idx[:, None], node_features], axis=1)

        return HomologGraphData(
            distance_features=torch.tensor(self.distance_features, dtype=torch.float),
            node_features=torch.tensor(node_features, dtype=torch.float),
            sequence=torch.tensor(self.wt_sequence_encoded, dtype=torch.long), # not used in model
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            mask_idx=torch.tensor(mask_idx, dtype=torch.bool), # position to read logits from at eval/inference time
            labels=torch.tensor(labels, dtype=torch.long),
        )
