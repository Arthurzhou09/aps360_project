import platform

import numpy as np
import pandas as pd
import torch

from data.data_class import DataClass, HomologGraphData, RESIDUE_LETTERS
from data.feature_utils import (align_sequence, build_distance_features, build_backbone_edge_index,
                                build_node_features, encode_aaindex_features)
from data.data_utils import parse_structure, load_cif_structure

if platform.system() != "Windows":
    PROCESSED_DATA_DIR = "/Users/arthurzhou/github/aps360_project/src/data/processed"
else:
    PROCESSED_DATA_DIR = 'C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed'


class MSAMaskedDataset(DataClass):
    """
    Masked-residue dataset driven by an MSA's own columns, for structure-conditioned
    SSL pretraining. 

    Only the structure nodes that carry a match column are modelled (215 ish of 263 for
    BLAT_ECOLX/1BTL). The rest are helded as non residue encoding (feature).

    node features (38): mask(1) + AA one-hot(20) + AAindex props(8)
                        + AAindex delta vs the structure residue(8) + covered(1)
    The covered channel is new relative to HomologMaskedDataset's 37.

    args:
        msa_data: DataFrame with ['seq_id', 'match_states'] from msa_process.py. Required when test=False.
        dms_data: DataFrame in dms_processed.csv format. Required when test=True.
        column_map: DataFrame with ['match_column', 'structure_node'] from msa_process.py.
        pdb_id: PDB ID of the WT protein structure to load.
        test: zero-shot DMS-scoring mode instead of pretraining mode.
        directed: whether to create directed edges in the graph.
        max_neighbours: number of nearest neighbours for edge construction.
        seed: base seed for the per-sample mask RNG (pretraining mode only).
        mask_ratio: fraction of a sequence's occupied columns to mask per sample.
    returns:
        HomologGraphData
    """

    def __init__(self, msa_data: pd.DataFrame = None, dms_data: pd.DataFrame = None,
                 column_map: pd.DataFrame = None, pdb_id: str = "1BTL", test: bool = False,
                 directed=True, max_neighbours=16, seed=1012, mask_ratio: float = 0.15,
                 radius: float = None):
        self.wt_sequence, self.atomic_pos = parse_structure(
            load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))
        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")
        aa_to_value, _ = encode_aaindex_features(self.aa_index)
        self.aa_property_matrix = np.stack([aa_to_value[aa] for aa in RESIDUE_LETTERS])
        self.test = test
        self.seed = seed
        self.mask_ratio = mask_ratio
        self._epoch = 0

        if column_map is None:
            column_map = pd.read_csv(f"{PROCESSED_DATA_DIR}/msa/msa_columns.csv")
        # match column i sits on structure node column_nodes[i]
        self.column_nodes = column_map["structure_node"].to_numpy()
        self.covered_nodes = np.zeros(len(self.wt_sequence), dtype=bool)
        self.covered_nodes[self.column_nodes] = True

        # static features: one fixed WT structure graph, shared by every sample
        self.distance_features = build_distance_features(self.atomic_pos, k=max_neighbours, directed=directed, radius=radius)
        self.edge_index = build_backbone_edge_index(self.atomic_pos, k=max_neighbours, directed=directed, radius=radius)
        self.distance_features_tensor = torch.tensor(self.distance_features, dtype=torch.float)
        self.edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long)
        self.wt_sequence_encoded_tensor = torch.tensor(self.wt_sequence_encoded, dtype=torch.long)

        if test:
            self._init_test(dms_data)
        else:
            self.msa_data = msa_data.reset_index(drop=True)
            # encode every sequence onto the structure once: -1 where gapped or uncovered
            self.encoded_sequences = []
            self.valid_positions = []
            for states in self.msa_data["match_states"]:
                encoded = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)
                for column, aa in enumerate(states):
                    if aa in RESIDUE_LETTERS:
                        encoded[self.column_nodes[column]] = RESIDUE_LETTERS.index(aa)
                self.encoded_sequences.append(encoded)
                self.valid_positions.append(np.flatnonzero(encoded != -1))

    def _init_test(self, dms_data):
        """Zero-shot mode: identical setup to HomologMaskedDataset's test branch."""
        single_sequence = dms_data.loc[dms_data['Single'] == 1]['Experiment Sequence'].iloc[0] if (dms_data['Single'] == 1).any() else None
        pair_sequence = dms_data.loc[dms_data['Single'] == 0]['Experiment Sequence'].iloc[0] if (dms_data['Single'] == 0).any() else None

        self.alignment_mappings = {
            1: align_sequence(single_sequence, self.wt_sequence)[0] if single_sequence is not None else {},
            0: align_sequence(pair_sequence, self.wt_sequence)[0] if pair_sequence is not None else {},
        }
        self.wt_experimental_encoded_sequences = {}
        for is_single, sequence in ((1, single_sequence), (0, pair_sequence)):
            if sequence is None:
                self.wt_experimental_encoded_sequences[is_single] = None
                continue
            encoded = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)
            for dms_idx, wt_idx in self.alignment_mappings[is_single].items():
                encoded[wt_idx] = RESIDUE_LETTERS.index(sequence[dms_idx])
            self.wt_experimental_encoded_sequences[is_single] = encoded

        s_mask = (dms_data['Single'] == 1) & (dms_data['Ambler Index'].isin(self.alignment_mappings[1].keys()))
        p_mask = (dms_data['Single'] == 0) & (dms_data['Ambler Index'].isin(self.alignment_mappings[0].keys()))
        kept = dms_data.loc[s_mask | p_mask].reset_index(drop=True)

        # a mutation at a structure node with no MSA column cannot be scored: the model was
        # never trained to predict there. Dropping them here keeps the reported n honest.
        node_of = lambda r: self.alignment_mappings[r['Single']][r['Ambler Index']]
        in_msa = kept.apply(lambda r: self.covered_nodes[node_of(r)], axis=1)
        if (~in_msa).any():
            print(f"MSAMaskedDataset: dropped {int((~in_msa).sum())}/{len(kept)} DMS rows "
                  f"whose mutated site has no MSA column")
        self.dms_data = kept.loc[in_msa].reset_index(drop=True)

    def __len__(self):
        return len(self.dms_data) if self.test else len(self.msa_data)

    def set_epoch(self, epoch: int):
        """
        Advance the masking stream so each epoch masks different columns. Same reason as
        HomologMaskedDataset.set_epoch: a self.rng attribute resets whenever DataLoader
        workers are respawned, collapsing every epoch onto identical masks.

        Do not set persistent_workers=True on a loader over this dataset - persistent
        workers keep a stale _epoch and reintroduce that bug.
        """
        self._epoch = epoch

    def _mask_rng(self, idx):
        return np.random.default_rng((self.seed * 1_000_003 + self._epoch * 9_176 + idx) % (2**63))

    def _delta_features(self, visible_encoded_sequence):
        """AAindex delta of the shown residue against the structure's own residue, zero where
        masked, gapped or uncovered. Same definition Tem1BetaLactamaseDataset uses."""
        delta = np.zeros((len(self.wt_sequence), self.aa_property_matrix.shape[1]), dtype=float)
        visible = visible_encoded_sequence != -1
        delta[visible] = (self.aa_property_matrix[visible_encoded_sequence[visible]]
                          - self.aa_property_matrix[self.wt_sequence_encoded[visible]])
        return delta

    def _assemble(self, encoded_sequence, mask_idx):
        """Build the (N, 38) node feature block shared by both modes."""
        node_features = build_node_features(encoded_sequence, self.aa_index)
        covered = np.zeros(len(self.wt_sequence), dtype=float)
        # "covered" means the alignment has a column here AND this sequence occupies it, or
        # the position is masked (so the model knows a prediction is wanted there)
        covered[(encoded_sequence != -1) | mask_idx] = 1.0
        return np.concatenate(
            [mask_idx[:, None], node_features, self._delta_features(encoded_sequence), covered[:, None]], axis=1)

    def __getitem__(self, idx):
        return self._get_test_item(idx) if self.test else self._get_pretrain_item(idx)

    def _get_pretrain_item(self, idx):
        encoded_sequence = self.encoded_sequences[idx]
        valid = self.valid_positions[idx]
        rng = self._mask_rng(idx)

        n_mask = max(1, int(round(self.mask_ratio * len(valid))))
        mask_pos = rng.choice(valid, size=min(n_mask, len(valid)), replace=False)

        true_aa_idx = encoded_sequence[mask_pos]
        masked = encoded_sequence.copy()
        masked[mask_pos] = -1

        mask_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mask_idx[mask_pos] = True

        labels = np.full(len(self.wt_sequence), fill_value=-100, dtype=int) # CrossEntropyLoss ignore_index
        labels[mask_pos] = true_aa_idx

        return HomologGraphData(
            distance_features=self.distance_features_tensor,
            node_features=torch.tensor(self._assemble(masked, mask_idx), dtype=torch.float),
            sequence=self.wt_sequence_encoded_tensor,
            edge_index=self.edge_index_tensor,
            mask_idx=torch.tensor(mask_idx, dtype=torch.bool),
            labels=torch.tensor(labels, dtype=torch.long),
        )

    def _get_test_item(self, idx):
        sample = self.dms_data.iloc[idx]
        is_single = sample['Single']
        alignment_mapping = self.alignment_mappings[is_single]
        wt_experimental = self.wt_experimental_encoded_sequences[is_single]
        node_idx = alignment_mapping[sample['Ambler Index']]

        code = sample['Code'].split("_")
        is_pair = not code[1].isnumeric()

        masked = wt_experimental.copy()
        # positions outside the alignment are not modelled, so blank them rather than
        # feeding the model context it never saw during pretraining
        masked[~self.covered_nodes] = -1

        mask_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mut_aa_idx = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)

        mask_idx[node_idx] = True
        masked[node_idx] = -1
        mut_aa_idx[node_idx] = RESIDUE_LETTERS.index(code[3] if is_pair else code[2])

        if is_pair and (sample['Ambler Index'] + 1) in alignment_mapping:
            node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
            if self.covered_nodes[node_idx_2]:
                mask_idx[node_idx_2] = True
                masked[node_idx_2] = -1
                mut_aa_idx[node_idx_2] = RESIDUE_LETTERS.index(code[4])

        labels = np.full(len(self.wt_sequence), fill_value=-100, dtype=int)
        labels[mask_idx] = wt_experimental[mask_idx]

        return HomologGraphData(
            distance_features=self.distance_features_tensor,
            node_features=torch.tensor(self._assemble(masked, mask_idx), dtype=torch.float),
            sequence=self.wt_sequence_encoded_tensor,
            edge_index=self.edge_index_tensor,
            mask_idx=torch.tensor(mask_idx, dtype=torch.bool),
            labels=torch.tensor(labels, dtype=torch.long),
            mut_aa_idx=torch.tensor(mut_aa_idx, dtype=torch.long),
            fitness=torch.tensor(sample['Fitness'], dtype=torch.float),
        )
