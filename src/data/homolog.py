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
    Masked-residue dataset for the structure-conditioned GNN. Has two modes:

    - test=False (default, self-supervised pretraining): homolog sequences (same
      protein family as the WT structure) are threaded onto the WT structure graph,
      one aligned residue is masked per sample, and the label is the true amino acid
      identity at that position. No DMS fitness labels are used here.
    - test=True (zero-shot DMS scoring): dms_data (dms_processed.csv, single and/or
      pair mutants) is used instead. The known mutated position(s) are masked (rather
      than a random position), labels holds the WT identity there, and mut_aa_idx/
      fitness are attached so log P(mutant) - log P(WT) can be scored against measured
      fitness without ever training on it.

    args:
        homolog_data: DataFrame with columns ['homolog_ID', 'percent_identity', 'sequence'] (from homolog_process.py / split_by_cluster). Required when test=False.
        dms_data: DataFrame in dms_processed.csv format (same as Tem1BetaLactamaseDataset). Required when test=True.
        pdb_id: PDB ID of the WT protein structure to load.
        test: whether to run in zero-shot DMS-scoring mode instead of pretraining mode.
        directed: Whether to create directed edges in the graph.
        max_neighbours: Maximum number of nearest neighbors to consider for edge construction.
        seed: seed for the per-sample mask position RNG (pretraining mode only).
    returns:
        HomologGraphData
    """

    def __init__(self, homolog_data: pd.DataFrame = None, dms_data: pd.DataFrame = None, pdb_id: str = "1BTL",
                 test: bool = False, directed=True, max_neighbours=16, seed=1012, mask_ratio: float = 0.15):
        self.wt_sequence, self.atomic_pos = parse_structure(load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))
        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")
        aa_to_value, _ = encode_aaindex_features(self.aa_index)
        self.aa_property_matrix = np.stack([aa_to_value[aa] for aa in RESIDUE_LETTERS]) # (20, n_props)
        self.test = test
        self.seed = seed
        self.mask_ratio = mask_ratio
        self._epoch = 0

        # static features shared across all samples: same WT structure graph.
        self.distance_features = build_distance_features(self.atomic_pos, k=max_neighbours, directed=directed)
        self.edge_index = build_backbone_edge_index(self.atomic_pos, k=max_neighbours, directed=directed)
        self.distance_features_tensor = torch.tensor(self.distance_features, dtype=torch.float)
        self.edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long)
        self.wt_sequence_encoded_tensor = torch.tensor(self.wt_sequence_encoded, dtype=torch.long)

        if test:
            # align the single/pair experimental sequences onto the WT structure once,
            # same approach as Tem1BetaLactamaseDataset.
            if (dms_data['Single'] == 1).any():
                single_sequence = dms_data.loc[dms_data['Single'] == 1]['Experiment Sequence'].iloc[0]
            else:
                single_sequence = None
            if (dms_data['Single'] == 0).any():
                pair_sequence = dms_data.loc[dms_data['Single'] == 0]['Experiment Sequence'].iloc[0]
            else:
                pair_sequence = None

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

            # keep only rows whose mutated position(s) actually aligned to the WT structure
            s_mask = (dms_data['Single'] == 1) & (dms_data['Ambler Index'].isin(self.alignment_mappings[1].keys()))
            p_mask = (dms_data['Single'] == 0) & (dms_data['Ambler Index'].isin(self.alignment_mappings[0].keys()))
            self.dms_data = dms_data.loc[s_mask | p_mask].reset_index(drop=True)
        else:
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

    def __len__(self):
        return len(self.dms_data) if self.test else len(self.homolog_data)

    def set_epoch(self, epoch: int):
        """
        Advance the masking RNG stream so each epoch masks different positions.

        Needed because the mask RNG cannot live on the dataset as instance state: with
        num_workers > 0 the DataLoader respawns workers from a pickled copy of the parent
        dataset every epoch, so a self.rng attribute resets to its seed each time and every
        epoch masks the identical positions (verified: 100% identical at num_workers=4).
        Seeding per (epoch, index) instead keeps runs reproducible without that collapse.

        Do NOT set persistent_workers=True on a loader over this dataset: persistent workers
        are never re-pickled, so they would keep their stale _epoch and silently reintroduce
        the identical-masks-every-epoch bug this method exists to fix.
        """
        self._epoch = epoch

    def _mask_rng(self, idx):
        return np.random.default_rng((self.seed * 1_000_003 + self._epoch * 9_176 + idx) % (2**63))

    def _delta_features(self, visible_encoded_sequence):
        """
        AAindex property delta of the sequence being shown against the structure's own
        (PDB WT) residue at each position, zero wherever they agree.

        Same feature Tem1BetaGNN gets, under the same definition - "how does the sequence
        threaded onto this structure differ from the structure's native residue here" -
        which for a mutant sequence is nonzero only at the mutated sites, and for a homolog
        is nonzero wherever it has diverged. Masked and unaligned positions (encoded -1)
        stay zero: their identity is the label, so a nonzero delta there would leak it.

        args:
            visible_encoded_sequence: encoded residues (0-19), -1 at masked/unaligned positions
        returns:
            delta: (N, n_props) array
        """
        delta = np.zeros((len(self.wt_sequence), self.aa_property_matrix.shape[1]), dtype=float)
        visible = visible_encoded_sequence != -1
        delta[visible] = (self.aa_property_matrix[visible_encoded_sequence[visible]]
                          - self.aa_property_matrix[self.wt_sequence_encoded[visible]])
        return delta

    def __getitem__(self, idx):
        if self.test:
            return self._get_test_item(idx)
        return self._get_pretrain_item(idx)

    def _get_pretrain_item(self, idx):
        """
        Returns a single self-supervised pretraining sample with a random subset of its
        aligned residues masked (mask_ratio of them, at least one). The mask positions are
        reseeded per epoch via set_epoch, so repeated epochs see different positions for
        the same homolog.

        Masking a fraction rather than a single residue is what makes this affordable:
        the model runs message passing over all ~263 nodes either way, so masking one
        position threw away >99% of each forward pass (0.38% of nodes carried a label).
        At mask_ratio=0.15 the same compute supervises ~40 positions instead of 1.
        """
        encoded_sequence = self.encoded_sequences[idx]
        valid = self.valid_positions[idx]
        rng = self._mask_rng(idx)

        n_mask = max(1, int(round(self.mask_ratio * len(valid))))
        mask_pos = rng.choice(valid, size=min(n_mask, len(valid)), replace=False)

        true_aa_idx = encoded_sequence[mask_pos]
        masked_encoded_sequence = encoded_sequence.copy()
        masked_encoded_sequence[mask_pos] = -1 # hide the residue identity, same convention as unaligned/gap positions

        mask_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mask_idx[mask_pos] = True

        # -100 is the ignore_index convention for nn.CrossEntropyLoss
        labels = np.full(len(self.wt_sequence), fill_value=-100, dtype=int)
        labels[mask_pos] = true_aa_idx

        # build node features: masked position gets zeroed one-hot/aaindex features (aa_idx == -1)
        node_features = build_node_features(masked_encoded_sequence, self.aa_index)
        node_features = np.concatenate(
            [mask_idx[:, None], node_features, self._delta_features(masked_encoded_sequence)], axis=1)

        return HomologGraphData(
            distance_features=self.distance_features_tensor,
            node_features=torch.tensor(node_features, dtype=torch.float),
            sequence=self.wt_sequence_encoded_tensor, # not used in model
            edge_index=self.edge_index_tensor,
            mask_idx=torch.tensor(mask_idx, dtype=torch.bool), # position to read logits from at eval/inference time
            labels=torch.tensor(labels, dtype=torch.long),
        )

    def _get_test_item(self, idx):
        """
        Returns a single zero-shot scoring sample: the known mutated position(s) are
        masked (single mutants: 1 position, pair mutants: 2), starting from the WT-
        aligned sequence rather than the actual mutant sequence, so the model always
        conditions on a "wild type everywhere else" background - the standard masked-
        marginal setup. labels holds the WT identity and mut_aa_idx the mutant identity
        at the masked position(s), so log P(mutant) - log P(WT) can be read off after
        the forward pass.
        """
        sample = self.dms_data.iloc[idx]
        is_single = sample['Single']
        alignment_mapping = self.alignment_mappings[is_single]
        wt_experimental_encoded_sequence = self.wt_experimental_encoded_sequences[is_single]
        node_idx = alignment_mapping[sample['Ambler Index']]

        code = sample['Code'].split("_")
        is_pair = not code[1].isnumeric()

        masked_encoded_sequence = wt_experimental_encoded_sequence.copy()
        mask_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mut_aa_idx = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)

        mask_idx[node_idx] = True
        masked_encoded_sequence[node_idx] = -1
        mut_aa_idx[node_idx] = RESIDUE_LETTERS.index(code[3] if is_pair else code[2])

        if is_pair and (sample['Ambler Index'] + 1) in alignment_mapping:
            node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
            mask_idx[node_idx_2] = True
            masked_encoded_sequence[node_idx_2] = -1
            mut_aa_idx[node_idx_2] = RESIDUE_LETTERS.index(code[4])

        labels = np.full(len(self.wt_sequence), fill_value=-100, dtype=int)
        labels[mask_idx] = wt_experimental_encoded_sequence[mask_idx]

        node_features = build_node_features(masked_encoded_sequence, self.aa_index)
        node_features = np.concatenate(
            [mask_idx[:, None], node_features, self._delta_features(masked_encoded_sequence)], axis=1)

        return HomologGraphData(
            distance_features=self.distance_features_tensor,
            node_features=torch.tensor(node_features, dtype=torch.float),
            sequence=self.wt_sequence_encoded_tensor, # not used in model
            edge_index=self.edge_index_tensor,
            mask_idx=torch.tensor(mask_idx, dtype=torch.bool),
            labels=torch.tensor(labels, dtype=torch.long),
            mut_aa_idx=torch.tensor(mut_aa_idx, dtype=torch.long),
            fitness=torch.tensor(sample['Fitness'], dtype=torch.float),
        )
