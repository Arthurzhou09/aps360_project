from sys import platform
import torch
from data.data_class import DataClass, ProteinGraphData
from data.feature_utils import *
from data.feature_utils import _one_hot_aa # leading underscore, so not picked up by import *
from data.data_utils import *
import pandas as pd
import numpy as np
import platform

if platform.system() != "Windows":
    PROCESSED_DATA_DIR = "/Users/arthurzhou/github/aps360_project/src/data/processed"
else:
    PROCESSED_DATA_DIR = 'C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed'


class Tem1BetaLactamaseDataset(DataClass):
    """
    Dataset class for TEM-1 beta-lactamase data.

    args:
        dms_data: DataFrame containing the DMS data.
        pdb_id: PDB ID of the protein structure to load.
        directed: Whether to create directed edges in the graph.
        max_neighbours: Maximum number of nearest neighbors to consider for edge construction.
    returns:
        ProteinGraphData 
    """

    def __init__(self, dms_data: pd.DataFrame, pdb_id:str, directed = True, max_neighbours=16):
        self.wt_sequence, self.atomic_pos = parse_structure(load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))
        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")
        self.aa_to_value, _ = encode_aaindex_features(self.aa_index) # AA letter -> standardized property vector, for the mutation delta feature

        # align the experimental sequence with pdb wt. (single and pair)
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
            0: align_sequence(pair_sequence, self.wt_sequence)[0] if pair_sequence is not None else {} ,
        }

        if single_sequence is not None:
            # encode the sequence with respect to PDB WT.
            single_wt_experimental_encoded_sequence = np.full(len(self.wt_sequence),fill_value=-1, dtype=int)
            for dms_idx, wt_idx in self.alignment_mappings[1].items():
                aa = single_sequence[dms_idx]
                single_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        else:
            single_wt_experimental_encoded_sequence = None

        if pair_sequence is not None:
            pair_wt_experimental_encoded_sequence = np.full(len(self.wt_sequence),fill_value=-1, dtype=int)
            for dms_idx, wt_idx in self.alignment_mappings[0].items():
                aa = pair_sequence[dms_idx]
                pair_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        else:
            pair_wt_experimental_encoded_sequence = None

        self.wt_experimental_encoded_sequences = {
            1: single_wt_experimental_encoded_sequence, 
            0: pair_wt_experimental_encoded_sequence,
        }

        # shorten the dms set to have only valid resides
        s_mask = (dms_data['Single'] == 1) & (dms_data['Ambler Index'].isin(self.alignment_mappings[1].keys()))
        p_mask = (dms_data['Single'] == 0) & (dms_data['Ambler Index'].isin(self.alignment_mappings[0].keys()))
        self.dms = dms_data.loc[s_mask | p_mask].reset_index(drop=True)

        # static features and labels
        self.labels=self.dms['Fitness'].to_numpy(copy=True)
        self.distance_features = build_distance_features(self.atomic_pos, k=max_neighbours, directed=directed)
        self.edge_index = build_backbone_edge_index(self.atomic_pos, k=max_neighbours, directed=directed)

        # the graph is the same WT structure for every sample, so build these tensors once
        # instead of re-materializing (E, 128) floats per __getitem__ call
        self.distance_features_tensor = torch.tensor(self.distance_features, dtype=torch.float)
        self.edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long)
        self.wt_sequence_encoded_tensor = torch.tensor(self.wt_sequence_encoded, dtype=torch.long)

        # the WT background is identical across samples too: only the mutated site(s) change,
        # so build the WT node-feature block once and overwrite the mutated rows per sample.
        self.wt_node_features = {
            single: build_node_features(encoded, self.aa_index) if encoded is not None else None
            for single, encoded in self.wt_experimental_encoded_sequences.items()
        }

    def __len__(self):
        return len(self.dms)

    def __getitem__(self, idx):
        """ 
        Returns a single sample.
        """
        sample = self.dms.iloc[idx]
        fitness_label = self.labels[idx]
        alignment_mapping = self.alignment_mappings[sample['Single']]
        node_idx = alignment_mapping[sample['Ambler Index']]

        # get one hot mutation index
        mutation_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mutation_idx[node_idx] = True

        # encode the sequence with respect to PDB WT.
        wt_experimental_encoded_sequence = self.wt_experimental_encoded_sequences[sample['Single']]

        # build the node sequence with the mutation in
        code = sample['Code'].split("_")
        node_idx_2 = None
        if code[1].isnumeric(): # single mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mut_aa_idx = RESIDUE_LETTERS.index(code[2])
            mutation_encoded_sequence[node_idx] = mut_aa_idx
        else: # pair mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mut_aa_idx = RESIDUE_LETTERS.index(code[3])
            mutation_encoded_sequence[node_idx] = mut_aa_idx
            if (sample['Ambler Index'] + 1) in alignment_mapping:
                node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
                mut_aa_idx_2 = RESIDUE_LETTERS.index(code[4])
                mutation_encoded_sequence[node_idx_2] = mut_aa_idx_2
                mutation_idx[node_idx_2] = True

        #build node features: mutant identity everywhere it changed, WT identity elsewhere.
        # Only the mutated row(s) differ from the cached WT background, and build_node_features
        # re-standardizes the whole AAindex table on every call, so patch the rows directly.
        aaindex_node_features = self.wt_node_features[sample['Single']].copy() #263, 28
        aaindex_node_features[node_idx] = np.concatenate([_one_hot_aa(mut_aa_idx), self.aa_to_value[RESIDUE_LETTERS[mut_aa_idx]]])
        if node_idx_2 is not None:
            aaindex_node_features[node_idx_2] = np.concatenate([_one_hot_aa(mut_aa_idx_2), self.aa_to_value[RESIDUE_LETTERS[mut_aa_idx_2]]])

        # AAindex property delta (mutant - WT) at the mutated site(s) only, zero elsewhere: a
        # compact signal for what changed (fitness tracks property CHANGES, not absolute mutant
        # identity) - the same delta_props feature build_mutation_features gives the MLP baseline.
        n_props = len(next(iter(self.aa_to_value.values())))
        delta_features = np.zeros((len(self.wt_sequence), n_props), dtype=float)
        delta_features[node_idx] = self.aa_to_value[RESIDUE_LETTERS[mut_aa_idx]] - self.aa_to_value[RESIDUE_LETTERS[wt_experimental_encoded_sequence[node_idx]]]
        if node_idx_2 is not None:
            delta_features[node_idx_2] = self.aa_to_value[RESIDUE_LETTERS[mut_aa_idx_2]] - self.aa_to_value[RESIDUE_LETTERS[wt_experimental_encoded_sequence[node_idx_2]]]

        # WT identity one-hot at the mutated site(s), zero elsewhere. Without this the model
        # only ever sees which residue is there NOW: the mutated node carries the mutant
        # one-hot, and the WT residue survives only as an 8-dim property delta. Which residue
        # was replaced is one of the strongest single predictors of fitness (losing a catalytic
        # Ser, a structural Gly or Pro is not interchangeable with losing a surface Ala), and
        # it is exactly what build_mutation_features already hands the MLP baseline.
        wt_onehot_features = np.zeros((len(self.wt_sequence), len(RESIDUE_LETTERS)), dtype=float)
        wt_onehot_features[node_idx] = _one_hot_aa(wt_experimental_encoded_sequence[node_idx])
        if node_idx_2 is not None:
            wt_onehot_features[node_idx_2] = _one_hot_aa(wt_experimental_encoded_sequence[node_idx_2])

        # first 37 columns (mask + one-hot + AAindex + delta) match HomologMaskedDataset's
        # layout, so the two models' encoders see the same features in the same order.
        aaindex_node_features = np.concatenate(
            [mutation_idx[:,None], aaindex_node_features, delta_features, wt_onehot_features], axis=1)

        protein_graph = ProteinGraphData(
            distance_features=self.distance_features_tensor,
            node_features=torch.tensor(aaindex_node_features, dtype=torch.float),
            sequence=self.wt_sequence_encoded_tensor, # not used  in model
            edge_index=self.edge_index_tensor,
            mutation_idx=torch.tensor(mutation_idx, dtype=torch.bool), # used by decoder to pool mutated residues
            fitness =torch.tensor(fitness_label, dtype=torch.float),
        )

        return protein_graph


class MLPDataset(DataClass):
    """
    Dataset class for MLP model.
    args:
        dms_data: DataFrame containing the DMS data.
        pdb_id: ID of the PDB structure.
    returns:
        aaindex_features: flattened features for each residue in the sequence.
        fitness_label: Fitness value for the mutation(s).
    """

    def __init__(self, dms_data: pd.DataFrame, pdb_id:str):
        self.wt_sequence, self.atomic_pos = parse_structure(load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))
        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")

        # align the experimental sequence with pdb wt. (single and pair)
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

         # encode the sequence with respect to PDB WT.
        if single_sequence is not None:
            single_wt_experimental_encoded_sequence = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)
            for dms_idx, wt_idx in self.alignment_mappings[1].items():
                aa = single_sequence[dms_idx]
                single_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        else:
            single_wt_experimental_encoded_sequence = None

        if pair_sequence is not None:
            pair_wt_experimental_encoded_sequence = np.full(len(self.wt_sequence), fill_value=-1, dtype=int)
            for dms_idx, wt_idx in self.alignment_mappings[0].items():
                aa = pair_sequence[dms_idx]
                pair_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        else:
            pair_wt_experimental_encoded_sequence = None

        self.wt_experimental_encoded_sequences = {
            1: single_wt_experimental_encoded_sequence, 
            0: pair_wt_experimental_encoded_sequence,
        }

        # shorten the dms set to have only valid resides
        s_mask = (dms_data['Single'] == 1) & (dms_data['Ambler Index'].isin(self.alignment_mappings[1].keys()))
        p_mask = (dms_data['Single'] == 0) & (dms_data['Ambler Index'].isin(self.alignment_mappings[0].keys()))
        self.dms = dms_data.loc[s_mask | p_mask].reset_index(drop=True)

        # static features and labels
        self.labels=self.dms['Fitness'].to_numpy(copy=True)
    
    def __len__(self):
        return len(self.dms)
    

    def __getitem__(self, idx):
        """
        Returns a single sample. Features are built from the mutated site(s)
        (WT properties, mutant properties, and their delta)
        """
        sample = self.dms.iloc[idx]
        alignment_mapping = self.alignment_mappings[sample['Single']]
        node_idx = alignment_mapping[sample['Ambler Index']]

        # encode the sequence with respect to PDB WT.
        wt_experimental_encoded_sequence = self.wt_experimental_encoded_sequences[sample['Single']]

        code = sample['Code'].split("_")
        is_pair = not code[1].isnumeric()

        wt_aa_1 = wt_experimental_encoded_sequence[node_idx]
        mut_aa_1 = RESIDUE_LETTERS.index(code[3] if is_pair else code[2])
        site_1_features = build_mutation_features(wt_aa_1, mut_aa_1, self.aa_index)

        # site 2 only exists for pair mutations with a valid adjacent alignment
        site_2_features = np.zeros_like(site_1_features)
        has_site_2 = False
        if is_pair and (sample['Ambler Index'] + 1) in alignment_mapping:
            node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
            wt_aa_2 = wt_experimental_encoded_sequence[node_idx_2]
            mut_aa_2 = RESIDUE_LETTERS.index(code[4])
            site_2_features = build_mutation_features(wt_aa_2, mut_aa_2, self.aa_index)
            has_site_2 = True

        aaindex_features = np.concatenate([site_1_features, site_2_features, [float(has_site_2)]])
        aaindex_features = torch.tensor(aaindex_features, dtype=torch.float)
        fitness_label = torch.tensor(self.labels[idx], dtype=torch.float)

        return aaindex_features, fitness_label
