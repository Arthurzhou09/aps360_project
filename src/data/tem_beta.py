from sys import platform

from data.data_class import DataClass, ProteinGraphData
from data.feature_utils import *
from data.data_utils import *
import pandas as pd
import numpy as np
import torch.nn as nn 
import torch
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

        # align the experimental sequence with pdb wt. (single and pair)
        single_sequence = dms_data.loc[dms_data['Single'] == 1]['Experiment Sequence'].iloc[0]
        pair_sequence = dms_data.loc[dms_data['Single'] == 0]['Experiment Sequence'].iloc[0]

        self.alignment_mappings = {
            1: align_sequence(single_sequence, self.wt_sequence),
            0: align_sequence(pair_sequence, self.wt_sequence),
        }

         # encode the sequence with respect to PDB WT.
        single_wt_experimental_encoded_sequence = np.zeros(len(self.wt_sequence), dtype=int)
        for dms_idx, wt_idx in self.alignment_mappings[1].items():
            aa = single_sequence[dms_idx]
            single_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        
        pair_wt_experimental_encoded_sequence = np.zeros(len(self.wt_sequence), dtype=int)
        for dms_idx, wt_idx in self.alignment_mappings[0].items():
            aa = pair_sequence[dms_idx]
            pair_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        
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
        if code[1].isnumeric(): # single mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mutation_encoded_sequence[node_idx] = RESIDUE_LETTERS.index(code[2])
        else: # pair mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mutation_encoded_sequence[node_idx] = RESIDUE_LETTERS.index(code[3])
            if (sample['Ambler Index'] + 1) in alignment_mapping:
                node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
                mutation_encoded_sequence[node_idx_2] = RESIDUE_LETTERS.index(code[4])

        #build node features
        aaindex_node_features = build_node_features(mutation_encoded_sequence, self.aa_index)
    
        protein_graph = ProteinGraphData(
            distance_features=torch.tensor(self.distance_features, dtype=torch.float),
            node_features=torch.tensor(aaindex_node_features, dtype=torch.float),
            sequence=torch.tensor(self.wt_sequence_encoded, dtype=torch.long), # not used  in model
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            mutation_idx=torch.tensor(mutation_idx, dtype=torch.bool), # not used rn in model
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
        single_sequence = dms_data.loc[dms_data['Single'] == 1]['Experiment Sequence'].iloc[0]
        pair_sequence = dms_data.loc[dms_data['Single'] == 0]['Experiment Sequence'].iloc[0]

        self.alignment_mappings = {
            1: align_sequence(single_sequence, self.wt_sequence),
            0: align_sequence(pair_sequence, self.wt_sequence),
        }

         # encode the sequence with respect to PDB WT.
        single_wt_experimental_encoded_sequence = np.zeros(len(self.wt_sequence), dtype=int)
        for dms_idx, wt_idx in self.alignment_mappings[1].items():
            aa = single_sequence[dms_idx]
            single_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        
        pair_wt_experimental_encoded_sequence = np.zeros(len(self.wt_sequence), dtype=int)
        for dms_idx, wt_idx in self.alignment_mappings[0].items():
            aa = pair_sequence[dms_idx]
            pair_wt_experimental_encoded_sequence[wt_idx] = RESIDUE_LETTERS.index(aa)
        
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
        Returns a single sample.
        """
        sample = self.dms.iloc[idx]
        alignment_mapping = self.alignment_mappings[sample['Single']]
        node_idx = alignment_mapping[sample['Ambler Index']]

        # get one hot mutation index
        mutation_idx = np.zeros(len(self.wt_sequence), dtype=bool)
        mutation_idx[node_idx] = True

        # encode the sequence with respect to PDB WT.
        wt_experimental_encoded_sequence = self.wt_experimental_encoded_sequences[sample['Single']]

        # build the node sequence with the mutation in
        code = sample['Code'].split("_")
        if code[1].isnumeric(): # single mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mutation_encoded_sequence[node_idx] = RESIDUE_LETTERS.index(code[2])
        else: # pair mutation
            mutation_encoded_sequence = wt_experimental_encoded_sequence.copy()
            mutation_encoded_sequence[node_idx] = RESIDUE_LETTERS.index(code[3])
            if (sample['Ambler Index'] + 1) in alignment_mapping:
                node_idx_2 = alignment_mapping[sample['Ambler Index'] + 1]
                mutation_encoded_sequence[node_idx_2] = RESIDUE_LETTERS.index(code[4])

        #build node features
        aaindex_features = build_node_features(mutation_encoded_sequence, self.aa_index)

        aaindex_features = torch.tensor(aaindex_features.flatten(), dtype=torch.float)
        fitness_label = torch.tensor(self.labels[idx], dtype=torch.float)

        return aaindex_features, fitness_label
