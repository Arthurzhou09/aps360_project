from src.data.data_class import DataClass, ProteinGraphData
from src.data.feature_utils import *
from src.data.data_utils import *
import pandas as pd
import numpy as np
import torch.nn as nn 
import torch

PROCESSED_DATA_DIR = "/Users/arthurzhou/github/aps360_project/src/data/processed"

class Tem1BetaLactamaseDataset(DataClass):
    """
    Dataset class for TEM-1 beta-lactamase data.
    """
    def __init__(self, dms_data: pd.DataFrame, pdb_id:str, directed = True, max_neighbours=None):

        # you should have the saved double single splits before here.
        self.dms= dms_data
        self.aa_index = pd.read_csv(f"{PROCESSED_DATA_DIR}/aa_index_data.csv")
        self.wt_sequence, self.atomic_pos = parse_structure(load_cif_structure(f"{PROCESSED_DATA_DIR}/{pdb_id}.cif", pdb_id))

        self.wt_sequence_encoded = np.array([RESIDUE_LETTERS.index(i) for i in self.wt_sequence])
        #compute reference variables 
        
        # for reference
        self.aaindex_id = self.aa_index['id'].to_numpy(copy=True)
        self.aaindex_aa_arr = self.aa_index.columns[1:].to_numpy(copy=True)

        # label array
        self.labels=self.dms['Fitness'].to_numpy(copy=True)

        # static features and labels
        self.distance_features = build_distance_features(self.atomic_pos, k=max_neighbours, directed=directed)
        self.edge_index = build_backbone_edge_index(self.atomic_pos, k=max_neighbours, directed=directed)
    

    def __len__(self):
        return len(self.dms_data)

    def __getitem__(self, idx):
        """ 
        Returns a single mutated sequence's features.
        """
        # build the full mutant sequence for node features + aaindex properties
        code = self.dms.iloc[idx]['Code'].copy()
        mutation_position = [int(i) for i in code.split("_") if i.isdigit()]
        fitness_label = self.labels[idx]

        #build node features
        aaindex_node_features = build_node_features(code, self.wt_sequence_encoded.copy(), self.aa_index)

        # select edge features
        distance_edge_features = self.distance_features.iloc[idx]

        mutation_residue_index = np.zeros(len(self.wt_sequence), dtype=bool)
        for pos in mutation_position:
            mutation_residue_index[pos] = True
    
        protein_graph = ProteinGraphData(
            distance_features=torch.tensor(distance_edge_features.values, dtype=torch.float),
            node_features=torch.tensor(aaindex_node_features, dtype=torch.float),
            sequence=torch.tensor(self.wt_sequence_encoded, dtype=torch.long),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            mutation_idx=torch.tensor(mutation_residue_index, dtype=torch.bool),
            fitness =torch.tensor(fitness_label, dtype=torch.float),
        )

        return protein_graph
