"""Utilities for building simple protein GNN inputs.

This module keeps the graph construction lightweight:
- backbone edges connect residue i to i+1
- node features combine residue identity with optional numeric features
- edge features are small, fixed-size attributes that can be extended later
"""

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.data_class import ProteinGraphData, DataClass

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

# euclidean distance matrix (N,N,3)
def _euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
	"""
	Euclidean distance matrix for atom coordinates
	args:
		coords: (N, x,3) array of atomic coordinates
	returns:   
		dist: (x, N, N) array of pairwise distances (ex: CA, N, C, , x = 4)
	"""
	all_distances = []
	for i in range(coords.shape[1]):
		cur_atom = coords[:, i, :] #N,3
		diff = cur_atom[:, None, :] - cur_atom[None, :, :] #N,1,3 - 1,N,3 -> N,N,3
		all_distances.append(np.sqrt(np.sum(diff**2, axis=-1)))

	return np.stack(all_distances, axis=0) #4,N,N

def _k_nearest_residues(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Get k-nearest CA residue based on distance matrix.
    args:
        distance_matrix: (N, N) array of pairwise distances
        k: number of nearest neighbors to return
    returns:
        nearest_indices: (N, k) array of indices of nearest neighbors for each residue
    """
    d_ca = distance_matrix[0].copy()
    np.fill_diagonal(d_ca, np.inf)
    nearest_indices = np.argsort(d_ca, axis=1)[:,:k]
    return nearest_indices


def build_backbone_edge_index(positions: np.ndarray, k: int = 20,directed:bool = True) -> np.ndarray:
	"""
	Call build distance features prefered, will return edge indices for knn backbone connectivity for a protein chain.

	Create knn backbone connectivity for a protein chain.
	args:
		positions: (N, 4, 3) array of atomic coordinates
		k: number of nearest neighbors to return
		directed: whether to create directed edges (i -> j) or undirected edges (i <-> j). 
	returns:
		edge_index: array of shape (2, E) with source and target indices for each edge (upper bound of 2*E for undirected).
	"""
	distance_matrix = _euclidean_distance_matrix(positions)
	nearest_neighbours = _k_nearest_residues(distance_matrix, k=k) #[N, k]

	num_residues = nearest_neighbours.shape[0]
	source = np.repeat(np.arange(0, num_residues), nearest_neighbours.shape[1])
	target = np.ravel(nearest_neighbours)

	if not directed:
		edge_index = np.vstack([
			np.concatenate([source, target]),
			np.concatenate([target, source]),
		])
		edge_index = np.unique(edge_index, axis=1)
	else:
		edge_index = np.vstack([source, target])

	return edge_index

def build_rbf(pos_1: np.ndarray, pos_2: np.ndarray, edge_indices: np.ndarray,
			  distance_min =2,
			  distance_max = 20,
			  rbf_count = 8,
			  ) -> np.ndarray:
	"""
	Compute gaussian radial basis functions between two sets of atomic positions.
	args:
		pos_1: (N, 3) array of atomic coordinates for atom type 1
		pos_2: (N, 3) array of atomic coordinates for atom type 2
		edge_indices: (2, E) array of source and target indices.
	"""
	euclidean_coord_vector = np.linalg.norm(pos_1[edge_indices[0], :] - pos_2[edge_indices[1], :], axis=-1)

	sigma = (distance_max - distance_min) / rbf_count
	centers = np.linspace(distance_min, distance_max, rbf_count)
	rbf = np.exp(-((euclidean_coord_vector[:, None] - centers[None, :] / sigma ** 2)))#(edge_count, rbf_count)
	return rbf


def build_distance_features(positions: np.ndarray, k: int = 20, directed: bool = True) -> np.ndarray:
	"""
	Compute knn atomic distance features (edge attributes).
	args:
		positions: (N,4, 3) array of atomic coordinates
		k: number of nearest neighbors to search.
		directed: whether to create directed edges (i -> j) or undirected edges (i <-> j).
	returns:
		edge_attr: (E, num_rbf) array of edge attributes for each edge in the graph.
	"""
	edge_index = build_backbone_edge_index(positions, k=k, directed=directed)

	rbf_features = []
	for atom_i in range(positions.shape[1]):
		for atom_j in range(positions.shape[1]):
			feat = build_rbf(positions[:, atom_i], positions[:, atom_j], edge_index,)
			
			rbf_features.append(feat)
	#(16, edge_count, rbf_count)
	return np.stack(rbf_features, axis=-1)


def encode_sequence_features(code: str, sequence: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
	"""
	One-hot encode a protein sequence.
	args:
		code: WT_ambler_Mut or WT1_WT2_ambler_Mut1_Mut2
		sequence: list of residue indices
	returns:
		seq: mutated sequence
		mut_indices: indices of mutated residues

	"""
	split = code.split("_")  # Extract the amino acid sequence from the 
	seq = sequence.copy() # list[int]
	

	assert seq[int(split[1]) - 1] == RESIDUE_LETTERS.index(split[0]), "WT amino acid does not match sequence at position"
	
	if split[1].isnumeric(): # 
		mutation_indices = [int(split[1])]
		seq[int(split[1]) - 1] = RESIDUE_LETTERS.index(split[2])  # Update the sequence with the mutation
	else:
		muts = [split[3], split[4]]
		mutation_indices = [int(split[2]), int(split[2]) +1]
		wts = [split[0], split[1]]
		for wt, pos, mut in zip(wts, mutation_indices, muts):
			assert seq[pos - 1] == RESIDUE_LETTERS.index(wt), "WT amino acid does not match sequence at position"
			seq[pos - 1] = RESIDUE_LETTERS.index(mut)  # Update the sequence with the mutation
	
	return np.array(seq), np.array(mutation_indices)


import pandas as pd
def encode_aaindex_features(aaindex_df: pd.DataFrame, sequence: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	build node features for each residue in the sequence.
	args:
		aa_index_df: DataFrame mapping aaindex IDs to lists of property values.
		sequence: resiudes encoded as integers
	returns:
		propeties: (N ,F), node indices are mapped to residue order
		aaindex_ids: list of aaindex record ids corresponding to the properties
	"""

	aa_order = aaindex_df.columns[1:].tolist()
	property_arr = aaindex_df.to_numpy()[:,1:]
	id_array = property_arr[:, 0]

	return property_arr, id_array, aa_order


def build_node_features(code, sequence, aaindex_df):
	"""
	Build node features.
	args:
		code: WT_ambler_Mut or WT1_WT2_ambler_Mut1_Mut2
		sequence: list of residue indices
		list_properties: list of dicts mapping amino acid index to property value.
	returns:
		node_features: (N, F) array of node features for each residue
	"""
	encoded_sequence, _ = encode_sequence_features(code, sequence)
	aaindex_features, _,_ = encode_aaindex_features(aaindex_df, encoded_sequence)
	node_features = np.concatenate([encoded_sequence[:,None], aaindex_features], axis=1) # (N, F)

	return node_features



