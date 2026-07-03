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
from data.data_class import ProteinGraphData, DataClass, RESIDUE_LETTERS
import pandas as pd


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
		sequence: list of residue indices referencing RESIDUE_LETTERS. WT sequence.
	returns:
		seq: mutated sequence
		mut_indices: indices of mutated residues

	"""
	# perform a basic alignment

	split = code.split("_") 
	seq = sequence.copy() 

	
	
	if split[1].isnumeric():
		assert seq[int(split[1])] == RESIDUE_LETTERS.index(split[0]), "WT amino acid does not match sequence at position: {} vs {} at idx {} AA {}".format(seq[int(split[1])], RESIDUE_LETTERS.index(split[0]), int(split[1]), split[0])
		mutation_indices = [int(split[1])]
		seq[mutation_indices[0]] = RESIDUE_LETTERS.index(split[2])  # Update the sequence with the mutation
	else:
		muts = [split[3], split[4]]
		mutation_indices = [int(split[2]), int(split[2]) +1]
		wts = [split[0], split[1]]
		for wt, pos, mut in zip(wts, mutation_indices, muts):
			assert seq[pos] == RESIDUE_LETTERS.index(wt), "WT amino acid does not match sequence at position: {} vs {} at idx {} AA {}".format(seq[pos], RESIDUE_LETTERS.index(wt), pos, wt)
			seq[pos] = RESIDUE_LETTERS.index(mut)  # Update the sequence with the mutation
	
	return np.array(seq), np.array(mutation_indices)


def encode_aaindex_features(aaindex_df: pd.DataFrame, sequence: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
	"""
	build node features for each residue in the sequence.
	args:
		aa_index_df: DataFrame mapping aaindex IDs to lists of property values.
		sequence: resiudes encoded as integers
	returns:
		aa_to_value: dict mapping amino acid index to property values
		id_array: list of aaindex record ids corresponding to the properties
	"""

	aa_to_value = {aa: aaindex_df[aa].to_numpy() for aa in RESIDUE_LETTERS}
	id_array = aaindex_df['id'].to_numpy()

	return aa_to_value, id_array


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
	aa_to_value, _ = encode_aaindex_features(aaindex_df, encoded_sequence)

	node_features = np.concatenate([encoded_sequence[:,None], np.array([aa_to_value[aa] for aa in sequence])], axis=1) # (N, F)

	return node_features



def align_sequence(wt_sequence: str, experiment_sequence: str, pdb_id,) -> str:
	"""
	Aligns the wild-type sequence with the experimental sequence.
	args:
		wt_sequence: wild-type amino acid sequence from PDB entry
		experiment_sequence: experimental amino acid sequence from DMS data
	returns:
		aligned_wt_seq: aligned wild-type sequence as a numpy array of indices
	"""

	if pdb_id == "1BTL":
		valid_residue_mask = np.zeros(len(wt_sequence), dtype=bool)
		valid_residue_mask[23:-1] = True  # Mark valid residues (excluding stop codon and activation region)
		# For 1BTL, the wild-type sequence starts at index 23 (0-based) and ends before the last residue
		aligned_wt_seq = wt_sequence
		aligned_experimental_seq = experiment_sequence[23:-1]
	
	if len(aligned_wt_seq) != len(aligned_experimental_seq):
		raise ValueError("Wild-type length {} vs experiment length {} do not match after alignment for pdb_id {}".format(len(aligned_wt_seq), len(aligned_experimental_seq), pdb_id))

	return aligned_wt_seq, valid_residue_mask



