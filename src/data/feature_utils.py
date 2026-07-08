"""
Utilities for building simple protein GNN inputs.
"""

import numpy as np
from data.data_class import ProteinGraphData, DataClass, RESIDUE_LETTERS
import pandas as pd
from Bio.Align import PairwiseAligner


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
	# return the upper diagonal matrix all the time
    nearest_indices = np.argsort(d_ca, axis=1)[:,:k]
    return nearest_indices


def build_backbone_edge_index(positions: np.ndarray, k: int = 20,directed:bool = False) -> np.ndarray:
	"""
	Call build distance features prefered, will return edge indices for knn spatial graph for a protein chain.

	args:
		positions: (N, 4, 3) array of atomic coordinates
		k: number of nearest neighbors to return
		directed: whether to create directed edges (i -> j) or undirected edges (i <->j). 
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
		edge_index = np.hstack([np.vstack([source, target]), np.vstack([target, source])])
		edge_index = np.unique(edge_index, axis=1)  #remove exact duplicates only.
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
	rbf = np.exp(-((euclidean_coord_vector[:, None] - centers[None, :])**2 / sigma ** 2))#(edge_count, rbf_count)
	return rbf


def build_distance_features(positions: np.ndarray, k: int = 16, directed: bool = False) -> np.ndarray:
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
	#( edge_count, 16*rbf_count)
	return np.concatenate(rbf_features, axis=-1)


def align_sequence(seq1, seq2) ->tuple[dict[int, int], object]:
	"""
	Global alignment of two sequences. Point mutations are included in the mapping.
	args:
		seq1: target sequence (experiment)
		seq2: reference sequence (pdb wt)
	returns:
		mapping: maapping[i] gives the residue index in seq2 for residue i in seq1. 
		alignment: the alignment object from Biopython PairwiseAligner
	"""
	aligner = PairwiseAligner()
	aligner.mode = "global"
	aligner.match_score = 1
	aligner.mismatch_score = -1
	aligner.open_gap_score = -2
	aligner.extend_gap_score = -0.5
	alignment = aligner.align(seq1, seq2)[0]

	mapping = {}

	for (s1_start, s1_end), (s2_start, s2_end) in zip(*alignment.aligned):
		for i1, i2 in zip(range(s1_start, s1_end), range(s2_start, s2_end)):
			mapping[i1] = i2

	return mapping, alignment



def encode_aaindex_features(aaindex_df: pd.DataFrame) -> tuple[dict, np.ndarray]:
	"""
	build node features for each residue in the sequence. Properties are standardized
	across the 20 canonical AAs.
	args:
		aa_index_df: DataFrame mapping aaindex IDs to lists of property values.
		sequence: resiudes encoded as integers
	returns:
		aa_to_value: dict mapping amino acid index to standardized property values
		id_array: list of aaindex record ids corresponding to the properties
	"""

	values = aaindex_df[RESIDUE_LETTERS].to_numpy(dtype=float) # (num_properties, 20)
	standardized = (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, keepdims=True)

	aa_to_value = {aa: standardized[:, i] for i, aa in enumerate(RESIDUE_LETTERS)}
	id_array = aaindex_df['id'].to_numpy()

	return aa_to_value, id_array


def _one_hot_aa(aa_idx: int) -> np.ndarray:
	"""
	One-hot encode an amino acid index (0-19), or all-zeros for -1/unknown.
	returns:
		one_hot: (20,) array of one-hot encoded amino acid identity
	"""
	one_hot = np.zeros(len(RESIDUE_LETTERS), dtype=float)
	if aa_idx != -1:
		one_hot[aa_idx] = 1.0
	return one_hot


def build_node_features( encoded_mutation_sequence: np.ndarray[int], aaindex_df: pd.DataFrame):
	"""
	Build node features
	args:
		encoded_mutation_sequence:  encoded mutation sequence (0-19)
		aaindex_df: DataFrame mapping aaindex IDs to lists of property values
	returns:
		node_features: (N, 20+ aa_prop_classes) array of node features for each residue
	"""
	aa_to_value, _ = encode_aaindex_features(aaindex_df)

	zero_feat = np.zeros(len(next(iter(aa_to_value.values()))), dtype=float)
	one_hot_features = np.array([_one_hot_aa(aa_idx) for aa_idx in encoded_mutation_sequence])
	aa_features = np.array([
		aa_to_value[RESIDUE_LETTERS[aa_idx]] if aa_idx != -1 else zero_feat
		for aa_idx in encoded_mutation_sequence])

	node_features = np.concatenate([one_hot_features, aa_features], axis=1) #[263, 20] cat [8] = 263 , 28

	return node_features


def build_mutation_features(wt_aa_idx: int, mut_aa_idx: int, aaindex_df: pd.DataFrame) -> np.ndarray:
	"""
	Build compact features describing mutations, instead of whole sequence for MLP.
	Amino acid identity is one-hot encoded rather than a raw ordinal index, since
	there is no meaningful ordering between amino acids.
	args:
		wt_aa_idx: encoded WT amino acid index (0-19)
		mut_aa_idx: encoded mutant amino acid index (0-19)
		aaindex_df: DataFrame mapping aaindex IDs to lists of property values
	returns:
		features: (2*20 + 3*P,) array: [wt_onehot, mut_onehot, wt_props, mut_props, delta_props]
	"""
	aa_to_value, _ = encode_aaindex_features(aaindex_df)

	wt_props = aa_to_value[RESIDUE_LETTERS[wt_aa_idx]]
	mut_props = aa_to_value[RESIDUE_LETTERS[mut_aa_idx]]
	delta_props = mut_props - wt_props

	return np.concatenate([_one_hot_aa(wt_aa_idx), _one_hot_aa(mut_aa_idx), wt_props, mut_props, delta_props])


