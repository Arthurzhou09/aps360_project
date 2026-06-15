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


RESIDUE_ORDER = list("ARNDCQEGHILKMFPSTWYV")
RESIDUE_TO_INDEX = {residue: index for index, residue in enumerate(RESIDUE_ORDER)}


@dataclass
class ProteinGraph:
	sequence: str
	node_features: np.ndarray
	edge_index: np.ndarray
	edge_attr: np.ndarray
	target: Optional[float] = None


def build_backbone_edge_index(nearest_neighbours: np.ndarray, directed:bool = True) -> np.ndarray:
    """
    Create sequential backbone connectivity for a protein chain.
	args:
        nearest_neighbours: array of shape (N, k) with indices of the k nearest neighbors for each of the N residues.
        directed: whether to create directed edges (i -> j) or undirected edges (i <-> j). 
    returns:
        edge_index: array of shape (2, E) with source and target indices for each edge (upper bound of 2*E for undirected).
    """
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


def build_backbone_edge_attr(num_residues: int, directed: bool = False) -> np.ndarray:
	"""Create simple edge attributes for backbone edges.

	Columns:
	- backbone flag
	- sequence distance
	- direction (+1 forward, -1 reverse)
	"""
	if num_residues < 2:
		return np.zeros((0, 3), dtype=np.float32)

	forward = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (num_residues - 1, 1))

	if not directed:
		return forward

	reverse = forward.copy()
	reverse[:, 2] = -1.0
	return np.concatenate([forward, reverse], axis=0)


def build_residue_one_hot(sequence: str) -> np.ndarray:
	"""One-hot encode a protein sequence over the 20 canonical amino acids."""
	encoded = np.zeros((len(sequence), len(RESIDUE_ORDER)), dtype=np.float32)
	for row, residue in enumerate(sequence):
		index = RESIDUE_TO_INDEX.get(residue)
		if index is not None:
			encoded[row, index] = 1.0
	return encoded


def build_residue_node_features(
	sequence: str,
	numeric_features: Optional[np.ndarray] = None,
	include_one_hot: bool = True,
) -> np.ndarray:
	"""Build node features for a residue-level GNN.

	Args:
		sequence: amino-acid sequence.
		numeric_features: optional array of shape (L, F) with extra residue features.
		include_one_hot: whether to include residue identity.

	Returns:
		Array of shape (L, D).
	"""
	features = []

	if include_one_hot:
		features.append(build_residue_one_hot(sequence))

	if numeric_features is not None:
		numeric_features = np.asarray(numeric_features, dtype=np.float32)
		if numeric_features.shape[0] != len(sequence):
			raise ValueError(
				"numeric_features must have one row per residue "
				f"(expected {len(sequence)}, got {numeric_features.shape[0]})."
			)
		features.append(numeric_features)

	if not features:
		return np.zeros((len(sequence), 0), dtype=np.float32)

	return np.concatenate(features, axis=1)


def build_protein_graph(
	sequence: str,
	numeric_features: Optional[np.ndarray] = None,
	bidirectional_edges: bool = True,
) -> ProteinGraph:
	"""Create a minimal residue-level protein graph skeleton."""
	node_features = build_residue_node_features(sequence, numeric_features=numeric_features)
	edge_index = build_backbone_edge_index(len(sequence), directed=not bidirectional_edges)
	edge_attr = build_backbone_edge_attr(len(sequence), directed=not bidirectional_edges)
	return ProteinGraph(
		sequence=sequence,
		node_features=node_features,
		edge_index=edge_index,
		edge_attr=edge_attr,
	)


class ProteinGraphDataset(Dataset):
	"""Torch dataset wrapper for precomputed protein graphs."""

	def __init__(self, graphs: Sequence[ProteinGraph]):
		self.graphs = list(graphs)

	def __len__(self) -> int:
		return len(self.graphs)

	def __getitem__(self, index: int):
		graph = self.graphs[index]
		item = {
			"sequence": graph.sequence,
			"node_features": torch.as_tensor(graph.node_features, dtype=torch.float32),
			"edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
			"edge_attr": torch.as_tensor(graph.edge_attr, dtype=torch.float32),
		}
		if graph.target is not None:
			item["target"] = torch.tensor(graph.target, dtype=torch.float32)
		return item


def make_protein_graph_loader(
	graphs: Sequence[ProteinGraph],
	batch_size: int = 1,
	shuffle: bool = False,
	num_workers: int = 0,
) -> DataLoader:
	"""Create a DataLoader for a list of protein graphs."""
	return DataLoader(
		ProteinGraphDataset(graphs),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
	)