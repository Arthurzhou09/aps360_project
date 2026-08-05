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
		coords: (N, 4,3) array of atomic coordinates
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


def build_backbone_edge_index(positions: np.ndarray, k: int = 20,directed:bool = False,
							  radius: float = None) -> np.ndarray:
	"""
	Call build distance features prefered, will return edge indices for a spatial graph.

	Two neighbourhood definitions:

	- k nearest (radius=None): every residue gets exactly k neighbours. 
	- radius cutoff (radius=R): every residue is linked to whatever lies within R angstroms
	  of its CA. 

	args:
		positions: (N, 4, 3) array of atomic coordinates
		k: number of nearest neighbors to return (ignored when radius is given)
		directed: whether to create directed edges (i -> j) or undirected edges (i <->j).
		radius: CA-CA cutoff in angstroms. None uses the k-nearest rule instead.
	returns:
		edge_index: array of shape (2, E) with source and target indices for each edge (upper bound of 2*E for undirected).
	"""
	distance_matrix = _euclidean_distance_matrix(positions)

	# k <= 0 would slice as np.argsort(...)[:, :k], which Python reads as "all but the last
	# |k| columns" - k=-1 silently yields 262 neighbours per residue and a complete graph.
	# There is no sentinel that disables the k-nearest rule; set radius to switch modes.
	if radius is None and k <= 0:
		raise ValueError(f"k must be positive, got {k}. To use a distance cutoff instead of "
						 f"k-nearest, pass radius=<angstroms>; k is then ignored.")

	if radius is not None:
		d_ca = distance_matrix[0].copy()
		np.fill_diagonal(d_ca, np.inf) # no self loops, matching the k-nearest path
		source, target = np.nonzero(d_ca < radius)
		if len(source) == 0:
			raise ValueError(f"radius {radius}A produced no edges")
		isolated = np.setdiff1d(np.arange(d_ca.shape[0]), np.unique(source))
		if len(isolated):
			# an isolated node never receives a message, so its embedding is its input alone
			print(f"build_backbone_edge_index: {len(isolated)} residues have no neighbour within {radius}A")
	else:
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
			  distance_max = 12,
			  rbf_count = 6,
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


def build_distance_features(positions: np.ndarray, k: int = 16, directed: bool = False,
							radius: float = None, edge_index: np.ndarray = None) -> np.ndarray:
	"""
	Compute atomic distance features (edge attributes) for the spatial graph.
	args:
		positions: (N,4, 3) array of atomic coordinates
		k: number of nearest neighbors to search.
		directed: whether to create directed edges (i -> j) or undirected edges (i <-> j).
		radius: CA-CA cutoff in angstroms; None uses k-nearest. MUST match whatever is passed
			to build_backbone_edge_index, or the edge attributes are attached to the wrong edges.
		edge_index: prebuilt (2, E) edge index to attach features to, e.g. from
			build_dci_edge_index. Given this, k/radius are ignored - passing the edge index
			explicitly is the only safe way to keep attributes aligned with a graph this
			function did not construct itself.
	returns:
		edge_attr: (E, num_rbf) array of edge attributes for each edge in the graph.
	"""
	if edge_index is None:
		edge_index = build_backbone_edge_index(positions, k=k, directed=directed, radius=radius)

	rbf_features = []
	for atom_i in range(positions.shape[1]):
		for atom_j in range(positions.shape[1]):
			feat = build_rbf(positions[:, atom_i], positions[:, atom_j], edge_index,)

			rbf_features.append(feat)
	#( edge_count, 16*rbf_count)
	return np.concatenate(rbf_features, axis=-1)

from data import dfi_calc # sibling module; a bare "import dfi_calc" fails since src/ is what's on sys.path
from Bio.PDB import PDBIO
def build_dci_perturbation_matrix(structure, cache_path: str = None) -> np.ndarray:
	"""
	Normalized perturbation matrix from dfi_calc.

	perturbation[i, j] is the displacement response of residue i to a unit force applied at
	residue j. It is not symmetric. dfi_calc.pdb_reader parses fixed column PDB ATOM records, so a Bio.PDB structure loaded from mmCIF is written to a temporary PDB first.

	args:
		structure: Bio.PDB structure (from load_cif_structure)
		cache_path: optional .npy path to read/write the matrix
	returns:
		perturbation: (N, N) matrix, rows indexed the same as parse_structure's residues
	"""
	import os, tempfile

	if cache_path is not None and os.path.exists(cache_path):
		return np.load(cache_path)

	handle, pdb_path = tempfile.mkstemp(suffix=".pdb")
	os.close(handle)
	try:
		writer = PDBIO()
		writer.set_structure(structure)
		writer.save(pdb_path)
		atoms = dfi_calc.pdb_reader(pdb_path, CAonly=True, noalc=True)
	finally:
		os.remove(pdb_path)

	x, y, z = dfi_calc.getcoords(atoms)
	n_residues = len(atoms)
	inverse_hessian = dfi_calc.calc_covariance(n_residues, x, y, z)

	directions = np.vstack(([1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]))
	directions = directions / np.linalg.norm(directions, axis=1)[:, None]
	perturbation = dfi_calc.calcperturbMat(inverse_hessian, directions, n_residues)

	if cache_path is not None:
		np.save(cache_path, perturbation)
	return perturbation


def _union_edges(dci_edges: np.ndarray, spatial_edges: np.ndarray):
	"""
	Union two directed edge sets, keeping a flag for which set each edge came from.

	args:
		dci_edges, spatial_edges: (2, E) directed edge indices
	returns:
		edge_index: (2, E_union) deduplicated edges
		source_flags: (E_union, 2) float array, columns [from_dci, from_spatial]
	"""
	both = np.hstack([dci_edges, spatial_edges])
	edge_index, inverse = np.unique(both, axis=1, return_inverse=True)

	source_flags = np.zeros((edge_index.shape[1], 2), dtype=float)
	source_flags[inverse[:dci_edges.shape[1]], 0] = 1.0
	source_flags[inverse[dci_edges.shape[1]:], 1] = 1.0
	return edge_index, source_flags


def _remove_additive_driver_strength(asymmetry: np.ndarray) -> np.ndarray:
	"""
	Strip the responder-independent part of the DCI asymmetry, leaving the pairwise residual. DCI normalizes each responder's (i) DFI , so flexible residues have large A[i,j] v.s all j's. When using k threshold, this produces graphs where a few flexible resiude dominate the edges (really large columns in A). keeping the K top drivers for every responder would mean that every responder might choose a small subset of drivers.

	args:
		asymmetry: (N, N) antisymmetric DCI asymmetry matrix with a zero diagonal
	returns:
		residual: (N, N) antisymmetric matrix with the global driver-strength term removed
	"""
	driver_strength = asymmetry.mean(0)
	return asymmetry - (driver_strength[None, :] - driver_strength[:, None])


def build_dci_edge_index(structure, k: int = 12, threshold: float = None, cache_path: str = None,
						 return_asymmetry: bool = False, remove_driver_strength: bool = True):
	"""
	Directed edge index from the asymmetry of the dynamic coupling index.

	DCI[i, j] normalizes the perturbation response of i to j by i's own overall flexibility
	(its DFI, the row mean), so DCI[i, j] = N * perturbation[i, j] / sum_k perturbation[i, k]. For edge connectivity we use the difference:
		A[i, j] = DCI[i, j] - DCI[j, i]
	coupling can therefore be long range and graphs should be asymetrical.

	Edges are ranked on the pairwise residual of A rather than on A itself. The asymmetry matrix is a raw one.

	args:
		structure: Bio.PDB structure (from load_cif_structure)
		k: maximum number of strongest drivers to keep per residue
		threshold: minimum ranked score for an edge to survive. None keeps all k. NOTE this is
			in the units of whatever is ranked: with remove_driver_strength (the default) that
			is the residual.
		cache_path: opt. .npy path for the perturbation matrix (see build_dci_perturbation_matrix)
		return_asymmetry: also return the (N, N) raw asymmetry matrix
		remove_driver_strength: rank on the pairwise residual instead of raw A. 
	returns:
		edge_index: (2, E) array of directed edges, source=driver, target=responder.
			E == N*k only when threshold is None.
		asymmetry: (N, N) raw matrix, only when return_asymmetry is True
	"""
	perturbation = build_dci_perturbation_matrix(structure, cache_path=cache_path)
	n_residues = perturbation.shape[0]

	dci = perturbation / (perturbation.sum(1, keepdims=True) / n_residues)
	asymmetry = dci - dci.T
	np.fill_diagonal(asymmetry, 0.0) # zero by antisymmetry; set exactly, so the column means below are clean

	ranked = _remove_additive_driver_strength(asymmetry) if remove_driver_strength else asymmetry.copy()
	np.fill_diagonal(ranked, -np.inf) # never a self loop

	drivers = np.argsort(-ranked, axis=1)[:, :k] # per responder i, its k strongest drivers j
	responders = np.repeat(np.arange(n_residues), k)
	drivers = drivers.ravel()

	if threshold is not None:
		keep = ranked[responders, drivers] > threshold
		drivers, responders = drivers[keep], responders[keep]
		if len(drivers) == 0:
			raise ValueError(f"DCI threshold {threshold} kept no edges; the maximum ranked score "
							 f"on this structure is {ranked[np.isfinite(ranked)].max():.3f}")

	# out-degree is the diagnostic that matters: a healthy coupling graph spreads driving
	# across the structure, a degenerate one routes most edges through a few hubs
	in_degree = np.bincount(responders, minlength=n_residues)
	out_degree = np.bincount(drivers, minlength=n_residues)
	isolated = int((in_degree == 0).sum())
	print(f"build_dci_edge_index: k={k} threshold={threshold} "
		  f"driver_strength_removed={remove_driver_strength} -> {len(drivers)} edges, "
		  f"in-degree mean {in_degree.mean():.1f} (0-{in_degree.max()}), "
		  f"{int((out_degree > 0).sum())}/{n_residues} residues act as a driver "
		  f"(max out-degree {out_degree.max()})"
		  + (f", {isolated} residues receive none" if isolated else ""))

	edge_index = np.vstack([drivers, responders]) # source=driver, target=responder

	if return_asymmetry:
		return edge_index, asymmetry
	return edge_index


from Bio.PDB.SASA import ShrakeRupley
def build_structural_features(atomic_positions: np.ndarray, structure=None, chain_id: str = "A",
							  active_site_index: int = 44) -> np.ndarray:
	"""
	Per-residue structural descriptors of how buried / central a position is, standardized.
	args:
		atomic_positions: (N, 4, 3) CA/N/C/O coordinates from parse_structure
		structure: Bio.PDB structure for SASA. If None, SASA is omitted and replaced by zeros
			(keeps the feature width fixed so checkpoints stay loadable).
		chain_id: chain to read residues from for SASA
		active_site_index: structure index of the catalytic residue. Default 44 is TEM-1's
			Ser70 (Ambler 70; structure index 0 == Ambler 26) - verified to be a serine in
			the S-x-x-K motif. Change this for a different protein.
	returns:
		features: (N, 4) standardized [contact_8, contact_12, sasa, distance_to_active_site]
	"""
	ca = atomic_positions[:, 0, :]
	distances = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
	np.fill_diagonal(distances, np.inf) # a residue is not its own contact

	contact_8 = (distances < 8.0).sum(1).astype(float)
	contact_12 = (distances < 12.0).sum(1).astype(float)
	distance_to_active_site = np.linalg.norm(ca - ca[active_site_index], axis=1)

	sasa = np.zeros(len(ca), dtype=float)
	if structure is not None:
		ShrakeRupley().compute(structure[0], level="R")
		values = [r.sasa for r in structure[0][chain_id].get_residues() if r.id[0] == " " and hasattr(r, "sasa")]
		sasa[:min(len(values), len(ca))] = values[:len(ca)]

	features = np.column_stack([contact_8, contact_12, sasa, distance_to_active_site])
	# raw ranges differ by magnitudes, so an unstandardized SASA would dominate input_proj 
	spread = features.std(0)
	spread[spread == 0] = 1.0
	return (features - features.mean(0)) / spread


PCA19_AA_ORDER = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
				  "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]


def load_pca_aa_features(path: str) -> dict:
	"""
	Load a (20, 19) PCA-compressed AAindex table from https://pmc.ncbi.nlm.nih.gov/articles/PMC12557479/#abstract1


	args:
		path: path to pca-19.npy, shape (20, 19)
	returns:
		aa_to_value: dict mapping amino acid letter -> (19,) component vector
	"""
	table = np.load(path)
	if table.shape != (len(PCA19_AA_ORDER), 19):
		raise ValueError(f"expected a (20, 19) table in {path}, got {table.shape}")

	source = {aa: table[i] for i, aa in enumerate(PCA19_AA_ORDER)}
	return {aa: source[aa].astype(float) for aa in RESIDUE_LETTERS}


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


def build_node_features( encoded_mutation_sequence: np.ndarray[int], aaindex_df: pd.DataFrame, aa_to_value: dict = None):
	"""
	Build node features
	args:
		encoded_mutation_sequence:  encoded mutation sequence (0-19)
		aaindex_df: DataFrame mapping aaindex IDs to lists of property values
		aa_to_value: optional precomputed AA letter -> property vector. Pass this to use a
			different property set (e.g. load_pca_aa_features' 19 components) or simply to
			skip re-standardizing aaindex_df on every call, which is a pandas round trip.
	returns:
		node_features: (N, 20+ aa_prop_classes) array of node features for each residue
	"""
	if aa_to_value is None:
		aa_to_value, _ = encode_aaindex_features(aaindex_df)

	zero_feat = np.zeros(len(next(iter(aa_to_value.values()))), dtype=float)
	one_hot_features = np.array([_one_hot_aa(aa_idx) for aa_idx in encoded_mutation_sequence])
	aa_features = np.array([
		aa_to_value[RESIDUE_LETTERS[aa_idx]] if aa_idx != -1 else zero_feat
		for aa_idx in encoded_mutation_sequence])

	node_features = np.concatenate([one_hot_features, aa_features], axis=1) #[263, 20] cat [8] = 263 , 28

	return node_features


def build_mutation_features(wt_aa_idx: int, mut_aa_idx: int, aaindex_df: pd.DataFrame, aa_to_value: dict = None) -> np.ndarray:
	"""
	Build compact features describing mutations, instead of whole sequence for MLP.
	Amino acid identity is one-hot encoded rather than a raw ordinal index, since
	there is no meaningful ordering between amino acids.
	args:
		wt_aa_idx: encoded WT amino acid index (0-19)
		mut_aa_idx: encoded mutant amino acid index (0-19)
		aaindex_df: DataFrame mapping aaindex IDs to lists of property values
		aa_to_value: optional precomputed AA letter -> property vector, so the caller can
			select a property set (see load_pca_aa_features) and avoid re-standardizing
			aaindex_df on every call
	returns:
		features: (2*20 + 3*P,) array: [wt_onehot, mut_onehot, wt_props, mut_props, delta_props]
	"""
	if aa_to_value is None:
		aa_to_value, _ = encode_aaindex_features(aaindex_df)

	wt_props = aa_to_value[RESIDUE_LETTERS[wt_aa_idx]]
	mut_props = aa_to_value[RESIDUE_LETTERS[mut_aa_idx]]
	delta_props = mut_props - wt_props

	return np.concatenate([_one_hot_aa(wt_aa_idx), _one_hot_aa(mut_aa_idx), wt_props, mut_props, delta_props])


