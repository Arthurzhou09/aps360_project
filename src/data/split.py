


import numpy as np
import pandas as pd
import re
import os

def get_positions(code):
    """
    Extract mutatation residue positions from Sequence Index (Ambler Index).
    returns:
        list of residue positions (Ambler)
    """
    return [int(x) for x in re.findall(r'_([0-9]+)_', code)]

def row_structural_positions(df, wt_sequence) -> pd.Series:
    """
    Map every DMS row to the set of *structural* residue indices (0..len(wt_sequence)-1)
    it mutates.

    This exists because 'Ambler Index' is an index into the row's own
    'Experiment Sequence', and the single-mutant and pair-mutant experiments use
    *different* sequences (287 vs 281 aa for TEM-1). 187 of 258 shared Ambler indices
    therefore point at a different residue depending on the row's 'Single' value, so
    grouping on the raw index does not group by position at all - it leaks residues
    across splits. Aligning each experiment onto the PDB WT first gives a key that
    actually identifies a residue.

    args:
        df: dms_processed.csv format, with 'Single', 'Ambler Index' and 'Experiment Sequence'
        wt_sequence: reference (PDB) WT sequence
    returns:
        positions: Series of frozensets of structural node indices, aligned to df.index.
            Empty frozenset for rows whose site(s) did not align onto the structure.
    """
    from data.data_utils import align_dms_experiment_sequences

    alignment_mappings, _ = align_dms_experiment_sequences(df, wt_sequence)

    def _positions(row):
        mapping = alignment_mappings[row['Single']]
        ambler = row['Ambler Index']
        if ambler not in mapping:
            return frozenset()
        nodes = {mapping[ambler]}
        if row['Single'] == 0:
            # pair mutants are always at adjacent Ambler positions (Code = WT1_WT2_pos_MUT1_MUT2)
            if (ambler + 1) not in mapping:
                return frozenset()
            nodes.add(mapping[ambler + 1])
        return frozenset(nodes)

    return df.apply(_positions, axis=1)


def split_by_structural_position(
    df,
    wt_sequence,
    train_frac=0.8,
    val_frac=0.1,
    seed=1012,
    n_blocks=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Position-held-out split keyed on structural residue index rather than 'Ambler Index'
    (see row_structural_positions for why the raw index leaks).

    Two grouping modes:

    - n_blocks=None: each residue is assigned independently, like the original
      split_by_position. Correct and lossless for single mutants.
    - n_blocks=K: residues are grouped into K *contiguous* segments and whole segments
      are assigned. Needed whenever pair mutants are present: pair mutants tile every
      consecutive position (the gap between successive pair start positions is exactly 1
      at all 280 of them), so per-residue assignment necessarily splits adjacent pairs
      across splits - 3809 of 17857 rows (21%) of the combined set belong to no split at
      all under the original rule and are silently discarded. Contiguous segments only
      break pairs that straddle a segment boundary, so the loss drops to ~K rows.

    Rows are assigned to a split only if *all* their mutated residues fall in it;
    the few remaining straddling rows are dropped and reported.

    args:
        df: dms_processed.csv format
        wt_sequence: reference (PDB) WT sequence
        n_blocks: number of contiguous residue segments, or None for per-residue assignment.
            Defaults to 20 when the data contains pair mutants, None otherwise.
    returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    positions = row_structural_positions(df, wt_sequence)

    if n_blocks is None and (df['Single'] == 0).any():
        n_blocks = 20

    all_nodes = sorted({n for s in positions for n in s})
    if not all_nodes:
        raise ValueError("No DMS rows aligned onto the WT structure.")

    if n_blocks is None:
        groups = {node: node for node in all_nodes}
        group_ids = list(all_nodes)
    else:
        # contiguous segments over the structure, sized to hold roughly equal residue counts
        edges = np.linspace(0, len(all_nodes), n_blocks + 1).astype(int)
        groups = {}
        for block_id, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            for node in all_nodes[lo:hi]:
                groups[node] = block_id
        group_ids = sorted(set(groups.values()))

    row_groups = positions.apply(lambda s: frozenset(groups[n] for n in s) if s else frozenset())

    # size each group by how many rows it would carry, so the greedy fill tracks row counts
    group_sizes = {g: 0 for g in group_ids}
    for gs in row_groups:
        if len(gs) == 1:
            group_sizes[next(iter(gs))] += 1

    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)

    fractions = {'train': train_frac, 'val': val_frac, 'test': 1 - train_frac - val_frac}
    total = sum(group_sizes.values())
    targets = {name: frac * total for name, frac in fractions.items()}
    counts = {'train': 0, 'val': 0, 'test': 0}
    assignment = {}

    # fill whichever split is furthest below its target share, same rule as split_by_cluster
    for group in group_ids:
        split_name = min(counts, key=lambda name: counts[name] / targets[name] if targets[name] > 0 else float('inf'))
        assignment[group] = split_name
        counts[split_name] += group_sizes[group]

    def _split_of(group_set):
        if not group_set:
            return None
        names = {assignment[g] for g in group_set}
        return names.pop() if len(names) == 1 else None # straddles a boundary

    row_split = row_groups.apply(_split_of)
    unaligned = int((row_groups.apply(len) == 0).sum())
    straddling = int(row_split.isna().sum()) - unaligned
    if unaligned:
        # these never reach the model anyway: the Dataset filters them out on construction
        print(f"split_by_structural_position: {unaligned}/{len(df)} rows did not align onto the structure")
    if straddling:
        print(f"split_by_structural_position: dropped {straddling}/{len(df)} rows "
              f"({100*straddling/len(df):.1f}%) whose mutated residues straddle a split boundary")

    train = df[row_split == 'train'].reset_index(drop=True)
    val = df[row_split == 'val'].reset_index(drop=True)
    test = df[row_split == 'test'].reset_index(drop=True)

    return train, val, test


def split_by_position(
    df,
    train_frac=0.8,
    val_frac=0.1,
    seed=1012,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED - use split_by_structural_position, which keys on the structural residue
    index instead of the raw 'Ambler Index'. This version leaks residues across splits
    whenever single and pair mutants are mixed, and silently drops every double mutant
    that straddles a split. Kept only so existing notebooks keep running.

    Split the dms data by mutation position. All mutations in the same residue position will be
    in the same split.

    returns:
        train_df, val_df, test_df
    """
    df = df.copy()

    df["positions"] = df.apply(lambda x: [x['Ambler Index']] if x['Single'] ==1 else [x['Ambler Index'], x['Ambler Index'] + 1], axis=1)

    # unique residue positions as a set
    all_positions = sorted(
        {p for lst in df.positions for p in lst}
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_positions)
    n = len(all_positions)

    train_pos = set(all_positions[:int(train_frac*n)])
    val_pos = set(all_positions[int(train_frac*n):
                                int((train_frac+val_frac)*n)])
    test_pos = set(all_positions[int((train_frac+val_frac)*n):])    

    # check if mutated positions are contained within their repspective sets
    train_mask = df.positions.apply(lambda x: set(x) <= train_pos)
    val_mask   = df.positions.apply(lambda x: set(x) <= val_pos)
    test_mask  = df.positions.apply(lambda x: set(x) <= test_pos)

    train = df[train_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)

    return train, val, test



def _build_kmers(sequence: str, k: int = 5) -> set:
    """
    Build a set of k-mers from a sequence. Used for splitting by k-mer overlap.
    """
    return {sequence[i:i+k] for i in range(len(sequence)- k + 1)}

def _anchor_kmers(kmers: set, num_anchors: int = 8) -> tuple:
    """
    The num_anchors lexicographically smallest k-mers in a set. Used as cheap bucket
    keys for candidate-cluster lookup: a stand-in for MinHash/LSH banding, but using
    sort order instead of a hash function since it's simpler and just as effective at
    this alphabet size (20 amino acids, k=5 -> 3.2M possible k-mers, so collisions on
    "smallest k-mer" reliably mean real sequence overlap). Near-duplicate sequences
    share almost all their k-mers, so they usually share at least one of these anchors.
    """
    return tuple(sorted(kmers)[:num_anchors])


def cluster_homologs(id_to_sequence: dict[str, str], k: int = 5, threshold=0.5, num_anchors: int = 8):
    """
    Cluster homolog ids by k-mer overlap (Jaccard similarity). A naive version of this
    (compare every new sequence against every existing cluster) is O(n * num_clusters)
    full set operations, which does not finish in reasonable time at family scale (tens
    of thousands of sequences). Instead, candidate clusters are looked up via an anchor
    k-mer index: only clusters that share at least one of a sequence's rarest k-mers
    are checked at all. Among those candidates, a k-mer-set size ratio (min/max) - a
    hard upper bound on Jaccard similarity - prunes obviously-too-different pairs before
    the full intersection/union is computed.

    This trades a small amount of recall (two similar sequences that happen to share
    none of their num_anchors smallest k-mers won't be compared, and end up as separate
    clusters) for tractable runtime. That failure mode is conservative: it can only
    over-split into more/smaller clusters, never merge dissimilar sequences.

    args:
        id_to_sequence: dictionary mapping homolog_ID to sequence
    """
    id_to_cluster = {}
    cluster_kmers = []
    cluster_sizes = []
    anchor_buckets = {} # anchor k-mer -> set of cluster ids whose representative has it as an anchor
    sorted_ids = sorted(id_to_sequence, key=lambda x: len(id_to_sequence[x]), reverse=True)

    for progress, record_id in enumerate(sorted_ids):
        if progress % 5000 == 0:
            print(f"Clustering homologs: {progress}/{len(sorted_ids)}, {len(cluster_kmers)} clusters so far")

        kmers = _build_kmers(id_to_sequence[record_id], k)
        kmers_size = len(kmers)
        anchors = _anchor_kmers(kmers, num_anchors=num_anchors)

        candidates = set()
        for anchor in anchors:
            candidates |= anchor_buckets.get(anchor, set())

        best_cluster = None
        best_similarity = 0.0
        for cluster_id in candidates:
            existing_kmers = cluster_kmers[cluster_id]
            existing_size = cluster_sizes[cluster_id]
            if min(kmers_size, existing_size) / max(kmers_size, existing_size) < threshold:
                continue # size ratio alone already rules this cluster out

            union = len(kmers | existing_kmers)
            similarity =  len(kmers & existing_kmers) / union if union > 0 else 0.0
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id

        if best_cluster is not None and best_similarity >= threshold:
            cluster_id = best_cluster
        else:
            # a new cluster id is formed and a new representative kmer cluster is born
            cluster_id = len(cluster_kmers)
            cluster_kmers.append(kmers)
            cluster_sizes.append(kmers_size)
        id_to_cluster[record_id] = cluster_id

        for anchor in anchors:
            anchor_buckets.setdefault(anchor, set()).add(cluster_id)

    return id_to_cluster



    
def split_by_homolog(
    df,
    train_frac=0.8,
    val_frac=0.1,
    seed=1012,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split homolog_processed.csv data by homolog_ID. All rows for the same homolog end up
    in the same split. 
    close relatives can still end up split across train/val/test.

    returns:
        train_df, val_df, test_df
    """
    df = df.copy()

    all_ids = sorted(df['homolog_ID'].unique())

    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n = len(all_ids)

    train_ids = set(all_ids[:int(train_frac*n)])
    val_ids = set(all_ids[int(train_frac*n):
                          int((train_frac+val_frac)*n)])
    test_ids = set(all_ids[int((train_frac+val_frac)*n):])

    train = df[df['homolog_ID'].isin(train_ids)].reset_index(drop=True)
    val = df[df['homolog_ID'].isin(val_ids)].reset_index(drop=True)
    test = df[df['homolog_ID'].isin(test_ids)].reset_index(drop=True)

    return train, val, test



def split_by_cluster(
    df,
    train_frac=0.8,
    val_frac=0.1,
    seed=1012,
    k=5,
    threshold=0.5,
    cache_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by jaccard similarity of k-mers. All homologs with similar sequences (k-mer overlap) will be in the same split. Splits are not guranteed to be exactly the specified fraction.
    args:
        df: homolog_processed.csv dataframe
        k: k-mer size for clustering
        threshold: jaccard similarity threshold for clustering
        cache_path: optional path to save/reuse the homolog_ID -> cluster assignment.
            Clustering the full family (tens of thousands of sequences) takes a few
            minutes, and pretrain.py/infer_pretrain.py each call this independently, so
            passing the same cache_path to both avoids recomputing it on every run.
    returns:
        train_df, val_df, test_df
    """
    df = df.copy()

    if cache_path is not None and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        id_to_cluster = dict(zip(cached['homolog_ID'], cached['cluster']))
    else: # clustter and save if it does not exist
        sequences = dict(zip(df['homolog_ID'], df['sequence']))
        id_to_cluster = cluster_homologs(sequences, k=k, threshold=threshold)
        if cache_path is not None:
            pd.DataFrame({'homolog_ID': list(id_to_cluster.keys()), 'cluster': list(id_to_cluster.values())}).to_csv(cache_path, index=False)

    df['cluster'] = df['homolog_ID'].map(id_to_cluster)

    cluster_sizes = df.groupby('cluster').size().to_dict()
    cluster_ids= list(cluster_sizes.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)
    cluster_ids.sort(key=lambda c: cluster_sizes[c], reverse=True) #keeps shuffle order (by size) for ties 


    fractions = {'train': train_frac, 'val': val_frac, 'test': 1 - train_frac - val_frac}
    targets = {name: frac* sum(cluster_sizes.values()) for name, frac in fractions.items()}
    counts = {'train': 0, 'val': 0, 'test': 0}
    assignment = {'train': set(), 'val': set(), 'test': set()}

    # we add to smallest split/frac ratio, splits are not going to be perfect fractions we want them to be
    for cluster in cluster_ids:
        split_name = min(counts, key=lambda name: counts[name] / targets[name] if targets[name] > 0 else float('inf'))
        assignment[split_name].add(cluster)
        counts[split_name] += cluster_sizes[cluster]

    train = df[df['cluster'].isin(assignment['train'])].drop(columns='cluster').reset_index(drop=True)
    val = df[df['cluster'].isin(assignment['val'])].drop(columns='cluster').reset_index(drop=True)
    test = df[df['cluster'].isin(assignment['test'])].drop(columns='cluster').reset_index(drop=True)

    return train, val, test

