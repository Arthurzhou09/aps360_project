"""
Process a ProteinGym-style a2m MSA (e.g. BLAT_ECOLX_full_11-26-2021_b02.a2m) into the
form the structure GNN consumes.

homolog_process.py: that pipeline reads the Pfam
PF00144 *full* alignment, throws the alignment away (extract_family_sequences strips every
gap), and then re-derives a correspondence per sequence with a pairwise aligner that has no
substitution matrix. Two things go wrong with that.

  1. PF00144 is the penicillin-binding transpeptidase superfamily, not TEM-1's own family.
     It is 61,457 sequences over 2,732 columns for a 263-residue protein, TEM-1 is not a
     member, and its E. coli entries are AmpC/AmpH - class C enzymes, a different fold from
     TEM-1's class A.
  2. filter_homologs_by_identity scores identity as matches/len(mapping), i.e. over whatever
     subregion happened to align, so a sequence matching a 60-residue core passes a "40%
     identity to TEM-1" filter. Under a full-length definition only ~4.5% of the kept set
     qualifies.

An a2m alignment already carries the column correspondence, computed once by a profile HMM,
so it is used directly here: no per-sequence realignment happens anywhere in this file.

a2m conventions: uppercase = match state, lowercase = insertion, '-' = gap in a match
state, '.' = gap in an insertion. Match columns are exactly the uppercase/'-' positions of
the query (the first record), which for this file is BLAT_ECOLX itself.

usage:
    python src/data/process/msa_process.py --a2m src/data/raw/BLAT_ECOLX_full_11-26-2021_b02.a2m
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_class import RESIDUE_LETTERS
from data.data_utils import load_cif_structure, parse_structure
from data.feature_utils import align_sequence


def read_a2m(path: str):
    """
    Read an a2m file into (id, raw_sequence) pairs, preserving case and gap characters.
    """
    records = []
    name, buffer = None, []
    with open(path, "r") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buffer)))
                name, buffer = line[1:], []
            else:
                buffer.append(line)
    if name is not None:
        records.append((name, "".join(buffer)))
    return records


def match_state_columns(query_a2m: str) -> list[int]:
    """
    Positions of the match states in the raw a2m string, taken from the query record.
    Match states are the uppercase and '-' positions; lowercase and '.' are insertions and
    carry no alignment column.
    """
    return [i for i, c in enumerate(query_a2m) if c.isupper() or c == "-"]


def query_column_to_residue(query_a2m: str) -> dict[int, int]:
    """
    Map each match column of the query to the query's own residue index (0-based, ungapped).
    """
    column_to_residue = {}
    residue_index = 0
    for i, c in enumerate(query_a2m):
        if c == "." or c == "-":
            continue # gap: consumes no query residue
        if c.isupper():
            column_to_residue[i] = residue_index
        residue_index += 1 # lowercase inserts consume a residue but get no column
    return column_to_residue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m", type=str, required=True, help="Path to the ProteinGym .a2m alignment.")
    parser.add_argument("--pdb_dir", type=str, default=r"C:\Users\Arthur Zhou\GitHub\aps360_project\src\data\processed")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID of the WT reference structure.")
    parser.add_argument("--min_coverage", type=float, default=0.5,
                        help="Drop sequences occupying fewer than this fraction of the match columns.")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\Users\Arthur Zhou\GitHub\aps360_project\src\data\processed\msa")
    args = parser.parse_args()

    wt_sequence, _ = parse_structure(load_cif_structure(f"{args.pdb_dir}/{args.pdb_id}.cif", args.pdb_id))
    records = read_a2m(args.a2m)
    query_id, query_a2m = records[0]
    print(f"{len(records)} sequences; query record is {query_id}")

    columns = match_state_columns(query_a2m)
    column_to_residue = query_column_to_residue(query_a2m)
    query_ungapped = "".join(c for c in query_a2m if c not in ".-").upper()

    # the ONE alignment in this pipeline: query onto the structure. Every other sequence
    # inherits its correspondence from the MSA columns.
    query_to_structure, _ = align_sequence(query_ungapped, wt_sequence)
    identity = sum(1 for i, j in query_to_structure.items() if query_ungapped[i] == wt_sequence[j]) / len(query_to_structure)
    print(f"query vs structure: {len(query_to_structure)} residues mapped, identity {identity:.4f}")
    if identity < 0.9:
        print("WARNING: query and structure disagree badly - is this a2m for the same protein as the PDB?")

    # match column -> structure node index
    column_to_node = {c: query_to_structure[r] for c, r in column_to_residue.items() if r in query_to_structure}
    kept_columns = [c for c in columns if c in column_to_node]
    print(f"match columns: {len(columns)}, mapping onto a structure node: {len(kept_columns)}")
    print(f"structure nodes covered: {len(set(column_to_node.values()))} of {len(wt_sequence)}")

    # extract each sequence's residues at the kept match columns, in one pass, no realignment
    rows = []
    n_columns = len(kept_columns)
    for progress, (seq_id, raw) in enumerate(records):
        if progress % 25000 == 0:
            print(f"  extracting {progress}/{len(records)}")
        states = "".join(raw[c] for c in kept_columns).upper()
        # '.' cannot appear in a match column, but guard anyway; anything non-canonical
        # becomes a gap so downstream code only ever sees RESIDUE_LETTERS or '-'
        states = "".join(c if c in RESIDUE_LETTERS else "-" for c in states)
        coverage = 1.0 - states.count("-") / n_columns
        if coverage < args.min_coverage:
            continue
        rows.append({"seq_id": seq_id, "match_states": states, "coverage": round(coverage, 4)})

    msa_df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    msa_df.to_csv(os.path.join(args.output_dir, "msa_processed.csv"), index=False)

    # column -> structure node, so the dataset never has to redo any of the above
    pd.DataFrame({
        "match_column": np.arange(n_columns),
        "structure_node": [column_to_node[c] for c in kept_columns],
        "query_residue": [column_to_residue[c] for c in kept_columns],
        "wt_aa": [wt_sequence[column_to_node[c]] for c in kept_columns],
    }).to_csv(os.path.join(args.output_dir, "msa_columns.csv"), index=False)

    print(f"kept {len(msa_df)}/{len(records)} sequences at coverage >= {args.min_coverage}")
    print(f"mean coverage {msa_df.coverage.mean():.3f}")
    print(f"saved to {args.output_dir}")
