import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_utils import load_cif_structure, parse_structure
from data.feature_utils import align_sequence
from data.data_class import RESIDUE_LETTERS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--homolog_dir', type=str, required=True, help='Path to the directory containing homolog_processed.csv')
    parser.add_argument('--pdb_dir', type=str, default='C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed', help='Path to the directory containing the WT PDB/CIF structure')
    parser.add_argument('--pdb_id', type=str, default='1BTL', help='PDB ID of the WT reference structure')
    parser.add_argument('--pseudocount', type=float, default=1.0, help='pseudocount added to every AA count at each position, so AA never observed at a position gets non -inf log score.')
    parser.add_argument('--output_dir', type=str, default='C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed\\homologs', help='Path to save the PSSM as a CSV file')
    args = parser.parse_args()

    wt_sequence, _ = parse_structure(load_cif_structure(f"{args.pdb_dir}/{args.pdb_id}.cif", args.pdb_id))
    homolog_df = pd.read_csv(os.path.join(args.homolog_dir, "homolog_processed.csv"))

    # count observed amino acids at each WT-aligned position across the homolog family. We use alignment to WT instead of a true MSA.
    counts = np.zeros((len(wt_sequence), len(RESIDUE_LETTERS)), dtype=float)
    counts[np.arange(len(wt_sequence)), [RESIDUE_LETTERS.index(aa) for aa in wt_sequence]] += 1 # WT itself is a family member too

    for i, sequence in enumerate(homolog_df['sequence']):
        if i % 5000 == 0:
            print(f"Building PSSM: {i}/{len(homolog_df)} homologs")
        mapping, _ = align_sequence(sequence, wt_sequence)
        for homolog_idx, wt_idx in mapping.items():
            aa = sequence[homolog_idx]
            if aa in RESIDUE_LETTERS:
                counts[wt_idx, RESIDUE_LETTERS.index(aa)] += 1

    # Apply pseudocount and make matrix
    smoothed = counts + args.pseudocount
    frequencies = smoothed / smoothed.sum(axis=1, keepdims=True)

    pssm_df = pd.DataFrame(frequencies, columns=RESIDUE_LETTERS)
    pssm_df.insert(0, 'wt_position', np.arange(len(wt_sequence)))
    pssm_df.insert(1, 'n_observed', counts.sum(axis=1).astype(int))

    os.makedirs(args.output_dir, exist_ok=True)
    pssm_df.to_csv(os.path.join(args.output_dir, "pssm.csv"), index=False)
    print(f"Processed PSSM saved to {os.path.join(args.output_dir, 'pssm.csv')}")
