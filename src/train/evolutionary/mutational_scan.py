import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_class import RESIDUE_LETTERS
from data.data_utils import load_cif_structure, parse_structure, align_dms_experiment_sequences
from train.evolutionary.infer import score_dms_data
from train.train_utils import safe_pearson, safe_spearman


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the directory containing dms_processed.csv (single mutants - pairs don't fit a position x AA grid).")
    parser.add_argument("--pssm_path", type=str, required=True, help="Path to pssm.csv (from pssm_process.py).")
    parser.add_argument("--pdb_dir", type=str, default="C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed", help="Path to the directory containing the WT PDB/CIF structure.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--output_dir", type=str, default="./src/train/evolutionary/output/mutational_scan", help="Directory to save the scan CSV/JSON.")
    args = parser.parse_args()

    wt_sequence, _ = parse_structure(load_cif_structure(f"{args.pdb_dir}/{args.pdb_id}.cif", args.pdb_id))
    wt_aa_idx = np.array([RESIDUE_LETTERS.index(aa) for aa in wt_sequence])
    n_positions = len(wt_sequence)

    # predicted score grid: log freq(mutant) - log freq(WT) for every position x amino acid,
    # straight from the PSSM - no DMS data needed for this half at all.
    pssm_df = pd.read_csv(args.pssm_path)
    pssm = pssm_df[RESIDUE_LETTERS].to_numpy()
    log_pssm = np.log(pssm)
    predicted_grid = log_pssm - log_pssm[np.arange(n_positions), wt_aa_idx][:, None]

    # actual fitness grid: NaN by default, filled in from measured single mutants only.
    dms_data = pd.read_csv(os.path.join(args.processed_dms, "dms_processed.csv"))
    dms_data = dms_data[dms_data['Single'] == 1]
    alignment_mappings, wt_experimental_encoded_sequences = align_dms_experiment_sequences(dms_data, wt_sequence)
    mapping = alignment_mappings[1]

    actual_grid = np.full((n_positions, len(RESIDUE_LETTERS)), np.nan)
    for _, row in dms_data.iterrows():
        if row['Ambler Index'] not in mapping:
            continue
        wt_idx = mapping[row['Ambler Index']]
        mut_aa = row['Code'].split("_")[2]
        actual_grid[wt_idx, RESIDUE_LETTERS.index(mut_aa)] = row['Fitness']

    # WT-to-itself isn't a real mutation - mask it out of both grids rather than showing a
    # meaningless "score of 0" / "fitness of nothing measured" cell.
    predicted_grid[np.arange(n_positions), wt_aa_idx] = np.nan
    actual_grid[np.arange(n_positions), wt_aa_idx] = np.nan

    # Correlations come from score_dms_data, the same implementation evolutionary/infer.py
    # reports and the GNN zero-shot is compared against, rather than being recomputed off
    # this grid. The grid is built for the heatmap and is not a scoring path: it references
    # every mutation against the 1BTL structure residue rather than the experiment's own WT
    # (they differ at 2 aligned positions)
    scored = score_dms_data(dms_data, alignment_mappings, wt_experimental_encoded_sequences, pssm)
    pearson_r = safe_pearson(scored["evolutionary_score"], scored["Fitness"])
    spearman_rho = safe_spearman(scored["evolutionary_score"], scored["Fitness"])
    measured_mask = ~np.isnan(actual_grid)
    print(f"n scored mutations: {len(scored)}, Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}")
    print(f"({int(measured_mask.sum())} measured grid cells retained for the heatmap)")

    # long-format CSV, for reuse in a notebook or elsewhere
    records = []
    for pos in range(n_positions):
        for aa_i, aa in enumerate(RESIDUE_LETTERS):
            if aa_i == wt_aa_idx[pos]:
                continue
            actual = actual_grid[pos, aa_i]
            records.append({
                "position": pos,
                "wt_aa": wt_sequence[pos],
                "mut_aa": aa,
                "predicted_score": round(float(predicted_grid[pos, aa_i]), 4),
                "actual_fitness": None if np.isnan(actual) else round(float(actual), 4),
            })

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, "mutational_scan.csv"), index=False)