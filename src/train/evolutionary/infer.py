import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_class import RESIDUE_LETTERS
from data.data_utils import load_cif_structure, parse_structure, align_dms_experiment_sequences, real_mutation_mask
from train.train_utils import safe_pearson, safe_spearman


def score_dms_data(dms_data, alignment_mappings, wt_experimental_encoded_sequences, pssm):
    """
    Conservation-based heuristic score, one row per scoreable DMS mutation:
    log(freq(mutant)) - log(freq(WT)) at each mutated position, summed across sites
    (single: 1 site, pair: 2 sites) - the same additive convention used for the GNN's
    zero-shot score, so the two are directly comparable. Mirrors exactly how
    positions/mutant identities are parsed in HomologMaskedDataset's test mode.

    Rows dropped: those whose position(s) don't align to the WT structure, and those that
    do not change any residue (see real_mutation_mask - a log-ratio score is identically 0
    for those and their fitness is WT-level, which inflates Spearman by ~0.08).
    gnn/infer_zero_shot.py applies the same mask, so both report on the same rows.

    args:
        dms_data: DataFrame in dms_processed.csv format
        alignment_mappings, wt_experimental_encoded_sequences: from align_dms_experiment_sequences
        pssm: (N, 20) array of per-position amino acid frequencies (from pssm_process.py)
    returns:
        results_df: DataFrame with columns ['Single', 'Code', 'Fitness', 'evolutionary_score']
    """
    dms_data = dms_data.loc[real_mutation_mask(dms_data, alignment_mappings, wt_experimental_encoded_sequences)]

    rows = []
    for _, sample in dms_data.iterrows():
        is_single = sample['Single']
        alignment_mapping = alignment_mappings[is_single]
        wt_experimental_encoded_sequence = wt_experimental_encoded_sequences[is_single]

        if sample['Ambler Index'] not in alignment_mapping:
            continue

        node_idx = alignment_mapping[sample['Ambler Index']]
        code = sample['Code'].split("_")
        is_pair = not code[1].isnumeric()

        positions = [node_idx]
        mut_aas = [code[3] if is_pair else code[2]]
        if is_pair and (sample['Ambler Index'] + 1) in alignment_mapping:
            positions.append(alignment_mapping[sample['Ambler Index'] + 1])
            mut_aas.append(code[4])

        score = 0.0
        for wt_idx, mut_aa in zip(positions, mut_aas):
            wt_aa_idx = wt_experimental_encoded_sequence[wt_idx]
            mut_aa_idx = RESIDUE_LETTERS.index(mut_aa)
            score += np.log(pssm[wt_idx, mut_aa_idx]) - np.log(pssm[wt_idx, wt_aa_idx])

        rows.append({"Single": sample['Single'], "Code": sample['Code'], "Fitness": sample['Fitness'], "evolutionary_score": score})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the directory containing dms_processed.csv (single and/or pair mutants).")
    parser.add_argument("--pssm_path", type=str, required=True, help="Path to pssm.csv (from pssm_process.py).")
    parser.add_argument("--pdb_dir", type=str, default="C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed", help="Path to the directory containing the WT PDB/CIF structure.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--output_dir", type=str, default="./src/train/evolutionary/output/inference", help="Directory to save scoring results.")
    parser.add_argument("--dont_save_results", action="store_true", help="Skip saving predictions/summary to output_dir.")
    args = parser.parse_args()

    wt_sequence, _ = parse_structure(load_cif_structure(f"{args.pdb_dir}/{args.pdb_id}.cif", args.pdb_id))
    dms_data = pd.read_csv(os.path.join(args.processed_dms, "dms_processed.csv"))
    alignment_mappings, wt_experimental_encoded_sequences = align_dms_experiment_sequences(dms_data, wt_sequence)

    pssm_df = pd.read_csv(args.pssm_path)
    pssm = pssm_df[RESIDUE_LETTERS].to_numpy()

    results_df = score_dms_data(dms_data, alignment_mappings, wt_experimental_encoded_sequences, pssm)

    pearson_r = safe_pearson(results_df["Fitness"], results_df["evolutionary_score"])
    spearman_rho = safe_spearman(results_df["Fitness"], results_df["evolutionary_score"])
    print(f"n={len(results_df)}, Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(args.output_dir, "evolutionary_predictions.csv"), index=False)
        with open(os.path.join(args.output_dir, "evolutionary_summary.json"), "w") as f:
            json.dump({
                "n_samples": int(len(results_df)),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
            }, f, indent=2)
        print(f"Saved evolutionary baseline results to {args.output_dir}")
