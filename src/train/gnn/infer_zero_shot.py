import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.homolog import HomologMaskedDataset
from model.gnn import Tem1MaskedResidueGNN
from train.train_utils import load_config, safe_pearson, safe_spearman


def run_zero_shot_inference(model, data_loader, device):
    """
    Zero-shot scoring: for each DMS mutation, sum log P(mutant) - log P(WT) over its
    masked position (1 for single mutants, 2 for pair mutants), read off from the
    model's own softmax at each masked node. Fitness carried along here to correlate against it afterward.
    Also tracks wildtype-recovery accuracy: same definition as infer_pretrain.py's
    accuracy (top-1 prediction matches the true label), but measured over the DMS
    structure's masked positions rather than the held-out homolog test set.
    returns:
        scores: (N,) zero-shot scores, one per DMS row
        fitness: (N,) DMS fitness values, same order
        accuracy: fraction of masked positions where the model's top-1 prediction
            matches the true wildtype residue there
    """
    model.to(device)
    model.eval()

    all_scores = []
    all_fitness = []
    total_correct = 0
    total_masked = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch.node_features, batch.edge_index, batch.distance_features)
            log_probs = torch.log_softmax(logits, dim=-1)

            mask = batch.labels != -100 # masked position(s): 1 node for singles, 2 for pairs
            masked_labels = batch.labels[mask]
            wt_logp = log_probs[mask, masked_labels]
            mut_logp = log_probs[mask, batch.mut_aa_idx[mask]]
            per_node_score = mut_logp - wt_logp

            predictions = log_probs[mask].argmax(dim=-1)
            total_correct += (predictions == masked_labels).sum().item()
            total_masked += masked_labels.numel()

            # sum per graph: a pair mutant's two masked nodes collapse into one score
            graph_ids = batch.batch[mask]
            score = torch.zeros(batch.num_graphs, device=device).scatter_add_(0, graph_ids, per_node_score)

            all_scores.append(score.cpu().numpy())
            all_fitness.append(batch.fitness.cpu().numpy())

            if total_masked % 1000 == 0:
                print(f"Zero-shot inference: {total_masked} masked positions processed")

    scores = np.concatenate(all_scores)
    fitness = np.concatenate(all_fitness)
    accuracy = total_correct / total_masked
    return scores, fitness, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the directory containing dms_processed.csv (single and/or pair mutants).")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file (same one used for pretraining).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained masked-residue model checkpoint.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. Must match training. False if not specified.")

    parser.add_argument("--output_dir", type=str, default="./src/train/gnn/output_pretrain/zero_shot", help="Directory to save zero-shot scoring results.")
    parser.add_argument("--dont_save_results", action="store_true", help="Skip saving predictions/summary to output_dir.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # data load: dms_processed.csv (single and/or pair mutants), scored zero-shot - never
    # used for training, so there is no train/val/test split to worry about here.
    dms_df = pd.read_csv(os.path.join(args.processed_dms, "dms_processed.csv"))
    dataset = HomologMaskedDataset(dms_data=dms_df, pdb_id=args.pdb_id, test=True, directed=args.directed, max_neighbours=cfg.data.neighbours)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model load
    checkpoint = torch.load(args.model_path, weights_only=True)
    model_hps = checkpoint['config']
    model = Tem1MaskedResidueGNN(
        node_in_channels=model_hps['node_in_channels'],
        edge_features_dim=model_hps['edge_features_dim'],
        hidden_channels=model_hps['hidden_channels'],
        head_hidden_channels=model_hps['head_hidden_channels'],
        mp_layers=model_hps['mp_layers'],
        head_layers=model_hps['head_layers'],
        dropout=model_hps.get('dropout', 0.0),
        encoder_rounds = model_hps['encoder_rounds'],
        decoder_rounds = model_hps['decoder_rounds']
    )
    model.load_state_dict(checkpoint['model_state'])

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scores, fitness, accuracy = run_zero_shot_inference(model, data_loader, device)

    # dataset.dms_data has the same row order as data_loader's output (shuffle=False)
    results_df = dataset.dms_data[["Single", "Code", "Fitness"]].copy()
    results_df["zero_shot_score"] = scores

    pearson_r = safe_pearson(results_df["Fitness"], results_df["zero_shot_score"])
    spearman_rho = safe_spearman(results_df["Fitness"], results_df["zero_shot_score"])
    print(f"n={len(results_df)}, accuracy: {accuracy:.4f}, Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(args.output_dir, "zero_shot_predictions.csv"), index=False)
        with open(os.path.join(args.output_dir, "zero_shot_summary.json"), "w") as f:
            json.dump({
                "n_samples": int(len(results_df)),
                "accuracy": accuracy,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
            }, f, indent=2)
        print(f"Saved zero-shot results to {args.output_dir}")
