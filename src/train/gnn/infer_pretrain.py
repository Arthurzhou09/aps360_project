import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_class import RESIDUE_LETTERS
from data.homolog import HomologMaskedDataset
from data.split import split_by_cluster
from model.gnn import Tem1MaskedResidueGNN
from train.train_utils import load_config


def run_inference(model, data_loader, device):
    """
    Inference for the masked-residue model. Applies softmax over the per-node logits,
    then reads off predictions only at each graph's masked position (labels != -100).
    returns:
        mean_nll: cross-entropy of the true residue, averaged over masked positions
        probs: (N, 20) array of softmax probabilities at each masked position
        true_labels: (N,) array of true amino acid indices (0-19) at each masked position
    """
    model.to(device)
    model.eval()

    total_nll = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch.node_features, batch.edge_index, batch.distance_features)

            mask = batch.labels != -100 # one masked position per graph
            masked_logits = logits[mask]
            masked_labels = batch.labels[mask]

            nll = torch.nn.functional.cross_entropy(masked_logits, masked_labels, reduction='sum')
            total_nll += nll.item()
            total_samples += masked_labels.numel()

            probs = torch.softmax(masked_logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(masked_labels.cpu().numpy())

    mean_nll = total_nll / total_samples
    probs = np.concatenate(all_probs, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    return mean_nll, probs, true_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, required=True, help="Path to the directory containing homolog_processed.csv.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file (same one used for pretraining).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. Must match training. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed used for the train/val/test split. Must match training.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to run inference on.")
    parser.add_argument("--top_k", type=int, default=3, help="k for top-k accuracy.")

    parser.add_argument("--output_dir", type=str, default="./src/train/gnn/output_pretrain/inference", help="Directory to save inference results.")
    parser.add_argument("--dont_save_results", action="store_true", help="Skip saving predictions/summary to output_dir.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference. Must match training.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # data load: same cluster-aware split as pretrain.py (same cache file), so args.split == "test" is genuinely held out.
    homolog_df = pd.read_csv(args.processed + "/homolog_processed.csv")
    cluster_cache_path = os.path.join(args.processed, "homolog_clusters_cache.csv")
    train_df, val_df, test_df = split_by_cluster(homolog_df, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
                                                   k=cfg.data.cluster_kmer_k, threshold=cfg.data.cluster_similarity_threshold,
                                                   cache_path=cluster_cache_path)

    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]

    print(f"Running inference on {args.split} split with {len(split_df)} homologs.")

    dataset = HomologMaskedDataset(split_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, seed=args.seed)
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
    )
    model.load_state_dict(checkpoint['model_state'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mean_nll, probs, true_labels = run_inference(model, data_loader, device)

    predictions = np.argmax(probs, axis=-1)
    accuracy = float(np.mean(predictions == true_labels))

    top_k_preds = np.argsort(probs, axis=-1)[:, -args.top_k:]
    top_k_accuracy = float(np.mean([label in row for label, row in zip(true_labels, top_k_preds)]))

    true_aa_probability = probs[np.arange(len(true_labels)), true_labels]
    perplexity = float(np.exp(mean_nll))

    print(f"{args.split} accuracy: {accuracy:.4f}, top_{args.top_k}_accuracy: {top_k_accuracy:.4f}, "
          f"nll: {mean_nll:.4f}, perplexity: {perplexity:.4f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        results_df = pd.DataFrame({
            "true_aa": [RESIDUE_LETTERS[i] for i in true_labels],
            "predicted_aa": [RESIDUE_LETTERS[i] for i in predictions],
            "true_aa_probability": true_aa_probability,
            "correct": predictions == true_labels,
        })
        # full softmax distribution, one column per amino acid, for downstream analysis
        prob_df = pd.DataFrame(probs, columns=[f"prob_{aa}" for aa in RESIDUE_LETTERS])
        results_df = pd.concat([results_df, prob_df], axis=1)
        results_df.to_csv(os.path.join(args.output_dir, f"{args.split}_predictions.csv"), index=False)

        with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w") as f:
            json.dump({
                "split": args.split,
                "accuracy": accuracy,
                f"top_{args.top_k}_accuracy": top_k_accuracy,
                "mean_nll": mean_nll,
                "perplexity": perplexity,
                "n_samples": int(len(true_labels)),
            }, f, indent=2)

        print(f"Saved inference results to {args.output_dir}")
