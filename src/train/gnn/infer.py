import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import Tem1BetaLactamaseDataset
from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_structural_position
from model.gnn import Tem1BetaGNN
from train.train_utils import standardize, safe_pearson, safe_spearman
from train.gnn.run import load_config


def run_inference(model, data_loader, criterion, device):
    """
    inference some stuff. use "mean" on mse to match for now.
    returns:
        mean_loss: average loss over the dataset
        predictions: (N,) array of model predictions, in the same (min-max normalized) scale as training
        targets: (N,) array of ground-truth fitness values, in the same normalized scale
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    groups = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch.node_features, batch.edge_index, batch.distance_features, batch.batch, batch.mutation_idx).squeeze(-1)
            loss = criterion(pred, batch.fitness)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            predictions.append(pred.cpu().numpy())
            targets.append(batch.fitness.cpu().numpy())
            groups.append(batch.is_single.cpu().numpy().reshape(-1))

    mean_loss = total_loss / total_samples
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    groups = np.concatenate(groups)
    return mean_loss, predictions, targets, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration JSON file (same one used for training).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")

    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--directed", action="store_true", help="Whether to create directed edges in the graph. Must match training. False if not specified.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed used for the train/val/test split. Must match training.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to run inference on.")

    parser.add_argument("--output_dir", type=str, default="./src/train/gnn/output/inference", help="Directory to save inference results.")
    parser.add_argument("--dont_save_results", action="store_true", help="Save loss and predictions (normalized and unscaled) to output_dir.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference. Must match training.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # data load: must mirror run.py exactly or the "held out" split is not the held out split
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    dms = load_dms(args.processed_dms, wt_sequence)
    train_df, val_df, test_df = split_by_structural_position(
        dms, wt_sequence, train_frac=cfg.data.train_size, val_frac=cfg.data.val_size, seed=args.seed,
        n_blocks=getattr(cfg.data, "n_blocks", None))
    train_df, val_df, test_df, (train_mean, train_std) = standardize(train_df, val_df, test_df)

    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    dataset = Tem1BetaLactamaseDataset(dms_data=split_df, pdb_id=args.pdb_id, directed=args.directed, max_neighbours=cfg.data.neighbours, radius=getattr(cfg.data, "radius", None), dci_k=getattr(cfg.data, "dci_k", None), dci_threshold=getattr(cfg.data, "dci_threshold", None), dci_add_spatial=getattr(cfg.data, "dci_add_spatial", False),
                                       aa_features=getattr(cfg.data, "aa_features", "aaindex8"))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model load
    checkpoint = torch.load(args.model_path, weights_only=True)
    model_hps = checkpoint['config']
    model = Tem1BetaGNN(
        node_in_channels=model_hps['node_in_channels'],
        edge_features_dim=model_hps['edge_features_dim'],
        hidden_channels=model_hps['hidden_channels'],
        reg_hidden_channels=model_hps['reg_hidden_channels'],
        mp_layers=model_hps['mp_layers'],
        head_layers=model_hps['head_layers'],
        dropout=model_hps.get('dropout', 0.0), #some old models without dropout
        # checkpoints predating mp_rounds have neither key and were single-round
        #encoder_rounds=model_hps['encoder_rounds'],
        #decoder_rounds=model_hps['decoder_rounds'],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = torch.nn.MSELoss(reduction='mean')
    loss, predictions, targets, groups = run_inference(model, data_loader, criterion, device)

    # Spearman is the headline number: it is what the evolutionary PSSM baseline and the
    # zero-shot self-supervised scores are reported on, so it is the only directly
    # comparable metric across the three. MSE here is on standardized labels, so a
    # constant predictor scores ~1.0 and the value reads as 1 - R^2.
    pearson_r = safe_pearson(predictions, targets)
    spearman_rho = safe_spearman(predictions, targets)
    baseline_mse = float(((targets - targets.mean()) ** 2).mean())
    print(f"{args.split} loss: {loss:.6f} (constant-predictor baseline {baseline_mse:.6f}), "
          f"Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}, n={len(predictions)}")

    # per-group, because the combined figure above is partly earned by separating singles
    # from doubles - counting mutations alone scores ~0.32 on the combined test split
    per_group = {}
    for name, flag in (("single", 1), ("double", 0)):
        sub = groups == flag
        if sub.sum() > 1:
            per_group[f"spearman_{name}"] = safe_spearman(predictions[sub], targets[sub])
            per_group[f"n_{name}"] = int(sub.sum())
            print(f"  {name:6s} n={int(sub.sum()):5d}  Spearman rho: {per_group[f'spearman_{name}']:.4f}")
    both = [v for k, v in per_group.items() if k.startswith("spearman_")]
    if len(both) == 2:
        per_group["spearman_balanced"] = float(np.mean(both))
        # what a predictor that only counts mutations would score, as a floor for the combined number
        per_group["spearman_count_only"] = safe_spearman(groups.astype(float), targets)
        print(f"  balanced (mean of the two): {per_group['spearman_balanced']:.4f}")
        print(f"  count-only trivial baseline on combined: {per_group['spearman_count_only']:.4f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # invert the standardization so 'Fitness' is on the same raw scale as the zero-shot
        # and evolutionary prediction files (neither of those standardizes at all)
        predictions_unscaled = predictions * train_std + train_mean
        targets_unscaled = targets * train_std + train_mean

        # Schema matches gnn/infer_zero_shot.py and evolutionary/infer.py: ['Single', 'Code',
        # 'Fitness', <model>_score]. Without 'Code' these predictions cannot be merged with
        # the other two, so any paired model-vs-model comparison has to recompute them by
        # hand. dataset.dms is in loader order because the loader runs with shuffle=False.
        results_df = dataset.dms[["Single", "Code"]].copy()
        results_df["Fitness"] = targets_unscaled
        results_df["supervised_score"] = predictions_unscaled
        # kept for anything already reading these names
        results_df["target_normalized"] = targets
        results_df["target_original"] = targets_unscaled
        results_df["prediction_normalized"] = predictions
        results_df["prediction_original"] = predictions_unscaled
        results_df.to_csv(os.path.join(args.output_dir, f"{args.split}_predictions.csv"), index=False)

        with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w") as f:
            json.dump({"split": args.split, "loss": loss, "baseline_loss": baseline_mse,
                       "pearson_r": pearson_r, "spearman_rho": spearman_rho,
                       "n_samples": int(len(predictions)), **per_group}, f, indent=2)

        print(f"Saved inference results to {args.output_dir}")
