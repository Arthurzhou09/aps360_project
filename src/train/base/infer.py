import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.tem_beta import MLPDataset
from data.data_utils import load_cif_structure, parse_structure, load_dms
from data.split import split_by_random, split_by_structural_position, held_out_doubles
from model.mlp import MLP
from train.train_utils import standardize, safe_pearson, safe_spearman


def run_inference(model, data_loader, criterion, device):
    """
    infer some stuff. use "mean" on mse to match for now.
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
            x, y, g = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            groups.append(g.numpy().reshape(-1))

    mean_loss = total_loss / total_samples
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    groups = np.concatenate(groups)

    return mean_loss, predictions, targets, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--aa_features", type=str, default=None, choices=["aaindex8", "pca19"], help="Per-residue property set. Must match what the checkpoint was trained with.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./src/train/base/inference", help="Directory to save inference results.")

    # the split has to be reproduced exactly or the "held out" split is not held out. Pass the
    # same --config baseline.py was given and these are read from it instead of retyped.
    parser.add_argument("--config", type=str, default=None, help="The config used for training; fills the split settings below.")
    parser.add_argument("--train_size", type=float, default=None, help="Fraction of data used for training.")
    parser.add_argument("--val_size", type=float, default=None, help="Fraction of data used for validation.")
    parser.add_argument("--n_blocks", type=int, default=None, help="Contiguous residue blocks used for the split.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed used for the train/val/test split.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "double", "random"], help="Which split to run inference on. 'double' scores held-out DOUBLE mutants and needs --processed_dms to point at the combined 'both' directory.")
    parser.add_argument("--doubles_scope", type=str, default="unseen", choices=["unseen", "all"], help="--split double only. 'unseen' keeps doubles with neither residue in the training split; 'all' keeps every aligned double (leaky, for contrast).")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--dont_save_results", action="store_true", help="Save loss and predictions (normalized and unscaled) to output_dir.")

    args = parser.parse_args()

    # resolve the split settings the same way baseline.py does, so --config alone is enough
    # to reproduce the training split
    settings = {"train_size": 0.8, "val_size": 0.1, "n_blocks": None, "aa_features": "aaindex8", "seed": 1012}
    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        for key in ("train_size", "val_size", "n_blocks", "aa_features"):
            if key in cfg_json.get("data", {}):
                settings[key] = cfg_json["data"][key]
        if "seed" in cfg_json:
            settings["seed"] = cfg_json["seed"]
    for key in ("train_size", "val_size", "n_blocks", "aa_features", "seed"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)

    # data loading: must mirror baseline.py exactly, or the "held out" split is not the held out split
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    dms = load_dms(args.processed_dms, wt_sequence)
    if args.split == "random":
        print("random split")
        train_df, val_df, test_df = split_by_random(dms, train_frac=settings["train_size"], val_frac=settings["val_size"], seed=args.seed)
        _, _, split_df, (train_mean, train_std) = standardize(train_df, val_df, test_df)
        print(f"n = {len(split_df)}")
    elif args.split == "double":
        # doubles the singles-trained model never saw, plus the singles train split their
        # labels are standardized against, so the MSE stays on the same scale as the
        # train/val/test numbers. Identical row selection to gnn/infer.py by construction:
        # both call held_out_doubles with the split settings their checkpoint was trained on.
        split_df, reference_train = held_out_doubles(
            dms, wt_sequence, train_frac=settings["train_size"], val_frac=settings["val_size"],
            seed=settings["seed"], n_blocks=settings["n_blocks"], scope=args.doubles_scope)
        _, split_df, _, (train_mean, train_std) = standardize(reference_train, split_df, None)
    else:
        train_df, val_df, test_df = split_by_structural_position(
            dms, wt_sequence, train_frac=settings["train_size"], val_frac=settings["val_size"],
            seed=settings["seed"], n_blocks=settings["n_blocks"])
        train_df, val_df, test_df, (train_mean, train_std) = standardize(train_df, val_df, test_df)
        split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    dataset = MLPDataset(dms_data=split_df, pdb_id=args.pdb_id, aa_features=settings["aa_features"])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model load
    checkpoint = torch.load(args.model_path, weights_only=True)
    model_hps = checkpoint['config']
    model = MLP(input_dim=model_hps['input_dim'], bottleneck_hidden_dim=model_hps['bottleneck_hidden_dim'])
    if model_hps['input_dim'] != dataset.feature_dim:
        raise ValueError(f"checkpoint expects {model_hps['input_dim']} features but MLPDataset produces "
                         f"{dataset.feature_dim} for aa_features={settings['aa_features']!r}; "
                         f"pass the --config or --aa_features the checkpoint was trained with")
    model.load_state_dict(checkpoint['model_state'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = torch.nn.MSELoss(reduction='mean')
    loss, predictions, targets, groups = run_inference(model, data_loader, criterion, device)

    # same metrics gnn/infer.py reports, so the baseline and the GNN are directly comparable
    pearson_r = safe_pearson(predictions, targets)
    spearman_rho = safe_spearman(predictions, targets)
    baseline_mse = float(((targets - targets.mean()) ** 2).mean())
    print(f"{args.split} loss: {loss:.6f} (constant-predictor baseline {baseline_mse:.6f}), "
          f"Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}, n={len(predictions)}")

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
        per_group["spearman_count_only"] = safe_spearman(groups.astype(float), targets)
        print(f"  balanced (mean of the two): {per_group['spearman_balanced']:.4f}")
        print(f"  count-only trivial baseline on combined: {per_group['spearman_count_only']:.4f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # invert
        predictions_unscaled = predictions * train_std + train_mean
        targets_unscaled = targets * train_std + train_mean

        # 'Single' and 'Code' identify the row, so these predictions can be merged against
        # gnn/infer.py's on the same split - the point of scoring both models on one
        # held-out double set. Without a key the two files can only be compared in aggregate.
        # dataset.dms is in loader order because the loader runs with shuffle=False.
        results_df = dataset.dms[["Single", "Code"]].copy()
        results_df["Fitness"] = targets_unscaled
        results_df["baseline_score"] = predictions_unscaled
        results_df["target_normalized"] = targets
        results_df["target_original"] = targets_unscaled
        results_df["prediction_normalized"] = predictions
        results_df["prediction_original"] = predictions_unscaled
        results_df.to_csv(os.path.join(args.output_dir, f"{args.split}_predictions.csv"), index=False)

        with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w") as f:
            json.dump({"split": args.split, "loss": loss, "baseline_loss": baseline_mse,
                       "pearson_r": pearson_r, "spearman_rho": spearman_rho,
                       "n_samples": int(len(predictions)), **per_group}, f, indent=2)

        print(f"saved inference results to {args.output_dir}")
