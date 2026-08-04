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
from data.data_utils import load_cif_structure, parse_structure
from data.split import split_by_structural_position
from model.mlp import MLP
from train.train_utils import standardize


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

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    mean_loss = total_loss / total_samples
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return mean_loss, predictions, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dms", type=str, required=True, help="Path to the dms directory.")
    parser.add_argument("--pdb_id", type=str, default="1BTL", help="PDB ID for the protein structure.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./src/train/base/output/inference", help="Directory to save inference results.")

    # USE the same split seed. Should be in the config.json
    parser.add_argument("--train_size", type=float, default=0.8, help="Fraction of data used for training.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of data used for validation.")
    parser.add_argument("--seed", type=int, default=1012, help="Random seed used for the train/val/test split.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to run inference on.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--dont_save_results", action="store_true", help="Save loss and predictions (normalized and unscaled) to output_dir.")

    args = parser.parse_args()

    # data loading
    dms = pd.read_csv(args.processed_dms + "/dms_processed.csv")
    # must mirror baseline.py exactly, or the "held out" split is not the held out split
    pdb_dir = os.path.dirname(args.processed_dms.rstrip("/\\"))
    wt_sequence, _ = parse_structure(load_cif_structure(os.path.join(pdb_dir, f"{args.pdb_id}.cif"), args.pdb_id))
    train_df, val_df, test_df = split_by_structural_position(dms, wt_sequence, train_frac=args.train_size, val_frac=args.val_size, seed=args.seed)
    train_df, val_df, test_df, (train_mean, train_std) = standardize(train_df, val_df, test_df)

    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    dataset = MLPDataset(dms_data=split_df, pdb_id=args.pdb_id)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model load
    checkpoint = torch.load(args.model_path, weights_only=True)
    model_hps = checkpoint['config']
    model = MLP(input_dim=model_hps['input_dim'], bottleneck_hidden_dim=model_hps['bottleneck_hidden_dim'])
    model.load_state_dict(checkpoint['model_state'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = torch.nn.MSELoss(reduction='mean')
    loss, predictions, targets = run_inference(model, data_loader, criterion, device)
    print(f"{args.split} loss: {loss:.6f}")

    if not args.dont_save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # invert
        predictions_unscaled = predictions * train_std + train_mean
        targets_unscaled = targets * train_std + train_mean

        results_df = pd.DataFrame({
            "target_normalized": targets,
            "target_original": targets_unscaled,
            "prediction_normalized": predictions,
            "prediction_original": predictions_unscaled,
        })
        results_df.to_csv(os.path.join(args.output_dir, f"{args.split}_predictions.csv"), index=False)

        with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w") as f:
            json.dump({"split": args.split, "loss": loss, "n_samples": int(len(predictions))}, f, indent=2)

        print(f"saved inference results to {args.output_dir}")
